import argparse
import os
import re
import shutil
import sys
import warnings
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import qr
from scipy.special import expit, logit
from scipy.stats import chi2, norm, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score
from sklearn.preprocessing import RobustScaler
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

TRAIN_SIZE = 0.60
VALIDATION_SIZE = 0.20
RARE_LEVEL_MIN_COUNT = 30
SLA_GRACE_STEP_HOURS = 0.25
SLA_GRACE_MAX_HOURS = 24.0
CANCELLATION_PENALTY_QUANTILE = 0.95
MIN_POSITIVE_BURDEN_HOURS = 0.25
TOP_K_RATES = [0.05, 0.10, 0.20, 0.30]
UTILITY_FALSE_ALERT_COSTS = [1, 3, 6, 12, 24]
CALIBRATION_GROUPS = 10
CONFORMAL_ALPHA = 0.10
TRIAGE_MARGINS = [0.01, 0.02, 0.05, 0.10]
BORDERLINE_CASE_MARGIN = 0.05
MAX_BORDERLINE_CASES = 5000
RANDOM_STATE = 2026


def snake(x):
    x = str(x).strip()
    x = re.sub(r"[()]+", "", x)
    x = re.sub(r"[^0-9a-zA-Z]+", "_", x)
    x = re.sub(r"_+", "_", x)
    return x.strip("_").lower()


def clean_text(s):
    z = s.astype(str).str.strip().replace(["nan", "None", "none", "NULL", "null", "", "NaN"], np.nan)
    return z.map(lambda v: snake(v) if pd.notna(v) else np.nan)


def parse_dt(s):
    z = pd.to_datetime(s, format="%m/%d/%Y %H:%M", errors="coerce")
    if z.notna().mean() < 0.95:
        z = pd.to_datetime(s, errors="coerce")
    return z


def prob_clip(p):
    return np.clip(np.asarray(p, dtype=float), 1e-8, 1 - 1e-8)


def auc_rank(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return np.nan
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision_rank(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    n_pos = int(y.sum())
    if n_pos == 0:
        return np.nan
    order = np.argsort(-p)
    ys = y[order]
    tp = np.cumsum(ys)
    precision = tp / (np.arange(len(y)) + 1)
    return float((precision * ys).sum() / n_pos)


def safe_spearman(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if len(y) < 3 or np.std(y) == 0 or np.std(p) == 0:
        return np.nan, np.nan
    r, pv = spearmanr(y, p)
    return float(r), float(pv)


def modal(frame, order_col, value_col):
    if value_col not in frame.columns:
        return pd.DataFrame({order_col: frame[order_col].drop_duplicates(), value_col: "unknown"})
    out = frame.dropna(subset=[value_col]).groupby([order_col, value_col], sort=False).size().reset_index(name="n")
    out = out.sort_values([order_col, "n", value_col], ascending=[True, False, True]).drop_duplicates(order_col)
    return out[[order_col, value_col]]


def missingness_table(frame, stage):
    n = len(frame)
    rows = []
    for c in frame.columns:
        miss = int(frame[c].isna().sum())
        rows.append({"stage": stage, "column": c, "missing_n": miss, "missing_rate": float(miss / n) if n else np.nan})
    return pd.DataFrame(rows)


def data_quality_line_table(frame, stage):
    rows = []
    def add(item, value):
        rows.append({"stage": stage, "item": item, "value": int(value) if isinstance(value, (np.integer, int, bool)) else value})
    add("rows", len(frame))
    add("columns", frame.shape[1])
    add("duplicate_line_rows", int(frame.duplicated().sum()))
    if "order_id" in frame.columns:
        add("unique_order_ids", int(frame["order_id"].nunique(dropna=True)))
        add("missing_order_id", int(frame["order_id"].isna().sum()))
    if "order_item_quantity" in frame.columns:
        add("quantity_nonpositive", int(pd.to_numeric(frame["order_item_quantity"], errors="coerce").le(0).sum()))
    if "order_item_product_price" in frame.columns:
        add("product_price_nonpositive", int(pd.to_numeric(frame["order_item_product_price"], errors="coerce").le(0).sum()))
    if "order_item_discount_rate" in frame.columns:
        d = pd.to_numeric(frame["order_item_discount_rate"], errors="coerce")
        add("discount_outside_0_1", int(((d < 0) | (d > 1)).sum()))
    if "days_for_shipping_real" in frame.columns:
        add("real_shipping_days_negative", int(pd.to_numeric(frame["days_for_shipping_real"], errors="coerce").lt(0).sum()))
    if "days_for_shipment_scheduled" in frame.columns:
        add("scheduled_shipping_days_negative", int(pd.to_numeric(frame["days_for_shipment_scheduled"], errors="coerce").lt(0).sum()))
    if "order_date_dateorders" in frame.columns and "shipping_date_dateorders" in frame.columns:
        od = frame["order_date_dateorders"]
        sd = frame["shipping_date_dateorders"]
        if not pd.api.types.is_datetime64_any_dtype(od):
            od = parse_dt(od)
        if not pd.api.types.is_datetime64_any_dtype(sd):
            sd = parse_dt(sd)
        add("missing_order_datetime", int(od.isna().sum()))
        add("missing_shipping_datetime", int(sd.isna().sum()))
        add("shipping_before_order", int((sd < od).sum()))
    return pd.DataFrame(rows)


def data_quality_order_table(orders, stage):
    rows = []
    def add(item, value):
        rows.append({"stage": stage, "item": item, "value": int(value) if isinstance(value, (np.integer, int, bool)) else value})
    add("orders", len(orders))
    add("duplicate_order_id_rows", int(orders["order_id"].duplicated().sum()) if "order_id" in orders.columns else 0)
    if "elapsed_hours_timestamp" in orders.columns:
        add("negative_elapsed_hours", int(orders["elapsed_hours_timestamp"].lt(0).sum()))
        add("missing_elapsed_hours", int(orders["elapsed_hours_timestamp"].isna().sum()))
    if "actual_shipping_hours" in orders.columns:
        add("actual_shipping_hours_negative", int(orders["actual_shipping_hours"].lt(0).sum()))
        add("actual_shipping_hours_over_30_days", int(orders["actual_shipping_hours"].gt(30 * 24).sum()))
    if "scheduled_shipping_hours" in orders.columns:
        add("scheduled_shipping_hours_negative", int(orders["scheduled_shipping_hours"].lt(0).sum()))
    if "order_lines" in orders.columns:
        add("orders_with_more_than_10_lines", int(orders["order_lines"].gt(10).sum()))
    if "quantity_sum" in orders.columns:
        add("quantity_sum_nonpositive", int(orders["quantity_sum"].le(0).sum()))
    if "discount_rate_mean" in orders.columns:
        add("mean_discount_outside_0_1", int(((orders["discount_rate_mean"] < 0) | (orders["discount_rate_mean"] > 1)).sum()))
    return pd.DataFrame(rows)


def calibrate_sla(train):
    rows = []
    grid = np.arange(0, SLA_GRACE_MAX_HOURS + SLA_GRACE_STEP_HOURS / 2, SLA_GRACE_STEP_HOURS)
    modes = sorted(train["shipping_mode"].dropna().astype(str).unique())
    for mode in modes:
        g = train[(train["shipping_mode"].astype(str) == mode) & (~train["delivery_status"].eq("shipping_canceled")) & (~train["order_status"].isin(["canceled", "suspected_fraud"]))].copy()
        if len(g) == 0:
            rows.append({"shipping_mode": mode, "scheduled_days_mode": 0.0, "grace_hours": 0.0, "agreement": np.nan, "mcc": np.nan, "n_train_delivered": 0, "raw_late_rate": np.nan})
            continue
        y = g["delivery_status"].eq("late_delivery").astype(int).to_numpy()
        actual = g["actual_shipping_hours"].to_numpy(float)
        scheduled = g["scheduled_shipping_hours"].to_numpy(float)
        best = (0.0, -1.0, -1.0)
        for theta in grid:
            pred = (actual > scheduled + theta).astype(int)
            agreement = float((pred == y).mean())
            mcc = float(matthews_corrcoef(y, pred)) if len(np.unique(y)) == 2 and len(np.unique(pred)) == 2 else 0.0
            if agreement > best[1] or (abs(agreement - best[1]) <= 1e-12 and mcc > best[2]):
                best = (float(theta), agreement, mcc)
        md = g["days_for_shipment_scheduled"].mode()
        rows.append({"shipping_mode": mode, "scheduled_days_mode": float(md.iloc[0]) if len(md) else np.nan, "grace_hours": best[0], "agreement": best[1], "mcc": best[2], "n_train_delivered": int(len(g)), "raw_late_rate": float(y.mean())})
    out = pd.DataFrame(rows)
    out["grace_hours"] = out["grace_hours"].fillna(0.0)
    return out


def apply_outcome(frame, sla, penalty=None, min_positive=None):
    x = frame.merge(sla[["shipping_mode", "grace_hours"]], on="shipping_mode", how="left")
    x["grace_hours"] = x["grace_hours"].fillna(0.0)
    x["deadline_hours"] = x["scheduled_shipping_hours"] + x["grace_hours"]
    x["lateness_hours"] = np.maximum(0.0, x["actual_shipping_hours"] - x["deadline_hours"])
    x["lateness_minutes"] = x["lateness_hours"] * 60
    x["is_canceled"] = (x["delivery_status"].eq("shipping_canceled") | x["order_status"].isin(["canceled", "suspected_fraud"])).astype(int)
    x["raw_late_status"] = x["delivery_status"].eq("late_delivery").astype(int)
    x["reconstructed_late"] = x["lateness_hours"].gt(0).astype(int)
    x["problem_delivery"] = ((x["is_canceled"].eq(1)) | (x["raw_late_status"].eq(1)) | (x["reconstructed_late"].eq(1))).astype(int)
    if min_positive is None:
        pos = x.loc[x["is_canceled"].eq(0) & x["lateness_hours"].gt(0), "lateness_hours"]
        min_positive = max(MIN_POSITIVE_BURDEN_HOURS, float(pos.min())) if len(pos) else MIN_POSITIVE_BURDEN_HOURS
    x["delivered_problem_burden_hours"] = 0.0
    m = x["is_canceled"].eq(0) & x["problem_delivery"].eq(1)
    x.loc[m, "delivered_problem_burden_hours"] = np.maximum(x.loc[m, "lateness_hours"].to_numpy(float), float(min_positive))
    if penalty is None:
        posb = x.loc[m, "delivered_problem_burden_hours"]
        penalty = float(np.nanquantile(posb, CANCELLATION_PENALTY_QUANTILE)) if len(posb) else 24.0
        if not np.isfinite(penalty) or penalty <= 0:
            penalty = 24.0
    x["cancellation_penalty_hours"] = float(penalty)
    x["severity_hours_if_problem"] = np.where(x["is_canceled"].eq(1), float(penalty), x["delivered_problem_burden_hours"])
    x.loc[x["problem_delivery"].eq(0), "severity_hours_if_problem"] = np.nan
    x["delivery_burden_hours"] = np.where(x["problem_delivery"].eq(1), x["severity_hours_if_problem"], 0.0)
    x["delivery_burden_minutes"] = x["delivery_burden_hours"] * 60
    x["clean_delivery_outcome"] = np.select([x["is_canceled"].eq(1), x["problem_delivery"].eq(1) & x["is_canceled"].eq(0)], ["canceled", "late_delivered"], default="non_late_delivered")
    labels = []
    for c, problem, h in zip(x["is_canceled"].to_numpy(), x["problem_delivery"].to_numpy(), x["delivered_problem_burden_hours"].to_numpy()):
        if c == 1:
            labels.append("canceled")
        elif problem == 0:
            labels.append("non_late")
        elif h <= 6:
            labels.append("late_0_6h")
        elif h <= 24:
            labels.append("late_6_24h")
        elif h <= 48:
            labels.append("late_1_2d")
        elif h <= 96:
            labels.append("late_2_4d")
        else:
            labels.append("late_over_4d")
    cats = ["non_late", "late_0_6h", "late_6_24h", "late_1_2d", "late_2_4d", "late_over_4d", "canceled"]
    x["delay_duration_band"] = pd.Categorical(labels, categories=cats, ordered=True)
    return x, float(penalty), float(min_positive)


def add_time_features(x):
    x = x.copy()
    month = x["order_date"].dt.month
    dow = x["order_date"].dt.dayofweek
    hour = x["order_date"].dt.hour
    x["month_sin"] = np.sin(2 * np.pi * month / 12)
    x["month_cos"] = np.cos(2 * np.pi * month / 12)
    x["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    x["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    x["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    x["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return x


def build_dummies(train, frames, cols):
    tr = pd.DataFrame(index=train.index)
    outs = [pd.DataFrame(index=f.index) for f in frames]
    refs = []
    unseen_rows = []
    for col in cols:
        counts = train[col].dropna().astype(str).value_counts()
        if len(counts) == 0:
            continue
        frequent = counts[counts >= RARE_LEVEL_MIN_COUNT].index.tolist()
        if not frequent:
            frequent = [counts.index[0]]
        ref = counts.index[0] if counts.index[0] in frequent else frequent[0]
        levels = [ref] + sorted([v for v in frequent if v != ref]) + ["__other__"]
        refs.append({"variable": col, "reference_level": ref, "raw_train_levels": int(len(counts)), "encoded_levels": int(len(levels)), "rare_levels_grouped": int((counts < RARE_LEVEL_MIN_COUNT).sum())})
        mapped = train[col].astype(str).where(train[col].astype(str).isin(frequent), "__other__")
        for level in levels[1:]:
            tr[f"{col}_{level}"] = mapped.eq(level).astype(float).to_numpy()
        train_levels = set(counts.index.tolist())
        for split_name, f, out in zip(["validation", "test"], frames, outs):
            raw_levels = set(f[col].dropna().astype(str).unique())
            unseen = sorted(list(raw_levels - train_levels))
            unseen_rows.append({"split": split_name, "variable": col, "unseen_levels_n": len(unseen), "unseen_rows_n": int(f[col].astype(str).isin(unseen).sum()) if unseen else 0, "unseen_levels": "|".join(unseen)})
            mapped_f = f[col].astype(str).where(f[col].astype(str).isin(frequent), "__other__")
            for level in levels[1:]:
                out[f"{col}_{level}"] = mapped_f.eq(level).astype(float).to_numpy()
    return tr, outs, pd.DataFrame(refs), pd.DataFrame(unseen_rows)


def rank_prune(xtr, xva, xte):
    xtr = xtr.loc[:, xtr.std(ddof=0) > 0]
    xva = xva[xtr.columns]
    xte = xte[xtr.columns]
    a = xtr.to_numpy(float)
    _, r, piv = qr(a, mode="economic", pivoting=True)
    diag = np.abs(np.diag(r)) if r.size else np.array([])
    tol = np.finfo(float).eps * max(a.shape) * max(float(diag[0]), 1.0) if len(diag) else 0.0
    rank = int((diag > tol).sum())
    keep = sorted(piv[:rank])
    keep_cols = xtr.columns[keep].tolist()
    dropped = [c for c in xtr.columns if c not in keep_cols]
    return xtr[keep_cols], xva[keep_cols], xte[keep_cols], pd.DataFrame({"dropped_collinear_term": dropped})


def logistic_fit(y, x, c_value=10.0):
    xm = np.asarray(x, float)
    y = np.asarray(y, int)
    clf = LogisticRegression(fit_intercept=False, penalty="l2", C=c_value, solver="liblinear", max_iter=500, random_state=RANDOM_STATE)
    clf.fit(xm, y)
    beta = clf.coef_.ravel()
    p = expit(xm @ beta)
    w = np.clip(p * (1 - p), 1e-8, None)
    bread = np.linalg.pinv((xm.T * w) @ xm)
    scores = xm * (y - p)[:, None]
    n, k = xm.shape
    cov = bread @ (scores.T @ scores) @ bread * n / max(n - k, 1)
    return pd.Series(beta, index=x.columns), pd.DataFrame(cov, index=x.columns, columns=x.columns)


def linear_fit_log(y, x):
    xm = np.asarray(x, float)
    yy = np.log(np.asarray(y, float))
    xtx_inv = np.linalg.pinv(xm.T @ xm)
    beta = xtx_inv @ xm.T @ yy
    resid = yy - xm @ beta
    n, k = xm.shape
    cov = xtx_inv @ ((xm.T * resid ** 2) @ xm) @ xtx_inv * n / max(n - k, 1)
    smearing = float(np.mean(np.exp(resid)))
    return pd.Series(beta, index=x.columns), pd.DataFrame(cov, index=x.columns, columns=x.columns), smearing


def coef_table(beta, cov, model, effect_name):
    se = pd.Series(np.sqrt(np.maximum(np.diag(cov), 0)), index=beta.index)
    z = beta / se.replace(0, np.nan)
    p = pd.Series(2 * norm.sf(np.abs(z.to_numpy(float))), index=beta.index)
    out = pd.DataFrame({"model": model, "term": beta.index, "beta": beta.values, "se": se.values, "z": z.values, "p_value": p.values})
    out["beta_ci_low"] = out["beta"] - 1.96 * out["se"]
    out["beta_ci_high"] = out["beta"] + 1.96 * out["se"]
    out["abs_beta"] = out["beta"].abs()
    mask = out["term"].ne("const")
    out["p_holm"] = np.nan
    out["q_bh"] = np.nan
    out["reject_holm_05"] = False
    out["reject_bh_05"] = False
    if mask.sum():
        reject_holm, p_holm, _, _ = multipletests(out.loc[mask, "p_value"], method="holm")
        reject_bh, q_bh, _, _ = multipletests(out.loc[mask, "p_value"], method="fdr_bh")
        out.loc[mask, "p_holm"] = p_holm
        out.loc[mask, "q_bh"] = q_bh
        out.loc[mask, "reject_holm_05"] = reject_holm
        out.loc[mask, "reject_bh_05"] = reject_bh
    out[effect_name] = np.exp(np.clip(out["beta"], -700, 700))
    out[f"{effect_name}_ci_low"] = np.exp(np.clip(out["beta_ci_low"], -700, 700))
    out[f"{effect_name}_ci_high"] = np.exp(np.clip(out["beta_ci_high"], -700, 700))
    return out


def group_wald(beta, cov, groups, model):
    rows = []
    for name, terms in groups.items():
        terms = [t for t in terms if t in beta.index and t != "const"]
        if not terms:
            continue
        b = beta.loc[terms].to_numpy(float)
        v = cov.loc[terms, terms].to_numpy(float)
        stat = float(b.T @ np.linalg.pinv(v) @ b)
        df = int(len(terms))
        rows.append({"model": model, "factor": name, "df": df, "wald_chi2": stat, "p_value": float(chi2.sf(stat, df)), "mean_abs_beta": float(np.mean(np.abs(b)))})
    out = pd.DataFrame(rows)
    if len(out):
        out["reject_holm_05"], out["p_holm"], _, _ = multipletests(out["p_value"], method="holm")
        out["reject_bh_05"], out["q_bh"], _, _ = multipletests(out["p_value"], method="fdr_bh")
    return out


def metrics_binary(y, p, threshold, split, threshold_name="validation_mcc"):
    p = prob_clip(p)
    y = np.asarray(y, int)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"split": split, "threshold_name": threshold_name, "threshold": float(threshold), "n": int(len(y)), "prevalence": float(y.mean()), "auc": auc_rank(y, p), "average_precision": average_precision_rank(y, p), "accuracy": float(accuracy_score(y, pred)), "balanced_accuracy": float(balanced_accuracy_score(y, pred)), "precision": float(precision_score(y, pred, zero_division=0)), "recall": float(recall_score(y, pred, zero_division=0)), "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan, "npv": float(tn / (tn + fn)) if (tn + fn) else np.nan, "f1": float(f1_score(y, pred, zero_division=0)), "mcc": float(matthews_corrcoef(y, pred)) if len(np.unique(pred)) > 1 else 0.0, "brier_score": float(np.mean((p - y) ** 2)), "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def select_threshold(y, p):
    p = prob_clip(p)
    candidates = np.unique(np.r_[0, 0.5, np.quantile(p, np.linspace(0.01, 0.99, 149)), 1])
    rows = []
    for t in candidates:
        yhat = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
        rows.append({"threshold": float(t), "accuracy": accuracy_score(y, yhat), "balanced_accuracy": balanced_accuracy_score(y, yhat), "precision": precision_score(y, yhat, zero_division=0), "recall": recall_score(y, yhat, zero_division=0), "f1": f1_score(y, yhat, zero_division=0), "mcc": matthews_corrcoef(y, yhat) if len(np.unique(yhat)) > 1 else 0.0, "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    tab = pd.DataFrame(rows)
    return float(tab.loc[tab["mcc"].idxmax(), "threshold"]), tab


def calibration_table(y, p, split):
    d = pd.DataFrame({"y": np.asarray(y, int), "p": prob_clip(p)})
    d["bin"] = pd.qcut(d["p"].rank(method="first"), q=min(CALIBRATION_GROUPS, len(d)), labels=False, duplicates="drop") + 1
    tab = d.groupby("bin", as_index=False).agg(n=("y", "size"), observed_rate=("y", "mean"), mean_p=("p", "mean"), p_min=("p", "min"), p_max=("p", "max"))
    tab.insert(0, "split", split)
    return tab


def calibration_summary(y, p, split):
    tab = calibration_table(y, p, split)
    ece = float(np.sum(tab["n"] / tab["n"].sum() * np.abs(tab["observed_rate"] - tab["mean_p"])))
    mce = float(np.max(np.abs(tab["observed_rate"] - tab["mean_p"])))
    y = np.asarray(y, dtype=int)
    lp = logit(prob_clip(p))
    intercept, slope = np.nan, np.nan
    if len(np.unique(y)) == 2 and np.std(lp) > 0:
        xx = pd.DataFrame({"const": np.ones(len(lp)), "logit_p": lp})
        try:
            b, _ = logistic_fit(y, xx, c_value=1e6)
            intercept = float(b.loc["const"])
            slope = float(b.loc["logit_p"])
        except Exception:
            pass
    return {"split": split, "ece": ece, "mce": mce, "calibration_intercept": intercept, "calibration_slope": slope}


def regression_metrics(y, pred, split, target):
    y = np.asarray(y, dtype=float)
    pred = np.clip(np.asarray(pred, dtype=float), 1e-8, None)
    err = pred - y
    r, pv = safe_spearman(y, pred)
    return {"split": split, "target": target, "n": int(len(y)), "mean_observed_hours": float(np.mean(y)), "mean_predicted_hours": float(np.mean(pred)), "mean_prediction_ratio": float(np.mean(pred) / np.mean(y)) if np.mean(y) > 0 else np.nan, "mae_hours": float(np.mean(np.abs(err))), "rmse_hours": float(np.sqrt(np.mean(err ** 2))), "median_absolute_error_hours": float(np.median(np.abs(err))), "spearman_r": r, "spearman_p": pv}


def severity_calibration_table(frame, split):
    d = frame.loc[frame["problem_delivery"].eq(1), ["severity_hours_if_problem", "predicted_severity_hours_if_problem"]].copy()
    d = d.rename(columns={"severity_hours_if_problem": "observed", "predicted_severity_hours_if_problem": "predicted"})
    d["predicted"] = np.clip(d["predicted"], 1e-8, None)
    d["bin"] = pd.qcut(d["predicted"].rank(method="first"), q=min(CALIBRATION_GROUPS, len(d)), labels=False, duplicates="drop") + 1
    tab = d.groupby("bin", as_index=False).agg(n=("observed", "size"), mean_observed_hours=("observed", "mean"), mean_predicted_hours=("predicted", "mean"), median_observed_hours=("observed", "median"), pred_min=("predicted", "min"), pred_max=("predicted", "max"))
    tab.insert(0, "split", split)
    return tab


def severity_calibration_summary(frame, split):
    d = frame.loc[frame["problem_delivery"].eq(1), ["severity_hours_if_problem", "predicted_severity_hours_if_problem"]].copy()
    if len(d) < 5:
        return {"split": split, "intercept_log": np.nan, "slope_log": np.nan, "r2_log": np.nan}
    y = np.log(np.clip(d["severity_hours_if_problem"].to_numpy(float), 1e-8, None))
    x = np.column_stack([np.ones(len(d)), np.log(np.clip(d["predicted_severity_hours_if_problem"].to_numpy(float), 1e-8, None))])
    beta = np.linalg.pinv(x.T @ x) @ x.T @ y
    yhat = x @ beta
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {"split": split, "intercept_log": float(beta[0]), "slope_log": float(beta[1]), "r2_log": float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan}


def topk_table(frame, score, split):
    d = frame[["problem_delivery", "delivery_burden_hours", score]].sort_values(score, ascending=False)
    total_b = float(d["delivery_burden_hours"].sum())
    total_p = float(d["problem_delivery"].sum())
    mean_b = float(d["delivery_burden_hours"].mean())
    rows = []
    for k in TOP_K_RATES:
        n = max(1, int(np.ceil(k * len(d))))
        h = d.head(n)
        rows.append({"split": split, "score": score, "top_k_rate": k, "orders_flagged": n, "problem_rate": float(h["problem_delivery"].mean()), "mean_observed_burden_hours": float(h["delivery_burden_hours"].mean()), "captured_problem_share": float(h["problem_delivery"].sum() / total_p) if total_p else np.nan, "captured_burden_share": float(h["delivery_burden_hours"].sum() / total_b) if total_b else np.nan, "lift_over_mean_burden": float(h["delivery_burden_hours"].mean() / mean_b) if mean_b else np.nan})
    return pd.DataFrame(rows)


def utility_table(frame, score, split):
    d = frame[["problem_delivery", "delivery_burden_hours", score]].sort_values(score, ascending=False)
    rows = []
    for k in TOP_K_RATES:
        n = max(1, int(np.ceil(k * len(d))))
        h = d.head(n)
        captured = float(h["delivery_burden_hours"].sum())
        false_alert = int(h["problem_delivery"].eq(0).sum())
        for c in UTILITY_FALSE_ALERT_COSTS:
            rows.append({"split": split, "score": score, "top_k_rate": k, "orders_flagged": n, "captured_burden_hours": captured, "false_alerts": false_alert, "false_alert_cost_hours": c, "operational_utility_hours": captured - false_alert * c})
    return pd.DataFrame(rows)


def delay_bands_summary(frame, split):
    tab = frame.groupby("delay_duration_band", observed=False).agg(n=("order_id", "size"), problem_rate=("problem_delivery", "mean"), mean_lateness_hours=("lateness_hours", "mean"), mean_burden_hours=("delivery_burden_hours", "mean"), mean_p_problem=("p_problem", "mean"), mean_predicted_severity_hours=("predicted_severity_hours_if_problem", "mean"), mean_expected_burden_hours=("expected_burden_hours", "mean")).reset_index()
    tab.insert(0, "split", split)
    return tab


def risk_strata_table(frame, score, split, cuts):
    d = frame.copy()
    d["risk_stratum"] = pd.cut(d[score], [-np.inf, cuts[0], cuts[1], cuts[2], np.inf], labels=["low", "medium", "high", "very_high"])
    tab = d.groupby("risk_stratum", observed=False).agg(n=("order_id", "size"), problem_rate=("problem_delivery", "mean"), mean_score=(score, "mean"), mean_observed_burden_hours=("delivery_burden_hours", "mean"), median_observed_burden_hours=("delivery_burden_hours", "median")).reset_index()
    tab.insert(0, "split", split)
    return tab


def vif_sample(x):
    if len(x) == 0 or x.shape[1] == 0:
        return pd.DataFrame(columns=["term", "vif_ridge_sample"])
    z = x.sample(min(3000, len(x)), random_state=RANDOM_STATE).to_numpy(float)
    z = (z - z.mean(axis=0)) / np.maximum(z.std(axis=0), 1e-12)
    corr = np.nan_to_num(np.corrcoef(z, rowvar=False), nan=0.0, posinf=0.0, neginf=0.0)
    corr = corr + np.eye(corr.shape[0]) * 1e-6
    inv = np.linalg.pinv(corr)
    return pd.DataFrame({"term": x.columns, "vif_ridge_sample": np.diag(inv)})


def borderline_audit(frame, threshold, split):
    d = frame.copy()
    d["distance_to_threshold"] = np.abs(d["p_problem"] - threshold)
    rows = []
    base_pred = d["p_problem"].ge(threshold).astype(int)
    d["binary_correct"] = base_pred.eq(d["problem_delivery"].astype(int))
    for margin in TRIAGE_MARGINS:
        near = d[d["distance_to_threshold"].le(margin)]
        far = d[d["distance_to_threshold"].gt(margin)]
        rows.append({"split": split, "margin": margin, "zone": "near_threshold", "n": int(len(near)), "share": float(len(near) / len(d)) if len(d) else np.nan, "problem_rate": float(near["problem_delivery"].mean()) if len(near) else np.nan, "binary_accuracy": float(near["binary_correct"].mean()) if len(near) else np.nan, "error_rate": float(1 - near["binary_correct"].mean()) if len(near) else np.nan, "mean_expected_burden_hours": float(near["expected_burden_hours"].mean()) if len(near) else np.nan})
        rows.append({"split": split, "margin": margin, "zone": "outside_threshold_zone", "n": int(len(far)), "share": float(len(far) / len(d)) if len(d) else np.nan, "problem_rate": float(far["problem_delivery"].mean()) if len(far) else np.nan, "binary_accuracy": float(far["binary_correct"].mean()) if len(far) else np.nan, "error_rate": float(1 - far["binary_correct"].mean()) if len(far) else np.nan, "mean_expected_burden_hours": float(far["expected_burden_hours"].mean()) if len(far) else np.nan})
    return pd.DataFrame(rows)


def triage_metrics(frame, threshold, split):
    rows = []
    y = frame["problem_delivery"].astype(int).to_numpy()
    for margin in TRIAGE_MARGINS:
        p = frame["p_problem"].to_numpy(float)
        auto_alert = p >= threshold + margin
        auto_no_alert = p <= threshold - margin
        manual = ~(auto_alert | auto_no_alert)
        auto = auto_alert | auto_no_alert
        pred = np.where(auto_alert, 1, 0)
        row = {"split": split, "margin": margin, "threshold_low": float(threshold - margin), "threshold_high": float(threshold + margin), "n": int(len(frame)), "auto_decision_n": int(auto.sum()), "manual_review_n": int(manual.sum()), "manual_review_share": float(manual.mean()), "manual_review_problem_rate": float(y[manual].mean()) if manual.sum() else np.nan, "auto_alert_n": int(auto_alert.sum()), "auto_no_alert_n": int(auto_no_alert.sum())}
        if auto.sum():
            ya = y[auto]
            pa = pred[auto]
            tn, fp, fn, tp = confusion_matrix(ya, pa, labels=[0, 1]).ravel()
            row.update({"auto_accuracy": float(accuracy_score(ya, pa)), "auto_balanced_accuracy": float(balanced_accuracy_score(ya, pa)) if len(np.unique(ya)) == 2 else np.nan, "auto_precision": float(precision_score(ya, pa, zero_division=0)), "auto_recall": float(recall_score(ya, pa, zero_division=0)), "auto_specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan, "auto_f1": float(f1_score(ya, pa, zero_division=0)), "auto_mcc": float(matthews_corrcoef(ya, pa)) if len(np.unique(pa)) > 1 and len(np.unique(ya)) > 1 else 0.0, "auto_tn": int(tn), "auto_fp": int(fp), "auto_fn": int(fn), "auto_tp": int(tp)})
        else:
            row.update({"auto_accuracy": np.nan, "auto_balanced_accuracy": np.nan, "auto_precision": np.nan, "auto_recall": np.nan, "auto_specificity": np.nan, "auto_f1": np.nan, "auto_mcc": np.nan, "auto_tn": 0, "auto_fp": 0, "auto_fn": 0, "auto_tp": 0})
        rows.append(row)
    return pd.DataFrame(rows)



def _excel_safe_value(v):
    if pd.isna(v):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def write_results_workbook(path, sheets):
    import xlsxwriter
    path = Path(path)
    wb = xlsxwriter.Workbook(str(path), {"constant_memory": True, "nan_inf_to_errors": True})
    header_fmt = wb.add_format({"bold": True, "font_color": "white", "bg_color": "#1F4E78", "border": 1})
    text_fmt = wb.add_format({"text_wrap": False})
    num_fmt = wb.add_format({"num_format": "0.0000"})
    int_fmt = wb.add_format({"num_format": "#,##0"})
    date_fmt = wb.add_format({"num_format": "yyyy-mm-dd hh:mm"})
    used_names = set()
    for sheet_name, df in sheets.items():
        safe = re.sub(r"[\\/*?:\[\]]", "_", str(sheet_name)).strip()[:31] or "Sheet"
        base = safe
        i = 1
        while safe.lower() in used_names:
            suffix = f"_{i}"
            safe = (base[:31-len(suffix)] + suffix)[:31]
            i += 1
        used_names.add(safe.lower())
        ws = wb.add_worksheet(safe)
        df2 = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
        for col in df2.columns:
            if str(df2[col].dtype) == "category":
                df2[col] = df2[col].astype(str)
        rows, cols = df2.shape
        for j, col in enumerate(df2.columns):
            ws.write(0, j, str(col), header_fmt)
        for irow, row in enumerate(df2.itertuples(index=False, name=None), start=1):
            for j, v in enumerate(row):
                v = _excel_safe_value(v)
                if v is None:
                    continue
                if isinstance(v, pd.Timestamp):
                    ws.write_datetime(irow, j, v.to_pydatetime(), date_fmt)
                elif isinstance(v, (int,)):
                    ws.write_number(irow, j, v, int_fmt)
                elif isinstance(v, (float,)):
                    ws.write_number(irow, j, v, num_fmt)
                else:
                    ws.write(irow, j, str(v), text_fmt)
        ws.freeze_panes(1, 0)
        ws.autofilter(0, 0, max(rows, 1), max(cols - 1, 0))
        for j, col in enumerate(df2.columns):
            sample = df2[col].dropna().astype(str).head(200).tolist()
            max_len = max([len(str(col))] + [len(x) for x in sample]) if sample else len(str(col))
            width = min(max(max_len + 2, 10), 42)
            ws.set_column(j, j, width)
    wb.close()


def plot_figures(outdir, test, occ_cal, topk, delay, occurrence_coef, severity_coef):
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    y = test["problem_delivery"].astype(int).to_numpy()
    p = prob_clip(test["p_problem"].to_numpy(float))
    order = np.argsort(-p)
    ys = y[order]
    tps = np.cumsum(ys)
    fps = np.cumsum(1 - ys)
    positives = max(int(y.sum()), 1)
    negatives = max(int((1 - y).sum()), 1)
    tpr = tps / positives
    fpr = fps / negatives
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Occurrence ROC, test AUC={auc_rank(y, p):.4f}")
    plt.tight_layout()
    plt.savefig(figdir / "01_occurrence_roc.png", dpi=300, bbox_inches="tight")
    plt.close()
    cal = occ_cal[occ_cal["split"].eq("test")]
    plt.figure(figsize=(7, 6))
    plt.plot(cal["mean_p"], cal["observed_rate"], marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed problem rate")
    plt.title("Occurrence calibration by decile")
    plt.tight_layout()
    plt.savefig(figdir / "02_occurrence_calibration.png", dpi=300, bbox_inches="tight")
    plt.close()
    tk = topk[(topk["split"].eq("test")) & (topk["score"].eq("expected_burden_hours"))]
    plt.figure(figsize=(7, 6))
    plt.plot(tk["top_k_rate"], tk["captured_burden_share"], marker="o")
    plt.xlabel("Top-k monitoring capacity")
    plt.ylabel("Captured burden share")
    plt.title("Captured delivery burden by top-k capacity")
    plt.tight_layout()
    plt.savefig(figdir / "03_topk_captured_burden.png", dpi=300, bbox_inches="tight")
    plt.close()
    d = delay[delay["split"].eq("test")].copy()
    d["delay_duration_band"] = d["delay_duration_band"].astype(str)
    plt.figure(figsize=(10, 6))
    plt.bar(d["delay_duration_band"], d["n"])
    plt.xticks(rotation=35, ha="right")
    plt.xlabel("Delay duration band")
    plt.ylabel("Orders")
    plt.title("Delay duration bands, test set")
    plt.tight_layout()
    plt.savefig(figdir / "04_delay_duration_bands.png", dpi=300, bbox_inches="tight")
    plt.close()
    oc = occurrence_coef[occurrence_coef["term"].ne("const")].copy()
    oc["abs_effect"] = np.abs(np.log(np.clip(oc.get("odds_ratio", np.exp(oc["beta"])), 1e-12, None)))
    sv = severity_coef[severity_coef["term"].ne("const")].copy()
    sv["abs_effect"] = np.abs(np.log(np.clip(sv.get("severity_ratio", np.exp(sv["beta"])), 1e-12, None)))
    imp = pd.concat([
        oc[["term", "abs_effect"]].assign(model="occurrence"),
        sv[["term", "abs_effect"]].assign(model="severity")
    ], ignore_index=True)
    imp = imp.groupby("term", as_index=False)["abs_effect"].mean().sort_values("abs_effect", ascending=False).head(20).sort_values("abs_effect")
    plt.figure(figsize=(10, 8))
    plt.barh(imp["term"], imp["abs_effect"])
    plt.xlabel("Mean absolute log effect across occurrence/severity")
    plt.ylabel("Predictor")
    plt.title("Top interpretable effect signals")
    plt.tight_layout()
    plt.savefig(figdir / "05_factor_importance.png", dpi=300, bbox_inches="tight")
    plt.close()

def main(input_file):
    inp = Path(input_file).expanduser().resolve()
    if not inp.exists():
        raise FileNotFoundError(str(inp))
    out = inp.with_suffix("").parent / f"{inp.stem}_excel_png_two_part_glm_output"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    print("Reading data", flush=True)
    raw = pd.read_csv(inp, encoding="latin1", low_memory=False)
    required = ["Order Id", "Delivery Status", "Order Status", "Type", "Shipping Mode", "Customer Segment", "Order Region", "Department Name", "Order Item Quantity", "Order Item Product Price", "Order Item Discount Rate", "Order Profit Per Order", "order date (DateOrders)", "shipping date (DateOrders)", "Days for shipping (real)", "Days for shipment (scheduled)"]
    optional = ["Late_delivery_risk", "Market", "Category Name", "Sales", "Order Item Total"]
    missing_required = [c for c in required if c not in raw.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")
    selected = required + [c for c in optional if c in raw.columns]
    selected_table = pd.DataFrame({"original_column": selected, "clean_column": [snake(c) for c in selected]})
    data = raw[selected].copy()
    data = data.rename(columns={c: snake(c) for c in data.columns})
    missing_before = missingness_table(data, "selected_raw")
    quality_before = data_quality_line_table(data, "selected_raw")
    print("Preprocessing", flush=True)
    cat_cols = [c for c in ["delivery_status", "order_status", "type", "shipping_mode", "customer_segment", "market", "order_region", "department_name", "category_name"] if c in data.columns]
    num_cols = [c for c in ["order_id", "late_delivery_risk", "order_item_quantity", "order_item_product_price", "order_item_discount_rate", "order_profit_per_order", "days_for_shipping_real", "days_for_shipment_scheduled", "sales", "order_item_total"] if c in data.columns]
    for c in cat_cols:
        data[c] = clean_text(data[c])
    for c in num_cols:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data["order_date_dateorders"] = parse_dt(data["order_date_dateorders"])
    data["shipping_date_dateorders"] = parse_dt(data["shipping_date_dateorders"])
    if "market" not in data.columns:
        data["market"] = "unknown"
    if "category_name" not in data.columns:
        data["category_name"] = "unknown"
    if "sales" not in data.columns:
        data["sales"] = data["order_item_product_price"] * data["order_item_quantity"]
    if "order_item_total" not in data.columns:
        data["order_item_total"] = data["sales"]
    if "late_delivery_risk" not in data.columns:
        data["late_delivery_risk"] = np.nan
    needed = ["order_id", "delivery_status", "order_status", "type", "shipping_mode", "customer_segment", "order_region", "department_name", "order_item_quantity", "order_item_product_price", "order_item_discount_rate", "order_profit_per_order", "order_date_dateorders", "shipping_date_dateorders", "days_for_shipping_real", "days_for_shipment_scheduled"]
    rows_before_filter = len(data)
    data = data.dropna(subset=needed).copy()
    rows_after_filter = len(data)
    data["order_id"] = data["order_id"].astype(int)
    missing_after = missingness_table(data, "line_filtered")
    quality_after = data_quality_line_table(data, "line_filtered")
    department_modal = modal(data, "order_id", "department_name")
    category_modal = modal(data, "order_id", "category_name")
    order_counts = data.groupby("order_id", sort=False).size()
    conflicts = []
    for c in ["delivery_status", "order_status", "shipping_mode", "type", "customer_segment", "order_region", "late_delivery_risk"]:
        if c in data.columns:
            conflicts.append({"variable": c, "orders_with_conflicting_values": int((data.groupby("order_id")[c].nunique(dropna=True) > 1).sum())})
    orders = data.groupby("order_id", sort=False).agg(order_date=("order_date_dateorders", "first"), shipping_date=("shipping_date_dateorders", "first"), days_for_shipping_real=("days_for_shipping_real", "first"), days_for_shipment_scheduled=("days_for_shipment_scheduled", "first"), late_delivery_risk=("late_delivery_risk", "first"), delivery_status=("delivery_status", "first"), order_status=("order_status", "first"), type=("type", "first"), shipping_mode=("shipping_mode", "first"), customer_segment=("customer_segment", "first"), market=("market", "first"), order_region=("order_region", "first"), department_count=("department_name", "nunique"), category_count=("category_name", "nunique"), order_lines=("order_id", "size"), quantity_sum=("order_item_quantity", "sum"), quantity_mean=("order_item_quantity", "mean"), product_price_mean=("order_item_product_price", "mean"), discount_rate_mean=("order_item_discount_rate", "mean"), discount_rate_max=("order_item_discount_rate", "max"), sales_sum=("sales", "sum"), order_profit_first=("order_profit_per_order", "first")).reset_index()
    orders = orders.merge(department_modal, on="order_id", how="left").merge(category_modal, on="order_id", how="left")
    orders["elapsed_hours_timestamp"] = (orders["shipping_date"] - orders["order_date"]).dt.total_seconds() / 3600
    orders.loc[orders["elapsed_hours_timestamp"].lt(0), "elapsed_hours_timestamp"] = np.nan
    orders["actual_shipping_hours"] = np.maximum(orders["elapsed_hours_timestamp"], orders["days_for_shipping_real"] * 24)
    orders["scheduled_shipping_hours"] = orders["days_for_shipment_scheduled"] * 24
    orders = orders.dropna(subset=["order_id", "order_date", "shipping_date", "actual_shipping_hours", "scheduled_shipping_hours", "delivery_status", "order_status", "shipping_mode"]).copy()
    quality_orders = data_quality_order_table(orders, "order_level")
    orders = orders.sort_values(["order_date", "order_id"]).reset_index(drop=True)
    n = len(orders)
    cut_train = int(np.floor(TRAIN_SIZE * n))
    cut_val = int(np.floor((TRAIN_SIZE + VALIDATION_SIZE) * n))
    train_base = orders.iloc[:cut_train].copy().reset_index(drop=True)
    val_base = orders.iloc[cut_train:cut_val].copy().reset_index(drop=True)
    test_base = orders.iloc[cut_val:].copy().reset_index(drop=True)
    sla = calibrate_sla(train_base)
    train, penalty, min_positive = apply_outcome(train_base, sla, None, None)
    val, _, _ = apply_outcome(val_base, sla, penalty, min_positive)
    test, _, _ = apply_outcome(test_base, sla, penalty, min_positive)
    train = add_time_features(train)
    val = add_time_features(val)
    test = add_time_features(test)
    cats = [c for c in ["type", "shipping_mode", "customer_segment", "market", "order_region", "department_name"] if c in train.columns]
    nums = [c for c in ["department_count", "category_count", "order_lines", "quantity_sum", "quantity_mean", "product_price_mean", "discount_rate_mean", "discount_rate_max", "sales_sum", "order_profit_first", "month_sin", "month_cos", "dow_sin", "dow_cos", "hour_sin", "hour_cos"] if c in train.columns]
    model_needed = cats + nums + ["problem_delivery", "severity_hours_if_problem", "delivery_burden_hours"]
    train = train.dropna(subset=[c for c in model_needed if c != "severity_hours_if_problem"]).reset_index(drop=True)
    val = val.dropna(subset=[c for c in model_needed if c != "severity_hours_if_problem"]).reset_index(drop=True)
    test = test.dropna(subset=[c for c in model_needed if c != "severity_hours_if_problem"]).reset_index(drop=True)
    scaler = RobustScaler()
    xn_train = pd.DataFrame(scaler.fit_transform(train[nums]), columns=nums)
    xn_val = pd.DataFrame(scaler.transform(val[nums]), columns=nums)
    xn_test = pd.DataFrame(scaler.transform(test[nums]), columns=nums)
    xd_train, [xd_val, xd_test], refs, unseen = build_dummies(train, [val, test], cats)
    x_train = pd.concat([xn_train.reset_index(drop=True), xd_train.reset_index(drop=True)], axis=1).astype(float)
    x_val = pd.concat([xn_val.reset_index(drop=True), xd_val.reset_index(drop=True)], axis=1).astype(float)
    x_test = pd.concat([xn_test.reset_index(drop=True), xd_test.reset_index(drop=True)], axis=1).astype(float)
    x_train, x_val, x_test, dropped = rank_prune(x_train, x_val, x_test)
    x_train.insert(0, "const", 1.0)
    x_val.insert(0, "const", 1.0)
    x_test.insert(0, "const", 1.0)
    print("Fitting two-part GLM", flush=True)
    beta_occ, cov_occ = logistic_fit(train["problem_delivery"].astype(int).to_numpy(), x_train)
    sev_mask = train["problem_delivery"].eq(1).to_numpy()
    beta_sev, cov_sev, smearing = linear_fit_log(train.loc[sev_mask, "severity_hours_if_problem"].to_numpy(float), x_train.loc[sev_mask])
    p_train = expit(x_train.to_numpy(float) @ beta_occ.to_numpy(float))
    p_val = expit(x_val.to_numpy(float) @ beta_occ.to_numpy(float))
    p_test = expit(x_test.to_numpy(float) @ beta_occ.to_numpy(float))
    sev_train = np.exp(x_train.to_numpy(float) @ beta_sev.to_numpy(float)) * smearing
    sev_val = np.exp(x_val.to_numpy(float) @ beta_sev.to_numpy(float)) * smearing
    sev_test = np.exp(x_test.to_numpy(float) @ beta_sev.to_numpy(float)) * smearing
    threshold, threshold_candidates = select_threshold(val["problem_delivery"].astype(int).to_numpy(), p_val)
    val_pos = val["problem_delivery"].eq(1).to_numpy()
    conformal_resid = np.abs(np.log(np.clip(val.loc[val_pos, "severity_hours_if_problem"].to_numpy(float), 1e-8, None)) - np.log(np.clip(sev_val[val_pos], 1e-8, None)))
    q = float(np.quantile(conformal_resid, 1 - CONFORMAL_ALPHA)) if len(conformal_resid) else 0.0
    for frame, p, s in [(train, p_train, sev_train), (val, p_val, sev_val), (test, p_test, sev_test)]:
        frame["p_problem"] = prob_clip(p)
        frame["predicted_severity_hours_if_problem"] = np.clip(s, 1e-8, None)
        frame["expected_burden_hours"] = frame["p_problem"] * frame["predicted_severity_hours_if_problem"]
        frame["alert_problem"] = frame["p_problem"].ge(threshold).astype(int)
        frame["distance_to_threshold"] = np.abs(frame["p_problem"] - threshold)
        frame["triage_action_m05"] = np.select([frame["p_problem"].ge(threshold + 0.05), frame["p_problem"].le(threshold - 0.05)], ["auto_alert", "auto_no_alert"], default="manual_review")
        frame["severity_pi_lower_90"] = np.maximum(0, frame["predicted_severity_hours_if_problem"] * np.exp(-q))
        frame["severity_pi_upper_90"] = frame["predicted_severity_hours_if_problem"] * np.exp(q)
    print("Building outputs", flush=True)
    occ_metrics = pd.DataFrame([metrics_binary(train["problem_delivery"], p_train, threshold, "train"), metrics_binary(val["problem_delivery"], p_val, threshold, "validation"), metrics_binary(test["problem_delivery"], p_test, threshold, "test")])
    occ_cal = pd.concat([calibration_table(train["problem_delivery"], p_train, "train"), calibration_table(val["problem_delivery"], p_val, "validation"), calibration_table(test["problem_delivery"], p_test, "test")], ignore_index=True)
    occ_cal_summary = pd.DataFrame([calibration_summary(train["problem_delivery"], p_train, "train"), calibration_summary(val["problem_delivery"], p_val, "validation"), calibration_summary(test["problem_delivery"], p_test, "test")])
    sev_metrics = pd.DataFrame([regression_metrics(train.loc[train["problem_delivery"].eq(1), "severity_hours_if_problem"], train.loc[train["problem_delivery"].eq(1), "predicted_severity_hours_if_problem"], "train", "conditional_severity"), regression_metrics(val.loc[val["problem_delivery"].eq(1), "severity_hours_if_problem"], val.loc[val["problem_delivery"].eq(1), "predicted_severity_hours_if_problem"], "validation", "conditional_severity"), regression_metrics(test.loc[test["problem_delivery"].eq(1), "severity_hours_if_problem"], test.loc[test["problem_delivery"].eq(1), "predicted_severity_hours_if_problem"], "test", "conditional_severity")])
    sev_cal = pd.concat([severity_calibration_table(train, "train"), severity_calibration_table(val, "validation"), severity_calibration_table(test, "test")], ignore_index=True)
    sev_cal_summary = pd.DataFrame([severity_calibration_summary(train, "train"), severity_calibration_summary(val, "validation"), severity_calibration_summary(test, "test")])
    burden_metrics = pd.DataFrame([regression_metrics(train["delivery_burden_hours"], train["expected_burden_hours"], "train", "expected_burden"), regression_metrics(val["delivery_burden_hours"], val["expected_burden_hours"], "validation", "expected_burden"), regression_metrics(test["delivery_burden_hours"], test["expected_burden_hours"], "test", "expected_burden")])
    conformal = pd.DataFrame([{"split": "test", "alpha": CONFORMAL_ALPHA, "nominal_coverage": 1 - CONFORMAL_ALPHA, "observed_coverage": float(((test.loc[test["problem_delivery"].eq(1), "severity_hours_if_problem"] >= test.loc[test["problem_delivery"].eq(1), "severity_pi_lower_90"]) & (test.loc[test["problem_delivery"].eq(1), "severity_hours_if_problem"] <= test.loc[test["problem_delivery"].eq(1), "severity_pi_upper_90"])).mean()), "q_abs_log_residual": q}])
    occ_coef = coef_table(beta_occ, cov_occ, "occurrence_logistic", "odds_ratio")
    sev_coef = coef_table(beta_sev, cov_sev, "severity_log_linear", "severity_ratio")
    groups = {"all_predictors": x_train.columns.drop("const").tolist(), "numeric_all": [c for c in nums if c in x_train.columns], "time_cyclic": [c for c in ["month_sin", "month_cos", "dow_sin", "dow_cos", "hour_sin", "hour_cos"] if c in x_train.columns]}
    for c in cats:
        groups[c] = [v for v in x_train.columns if v.startswith(c + "_")]
    factor_tests = pd.concat([group_wald(beta_occ, cov_occ, groups, "occurrence_logistic"), group_wald(beta_sev, cov_sev, groups, "severity_log_linear")], ignore_index=True)
    topk = pd.concat([topk_table(train, "expected_burden_hours", "train"), topk_table(val, "expected_burden_hours", "validation"), topk_table(test, "expected_burden_hours", "test")], ignore_index=True)
    utility = pd.concat([utility_table(train, "expected_burden_hours", "train"), utility_table(val, "expected_burden_hours", "validation"), utility_table(test, "expected_burden_hours", "test")], ignore_index=True)
    cuts = np.quantile(val["expected_burden_hours"], [0.25, 0.50, 0.75])
    strata = pd.concat([risk_strata_table(train, "expected_burden_hours", "train", cuts), risk_strata_table(val, "expected_burden_hours", "validation", cuts), risk_strata_table(test, "expected_burden_hours", "test", cuts)], ignore_index=True)
    delay = pd.concat([delay_bands_summary(train, "train"), delay_bands_summary(val, "validation"), delay_bands_summary(test, "test")], ignore_index=True)
    outcome = pd.concat([train.assign(split="train"), val.assign(split="validation"), test.assign(split="test")], ignore_index=True).groupby(["split", "clean_delivery_outcome"], as_index=False).agg(n=("order_id", "size"), mean_burden_hours=("delivery_burden_hours", "mean"), mean_p_problem=("p_problem", "mean"), mean_expected_burden=("expected_burden_hours", "mean"))
    borderline = pd.concat([borderline_audit(train, threshold, "train"), borderline_audit(val, threshold, "validation"), borderline_audit(test, threshold, "test")], ignore_index=True)
    triage = pd.concat([triage_metrics(train, threshold, "train"), triage_metrics(val, threshold, "validation"), triage_metrics(test, threshold, "test")], ignore_index=True)
    test_borderline_cases = test[test["distance_to_threshold"].le(BORDERLINE_CASE_MARGIN)].sort_values(["distance_to_threshold", "expected_burden_hours"], ascending=[True, False]).head(MAX_BORDERLINE_CASES).copy()
    pred_cols = ["order_id", "order_date", "shipping_date", "delivery_status", "order_status", "type", "shipping_mode", "customer_segment", "market", "order_region", "department_name", "clean_delivery_outcome", "delay_duration_band", "actual_shipping_hours", "scheduled_shipping_hours", "grace_hours", "deadline_hours", "lateness_hours", "lateness_minutes", "is_canceled", "problem_delivery", "severity_hours_if_problem", "delivery_burden_hours", "p_problem", "predicted_severity_hours_if_problem", "expected_burden_hours", "alert_problem", "distance_to_threshold", "triage_action_m05", "severity_pi_lower_90", "severity_pi_upper_90"]
    pred = test[[c for c in pred_cols if c in test.columns]].copy()
    for d in [pred, test_borderline_cases]:
        for c in d.columns:
            if pd.api.types.is_datetime64_any_dtype(d[c]) or str(d[c].dtype) == "category":
                d[c] = d[c].astype(str)
    audit = pd.DataFrame([{"item": "input_file", "value": str(inp)}, {"item": "raw_rows", "value": int(len(raw))}, {"item": "raw_columns", "value": int(raw.shape[1])}, {"item": "selected_columns", "value": int(len(selected))}, {"item": "line_rows_before_filter", "value": int(rows_before_filter)}, {"item": "line_rows_after_filter", "value": int(rows_after_filter)}, {"item": "line_rows_removed", "value": int(rows_before_filter - rows_after_filter)}, {"item": "orders_before_aggregation", "value": int(order_counts.shape[0])}, {"item": "max_lines_per_order", "value": int(order_counts.max())}, {"item": "mean_lines_per_order", "value": float(order_counts.mean())}, {"item": "orders_after_aggregation", "value": int(len(orders))}, {"item": "train_rows", "value": int(len(train))}, {"item": "validation_rows", "value": int(len(val))}, {"item": "test_rows", "value": int(len(test))}, {"item": "features_after_qr_pruning", "value": int(x_train.shape[1] - 1)}, {"item": "framework", "value": "SLA-reconstructed two-part interpretable GLM"}, {"item": "occurrence_model", "value": "penalized logistic GLM with robust sandwich inference"}, {"item": "severity_model", "value": "log-linear severity GLM with Duan smearing"}, {"item": "threshold_selected_on", "value": "validation"}, {"item": "threshold_objective", "value": "mcc"}, {"item": "threshold", "value": float(threshold)}, {"item": "triage_rule_margin_default", "value": 0.05}, {"item": "cancellation_penalty_hours", "value": float(penalty)}, {"item": "minimum_positive_burden_hours", "value": float(min_positive)}, {"item": "smearing_factor", "value": float(smearing)}, {"item": "conformal_q_abs_log_residual", "value": float(q)}, {"item": "causal_scope", "value": "associational_not_causal"}, {"item": "csv_only", "value": True}, {"item": "figures_png", "value": 5}])
    readme = pd.DataFrame([{"section": "Framework", "description": "SLA-reconstructed two-part interpretable GLM: occurrence first, conditional severity second."}, {"section": "Occurrence", "description": "logit[P(problem_delivery=1|X)] = X beta."}, {"section": "Severity", "description": "log[E(severity_hours|problem_delivery=1,X)] = X gamma, corrected with Duan smearing."}, {"section": "Expected burden", "description": "expected_burden_hours = p_problem * predicted_severity_hours_if_problem."}, {"section": "Triage", "description": "Predictions near the selected validation threshold are marked for manual review using threshold +/- margin."}, {"section": "Delay bands", "description": "non_late, late_0_6h, late_6_24h, late_1_2d, late_2_4d, late_over_4d, canceled."}])
    scaler_info = pd.DataFrame({"variable": nums, "center_median_train": scaler.center_, "scale_iqr_train": scaler.scale_})
    model_equations = pd.DataFrame([{"equation": "Occurrence", "formula": "logit[P(problem_delivery=1|X)] = X beta"}, {"equation": "Severity", "formula": "log[E(severity_hours|problem_delivery=1,X)] = X gamma; prediction uses Duan smearing"}, {"equation": "Expected burden", "formula": "E(burden_hours|X) = P(problem_delivery=1|X) * E(severity_hours|problem_delivery=1,X)"}])
    quality_all = pd.concat([quality_before, quality_after, quality_orders], ignore_index=True)
    missing_all = pd.concat([missing_before, missing_after], ignore_index=True)
    audit.loc[audit["item"].eq("csv_only"), "item"] = "excel_workbook"
    audit.loc[audit["item"].eq("excel_workbook"), "value"] = True
    pred_sample = pred.head(2000).copy()
    borderline_cases_sheet = test_borderline_cases[[c for c in pred_cols if c in test_borderline_cases.columns]].copy()
    sheets = {
        "README": readme,
        "Data_Audit": audit,
        "Selected_Columns": selected_table,
        "Missingness": missing_all,
        "Data_Quality_Audit": quality_all,
        "Order_Conflicts": pd.DataFrame(conflicts),
        "Model_Equations": model_equations,
        "SLA_Grace": sla,
        "Outcome_Distribution": outcome,
        "Delay_Bands": delay,
        "Occurrence_Metrics": occ_metrics,
        "Occurrence_Calibration": occ_cal,
        "Occurrence_Cal_Summary": occ_cal_summary,
        "Severity_Metrics": sev_metrics,
        "Severity_Calibration": sev_cal,
        "Severity_Cal_Summary": sev_cal_summary,
        "Conformal_Intervals": conformal,
        "Expected_Burden_Metrics": burden_metrics,
        "Threshold_Candidates": threshold_candidates,
        "Borderline_Audit": borderline,
        "Triage_Metrics": triage,
        "TopK_Capture": topk,
        "Operational_Utility": utility,
        "Risk_Strata": strata,
        "Occurrence_Coef": occ_coef,
        "Severity_Coef": sev_coef,
        "Factor_Tests": factor_tests,
        "VIF_Sample": vif_sample(x_train.drop(columns=["const"])),
        "Dropped_Collinear": dropped,
        "Reference_Levels": refs,
        "Unseen_Levels": unseen,
        "Scaler_Train": scaler_info,
        "Borderline_Cases_Test": borderline_cases_sheet,
        "Pred_Test_Sample": pred_sample,
    }
    print("Writing prediction CSV", flush=True)
    pred.to_csv(out / "pred_test_full.csv", index=False, encoding="utf-8-sig")
    excel_path = out / "two_part_delivery_glm_results.xlsx"
    print("Writing Excel workbook", flush=True)
    write_results_workbook(excel_path, sheets)
    print("Creating figures", flush=True)
    plot_figures(out, test, occ_cal, topk, delay, occ_coef, sev_coef)
    with open(out / "model_summary.txt", "w", encoding="utf-8") as f:
        f.write("SLA-reconstructed two-part interpretable GLM\n")
        f.write("Occurrence: logit[P(problem_delivery=1|X)] = X beta\n")
        f.write("Severity: log[E(severity_hours|problem_delivery=1,X)] = X gamma, with Duan smearing correction\n")
        f.write("Expected burden: E(burden_hours|X) = P(problem_delivery=1|X) * E(severity_hours|problem_delivery=1,X)\n")
        f.write("Borderline triage: predictions near the validation-selected threshold are flagged for manual review.\n")
    zpath = inp.with_suffix("").parent / f"{inp.stem}_excel_png_two_part_glm_package.zip"
    if zpath.exists():
        zpath.unlink()
    print("Packaging outputs", flush=True)
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(Path(__file__).resolve(), arcname=Path(__file__).name)
        for p in out.rglob("*"):
            if p.is_file():
                z.write(p, arcname=str(p.relative_to(out.parent)))
    print("DONE", flush=True)
    print(f"Output folder: {out}", flush=True)
    print(f"Excel workbook: {excel_path}", flush=True)
    print(f"Prediction CSV: {out / 'pred_test_full.csv'}", flush=True)
    print(f"Figures folder: {out / 'figures'}", flush=True)
    print(f"Package ZIP: {zpath}", flush=True)
    return out, zpath


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", nargs="?", default=None)
    args = parser.parse_args()
    if args.input_file is None:
        matches = sorted(Path.cwd().glob("*DataCo*Supply*Chain*Dataset*.csv"))
        if not matches:
            raise FileNotFoundError("Provide input CSV path")
        input_file = matches[0]
    else:
        input_file = args.input_file
    main(input_file)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)

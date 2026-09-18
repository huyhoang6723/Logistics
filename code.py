import argparse
import hashlib
import os
import platform
import re
import shutil
import sys
import warnings
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels
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
PRIMARY_CANCELLATION_QUANTILE = 0.95
MIN_POSITIVE_BURDEN_HOURS = 0.25
OCCURRENCE_C = 10.0
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


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prob_clip(p):
    return np.clip(np.asarray(p, dtype=float), 1e-8, 1 - 1e-8)


def safe_spearman(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if len(y) < 3 or np.nanstd(y) == 0 or np.nanstd(p) == 0:
        return np.nan, np.nan
    r, pv = spearmanr(y, p, nan_policy="omit")
    return float(r), float(pv)


def auc_rank(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = pd.Series(p).rank(method="average").to_numpy(float)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision_rank(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    n_pos = int(y.sum())
    if n_pos == 0:
        return np.nan
    order = np.argsort(-p, kind="mergesort")
    ys = y[order]
    tp = np.cumsum(ys)
    precision = tp / (np.arange(len(y)) + 1)
    return float((precision * ys).sum() / n_pos)


def modal(frame, order_col, value_col):
    if value_col not in frame.columns:
        return pd.DataFrame({order_col: frame[order_col].drop_duplicates(), value_col: "unknown"})
    out = frame.dropna(subset=[value_col]).groupby([order_col, value_col], sort=False).size().reset_index(name="n")
    out = out.sort_values([order_col, "n", value_col], ascending=[True, False, True]).drop_duplicates(order_col)
    return out[[order_col, value_col]]


def missingness_table(frame, stage):
    n = len(frame)
    return pd.DataFrame([{"stage": stage, "column": c, "missing_n": int(frame[c].isna().sum()), "missing_rate": float(frame[c].isna().mean()) if n else np.nan} for c in frame.columns])


def data_quality_line_table(frame, stage):
    rows = []
    rows.append({"stage": stage, "item": "rows", "value": int(len(frame))})
    rows.append({"stage": stage, "item": "columns", "value": int(frame.shape[1])})
    rows.append({"stage": stage, "item": "duplicate_line_rows", "value": int(frame.duplicated().sum())})
    checks = {
        "missing_order_id": int(frame["order_id"].isna().sum()) if "order_id" in frame else np.nan,
        "quantity_nonpositive": int(pd.to_numeric(frame["order_item_quantity"], errors="coerce").le(0).sum()) if "order_item_quantity" in frame else np.nan,
        "product_price_nonpositive": int(pd.to_numeric(frame["order_item_product_price"], errors="coerce").le(0).sum()) if "order_item_product_price" in frame else np.nan,
        "real_shipping_days_negative": int(pd.to_numeric(frame["days_for_shipping_real"], errors="coerce").lt(0).sum()) if "days_for_shipping_real" in frame else np.nan,
        "scheduled_shipping_days_negative": int(pd.to_numeric(frame["days_for_shipment_scheduled"], errors="coerce").lt(0).sum()) if "days_for_shipment_scheduled" in frame else np.nan,
    }
    if "order_item_discount_rate" in frame:
        d = pd.to_numeric(frame["order_item_discount_rate"], errors="coerce")
        checks["discount_outside_0_1"] = int(((d < 0) | (d > 1)).sum())
    for k, v in checks.items():
        rows.append({"stage": stage, "item": k, "value": v})
    if "order_date_dateorders" in frame and "shipping_date_dateorders" in frame:
        od = frame["order_date_dateorders"] if pd.api.types.is_datetime64_any_dtype(frame["order_date_dateorders"]) else parse_dt(frame["order_date_dateorders"])
        sd = frame["shipping_date_dateorders"] if pd.api.types.is_datetime64_any_dtype(frame["shipping_date_dateorders"]) else parse_dt(frame["shipping_date_dateorders"])
        rows.extend([
            {"stage": stage, "item": "missing_order_datetime", "value": int(od.isna().sum())},
            {"stage": stage, "item": "missing_shipping_datetime", "value": int(sd.isna().sum())},
            {"stage": stage, "item": "shipping_before_order", "value": int((sd < od).sum())},
        ])
    return pd.DataFrame(rows)


def data_quality_order_table(orders, stage):
    rows = [
        {"stage": stage, "item": "orders", "value": int(len(orders))},
        {"stage": stage, "item": "duplicate_order_id_rows", "value": int(orders["order_id"].duplicated().sum())},
        {"stage": stage, "item": "negative_elapsed_hours", "value": int(orders["elapsed_hours_timestamp"].lt(0).sum())},
        {"stage": stage, "item": "missing_elapsed_hours", "value": int(orders["elapsed_hours_timestamp"].isna().sum())},
        {"stage": stage, "item": "actual_shipping_hours_negative", "value": int(orders["actual_shipping_hours"].lt(0).sum())},
        {"stage": stage, "item": "actual_shipping_hours_over_30_days", "value": int(orders["actual_shipping_hours"].gt(30 * 24).sum())},
        {"stage": stage, "item": "scheduled_shipping_hours_negative", "value": int(orders["scheduled_shipping_hours"].lt(0).sum())},
        {"stage": stage, "item": "orders_with_more_than_10_lines", "value": int(orders["order_lines"].gt(10).sum())},
        {"stage": stage, "item": "quantity_sum_nonpositive", "value": int(orders["quantity_sum"].le(0).sum())},
        {"stage": stage, "item": "mean_discount_outside_0_1", "value": int(((orders["discount_rate_mean"] < 0) | (orders["discount_rate_mean"] > 1)).sum())},
    ]
    return pd.DataFrame(rows)


def choose_grace(y, actual, scheduled):
    grid = np.arange(0, SLA_GRACE_MAX_HOURS + SLA_GRACE_STEP_HOURS / 2, SLA_GRACE_STEP_HOURS)
    rows = []
    for theta in grid:
        pred = (actual > scheduled + theta).astype(int)
        agreement = float((pred == y).mean())
        mcc = float(matthews_corrcoef(y, pred)) if len(np.unique(y)) == 2 and len(np.unique(pred)) == 2 else 0.0
        rows.append({"grace_hours": float(theta), "agreement": agreement, "mcc": mcc})
    tab = pd.DataFrame(rows)
    best_mcc = tab["mcc"].max()
    pool = tab[np.isclose(tab["mcc"], best_mcc)]
    best_agreement = pool["agreement"].max()
    pool = pool[np.isclose(pool["agreement"], best_agreement)]
    best = pool.sort_values("grace_hours").iloc[0]
    return float(best["grace_hours"]), float(best["agreement"]), float(best["mcc"]), tab


def calibrate_sla(train, mode_specific=True):
    delivered = train[(~train["delivery_status"].eq("shipping_canceled")) & (~train["order_status"].isin(["canceled", "suspected_fraud"]))].copy()
    summary = []
    grid_rows = []
    if mode_specific:
        groups = [(m, delivered[delivered["shipping_mode"].astype(str).eq(m)].copy()) for m in sorted(train["shipping_mode"].dropna().astype(str).unique())]
    else:
        groups = [("__global__", delivered)]
    for mode, g in groups:
        if len(g) == 0:
            summary.append({"shipping_mode": mode, "scheduled_days_mode": np.nan, "grace_hours": 0.0, "agreement": np.nan, "mcc": np.nan, "n_train_delivered": 0, "raw_late_rate": np.nan})
            continue
        y = g["delivery_status"].eq("late_delivery").astype(int).to_numpy()
        grace, agreement, mcc, tab = choose_grace(y, g["actual_shipping_hours"].to_numpy(float), g["scheduled_shipping_hours"].to_numpy(float))
        md = g["days_for_shipment_scheduled"].mode()
        summary.append({"shipping_mode": mode, "scheduled_days_mode": float(md.iloc[0]) if len(md) else np.nan, "grace_hours": grace, "agreement": agreement, "mcc": mcc, "n_train_delivered": int(len(g)), "raw_late_rate": float(y.mean())})
        tab.insert(0, "shipping_mode", mode)
        grid_rows.append(tab)
    summary = pd.DataFrame(summary)
    grid = pd.concat(grid_rows, ignore_index=True) if grid_rows else pd.DataFrame(columns=["shipping_mode", "grace_hours", "agreement", "mcc"])
    if not mode_specific:
        g = float(summary.loc[0, "grace_hours"]) if len(summary) else 0.0
        modes = sorted(train["shipping_mode"].dropna().astype(str).unique())
        summary = pd.DataFrame([{"shipping_mode": m, "scheduled_days_mode": np.nan, "grace_hours": g, "agreement": float(summary.loc[0, "agreement"]) if len(summary) else np.nan, "mcc": float(summary.loc[0, "mcc"]) if len(summary) else np.nan, "n_train_delivered": int(summary.loc[0, "n_train_delivered"]) if len(summary) else 0, "raw_late_rate": float(summary.loc[0, "raw_late_rate"]) if len(summary) else np.nan} for m in modes])
    return summary, grid


def zero_sla(train):
    return pd.DataFrame({"shipping_mode": sorted(train["shipping_mode"].dropna().astype(str).unique()), "grace_hours": 0.0})


def apply_outcome(frame, sla, penalty_quantile=PRIMARY_CANCELLATION_QUANTILE, penalty=None, min_positive=None, min_positive_mode="train_min", problem_rule="union"):
    x = frame.merge(sla[["shipping_mode", "grace_hours"]], on="shipping_mode", how="left")
    x["grace_hours"] = x["grace_hours"].fillna(0.0)
    x["deadline_hours"] = x["scheduled_shipping_hours"] + x["grace_hours"]
    x["lateness_hours"] = np.maximum(0.0, x["actual_shipping_hours"] - x["deadline_hours"])
    x["lateness_minutes"] = x["lateness_hours"] * 60
    x["is_canceled"] = (x["delivery_status"].eq("shipping_canceled") | x["order_status"].isin(["canceled", "suspected_fraud"])).astype(int)
    x["raw_late_status"] = x["delivery_status"].eq("late_delivery").astype(int)
    x["reconstructed_late"] = x["lateness_hours"].gt(0).astype(int)
    if problem_rule == "status_only":
        x["problem_delivery"] = ((x["is_canceled"].eq(1)) | (x["raw_late_status"].eq(1))).astype(int)
    elif problem_rule == "reconstructed_only":
        x["problem_delivery"] = ((x["is_canceled"].eq(1)) | (x["reconstructed_late"].eq(1))).astype(int)
    else:
        x["problem_delivery"] = ((x["is_canceled"].eq(1)) | (x["raw_late_status"].eq(1)) | (x["reconstructed_late"].eq(1))).astype(int)
    if min_positive is None:
        pos = x.loc[x["is_canceled"].eq(0) & x["lateness_hours"].gt(0), "lateness_hours"]
        observed_min = float(pos.min()) if len(pos) else MIN_POSITIVE_BURDEN_HOURS
        min_positive = MIN_POSITIVE_BURDEN_HOURS if min_positive_mode == "fixed_epsilon" else max(MIN_POSITIVE_BURDEN_HOURS, observed_min)
    x["delivered_problem_burden_hours"] = 0.0
    m = x["is_canceled"].eq(0) & x["problem_delivery"].eq(1)
    x.loc[m, "delivered_problem_burden_hours"] = np.maximum(x.loc[m, "lateness_hours"].to_numpy(float), float(min_positive))
    if penalty is None:
        posb = x.loc[m, "delivered_problem_burden_hours"]
        penalty = float(np.nanquantile(posb, penalty_quantile)) if len(posb) else 24.0
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
    lineage = []
    for col in cols:
        counts = train[col].dropna().astype(str).value_counts()
        if len(counts) == 0:
            continue
        frequent = counts[counts >= RARE_LEVEL_MIN_COUNT].index.tolist()
        if not frequent:
            frequent = [counts.index[0]]
        ref = counts.index[0] if counts.index[0] in frequent else frequent[0]
        levels = [ref] + sorted([v for v in frequent if v != ref]) + ["__other__"]
        refs.append({"variable": col, "reference_level": ref, "raw_train_levels": int(len(counts)), "encoded_levels_including_reference": int(len(levels)), "rare_levels_grouped": int((counts < RARE_LEVEL_MIN_COUNT).sum())})
        mapped = train[col].astype(str).where(train[col].astype(str).isin(frequent), "__other__")
        for level in levels[1:]:
            term = f"{col}_{level}"
            tr[term] = mapped.eq(level).astype(float).to_numpy()
            lineage.append({"encoded_term": term, "source_variable": col, "feature_type": "categorical_dummy", "level": level, "reference_level": ref})
        train_levels = set(counts.index.tolist())
        for split_name, f, out in zip(["validation", "test"], frames, outs):
            raw_levels = set(f[col].dropna().astype(str).unique())
            unseen = sorted(list(raw_levels - train_levels))
            unseen_rows.append({"split": split_name, "variable": col, "unseen_levels_n": len(unseen), "unseen_rows_n": int(f[col].astype(str).isin(unseen).sum()) if unseen else 0, "unseen_levels": "|".join(unseen)})
            mapped_f = f[col].astype(str).where(f[col].astype(str).isin(frequent), "__other__")
            for level in levels[1:]:
                out[f"{col}_{level}"] = mapped_f.eq(level).astype(float).to_numpy()
    return tr, outs, pd.DataFrame(refs), pd.DataFrame(unseen_rows), pd.DataFrame(lineage)


def rank_prune(xtr, xva, xte):
    pre_cols = xtr.columns.tolist()
    nonconstant = xtr.std(ddof=0) > 0
    constant_dropped = [c for c in xtr.columns if not bool(nonconstant[c])]
    xtr = xtr.loc[:, nonconstant]
    xva = xva[xtr.columns]
    xte = xte[xtr.columns]
    a = xtr.to_numpy(float)
    _, r, piv = qr(a, mode="economic", pivoting=True)
    diag = np.abs(np.diag(r)) if r.size else np.array([])
    tol = np.finfo(float).eps * max(a.shape) * max(float(diag[0]), 1.0) if len(diag) else 0.0
    rank = int((diag > tol).sum())
    keep = sorted(piv[:rank])
    keep_cols = xtr.columns[keep].tolist()
    collinear_dropped = [c for c in xtr.columns if c not in keep_cols]
    audit = pd.DataFrame([{"term": c, "pre_qr": True, "constant_dropped": c in constant_dropped, "collinear_dropped": c in collinear_dropped, "retained_post_qr": c in keep_cols} for c in pre_cols])
    return xtr[keep_cols], xva[keep_cols], xte[keep_cols], audit


def fit_occurrence_predictive(y, x, c_value=OCCURRENCE_C):
    clf = LogisticRegression(fit_intercept=False, penalty="l2", C=c_value, solver="liblinear", max_iter=2000, random_state=RANDOM_STATE)
    clf.fit(np.asarray(x, float), np.asarray(y, int))
    beta = pd.Series(clf.coef_.ravel(), index=x.columns)
    return clf, beta


def fit_occurrence_inference(y, x):
    xm = np.asarray(x, float)
    yy = np.asarray(y, int)
    beta = np.zeros(xm.shape[1], dtype=float)
    for _ in range(100):
        eta = np.clip(xm @ beta, -35, 35)
        p = expit(eta)
        w = np.clip(p * (1 - p), 1e-8, None)
        score = xm.T @ (yy - p)
        info = (xm.T * w) @ xm
        step = np.linalg.pinv(info) @ score
        beta_new = beta + step
        if np.max(np.abs(step)) < 1e-8:
            beta = beta_new
            break
        beta = beta_new
    eta = np.clip(xm @ beta, -35, 35)
    p = expit(eta)
    w = np.clip(p * (1 - p), 1e-8, None)
    bread = np.linalg.pinv((xm.T * w) @ xm)
    resid = yy - p
    meat = (xm.T * (resid ** 2)) @ xm
    n, k = xm.shape
    cov = bread @ meat @ bread * n / max(n - k, 1)
    return pd.Series(beta, index=x.columns), pd.DataFrame(cov, index=x.columns, columns=x.columns)

def fit_log_linear(y, x):
    xm = np.asarray(x, float)
    yy = np.log(np.asarray(y, float))
    xtx_inv = np.linalg.pinv(xm.T @ xm)
    beta = xtx_inv @ xm.T @ yy
    resid = yy - xm @ beta
    n, k = xm.shape
    cov = xtx_inv @ ((xm.T * (resid ** 2)) @ xm) @ xtx_inv * n / max(n - k, 1)
    smearing = float(np.mean(np.exp(resid)))
    return pd.Series(beta, index=x.columns), pd.DataFrame(cov, index=x.columns, columns=x.columns), smearing

def fit_direct_burden(y, x):
    yy = np.log1p(np.asarray(y, float))
    beta = np.linalg.pinv(np.asarray(x, float).T @ np.asarray(x, float)) @ np.asarray(x, float).T @ yy
    resid = yy - np.asarray(x, float) @ beta
    smearing = float(np.mean(np.exp(resid)))
    return pd.Series(beta, index=x.columns), smearing


def predict_occurrence(clf, x):
    return prob_clip(clf.predict_proba(np.asarray(x, float))[:, 1])


def predict_severity(beta, smearing, x):
    return np.clip(np.exp(np.asarray(x, float) @ beta.to_numpy(float)) * smearing, 1e-8, None)


def predict_direct_burden(beta, smearing, x):
    return np.clip(np.exp(np.asarray(x, float) @ beta.to_numpy(float)) * smearing - 1.0, 0.0, None)


def coef_table(beta, cov, model, effect_name):
    se = pd.Series(np.sqrt(np.maximum(np.diag(cov), 0)), index=beta.index)
    z = beta / se.replace(0, np.nan)
    p = pd.Series(2 * norm.sf(np.abs(z.to_numpy(float))), index=beta.index)
    out = pd.DataFrame({"model": model, "term": beta.index, "beta": beta.values, "se_hc1": se.values, "z": z.values, "p_value": p.values})
    out["beta_ci_low"] = out["beta"] - 1.96 * out["se_hc1"]
    out["beta_ci_high"] = out["beta"] + 1.96 * out["se_hc1"]
    mask = out["term"].ne("const")
    out["p_holm"] = np.nan
    out["q_bh"] = np.nan
    out["reject_holm_05"] = False
    out["reject_bh_05"] = False
    if mask.sum():
        rh, ph, _, _ = multipletests(out.loc[mask, "p_value"], method="holm")
        rb, qb, _, _ = multipletests(out.loc[mask, "p_value"], method="fdr_bh")
        out.loc[mask, "p_holm"] = ph
        out.loc[mask, "q_bh"] = qb
        out.loc[mask, "reject_holm_05"] = rh
        out.loc[mask, "reject_bh_05"] = rb
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
    candidates = np.unique(np.r_[0.0, 0.5, np.quantile(p, np.linspace(0.01, 0.99, 149)), 1.0])
    rows = []
    for t in candidates:
        yhat = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
        rows.append({"threshold": float(t), "accuracy": float(accuracy_score(y, yhat)), "balanced_accuracy": float(balanced_accuracy_score(y, yhat)), "precision": float(precision_score(y, yhat, zero_division=0)), "recall": float(recall_score(y, yhat, zero_division=0)), "f1": float(f1_score(y, yhat, zero_division=0)), "mcc": float(matthews_corrcoef(y, yhat)) if len(np.unique(yhat)) > 1 else 0.0, "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    tab = pd.DataFrame(rows)
    best = tab.sort_values(["mcc", "balanced_accuracy", "threshold"], ascending=[False, False, True]).iloc[0]
    return float(best["threshold"]), tab


def calibration_table(y, p, split):
    d = pd.DataFrame({"y": np.asarray(y, int), "p": prob_clip(p)})
    q = min(CALIBRATION_GROUPS, len(d))
    d["bin"] = pd.qcut(d["p"].rank(method="first"), q=q, labels=False, duplicates="drop") + 1
    tab = d.groupby("bin", as_index=False).agg(n=("y", "size"), observed_rate=("y", "mean"), mean_p=("p", "mean"), p_min=("p", "min"), p_max=("p", "max"))
    tab.insert(0, "split", split)
    return tab


def calibration_summary(y, p, split):
    tab = calibration_table(y, p, split)
    ece = float(np.sum(tab["n"] / tab["n"].sum() * np.abs(tab["observed_rate"] - tab["mean_p"])))
    mce = float(np.max(np.abs(tab["observed_rate"] - tab["mean_p"])))
    y = np.asarray(y, dtype=int)
    lp = logit(prob_clip(p))
    intercept = np.nan
    slope = np.nan
    try:
        xx = pd.DataFrame({"const": np.ones(len(lp)), "logit_p": lp})
        b, _ = fit_occurrence_inference(y, xx)
        intercept = float(b.loc["const"])
        slope = float(b.loc["logit_p"])
    except Exception:
        pass
    return {"split": split, "ece": ece, "mce": mce, "calibration_intercept": intercept, "calibration_slope": slope}

def regression_metrics(y, pred, split, target):
    y = np.asarray(y, dtype=float)
    pred = np.clip(np.asarray(pred, dtype=float), 0.0, None)
    err = pred - y
    r, pv = safe_spearman(y, pred)
    return {"split": split, "target": target, "n": int(len(y)), "mean_observed_hours": float(np.mean(y)), "mean_predicted_hours": float(np.mean(pred)), "mean_prediction_ratio": float(np.mean(pred) / np.mean(y)) if np.mean(y) > 0 else np.nan, "mae_hours": float(np.mean(np.abs(err))), "rmse_hours": float(np.sqrt(np.mean(err ** 2))), "median_absolute_error_hours": float(np.median(np.abs(err))), "spearman_r": r, "spearman_p": pv}


def continuous_calibration_table(observed, predicted, split, target):
    d = pd.DataFrame({"observed": np.asarray(observed, float), "predicted": np.asarray(predicted, float)})
    d["predicted"] = np.clip(d["predicted"], 0.0, None)
    q = min(CALIBRATION_GROUPS, len(d))
    d["bin"] = pd.qcut(d["predicted"].rank(method="first"), q=q, labels=False, duplicates="drop") + 1
    tab = d.groupby("bin", as_index=False).agg(n=("observed", "size"), mean_observed=("observed", "mean"), mean_predicted=("predicted", "mean"), median_observed=("observed", "median"), pred_min=("predicted", "min"), pred_max=("predicted", "max"))
    tab.insert(0, "target", target)
    tab.insert(0, "split", split)
    return tab


def log_calibration_summary(observed, predicted, split, target, use_log1p=False):
    y = np.asarray(observed, float)
    p = np.asarray(predicted, float)
    if len(y) < 5:
        return {"split": split, "target": target, "intercept": np.nan, "slope": np.nan, "r2": np.nan}
    yy = np.log1p(np.clip(y, 0, None)) if use_log1p else np.log(np.clip(y, 1e-8, None))
    xxv = np.log1p(np.clip(p, 0, None)) if use_log1p else np.log(np.clip(p, 1e-8, None))
    xx = np.column_stack([np.ones(len(xxv)), xxv])
    beta = np.linalg.pinv(xx.T @ xx) @ xx.T @ yy
    yhat = xx @ beta
    ss_res = float(np.sum((yy - yhat) ** 2))
    ss_tot = float(np.sum((yy - yy.mean()) ** 2))
    return {"split": split, "target": target, "intercept": float(beta[0]), "slope": float(beta[1]), "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan}


def finite_conformal_quantile(scores, alpha):
    s = np.sort(np.asarray(scores, float))
    n = len(s)
    if n == 0:
        return 0.0, 0
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)
    return float(s[k - 1]), k


def topk_table(frame, score, split):
    d = frame[["problem_delivery", "delivery_burden_hours", score]].sort_values(score, ascending=False, kind="mergesort")
    total_b = float(d["delivery_burden_hours"].sum())
    total_p = float(d["problem_delivery"].sum())
    mean_b = float(d["delivery_burden_hours"].mean())
    rows = []
    for k in TOP_K_RATES:
        n = max(1, int(np.ceil(k * len(d))))
        h = d.head(n)
        rows.append({"split": split, "score": score, "top_k_rate": k, "orders_flagged": n, "problem_rate": float(h["problem_delivery"].mean()), "mean_observed_burden_hours": float(h["delivery_burden_hours"].mean()), "captured_problem_share": float(h["problem_delivery"].sum() / total_p) if total_p else np.nan, "captured_burden_share": float(h["delivery_burden_hours"].sum() / total_b) if total_b else np.nan, "lift_over_mean_burden": float(h["delivery_burden_hours"].mean() / mean_b) if mean_b else np.nan})
    return pd.DataFrame(rows)


def random_topk_reference(frame, split):
    prevalence = float(frame["problem_delivery"].mean())
    return pd.DataFrame([{"split": split, "score": "random_expected", "top_k_rate": k, "orders_flagged": int(np.ceil(k * len(frame))), "problem_rate": prevalence, "mean_observed_burden_hours": float(frame["delivery_burden_hours"].mean()), "captured_problem_share": k, "captured_burden_share": k, "lift_over_mean_burden": 1.0} for k in TOP_K_RATES])


def utility_table(frame, score, split):
    d = frame[["problem_delivery", "delivery_burden_hours", score]].sort_values(score, ascending=False, kind="mergesort")
    rows = []
    for k in TOP_K_RATES:
        n = max(1, int(np.ceil(k * len(d))))
        h = d.head(n)
        captured = float(h["delivery_burden_hours"].sum())
        false_alert = int(h["problem_delivery"].eq(0).sum())
        for c in UTILITY_FALSE_ALERT_COSTS:
            rows.append({"split": split, "score": score, "top_k_rate": k, "orders_flagged": n, "captured_burden_hours": captured, "false_alerts": false_alert, "false_alert_cost_hours": c, "operational_utility_hours": captured - false_alert * c})
    return pd.DataFrame(rows)


def risk_strata_table(frame, score, split, cuts):
    d = frame.copy()
    d["risk_stratum"] = pd.cut(d[score], [-np.inf, cuts[0], cuts[1], cuts[2], np.inf], labels=["low", "medium", "high", "very_high"])
    tab = d.groupby("risk_stratum", observed=False).agg(n=("order_id", "size"), problem_rate=("problem_delivery", "mean"), mean_score=(score, "mean"), mean_observed_burden_hours=("delivery_burden_hours", "mean"), median_observed_burden_hours=("delivery_burden_hours", "median")).reset_index()
    tab.insert(0, "split", split)
    return tab


def delay_bands_summary(frame, split):
    tab = frame.groupby("delay_duration_band", observed=False).agg(n=("order_id", "size"), problem_rate=("problem_delivery", "mean"), mean_lateness_hours=("lateness_hours", "mean"), mean_burden_hours=("delivery_burden_hours", "mean"), mean_p_problem=("p_problem", "mean"), mean_predicted_severity_hours=("predicted_severity_hours_if_problem", "mean"), mean_expected_burden_hours=("expected_burden_hours", "mean")).reset_index()
    tab.insert(0, "split", split)
    return tab


def borderline_audit(frame, threshold, split):
    d = frame.copy()
    d["distance_to_threshold"] = np.abs(d["p_problem"] - threshold)
    d["binary_correct"] = d["p_problem"].ge(threshold).astype(int).eq(d["problem_delivery"].astype(int))
    rows = []
    for margin in TRIAGE_MARGINS:
        for zone, g in [("near_threshold", d[d["distance_to_threshold"].le(margin)]), ("outside_threshold_zone", d[d["distance_to_threshold"].gt(margin)])]:
            rows.append({"split": split, "margin": margin, "zone": zone, "n": int(len(g)), "share": float(len(g) / len(d)) if len(d) else np.nan, "problem_rate": float(g["problem_delivery"].mean()) if len(g) else np.nan, "binary_accuracy": float(g["binary_correct"].mean()) if len(g) else np.nan, "error_rate": float(1 - g["binary_correct"].mean()) if len(g) else np.nan, "mean_expected_burden_hours": float(g["expected_burden_hours"].mean()) if len(g) else np.nan})
    return pd.DataFrame(rows)


def triage_metrics(frame, threshold, split):
    rows = []
    y = frame["problem_delivery"].astype(int).to_numpy()
    p = frame["p_problem"].to_numpy(float)
    for margin in TRIAGE_MARGINS:
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


def vif_sample(x):
    if len(x) == 0 or x.shape[1] == 0:
        return pd.DataFrame(columns=["term", "vif_ridge_sample"])
    z = x.sample(min(3000, len(x)), random_state=RANDOM_STATE).to_numpy(float)
    z = (z - z.mean(axis=0)) / np.maximum(z.std(axis=0), 1e-12)
    corr = np.nan_to_num(np.corrcoef(z, rowvar=False), nan=0.0, posinf=0.0, neginf=0.0)
    corr = corr + np.eye(corr.shape[0]) * 1e-6
    inv = np.linalg.pinv(corr)
    return pd.DataFrame({"term": x.columns, "vif_ridge_sample": np.diag(inv)})


def raw_field_audit(raw_columns, used_raw, predictor_raw, outcome_raw, descriptive_raw, pii_raw):
    rows = []
    for c in raw_columns:
        if c in predictor_raw:
            role = "retained_predictor_source"
            reason = "retained for ex_ante order characterization"
        elif c in outcome_raw:
            role = "retained_outcome_or_audit_source"
            reason = "used only for outcome reconstruction, timing consistency, or audit"
        elif c in descriptive_raw:
            role = "retained_descriptive_or_aggregation_source"
            reason = "used for order-level aggregation or descriptive summaries"
        elif c in pii_raw:
            role = "dropped_sensitive_or_identifier"
            reason = "excluded from analysis because it is personally identifying, credential-like, or non-analytical"
        elif c in used_raw:
            role = "retained_other"
            reason = "retained by the reconstruction pipeline"
        else:
            role = "dropped_not_required"
            reason = "not required for the prespecified reconstruction or predictor set"
        rows.append({"raw_field": c, "clean_field": snake(c), "role": role, "reason": reason})
    return pd.DataFrame(rows)


def feature_audit_table(nums, cats, lineage, qr_audit):
    rows = []
    for c in nums:
        rows.append({"encoded_term": c, "source_variable": c, "feature_type": "numeric_or_cyclic", "level": "", "reference_level": ""})
    base = pd.concat([pd.DataFrame(rows), lineage], ignore_index=True)
    out = base.merge(qr_audit, left_on="encoded_term", right_on="term", how="left").drop(columns=["term"])
    out["pre_qr"] = out["pre_qr"].fillna(False)
    out["constant_dropped"] = out["constant_dropped"].fillna(False)
    out["collinear_dropped"] = out["collinear_dropped"].fillna(False)
    out["retained_post_qr"] = out["retained_post_qr"].fillna(False)
    out.insert(0, "feature_order", np.arange(1, len(out) + 1))
    return out


def attach_predictions(frames, p_list, s_list, direct_list, threshold, conformal_q):
    for frame, p, s, direct in zip(frames, p_list, s_list, direct_list):
        frame["p_problem"] = prob_clip(p)
        frame["predicted_severity_hours_if_problem"] = np.clip(s, 1e-8, None)
        frame["expected_burden_hours"] = frame["p_problem"] * frame["predicted_severity_hours_if_problem"]
        frame["direct_burden_hours"] = np.clip(direct, 0, None)
        frame["alert_problem"] = frame["p_problem"].ge(threshold).astype(int)
        frame["distance_to_threshold"] = np.abs(frame["p_problem"] - threshold)
        frame["triage_action_m05"] = np.select([frame["p_problem"].ge(threshold + BORDERLINE_CASE_MARGIN), frame["p_problem"].le(threshold - BORDERLINE_CASE_MARGIN)], ["auto_alert", "auto_no_alert"], default="manual_review")
        frame["severity_pi_lower_90"] = np.maximum(0, frame["predicted_severity_hours_if_problem"] * np.exp(-conformal_q))
        frame["severity_pi_upper_90"] = frame["predicted_severity_hours_if_problem"] * np.exp(conformal_q)


def fit_scenario(train, val, test, x_train, x_val, x_test, infer=False):
    clf, beta_occ_pen = fit_occurrence_predictive(train["problem_delivery"].astype(int).to_numpy(), x_train)
    p_train = predict_occurrence(clf, x_train)
    p_val = predict_occurrence(clf, x_val)
    p_test = predict_occurrence(clf, x_test)
    sev_mask = train["problem_delivery"].eq(1).to_numpy()
    beta_sev, cov_sev, smearing = fit_log_linear(train.loc[sev_mask, "severity_hours_if_problem"].to_numpy(float), x_train.loc[sev_mask])
    s_train = predict_severity(beta_sev, smearing, x_train)
    s_val = predict_severity(beta_sev, smearing, x_val)
    s_test = predict_severity(beta_sev, smearing, x_test)
    beta_direct, direct_smearing = fit_direct_burden(train["delivery_burden_hours"].to_numpy(float), x_train)
    direct_train = predict_direct_burden(beta_direct, direct_smearing, x_train)
    direct_val = predict_direct_burden(beta_direct, direct_smearing, x_val)
    direct_test = predict_direct_burden(beta_direct, direct_smearing, x_test)
    threshold, threshold_candidates = select_threshold(val["problem_delivery"].astype(int).to_numpy(), p_val)
    val_pos = val["problem_delivery"].eq(1).to_numpy()
    residuals = np.abs(np.log(np.clip(val.loc[val_pos, "severity_hours_if_problem"].to_numpy(float), 1e-8, None)) - np.log(np.clip(s_val[val_pos], 1e-8, None)))
    conformal_q, conformal_rank = finite_conformal_quantile(residuals, CONFORMAL_ALPHA)
    attach_predictions([train, val, test], [p_train, p_val, p_test], [s_train, s_val, s_test], [direct_train, direct_val, direct_test], threshold, conformal_q)
    extra = {"beta_occ_penalized": beta_occ_pen, "beta_sev": beta_sev, "cov_sev": cov_sev, "smearing": smearing, "beta_direct": beta_direct, "direct_smearing": direct_smearing, "threshold": threshold, "threshold_candidates": threshold_candidates, "conformal_q": conformal_q, "conformal_rank": conformal_rank}
    if infer:
        try:
            beta_occ_inf, cov_occ_inf = fit_occurrence_inference(train["problem_delivery"].astype(int).to_numpy(), x_train)
        except Exception:
            beta_occ_inf = beta_occ_pen.copy()
            p = prob_clip(p_train)
            w = np.clip(p * (1 - p), 1e-8, None)
            xm = x_train.to_numpy(float)
            bread = np.linalg.pinv((xm.T * w) @ xm)
            scores = xm * (train["problem_delivery"].astype(int).to_numpy() - p)[:, None]
            n, k = xm.shape
            cov_occ_inf = pd.DataFrame(bread @ (scores.T @ scores) @ bread * n / max(n - k, 1), index=x_train.columns, columns=x_train.columns)
        extra["beta_occ_inference"] = beta_occ_inf
        extra["cov_occ_inference"] = cov_occ_inf
    return train, val, test, extra


def scenario_metrics(name, train, val, test, extra):
    occ_test = metrics_binary(test["problem_delivery"], test["p_problem"], extra["threshold"], "test")
    cal_test = calibration_summary(test["problem_delivery"], test["p_problem"], "test")
    sev_test = regression_metrics(test.loc[test["problem_delivery"].eq(1), "severity_hours_if_problem"], test.loc[test["problem_delivery"].eq(1), "predicted_severity_hours_if_problem"], "test", "conditional_severity")
    burden_test = regression_metrics(test["delivery_burden_hours"], test["expected_burden_hours"], "test", "expected_burden")
    direct_test = regression_metrics(test["delivery_burden_hours"], test["direct_burden_hours"], "test", "direct_burden")
    pos = test["problem_delivery"].eq(1)
    coverage = float(((test.loc[pos, "severity_hours_if_problem"] >= test.loc[pos, "severity_pi_lower_90"]) & (test.loc[pos, "severity_hours_if_problem"] <= test.loc[pos, "severity_pi_upper_90"])).mean()) if pos.sum() else np.nan
    tk = topk_table(test, "expected_burden_hours", "test")
    top10 = tk.loc[np.isclose(tk["top_k_rate"], 0.10)].iloc[0]
    top30 = tk.loc[np.isclose(tk["top_k_rate"], 0.30)].iloc[0]
    return {"scenario": name, "train_problem_prevalence": float(train["problem_delivery"].mean()), "validation_problem_prevalence": float(val["problem_delivery"].mean()), "test_problem_prevalence": float(test["problem_delivery"].mean()), "threshold": float(extra["threshold"]), "test_auc": occ_test["auc"], "test_average_precision": occ_test["average_precision"], "test_precision": occ_test["precision"], "test_recall": occ_test["recall"], "test_specificity": occ_test["specificity"], "test_mcc": occ_test["mcc"], "test_brier": occ_test["brier_score"], "test_ece": cal_test["ece"], "test_calibration_intercept": cal_test["calibration_intercept"], "test_calibration_slope": cal_test["calibration_slope"], "test_severity_mae": sev_test["mae_hours"], "test_severity_rmse": sev_test["rmse_hours"], "test_severity_spearman": sev_test["spearman_r"], "test_expected_mae": burden_test["mae_hours"], "test_expected_rmse": burden_test["rmse_hours"], "test_expected_spearman": burden_test["spearman_r"], "test_expected_mean_ratio": burden_test["mean_prediction_ratio"], "test_direct_mae": direct_test["mae_hours"], "test_direct_rmse": direct_test["rmse_hours"], "test_direct_spearman": direct_test["spearman_r"], "conformal_90_coverage": coverage, "top10_captured_burden": float(top10["captured_burden_share"]), "top30_captured_burden": float(top30["captured_burden_share"]), "top30_lift": float(top30["lift_over_mean_burden"]), "smearing_factor": float(extra["smearing"]), "conformal_q_abs_log_residual": float(extra["conformal_q"])}


def topk_overlap(primary, other, score, k):
    n = max(1, int(np.ceil(k * len(primary))))
    a = set(primary.nlargest(n, score, keep="first")["order_id"].tolist())
    b = set(other.nlargest(n, score, keep="first")["order_id"].tolist())
    return float(len(a & b) / n)


def build_sensitivity(train_base, val_base, test_base, x_train, x_val, x_test, sla_mode, sla_global):
    preliminary, primary_penalty, _ = apply_outcome(train_base.copy(), sla_mode, penalty_quantile=PRIMARY_CANCELLATION_QUANTILE, min_positive_mode="train_min", problem_rule="union")
    scenarios = [
        {"scenario": "primary_q95", "sla": sla_mode, "sla_variant": "mode_specific_mcc_calibrated", "penalty_quantile": 0.95, "penalty_override": None, "penalty_source": "training_quantile", "problem_rule": "union", "min_positive_mode": "train_min"},
        {"scenario": "penalty_q90", "sla": sla_mode, "sla_variant": "mode_specific_mcc_calibrated", "penalty_quantile": 0.90, "penalty_override": None, "penalty_source": "training_quantile", "problem_rule": "union", "min_positive_mode": "train_min"},
        {"scenario": "penalty_q99", "sla": sla_mode, "sla_variant": "mode_specific_mcc_calibrated", "penalty_quantile": 0.99, "penalty_override": None, "penalty_source": "training_quantile", "problem_rule": "union", "min_positive_mode": "train_min"},
        {"scenario": "penalty_125pct_primary", "sla": sla_mode, "sla_variant": "mode_specific_mcc_calibrated", "penalty_quantile": np.nan, "penalty_override": 1.25 * primary_penalty, "penalty_source": "stress_125pct_primary_p95", "problem_rule": "union", "min_positive_mode": "train_min"},
        {"scenario": "penalty_150pct_primary", "sla": sla_mode, "sla_variant": "mode_specific_mcc_calibrated", "penalty_quantile": np.nan, "penalty_override": 1.50 * primary_penalty, "penalty_source": "stress_150pct_primary_p95", "problem_rule": "union", "min_positive_mode": "train_min"},
        {"scenario": "no_grace_q95", "sla": zero_sla(train_base), "sla_variant": "zero_grace", "penalty_quantile": 0.95, "penalty_override": None, "penalty_source": "training_quantile", "problem_rule": "union", "min_positive_mode": "train_min"},
        {"scenario": "global_grace_q95", "sla": sla_global, "sla_variant": "global_mcc_calibrated", "penalty_quantile": 0.95, "penalty_override": None, "penalty_source": "training_quantile", "problem_rule": "union", "min_positive_mode": "train_min"},
        {"scenario": "status_only_q95", "sla": sla_mode, "sla_variant": "mode_specific_mcc_calibrated", "penalty_quantile": 0.95, "penalty_override": None, "penalty_source": "training_quantile", "problem_rule": "status_only", "min_positive_mode": "train_min"},
        {"scenario": "reconstructed_only_q95", "sla": sla_mode, "sla_variant": "mode_specific_mcc_calibrated", "penalty_quantile": 0.95, "penalty_override": None, "penalty_source": "training_quantile", "problem_rule": "reconstructed_only", "min_positive_mode": "train_min"},
        {"scenario": "fixed_floor_025_q95", "sla": sla_mode, "sla_variant": "mode_specific_mcc_calibrated", "penalty_quantile": 0.95, "penalty_override": None, "penalty_source": "training_quantile", "problem_rule": "union", "min_positive_mode": "fixed_epsilon"},
    ]
    summary_rows = []
    setting_rows = []
    predictions = {}
    primary_full = None
    primary_extra = None
    for i, s in enumerate(scenarios):
        qval = PRIMARY_CANCELLATION_QUANTILE if pd.isna(s["penalty_quantile"]) else float(s["penalty_quantile"])
        tr, penalty, floor = apply_outcome(train_base.copy(), s["sla"], penalty_quantile=qval, penalty=s["penalty_override"], min_positive_mode=s["min_positive_mode"], problem_rule=s["problem_rule"])
        va, _, _ = apply_outcome(val_base.copy(), s["sla"], penalty_quantile=qval, penalty=penalty, min_positive=floor, min_positive_mode=s["min_positive_mode"], problem_rule=s["problem_rule"])
        te, _, _ = apply_outcome(test_base.copy(), s["sla"], penalty_quantile=qval, penalty=penalty, min_positive=floor, min_positive_mode=s["min_positive_mode"], problem_rule=s["problem_rule"])
        tr, va, te, extra = fit_scenario(tr, va, te, x_train, x_val, x_test, infer=(i == 0))
        row = scenario_metrics(s["scenario"], tr, va, te, extra)
        row["cancellation_penalty_quantile"] = s["penalty_quantile"]
        row["cancellation_penalty_hours"] = penalty
        row["penalty_source"] = s["penalty_source"]
        row["minimum_positive_burden_hours"] = floor
        row["sla_variant"] = s["sla_variant"]
        row["problem_rule"] = s["problem_rule"]
        row["min_positive_mode"] = s["min_positive_mode"]
        summary_rows.append(row)
        setting_rows.append({"scenario": s["scenario"], "sla_variant": s["sla_variant"], "problem_rule": s["problem_rule"], "cancellation_penalty_quantile": s["penalty_quantile"], "cancellation_penalty_hours": penalty, "penalty_source": s["penalty_source"], "min_positive_mode": s["min_positive_mode"], "minimum_positive_burden_hours": floor})
        predictions[s["scenario"]] = te[["order_id", "expected_burden_hours", "p_problem", "predicted_severity_hours_if_problem", "direct_burden_hours", "delivery_burden_hours", "problem_delivery"]].copy()
        if i == 0:
            primary_full = (tr, va, te)
            primary_extra = extra
    summary = pd.DataFrame(summary_rows)
    primary_row = summary.loc[summary["scenario"].eq("primary_q95")].iloc[0]
    stability_rows = []
    ptest = predictions["primary_q95"]
    for s in scenarios:
        name = s["scenario"]
        otest = predictions[name]
        merged = ptest[["order_id", "expected_burden_hours"]].merge(otest[["order_id", "expected_burden_hours"]], on="order_id", suffixes=("_primary", "_scenario"))
        rank_r, _ = safe_spearman(merged["expected_burden_hours_primary"], merged["expected_burden_hours_scenario"])
        stability_rows.append({"scenario": name, "score_spearman_vs_primary": rank_r, "top10_overlap_vs_primary": topk_overlap(ptest, otest, "expected_burden_hours", 0.10), "top30_overlap_vs_primary": topk_overlap(ptest, otest, "expected_burden_hours", 0.30)})
    stability = pd.DataFrame(stability_rows)
    summary = summary.merge(stability, on="scenario", how="left")
    summary["delta_test_expected_spearman_vs_primary"] = summary["test_expected_spearman"] - float(primary_row["test_expected_spearman"])
    summary["delta_top30_capture_vs_primary"] = summary["top30_captured_burden"] - float(primary_row["top30_captured_burden"])
    return primary_full, primary_extra, summary, pd.DataFrame(setting_rows), predictions

def _excel_safe_value(v):
    if pd.isna(v):
        return None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.bool_):
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
            safe = (base[:31 - len(suffix)] + suffix)[:31]
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
                elif isinstance(v, int):
                    ws.write_number(irow, j, v, int_fmt)
                elif isinstance(v, float):
                    ws.write_number(irow, j, v, num_fmt)
                else:
                    ws.write(irow, j, str(v), text_fmt)
        ws.freeze_panes(1, 0)
        if cols:
            ws.autofilter(0, 0, max(rows, 1), cols - 1)
        for j, col in enumerate(df2.columns):
            sample = df2[col].dropna().astype(str).head(200).tolist()
            max_len = max([len(str(col))] + [len(x) for x in sample]) if sample else len(str(col))
            ws.set_column(j, j, min(max(max_len + 2, 10), 42))
    wb.close()


def plot_pipeline(figdir):
    fig, ax = plt.subplots(figsize=(15, 4.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    boxes = [
        (0.02, "Order-level\nreconstruction"),
        (0.215, "Burden outcome\nreconstruction"),
        (0.41, "Occurrence + severity\nmodels"),
        (0.605, "Expected burden\nP(problem) × E(severity|problem)"),
        (0.80, "Decision support\ntop-k, strata, triage"),
    ]
    for x, label in boxes:
        ax.add_patch(plt.Rectangle((x, 0.34), 0.16, 0.32, fill=False, linewidth=1.5))
        ax.text(x + 0.08, 0.50, label, ha="center", va="center", fontsize=11)
    for x in [0.18, 0.375, 0.57, 0.765]:
        ax.annotate("", xy=(x + 0.025, 0.50), xytext=(x - 0.005, 0.50), arrowprops={"arrowstyle": "->", "lw": 1.4})
    ax.text(0.5, 0.16, "Training-derived preprocessing and grace calibration; validation-selected threshold and conformal calibration; untouched temporal test evaluation", ha="center", va="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(figdir / "00_conceptual_pipeline.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_figures(outdir, test, occurrence_calibration, expected_calibration, baseline_topk, sensitivity):
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    plot_pipeline(figdir)
    y = test["problem_delivery"].astype(int).to_numpy()
    p = prob_clip(test["p_problem"].to_numpy(float))
    thresholds = np.unique(np.sort(p))
    order = np.argsort(-p)
    ys = y[order]
    tps = np.cumsum(ys)
    fps = np.cumsum(1 - ys)
    tpr = tps / max(int(y.sum()), 1)
    fpr = fps / max(int((1 - y).sum()), 1)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Occurrence ROC, temporal test AUC={auc_rank(y, p):.4f}")
    plt.tight_layout()
    plt.savefig(figdir / "01_occurrence_roc.png", dpi=300, bbox_inches="tight")
    plt.close()
    cal = occurrence_calibration[occurrence_calibration["split"].eq("test")]
    plt.figure(figsize=(7, 6))
    plt.plot(cal["mean_p"], cal["observed_rate"], marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed problem rate")
    plt.title("Occurrence calibration by decile")
    plt.tight_layout()
    plt.savefig(figdir / "02_occurrence_calibration.png", dpi=300, bbox_inches="tight")
    plt.close()
    ecal = expected_calibration[expected_calibration["split"].eq("test")]
    lim = float(max(ecal["mean_observed"].max(), ecal["mean_predicted"].max())) if len(ecal) else 1.0
    plt.figure(figsize=(7, 6))
    plt.plot(ecal["mean_predicted"], ecal["mean_observed"], marker="o")
    plt.plot([0, lim], [0, lim], linestyle="--")
    plt.xlabel("Mean predicted burden, hours")
    plt.ylabel("Mean observed burden, hours")
    plt.title("Expected-burden calibration by decile")
    plt.tight_layout()
    plt.savefig(figdir / "03_expected_burden_calibration.png", dpi=300, bbox_inches="tight")
    plt.close()
    plt.figure(figsize=(8, 6))
    for score in baseline_topk["score"].unique():
        g = baseline_topk[baseline_topk["score"].eq(score)]
        plt.plot(g["top_k_rate"], g["captured_burden_share"], marker="o", label=score)
    plt.xlabel("Top-k monitoring capacity")
    plt.ylabel("Captured burden share")
    plt.title("Same-data ranking comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figdir / "04_same_data_topk_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    sens = sensitivity.copy()
    plt.figure(figsize=(11, 6))
    plt.bar(sens["scenario"], sens["top30_captured_burden"])
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Top-30% captured burden share")
    plt.title("Outcome-reconstruction sensitivity")
    plt.tight_layout()
    plt.savefig(figdir / "05_sensitivity_top30.png", dpi=300, bbox_inches="tight")
    plt.close()


def prepare_data(inp):
    raw = pd.read_csv(inp, encoding="latin1", low_memory=False)
    required = ["Order Id", "Delivery Status", "Order Status", "Type", "Shipping Mode", "Customer Segment", "Order Region", "Department Name", "Order Item Quantity", "Order Item Product Price", "Order Item Discount Rate", "Order Profit Per Order", "order date (DateOrders)", "shipping date (DateOrders)", "Days for shipping (real)", "Days for shipment (scheduled)"]
    optional = ["Late_delivery_risk", "Market", "Category Name", "Sales", "Order Item Total"]
    missing_required = [c for c in required if c not in raw.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")
    selected = required + [c for c in optional if c in raw.columns]
    selected_table = pd.DataFrame({"original_column": selected, "clean_column": [snake(c) for c in selected]})
    predictor_raw = {"Type", "Shipping Mode", "Customer Segment", "Order Region", "Department Name", "Market", "Category Name", "Order Item Quantity", "Order Item Product Price", "Order Item Discount Rate", "Order Profit Per Order", "Sales", "Order Item Total", "order date (DateOrders)"}
    outcome_raw = {"Delivery Status", "Order Status", "Late_delivery_risk", "shipping date (DateOrders)", "Days for shipping (real)", "Days for shipment (scheduled)"}
    descriptive_raw = {"Order Id"}
    pii_raw = {"Customer City", "Customer Country", "Customer Email", "Customer Fname", "Customer Id", "Customer Lname", "Customer Password", "Customer State", "Customer Street", "Customer Zipcode", "Latitude", "Longitude", "Order City", "Order Country", "Order Customer Id", "Order State", "Order Zipcode", "Product Description", "Product Image", "Product Name"}
    raw_audit = raw_field_audit(raw.columns.tolist(), set(selected), predictor_raw, outcome_raw, descriptive_raw, pii_raw)
    data = raw[selected].copy().rename(columns={c: snake(c) for c in selected})
    missing_before = missingness_table(data, "selected_raw")
    quality_before = data_quality_line_table(data, "selected_raw")
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
    orders = add_time_features(orders.sort_values(["order_date", "order_id"]).reset_index(drop=True))
    n = len(orders)
    cut_train = int(np.floor(TRAIN_SIZE * n))
    cut_val = int(np.floor((TRAIN_SIZE + VALIDATION_SIZE) * n))
    train_base = orders.iloc[:cut_train].copy().reset_index(drop=True)
    val_base = orders.iloc[cut_train:cut_val].copy().reset_index(drop=True)
    test_base = orders.iloc[cut_val:].copy().reset_index(drop=True)
    info = {"raw": raw, "selected": selected, "selected_table": selected_table, "raw_audit": raw_audit, "missing_all": pd.concat([missing_before, missing_after], ignore_index=True), "quality_all": pd.concat([quality_before, quality_after, quality_orders], ignore_index=True), "conflicts": pd.DataFrame(conflicts), "rows_before_filter": rows_before_filter, "rows_after_filter": rows_after_filter, "order_counts": order_counts, "orders": orders}
    return train_base, val_base, test_base, info


def prepare_design(train, val, test):
    cats = [c for c in ["type", "shipping_mode", "customer_segment", "market", "order_region", "department_name"] if c in train.columns]
    nums = [c for c in ["department_count", "category_count", "order_lines", "quantity_sum", "quantity_mean", "product_price_mean", "discount_rate_mean", "discount_rate_max", "sales_sum", "order_profit_first", "month_sin", "month_cos", "dow_sin", "dow_cos", "hour_sin", "hour_cos"] if c in train.columns]
    required = cats + nums
    keep_train = train[required].notna().all(axis=1)
    keep_val = val[required].notna().all(axis=1)
    keep_test = test[required].notna().all(axis=1)
    train = train.loc[keep_train].reset_index(drop=True)
    val = val.loc[keep_val].reset_index(drop=True)
    test = test.loc[keep_test].reset_index(drop=True)
    scaler = RobustScaler()
    xn_train = pd.DataFrame(scaler.fit_transform(train[nums]), columns=nums)
    xn_val = pd.DataFrame(scaler.transform(val[nums]), columns=nums)
    xn_test = pd.DataFrame(scaler.transform(test[nums]), columns=nums)
    xd_train, [xd_val, xd_test], refs, unseen, lineage = build_dummies(train, [val, test], cats)
    x_train_pre = pd.concat([xn_train.reset_index(drop=True), xd_train.reset_index(drop=True)], axis=1).astype(float)
    x_val_pre = pd.concat([xn_val.reset_index(drop=True), xd_val.reset_index(drop=True)], axis=1).astype(float)
    x_test_pre = pd.concat([xn_test.reset_index(drop=True), xd_test.reset_index(drop=True)], axis=1).astype(float)
    x_train, x_val, x_test, qr_audit = rank_prune(x_train_pre, x_val_pre, x_test_pre)
    x_train.insert(0, "const", 1.0)
    x_val.insert(0, "const", 1.0)
    x_test.insert(0, "const", 1.0)
    feature_audit = feature_audit_table(nums, cats, lineage, qr_audit)
    scaler_info = pd.DataFrame({"variable": nums, "center_median_train": scaler.center_, "scale_iqr_train": scaler.scale_})
    return train, val, test, x_train, x_val, x_test, {"cats": cats, "nums": nums, "refs": refs, "unseen": unseen, "feature_audit": feature_audit, "qr_audit": qr_audit, "scaler_info": scaler_info, "pre_qr_features": x_train_pre.columns.tolist(), "post_qr_features": x_train.columns.drop("const").tolist()}


def environment_table():
    return pd.DataFrame([
        {"component": "python", "version": platform.python_version()},
        {"component": "platform", "version": platform.platform()},
        {"component": "numpy", "version": np.__version__},
        {"component": "pandas", "version": pd.__version__},
        {"component": "scipy", "version": scipy.__version__},
        {"component": "scikit-learn", "version": sklearn.__version__},
        {"component": "statsmodels", "version": statsmodels.__version__},
        {"component": "matplotlib", "version": plt.matplotlib.__version__},
        {"component": "random_state", "version": str(RANDOM_STATE)},
    ])


def main(input_file, output_dir=None):
    inp = Path(input_file).expanduser().resolve()
    if not inp.exists():
        raise FileNotFoundError(str(inp))
    out = Path(output_dir).expanduser().resolve() if output_dir else inp.parent / f"{inp.stem}_revision_analysis_output"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    print("Reading and reconstructing order-level data", flush=True)
    train_base, val_base, test_base, info = prepare_data(inp)
    train_base, val_base, test_base, x_train, x_val, x_test, design = prepare_design(train_base, val_base, test_base)
    print("Calibrating training-derived grace terms", flush=True)
    sla_mode, sla_grid_mode = calibrate_sla(train_base, mode_specific=True)
    sla_global, sla_grid_global = calibrate_sla(train_base, mode_specific=False)
    sla_mode.insert(0, "sla_definition", "training_derived_mode_specific_grace_not_contractual_sla")
    sla_global.insert(0, "sla_definition", "training_derived_global_grace_not_contractual_sla")
    print("Running primary model and robustness scenarios", flush=True)
    primary_full, primary_extra, sensitivity, sensitivity_settings, sensitivity_predictions = build_sensitivity(train_base, val_base, test_base, x_train, x_val, x_test, sla_mode, sla_global)
    train, val, test = primary_full
    threshold = primary_extra["threshold"]
    print("Building primary statistical outputs", flush=True)
    occ_metrics = pd.DataFrame([metrics_binary(train["problem_delivery"], train["p_problem"], threshold, "train"), metrics_binary(val["problem_delivery"], val["p_problem"], threshold, "validation"), metrics_binary(test["problem_delivery"], test["p_problem"], threshold, "test")])
    occ_cal = pd.concat([calibration_table(train["problem_delivery"], train["p_problem"], "train"), calibration_table(val["problem_delivery"], val["p_problem"], "validation"), calibration_table(test["problem_delivery"], test["p_problem"], "test")], ignore_index=True)
    occ_cal_summary = pd.DataFrame([calibration_summary(train["problem_delivery"], train["p_problem"], "train"), calibration_summary(val["problem_delivery"], val["p_problem"], "validation"), calibration_summary(test["problem_delivery"], test["p_problem"], "test")])
    sev_metrics = pd.DataFrame([regression_metrics(g.loc[g["problem_delivery"].eq(1), "severity_hours_if_problem"], g.loc[g["problem_delivery"].eq(1), "predicted_severity_hours_if_problem"], split, "conditional_severity") for g, split in [(train, "train"), (val, "validation"), (test, "test")]])
    burden_metrics = pd.DataFrame([regression_metrics(g["delivery_burden_hours"], g["expected_burden_hours"], split, "expected_burden") for g, split in [(train, "train"), (val, "validation"), (test, "test")]])
    direct_metrics = pd.DataFrame([regression_metrics(g["delivery_burden_hours"], g["direct_burden_hours"], split, "direct_log1p_burden") for g, split in [(train, "train"), (val, "validation"), (test, "test")]])
    sev_cal = pd.concat([continuous_calibration_table(g.loc[g["problem_delivery"].eq(1), "severity_hours_if_problem"], g.loc[g["problem_delivery"].eq(1), "predicted_severity_hours_if_problem"], split, "conditional_severity") for g, split in [(train, "train"), (val, "validation"), (test, "test")]], ignore_index=True)
    sev_cal_summary = pd.DataFrame([log_calibration_summary(g.loc[g["problem_delivery"].eq(1), "severity_hours_if_problem"], g.loc[g["problem_delivery"].eq(1), "predicted_severity_hours_if_problem"], split, "conditional_severity", False) for g, split in [(train, "train"), (val, "validation"), (test, "test")]])
    expected_cal = pd.concat([continuous_calibration_table(g["delivery_burden_hours"], g["expected_burden_hours"], split, "expected_burden") for g, split in [(train, "train"), (val, "validation"), (test, "test")]], ignore_index=True)
    expected_cal_summary = pd.DataFrame([log_calibration_summary(g["delivery_burden_hours"], g["expected_burden_hours"], split, "expected_burden", True) for g, split in [(train, "train"), (val, "validation"), (test, "test")]])
    pos_test = test["problem_delivery"].eq(1)
    conformal = pd.DataFrame([{"split": "test", "alpha": CONFORMAL_ALPHA, "nominal_coverage": 1 - CONFORMAL_ALPHA, "calibration_n": int(val["problem_delivery"].sum()), "finite_sample_rank": int(primary_extra["conformal_rank"]), "q_abs_log_residual": float(primary_extra["conformal_q"]), "observed_coverage": float(((test.loc[pos_test, "severity_hours_if_problem"] >= test.loc[pos_test, "severity_pi_lower_90"]) & (test.loc[pos_test, "severity_hours_if_problem"] <= test.loc[pos_test, "severity_pi_upper_90"])).mean()), "mean_interval_width_hours": float((test.loc[pos_test, "severity_pi_upper_90"] - test.loc[pos_test, "severity_pi_lower_90"]).mean())}])
    occ_coef = coef_table(primary_extra["beta_occ_inference"], primary_extra["cov_occ_inference"], "occurrence_unpenalized_inference_refit", "odds_ratio")
    sev_coef = coef_table(primary_extra["beta_sev"], primary_extra["cov_sev"], "severity_log_linear", "severity_ratio")
    pen_coef = pd.DataFrame({"term": primary_extra["beta_occ_penalized"].index, "penalized_beta": primary_extra["beta_occ_penalized"].values, "penalized_odds_ratio": np.exp(np.clip(primary_extra["beta_occ_penalized"].values, -700, 700))})
    groups = {"all_predictors": x_train.columns.drop("const").tolist(), "numerical_variables": [c for c in design["nums"] if c in x_train.columns], "cyclic_time_variables": [c for c in ["month_sin", "month_cos", "dow_sin", "dow_cos", "hour_sin", "hour_cos"] if c in x_train.columns]}
    for c in design["cats"]:
        groups[c] = [v for v in x_train.columns if v.startswith(c + "_")]
    factor_tests = pd.concat([group_wald(primary_extra["beta_occ_inference"], primary_extra["cov_occ_inference"], groups, "occurrence_unpenalized_inference_refit"), group_wald(primary_extra["beta_sev"], primary_extra["cov_sev"], groups, "severity_log_linear")], ignore_index=True)
    baseline_topk = pd.concat([topk_table(test, "expected_burden_hours", "test"), topk_table(test, "p_problem", "test"), topk_table(test, "predicted_severity_hours_if_problem", "test"), topk_table(test, "direct_burden_hours", "test"), random_topk_reference(test, "test")], ignore_index=True)
    primary_topk_all = pd.concat([topk_table(g, "expected_burden_hours", split) for g, split in [(train, "train"), (val, "validation"), (test, "test")]], ignore_index=True)
    utility = pd.concat([utility_table(g, "expected_burden_hours", split) for g, split in [(train, "train"), (val, "validation"), (test, "test")]], ignore_index=True)
    cuts = np.quantile(val["expected_burden_hours"], [0.25, 0.50, 0.75])
    strata = pd.concat([risk_strata_table(g, "expected_burden_hours", split, cuts) for g, split in [(train, "train"), (val, "validation"), (test, "test")]], ignore_index=True)
    delay = pd.concat([delay_bands_summary(g, split) for g, split in [(train, "train"), (val, "validation"), (test, "test")]], ignore_index=True)
    outcome = pd.concat([train.assign(split="train"), val.assign(split="validation"), test.assign(split="test")], ignore_index=True).groupby(["split", "clean_delivery_outcome"], as_index=False).agg(n=("order_id", "size"), mean_burden_hours=("delivery_burden_hours", "mean"), mean_p_problem=("p_problem", "mean"), mean_expected_burden=("expected_burden_hours", "mean"))
    borderline = pd.concat([borderline_audit(g, threshold, split) for g, split in [(train, "train"), (val, "validation"), (test, "test")]], ignore_index=True)
    triage = pd.concat([triage_metrics(g, threshold, split) for g, split in [(train, "train"), (val, "validation"), (test, "test")]], ignore_index=True)
    test_borderline_cases = test[test["distance_to_threshold"].le(BORDERLINE_CASE_MARGIN)].sort_values(["distance_to_threshold", "expected_burden_hours"], ascending=[True, False]).head(MAX_BORDERLINE_CASES).copy()
    pred_cols = ["order_id", "order_date", "shipping_date", "delivery_status", "order_status", "type", "shipping_mode", "customer_segment", "market", "order_region", "department_name", "clean_delivery_outcome", "delay_duration_band", "actual_shipping_hours", "scheduled_shipping_hours", "grace_hours", "deadline_hours", "lateness_hours", "lateness_minutes", "is_canceled", "raw_late_status", "reconstructed_late", "problem_delivery", "severity_hours_if_problem", "delivery_burden_hours", "p_problem", "predicted_severity_hours_if_problem", "expected_burden_hours", "direct_burden_hours", "alert_problem", "distance_to_threshold", "triage_action_m05", "severity_pi_lower_90", "severity_pi_upper_90"]
    pred = test[[c for c in pred_cols if c in test.columns]].copy()
    for d in [pred, test_borderline_cases]:
        for c in d.columns:
            if pd.api.types.is_datetime64_any_dtype(d[c]) or str(d[c].dtype) == "category":
                d[c] = d[c].astype(str)
    primary_sens = sensitivity.loc[sensitivity["scenario"].eq("primary_q95")].iloc[0]
    audit = pd.DataFrame([
        {"item": "input_file", "value": str(inp)},
        {"item": "input_sha256", "value": sha256_file(inp)},
        {"item": "raw_rows", "value": int(len(info["raw"]))},
        {"item": "raw_columns", "value": int(info["raw"].shape[1])},
        {"item": "selected_raw_columns", "value": int(len(info["selected"]))},
        {"item": "line_rows_before_filter", "value": int(info["rows_before_filter"])},
        {"item": "line_rows_after_filter", "value": int(info["rows_after_filter"])},
        {"item": "orders_after_aggregation", "value": int(len(info["orders"]))},
        {"item": "train_rows", "value": int(len(train))},
        {"item": "validation_rows", "value": int(len(val))},
        {"item": "test_rows", "value": int(len(test))},
        {"item": "features_pre_qr", "value": int(len(design["pre_qr_features"]))},
        {"item": "features_post_qr", "value": int(len(design["post_qr_features"]))},
        {"item": "occurrence_prediction_model", "value": f"L2 logistic regression C={OCCURRENCE_C}"},
        {"item": "occurrence_inference_model", "value": "unpenalized logistic companion refit with HC1 covariance"},
        {"item": "severity_model", "value": "log-linear OLS with HC1 covariance and Duan smearing"},
        {"item": "direct_baseline", "value": "log1p linear model with smearing retransformation"},
        {"item": "sla_interpretation", "value": "training-derived reconstruction grace; not a contractual carrier SLA"},
        {"item": "primary_cancellation_penalty_quantile", "value": PRIMARY_CANCELLATION_QUANTILE},
        {"item": "primary_cancellation_penalty_hours", "value": float(primary_sens["cancellation_penalty_hours"])},
        {"item": "primary_minimum_positive_burden_hours", "value": float(primary_sens["minimum_positive_burden_hours"])},
        {"item": "threshold_selected_on", "value": "validation"},
        {"item": "threshold_objective", "value": "MCC with balanced-accuracy and lower-threshold tie breaks"},
        {"item": "threshold", "value": float(threshold)},
        {"item": "conformal_quantile_rule", "value": "finite-sample split-conformal order statistic on absolute log residuals"},
        {"item": "causal_scope", "value": "predictive_and_associational_not_causal"},
        {"item": "sensitivity_scenarios", "value": int(len(sensitivity))},
    ])
    readme = pd.DataFrame([
        {"section": "Primary estimand", "description": "Cancellation-aware semicontinuous delivery burden with a structural zero and positive conditional severity."},
        {"section": "Occurrence prediction", "description": f"Lightly penalized L2 logistic regression with C={OCCURRENCE_C}."},
        {"section": "Occurrence inference", "description": "Separate unpenalized logistic companion refit with HC1 covariance for odds ratios, confidence intervals, and factor-level Wald tests."},
        {"section": "Severity", "description": "Log-linear positive-severity model with HC1 covariance and Duan smearing."},
        {"section": "Expected burden", "description": "P(problem|X) multiplied by predicted E(severity|problem,X)."},
        {"section": "SLA wording", "description": "Grace terms are training-derived reconstruction tolerances calibrated by MCC; they are not contractual carrier SLAs."},
        {"section": "Sensitivity", "description": "Primary P95 penalty is compared with P90/P99, 125% and 150% high-penalty stress tests, no grace, global grace, status-only and reconstructed-only problem definitions, and a fixed 0.25-hour positive floor."},
        {"section": "Same-data benchmarks", "description": "Expected burden is compared with occurrence-only ranking, severity-only ranking, a direct log1p burden model, and a random expected reference."},
        {"section": "Validation", "description": "Chronological 60/20/20 split; preprocessing and outcome-calibration quantities are learned from training, threshold and conformal calibration from validation, and final claims from the temporal test set."},
    ])
    model_equations = pd.DataFrame([
        {"equation": "Occurrence", "formula": "logit[P(problem_delivery=1|X)] = X beta"},
        {"equation": "Severity", "formula": "log[E(severity_hours|problem_delivery=1,X)] = X gamma; prediction uses Duan smearing"},
        {"equation": "Expected burden", "formula": "E(burden_hours|X) = P(problem_delivery=1|X) * E(severity_hours|problem_delivery=1,X)"},
        {"equation": "Direct comparator", "formula": "log(1+burden_hours) = X delta + error; prediction uses smearing retransformation"},
    ])
    env = environment_table()
    final_features = pd.DataFrame({"feature_order": np.arange(1, len(design["post_qr_features"]) + 1), "encoded_feature_post_qr": design["post_qr_features"]})
    threshold_candidates = primary_extra["threshold_candidates"].copy()
    sheets = {
        "README": readme,
        "Data_Audit": audit,
        "Environment": env,
        "Raw_Field_Audit": info["raw_audit"],
        "Selected_Raw_Columns": info["selected_table"],
        "Feature_Audit": design["feature_audit"],
        "Final_Features_Post_QR": final_features,
        "QR_Audit": design["qr_audit"],
        "Reference_Levels": design["refs"],
        "Unseen_Levels": design["unseen"],
        "Scaler_Train": design["scaler_info"],
        "Missingness": info["missing_all"],
        "Data_Quality_Audit": info["quality_all"],
        "Order_Conflicts": info["conflicts"],
        "Model_Equations": model_equations,
        "SLA_Grace_Mode": sla_mode,
        "SLA_Grace_Global": sla_global,
        "SLA_Grid_Mode": sla_grid_mode,
        "SLA_Grid_Global": sla_grid_global,
        "Sensitivity_Settings": sensitivity_settings,
        "Sensitivity_Summary": sensitivity,
        "Outcome_Distribution": outcome,
        "Delay_Bands": delay,
        "Occurrence_Metrics": occ_metrics,
        "Occurrence_Calibration": occ_cal,
        "Occurrence_Cal_Summary": occ_cal_summary,
        "Severity_Metrics": sev_metrics,
        "Severity_Calibration": sev_cal,
        "Severity_Cal_Summary": sev_cal_summary,
        "Expected_Burden_Metrics": burden_metrics,
        "Expected_Burden_Cal": expected_cal,
        "Expected_Burden_Cal_Sum": expected_cal_summary,
        "Direct_Burden_Metrics": direct_metrics,
        "Conformal_Intervals": conformal,
        "Threshold_Candidates": threshold_candidates,
        "Borderline_Audit": borderline,
        "Triage_Metrics": triage,
        "TopK_Primary": primary_topk_all,
        "TopK_SameData_Baselines": baseline_topk,
        "Operational_Utility": utility,
        "Risk_Strata": strata,
        "Occurrence_Coef_Infer": occ_coef,
        "Occurrence_Coef_Penalized": pen_coef,
        "Severity_Coef": sev_coef,
        "Factor_Tests": factor_tests,
        "VIF_Sample": vif_sample(x_train.drop(columns=["const"])),
        "Borderline_Cases_Test": test_borderline_cases[[c for c in pred_cols if c in test_borderline_cases.columns]],
        "Pred_Test_Sample": pred.head(2000),
    }
    print("Writing reproducibility outputs", flush=True)
    pred.to_csv(out / "pred_test_full.csv", index=False, encoding="utf-8-sig")
    sensitivity.to_csv(out / "sensitivity_summary.csv", index=False, encoding="utf-8-sig")
    design["feature_audit"].to_csv(out / "feature_audit.csv", index=False, encoding="utf-8-sig")
    info["raw_audit"].to_csv(out / "raw_field_audit.csv", index=False, encoding="utf-8-sig")
    final_features.to_csv(out / "final_features_post_qr.csv", index=False, encoding="utf-8-sig")
    baseline_topk.to_csv(out / "same_data_topk_baselines.csv", index=False, encoding="utf-8-sig")
    sla_mode.to_csv(out / "sla_training_derived_grace.csv", index=False, encoding="utf-8-sig")
    excel_path = out / "two_part_delivery_glm_revision_results.xlsx"
    write_results_workbook(excel_path, sheets)
    plot_figures(out, test, occ_cal, expected_cal, baseline_topk, sensitivity)
    requirements = [f"numpy=={np.__version__}", f"pandas=={pd.__version__}", f"scipy=={scipy.__version__}", f"scikit-learn=={sklearn.__version__}", f"statsmodels=={statsmodels.__version__}", f"matplotlib=={plt.matplotlib.__version__}", "xlsxwriter"]
    (out / "requirements_runtime.txt").write_text("\n".join(requirements) + "\n", encoding="utf-8")
    (out / "analysis_notes.txt").write_text("Primary analysis uses a training-derived mode-specific grace calibrated by MCC. This grace is a reconstruction tolerance and not a contractual carrier SLA. The primary cancellation penalty is the training-set 95th percentile of positive delivered-problem burden. Robustness scenarios include the 90th and 99th percentiles, 125% and 150% high-penalty stress tests relative to the primary P95 penalty, zero grace, a globally calibrated grace, status-only and reconstructed-only problem definitions, and a fixed 0.25-hour positive floor. Occurrence probabilities used for prediction come from an L2-penalized logistic model. Odds-ratio inference uses a separate unpenalized logistic companion refit with HC1 covariance.\n", encoding="utf-8")
    script_path = Path(__file__).resolve()
    manifest_rows = []
    for p in sorted(out.rglob("*")):
        if p.is_file():
            manifest_rows.append({"file": str(p.relative_to(out)), "bytes": int(p.stat().st_size), "sha256": sha256_file(p)})
    manifest_rows.append({"file": script_path.name, "bytes": int(script_path.stat().st_size), "sha256": sha256_file(script_path)})
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(out / "file_manifest_sha256.csv", index=False, encoding="utf-8-sig")
    zpath = inp.parent / f"{inp.stem}_revision_analysis_package.zip"
    if zpath.exists():
        zpath.unlink()
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(script_path, arcname=script_path.name)
        for p in out.rglob("*"):
            if p.is_file():
                z.write(p, arcname=str(Path(out.name) / p.relative_to(out)))
    print("DONE", flush=True)
    print(f"Output folder: {out}", flush=True)
    print(f"Workbook: {excel_path}", flush=True)
    print(f"Package: {zpath}", flush=True)
    return out, zpath


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", nargs="?", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    if args.input_file is None:
        matches = sorted(Path.cwd().glob("*DataCo*Supply*Chain*Dataset*.csv"))
        if not matches:
            raise FileNotFoundError("Provide input CSV path")
        input_file = matches[0]
    else:
        input_file = args.input_file
    main(input_file, args.output_dir)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)

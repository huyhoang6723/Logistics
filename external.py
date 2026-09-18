import argparse
import hashlib
import json
import math
import os
import platform
import re
import sys
import zipfile
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import qr
from scipy.special import expit, logit
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

RANDOM_STATE = 2026
DEFAULT_RARE_MIN_COUNT = 30
DEFAULT_CALIBRATION_GROUPS = 10
DEFAULT_TOP_K = [0.05, 0.10, 0.20, 0.30]
DEFAULT_CONFORMAL_ALPHA = 0.10
DEFAULT_C = 10.0
MIN_POSITIVE_HOURS = 0.25
RESOLVED_STATUSES = {"delivered", "canceled", "unavailable"}
FAILURE_STATUSES = {"canceled", "unavailable"}


def snake(x):
    x = str(x).strip()
    x = re.sub(r"[()]+", "", x)
    x = re.sub(r"[^0-9a-zA-Z]+", "_", x)
    x = re.sub(r"_+", "_", x)
    return x.strip("_").lower()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_dt(s):
    return pd.to_datetime(s, errors="coerce")


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


def finite_sample_quantile(values, alpha):
    x = np.sort(np.asarray(values, dtype=float))
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, 0
    rank = int(np.ceil((len(x) + 1) * (1 - alpha)))
    rank = min(max(rank, 1), len(x))
    return float(x[rank - 1]), rank


def group_mode(frame, group_col, value_col, output_col):
    d = frame[[group_col, value_col]].dropna().copy()
    if len(d) == 0:
        return pd.DataFrame(columns=[group_col, output_col])
    c = d.groupby([group_col, value_col], sort=False).size().reset_index(name="n")
    c = c.sort_values([group_col, "n", value_col], ascending=[True, False, True]).drop_duplicates(group_col)
    return c[[group_col, value_col]].rename(columns={value_col: output_col})


def haversine_km(lat1, lon1, lat2, lon2):
    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon1 = np.radians(np.asarray(lon1, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    lon2 = np.radians(np.asarray(lon2, dtype=float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    a = np.clip(a, 0, 1)
    return 6371.0088 * 2 * np.arcsin(np.sqrt(a))


def find_results_file(base, requested=None):
    if requested:
        p = Path(requested).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Prior results workbook not found: {p}")
        return p
    preferred = [
        base / "two_part_delivery_glm_revision_results.xlsx",
        base / "results.xlsx",
    ]
    for p in preferred:
        if p.exists():
            return p
    candidates = sorted([p for p in base.glob("*.xlsx") if "external" not in p.name.lower()])
    for p in candidates:
        try:
            names = pd.ExcelFile(p).sheet_names
            if "Data_Audit" in names and "Occurrence_Metrics" in names and "Severity_Coef" in names:
                return p
        except Exception:
            pass
    raise FileNotFoundError("Could not find the prior DataCo results workbook in the script folder. Put two_part_delivery_glm_revision_results.xlsx beside this script or pass --results.")


def find_archive(base, requested=None):
    if requested:
        p = Path(requested).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Olist archive not found: {p}")
        return p
    preferred = [base / "archive.zip", base / "brazilian-ecommerce.zip"]
    for p in preferred:
        if p.exists():
            return p
    for p in sorted(base.glob("*.zip")):
        try:
            with zipfile.ZipFile(p) as z:
                names = set(Path(x).name for x in z.namelist())
                if "olist_orders_dataset.csv" in names and "olist_order_items_dataset.csv" in names:
                    return p
        except Exception:
            pass
    return None


def read_prior_settings(results_path):
    xl = pd.ExcelFile(results_path)
    data_audit = pd.read_excel(results_path, sheet_name="Data_Audit")
    audit = dict(zip(data_audit["item"].astype(str), data_audit["value"]))
    top_k = DEFAULT_TOP_K.copy()
    if "TopK_Primary" in xl.sheet_names:
        t = pd.read_excel(results_path, sheet_name="TopK_Primary")
        if "top_k_rate" in t.columns:
            vals = sorted(pd.to_numeric(t["top_k_rate"], errors="coerce").dropna().unique().tolist())
            if vals:
                top_k = vals
    alpha = DEFAULT_CONFORMAL_ALPHA
    if "Conformal_Intervals" in xl.sheet_names:
        c = pd.read_excel(results_path, sheet_name="Conformal_Intervals")
        if "alpha" in c.columns and c["alpha"].notna().any():
            alpha = float(c["alpha"].dropna().iloc[0])
    c_value = DEFAULT_C
    model_text = str(audit.get("occurrence_prediction_model", ""))
    m = re.search(r"C\s*=\s*([0-9.]+)", model_text)
    if m:
        c_value = float(m.group(1))
    q = float(audit.get("primary_cancellation_penalty_quantile", 0.95))
    prior_penalty = float(audit.get("primary_cancellation_penalty_hours", 96.0))
    prior_threshold = float(audit.get("threshold", np.nan))
    train_rows = float(audit.get("train_rows", 0))
    val_rows = float(audit.get("validation_rows", 0))
    test_rows = float(audit.get("test_rows", 0))
    total = train_rows + val_rows + test_rows
    if total > 0:
        split = (train_rows / total, val_rows / total, test_rows / total)
    else:
        split = (0.60, 0.20, 0.20)
    prior_features = pd.read_excel(results_path, sheet_name="Final_Features_Post_QR") if "Final_Features_Post_QR" in xl.sheet_names else pd.DataFrame()
    settings = {
        "results_path": str(results_path),
        "results_sha256": sha256_file(results_path),
        "occurrence_C": c_value,
        "cancellation_penalty_quantile": q,
        "dataco_primary_penalty_hours": prior_penalty,
        "dataco_threshold": prior_threshold,
        "conformal_alpha": alpha,
        "top_k_rates": top_k,
        "split_train": split[0],
        "split_validation": split[1],
        "split_test": split[2],
    }
    return settings, prior_features


def make_reader(base, archive_path):
    extracted = {p.name: p for p in base.glob("*.csv")}
    if "olist_orders_dataset.csv" in extracted:
        def read_csv(name, **kwargs):
            p = extracted.get(name)
            if p is None:
                raise FileNotFoundError(f"Missing required Olist CSV: {name}")
            return pd.read_csv(p, **kwargs)
        return read_csv, "extracted_csvs"
    if archive_path is None:
        raise FileNotFoundError("No extracted Olist CSV files and no Olist ZIP archive were found.")
    z = zipfile.ZipFile(archive_path)
    name_map = {Path(n).name: n for n in z.namelist()}
    def read_csv(name, **kwargs):
        if name not in name_map:
            raise FileNotFoundError(f"Missing required Olist CSV in archive: {name}")
        with z.open(name_map[name]) as f:
            return pd.read_csv(f, **kwargs)
    return read_csv, "zip_archive"


def load_olist_tables(read_csv, use_geolocation=True):
    orders = read_csv("olist_orders_dataset.csv")
    customers = read_csv("olist_customers_dataset.csv")
    items = read_csv("olist_order_items_dataset.csv")
    payments = read_csv("olist_order_payments_dataset.csv")
    products = read_csv("olist_products_dataset.csv")
    sellers = read_csv("olist_sellers_dataset.csv")
    try:
        translation = read_csv("product_category_name_translation.csv")
    except Exception:
        translation = pd.DataFrame(columns=["product_category_name", "product_category_name_english"])
    geolocation = None
    if use_geolocation:
        try:
            geolocation = read_csv("olist_geolocation_dataset.csv", usecols=["geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng"])
        except Exception:
            geolocation = None
    return orders, customers, items, payments, products, sellers, translation, geolocation


def build_order_level(orders, customers, items, payments, products, sellers, translation, geolocation):
    o = orders.copy()
    for c in ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date", "order_delivered_customer_date", "order_estimated_delivery_date"]:
        if c in o.columns:
            o[c] = parse_dt(o[c])
    raw_status = o["order_status"].astype(str).str.lower().str.strip()
    status_counts = raw_status.value_counts(dropna=False).rename_axis("order_status").reset_index(name="n_raw")
    o["order_status"] = raw_status
    o["is_resolved_status"] = o["order_status"].isin(RESOLVED_STATUSES)
    o["has_required_dates"] = o["order_purchase_timestamp"].notna() & o["order_estimated_delivery_date"].notna()
    o["has_delivered_date_if_delivered"] = (~o["order_status"].eq("delivered")) | o["order_delivered_customer_date"].notna()
    o["eligible_external"] = o["is_resolved_status"] & o["has_required_dates"] & o["has_delivered_date_if_delivered"]
    base = o.loc[o["eligible_external"]].copy()
    c = customers.copy()
    c = c.rename(columns={"customer_state": "customer_state", "customer_city": "customer_city", "customer_zip_code_prefix": "customer_zip"})
    base = base.merge(c[["customer_id", "customer_state", "customer_city", "customer_zip"]], on="customer_id", how="left")
    p = products.copy()
    if len(translation):
        p = p.merge(translation, on="product_category_name", how="left")
        p["product_category_use"] = p["product_category_name_english"].fillna(p["product_category_name"])
    else:
        p["product_category_use"] = p["product_category_name"]
    p["product_volume_cm3"] = pd.to_numeric(p["product_length_cm"], errors="coerce") * pd.to_numeric(p["product_height_cm"], errors="coerce") * pd.to_numeric(p["product_width_cm"], errors="coerce")
    s = sellers.rename(columns={"seller_state": "seller_state", "seller_city": "seller_city", "seller_zip_code_prefix": "seller_zip"}).copy()
    if geolocation is not None and len(geolocation):
        g = geolocation.copy()
        g["geolocation_zip_code_prefix"] = pd.to_numeric(g["geolocation_zip_code_prefix"], errors="coerce")
        geo = g.groupby("geolocation_zip_code_prefix", as_index=False).agg(geo_lat=("geolocation_lat", "median"), geo_lng=("geolocation_lng", "median"))
        cgeo = geo.rename(columns={"geolocation_zip_code_prefix": "customer_zip", "geo_lat": "customer_lat", "geo_lng": "customer_lng"})
        sgeo = geo.rename(columns={"geolocation_zip_code_prefix": "seller_zip", "geo_lat": "seller_lat", "geo_lng": "seller_lng"})
        base = base.merge(cgeo, on="customer_zip", how="left")
        s = s.merge(sgeo, on="seller_zip", how="left")
    else:
        base["customer_lat"] = np.nan
        base["customer_lng"] = np.nan
        s["seller_lat"] = np.nan
        s["seller_lng"] = np.nan
    it = items.merge(p, on="product_id", how="left").merge(s, on="seller_id", how="left")
    numeric_map = {
        "order_item_id": "order_lines",
        "price": "sales_sum",
        "freight_value": "freight_sum",
        "product_weight_g": "product_weight_mean",
        "product_volume_cm3": "product_volume_mean",
        "product_photos_qty": "product_photos_mean",
        "product_name_lenght": "product_name_length_mean",
        "product_description_lenght": "product_description_length_mean",
        "seller_lat": "seller_lat_mean",
        "seller_lng": "seller_lng_mean",
    }
    for src in numeric_map:
        if src in it.columns:
            it[src] = pd.to_numeric(it[src], errors="coerce")
    agg = it.groupby("order_id", as_index=False).agg(
        order_lines=("order_item_id", "count"),
        product_count=("product_id", "nunique"),
        seller_count=("seller_id", "nunique"),
        category_count=("product_category_use", "nunique"),
        seller_state_count=("seller_state", "nunique"),
        sales_sum=("price", "sum"),
        product_price_mean=("price", "mean"),
        product_price_max=("price", "max"),
        freight_sum=("freight_value", "sum"),
        freight_mean=("freight_value", "mean"),
        freight_max=("freight_value", "max"),
        product_weight_mean=("product_weight_g", "mean"),
        product_volume_mean=("product_volume_cm3", "mean"),
        product_photos_mean=("product_photos_qty", "mean"),
        product_name_length_mean=("product_name_lenght", "mean"),
        product_description_length_mean=("product_description_lenght", "mean"),
        seller_lat_mean=("seller_lat", "mean"),
        seller_lng_mean=("seller_lng", "mean"),
    )
    cat_mode = group_mode(it, "order_id", "product_category_use", "product_category_mode")
    seller_mode = group_mode(it, "order_id", "seller_state", "seller_state_mode")
    base = base.merge(agg, on="order_id", how="left").merge(cat_mode, on="order_id", how="left").merge(seller_mode, on="order_id", how="left")
    pay = payments.copy()
    for col in ["payment_installments", "payment_value"]:
        pay[col] = pd.to_numeric(pay[col], errors="coerce")
    payagg = pay.groupby("order_id", as_index=False).agg(
        payment_count=("payment_sequential", "count"),
        payment_type_count=("payment_type", "nunique"),
        payment_installments_mean=("payment_installments", "mean"),
        payment_installments_max=("payment_installments", "max"),
        payment_value_sum=("payment_value", "sum"),
    )
    paymode = group_mode(pay, "order_id", "payment_type", "payment_type_mode")
    base = base.merge(payagg, on="order_id", how="left").merge(paymode, on="order_id", how="left")
    base["promise_horizon_days"] = (base["order_estimated_delivery_date"] - base["order_purchase_timestamp"]).dt.total_seconds() / 86400
    base["freight_to_price_ratio"] = base["freight_sum"] / np.maximum(base["sales_sum"], 1e-8)
    base["items_per_seller"] = base["order_lines"] / np.maximum(base["seller_count"], 1)
    base["customer_seller_same_state"] = np.where(base["customer_state"].notna() & base["seller_state_mode"].notna(), (base["customer_state"] == base["seller_state_mode"]).astype(float), np.nan)
    base["distance_km"] = haversine_km(base["customer_lat"], base["customer_lng"], base["seller_lat_mean"], base["seller_lng_mean"])
    month = base["order_purchase_timestamp"].dt.month.astype(float)
    dow = base["order_purchase_timestamp"].dt.dayofweek.astype(float)
    hour = base["order_purchase_timestamp"].dt.hour.astype(float) + base["order_purchase_timestamp"].dt.minute.astype(float) / 60
    base["month_sin"] = np.sin(2 * np.pi * month / 12)
    base["month_cos"] = np.cos(2 * np.pi * month / 12)
    base["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    base["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    base["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    base["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    base = base.sort_values(["order_purchase_timestamp", "order_id"]).reset_index(drop=True)
    eligible_counts = pd.DataFrame([
        {"item": "raw_orders", "value": len(o)},
        {"item": "resolved_status_orders", "value": int(o["is_resolved_status"].sum())},
        {"item": "eligible_external_orders", "value": len(base)},
        {"item": "excluded_unresolved_status", "value": int((~o["is_resolved_status"]).sum())},
        {"item": "excluded_missing_purchase_or_estimate", "value": int((o["is_resolved_status"] & ~o["has_required_dates"]).sum())},
        {"item": "excluded_delivered_missing_actual_delivery", "value": int((o["is_resolved_status"] & o["has_required_dates"] & ~o["has_delivered_date_if_delivered"]).sum())},
        {"item": "delivered_orders_eligible", "value": int(base["order_status"].eq("delivered").sum())},
        {"item": "canceled_orders_eligible", "value": int(base["order_status"].eq("canceled").sum())},
        {"item": "unavailable_orders_eligible", "value": int(base["order_status"].eq("unavailable").sum())},
    ])
    return base, status_counts, eligible_counts


def split_chronological(frame, train_rate, validation_rate):
    n = len(frame)
    n_train = int(np.floor(train_rate * n))
    n_val = int(np.floor(validation_rate * n))
    train = frame.iloc[:n_train].copy()
    validation = frame.iloc[n_train:n_train + n_val].copy()
    test = frame.iloc[n_train + n_val:].copy()
    return train, validation, test


def lateness_hours(frame, deadline_mode):
    actual = frame["order_delivered_customer_date"]
    estimate = frame["order_estimated_delivery_date"]
    if deadline_mode == "end_of_estimated_day":
        deadline = estimate.dt.normalize() + pd.Timedelta(days=1)
    elif deadline_mode == "estimated_timestamp_midnight":
        deadline = estimate
    else:
        raise ValueError(deadline_mode)
    late = (actual - deadline).dt.total_seconds() / 3600
    return np.maximum(0.0, late.to_numpy(float)), deadline


def compute_training_penalty(train, deadline_mode, quantile):
    late, _ = lateness_hours(train, deadline_mode)
    delivered = train["order_status"].eq("delivered").to_numpy()
    pos = late[delivered & np.isfinite(late) & (late > 0)]
    if len(pos) == 0:
        return 24.0
    return float(np.quantile(np.maximum(pos, MIN_POSITIVE_HOURS), quantile))


def apply_outcome(frame, penalty_hours, deadline_mode):
    x = frame.copy()
    late, deadline = lateness_hours(x, deadline_mode)
    x["delivery_deadline"] = deadline
    x["lateness_hours"] = late
    x["is_failure_status"] = x["order_status"].isin(FAILURE_STATUSES).astype(int)
    x["is_late_delivered"] = (x["order_status"].eq("delivered") & pd.Series(late, index=x.index).gt(0)).astype(int)
    x["problem_delivery"] = ((x["is_failure_status"] == 1) | (x["is_late_delivered"] == 1)).astype(int)
    delivered_severity = np.maximum(late, MIN_POSITIVE_HOURS)
    severity = np.where(x["is_failure_status"].to_numpy() == 1, float(penalty_hours), delivered_severity)
    x["severity_hours_if_problem"] = np.where(x["problem_delivery"].to_numpy() == 1, severity, np.nan)
    x["delivery_burden_hours"] = np.where(x["problem_delivery"].to_numpy() == 1, severity, 0.0)
    x["clean_external_outcome"] = np.select(
        [x["is_failure_status"].eq(1), x["is_late_delivered"].eq(1)],
        ["canceled_or_unavailable", "late_delivered"],
        default="on_time_delivered",
    )
    return x


def prepare_features(train, validation, test, rare_min_count):
    numeric_cols = [
        "order_lines", "product_count", "seller_count", "category_count", "seller_state_count",
        "sales_sum", "product_price_mean", "product_price_max", "freight_sum", "freight_mean", "freight_max",
        "product_weight_mean", "product_volume_mean", "product_photos_mean", "product_name_length_mean",
        "product_description_length_mean", "payment_count", "payment_type_count", "payment_installments_mean",
        "payment_installments_max", "payment_value_sum", "promise_horizon_days", "freight_to_price_ratio",
        "items_per_seller", "customer_seller_same_state", "distance_km", "month_sin", "month_cos", "dow_sin",
        "dow_cos", "hour_sin", "hour_cos",
    ]
    categorical_cols = ["customer_state", "seller_state_mode", "product_category_mode", "payment_type_mode"]
    medians = {}
    centers = {}
    scales = {}
    xtr = pd.DataFrame(index=train.index)
    xva = pd.DataFrame(index=validation.index)
    xte = pd.DataFrame(index=test.index)
    scaler_rows = []
    missing_rows = []
    for c in numeric_cols:
        tr = pd.to_numeric(train[c], errors="coerce")
        va = pd.to_numeric(validation[c], errors="coerce")
        te = pd.to_numeric(test[c], errors="coerce")
        med = float(tr.median()) if tr.notna().any() else 0.0
        trf = tr.fillna(med)
        vaf = va.fillna(med)
        tef = te.fillna(med)
        q1 = float(trf.quantile(0.25))
        q3 = float(trf.quantile(0.75))
        iqr = q3 - q1
        if not np.isfinite(iqr) or abs(iqr) < 1e-12:
            iqr = 1.0
        medians[c] = med
        centers[c] = med
        scales[c] = iqr
        xtr[c] = (trf - med) / iqr
        xva[c] = (vaf - med) / iqr
        xte[c] = (tef - med) / iqr
        scaler_rows.append({"variable": c, "impute_median_train": med, "center_median_train": med, "scale_iqr_train": iqr})
        missing_rows.extend([
            {"split": "train", "variable": c, "missing_n": int(tr.isna().sum())},
            {"split": "validation", "variable": c, "missing_n": int(va.isna().sum())},
            {"split": "test", "variable": c, "missing_n": int(te.isna().sum())},
        ])
    ref_rows = []
    unseen_rows = []
    feature_rows = [{"encoded_term": c, "source_variable": c, "feature_type": "numeric_or_cyclic", "level": np.nan, "reference_level": np.nan} for c in numeric_cols]
    for c in categorical_cols:
        tr_raw = train[c].fillna("__missing__").astype(str)
        va_raw = validation[c].fillna("__missing__").astype(str)
        te_raw = test[c].fillna("__missing__").astype(str)
        counts = tr_raw.value_counts()
        frequent = counts[counts >= rare_min_count].index.tolist()
        if not frequent:
            frequent = [counts.index[0]] if len(counts) else ["__missing__"]
        ref = counts.index[0] if len(counts) and counts.index[0] in frequent else frequent[0]
        levels = [ref] + sorted([v for v in frequent if v != ref]) + ["__other__"]
        tr_map = tr_raw.where(tr_raw.isin(frequent), "__other__")
        va_map = va_raw.where(va_raw.isin(frequent), "__other__")
        te_map = te_raw.where(te_raw.isin(frequent), "__other__")
        train_levels = set(counts.index.tolist())
        for split_name, raw in [("validation", va_raw), ("test", te_raw)]:
            unseen = sorted(set(raw.unique()) - train_levels)
            unseen_rows.append({"split": split_name, "variable": c, "unseen_levels_n": len(unseen), "unseen_rows_n": int(raw.isin(unseen).sum()), "unseen_levels": "|".join(unseen)})
        ref_rows.append({"variable": c, "reference_level": ref, "raw_train_levels": int(len(counts)), "encoded_levels_including_reference": int(len(levels)), "rare_levels_grouped": int((counts < rare_min_count).sum())})
        for level in levels[1:]:
            term = f"{c}_{snake(level)}"
            xtr[term] = tr_map.eq(level).astype(float).to_numpy()
            xva[term] = va_map.eq(level).astype(float).to_numpy()
            xte[term] = te_map.eq(level).astype(float).to_numpy()
            feature_rows.append({"encoded_term": term, "source_variable": c, "feature_type": "categorical_dummy", "level": level, "reference_level": ref})
    pre_cols = xtr.columns.tolist()
    std = xtr.std(ddof=0)
    constant = std[std <= 0].index.tolist()
    xtr = xtr.drop(columns=constant)
    xva = xva[xtr.columns]
    xte = xte[xtr.columns]
    a = xtr.to_numpy(float)
    if a.shape[1]:
        _, r, piv = qr(a, mode="economic", pivoting=True)
        diag = np.abs(np.diag(r)) if r.size else np.array([])
        tol = np.finfo(float).eps * max(a.shape) * max(float(diag[0]), 1.0) if len(diag) else 0.0
        rank = int((diag > tol).sum())
        keep_idx = sorted(piv[:rank])
        keep = xtr.columns[keep_idx].tolist()
    else:
        keep = []
    collinear = [c for c in xtr.columns if c not in keep]
    xtr = xtr[keep]
    xva = xva[keep]
    xte = xte[keep]
    for x in [xtr, xva, xte]:
        x.insert(0, "const", 1.0)
    feature_audit = pd.DataFrame(feature_rows)
    feature_audit["pre_qr"] = feature_audit["encoded_term"].isin(pre_cols).astype(int)
    feature_audit["constant_dropped"] = feature_audit["encoded_term"].isin(constant).astype(int)
    feature_audit["collinear_dropped"] = feature_audit["encoded_term"].isin(collinear).astype(int)
    feature_audit["retained_post_qr"] = feature_audit["encoded_term"].isin(keep).astype(int)
    qr_audit = pd.DataFrame({"term": pre_cols})
    qr_audit["constant_dropped"] = qr_audit["term"].isin(constant).astype(int)
    qr_audit["collinear_dropped"] = qr_audit["term"].isin(collinear).astype(int)
    qr_audit["retained_post_qr"] = qr_audit["term"].isin(keep).astype(int)
    return xtr, xva, xte, pd.DataFrame(scaler_rows), pd.DataFrame(ref_rows), pd.DataFrame(unseen_rows), feature_audit, qr_audit, pd.DataFrame(missing_rows)


def logistic_fit(y, x, c_value):
    xm = np.asarray(x, float)
    y = np.asarray(y, int)
    clf = LogisticRegression(fit_intercept=False, penalty="l2", C=c_value, solver="liblinear", max_iter=1000, random_state=RANDOM_STATE)
    clf.fit(xm, y)
    beta = clf.coef_.ravel()
    p = expit(xm @ beta)
    return pd.Series(beta, index=x.columns), p


def logistic_predict(beta, x):
    return expit(np.asarray(x, float) @ beta.reindex(x.columns).to_numpy(float))


def linear_fit_log(y, x):
    xm = np.asarray(x, float)
    yy = np.log(np.maximum(np.asarray(y, float), 1e-8))
    beta = np.linalg.pinv(xm.T @ xm) @ xm.T @ yy
    resid = yy - xm @ beta
    smearing = float(np.mean(np.exp(resid)))
    return pd.Series(beta, index=x.columns), smearing, resid


def linear_predict_log(beta, smearing, x):
    return np.exp(np.asarray(x, float) @ beta.reindex(x.columns).to_numpy(float)) * smearing


def linear_fit_log1p(y, x):
    xm = np.asarray(x, float)
    yy = np.log1p(np.maximum(np.asarray(y, float), 0))
    beta = np.linalg.pinv(xm.T @ xm) @ xm.T @ yy
    resid = yy - xm @ beta
    smear = float(np.mean(np.exp(resid)))
    return pd.Series(beta, index=x.columns), smear


def linear_predict_log1p(beta, smear, x):
    z = np.exp(np.asarray(x, float) @ beta.reindex(x.columns).to_numpy(float)) * smear - 1
    return np.maximum(z, 0)


def metrics_binary(y, p, threshold, split, threshold_name):
    y = np.asarray(y, int)
    p = prob_clip(p)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "split": split,
        "threshold_name": threshold_name,
        "threshold": float(threshold),
        "n": int(len(y)),
        "prevalence": float(y.mean()),
        "auc": auc_rank(y, p),
        "average_precision": average_precision_rank(y, p),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "npv": float(tn / (tn + fn)) if (tn + fn) else np.nan,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)) if len(np.unique(pred)) > 1 else 0.0,
        "brier_score": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def select_threshold(y, p):
    y = np.asarray(y, int)
    p = prob_clip(p)
    candidates = np.unique(np.r_[0, 0.5, np.quantile(p, np.linspace(0.01, 0.99, 149)), 1])
    rows = []
    for t in candidates:
        pred = (p >= t).astype(int)
        mcc = matthews_corrcoef(y, pred) if len(np.unique(pred)) > 1 else 0.0
        bal = balanced_accuracy_score(y, pred)
        rows.append({"threshold": float(t), "mcc": float(mcc), "balanced_accuracy": float(bal)})
    tab = pd.DataFrame(rows).sort_values(["mcc", "balanced_accuracy", "threshold"], ascending=[False, False, True]).reset_index(drop=True)
    return float(tab.iloc[0]["threshold"]), tab


def calibration_table(y, p, split, groups=DEFAULT_CALIBRATION_GROUPS):
    d = pd.DataFrame({"y": np.asarray(y, int), "p": prob_clip(p)})
    q = min(groups, len(d))
    d["bin"] = pd.qcut(d["p"].rank(method="first"), q=q, labels=False, duplicates="drop") + 1
    out = d.groupby("bin", as_index=False).agg(n=("y", "size"), observed_rate=("y", "mean"), mean_p=("p", "mean"), p_min=("p", "min"), p_max=("p", "max"))
    out.insert(0, "split", split)
    return out


def calibration_summary(y, p, split):
    tab = calibration_table(y, p, split)
    ece = float(np.sum(tab["n"] / tab["n"].sum() * np.abs(tab["observed_rate"] - tab["mean_p"])))
    mce = float(np.max(np.abs(tab["observed_rate"] - tab["mean_p"])))
    y = np.asarray(y, int)
    lp = logit(prob_clip(p))
    intercept = np.nan
    slope = np.nan
    if len(np.unique(y)) == 2 and np.std(lp) > 0:
        x = np.column_stack([np.ones(len(lp)), lp])
        clf = LogisticRegression(fit_intercept=False, penalty=None, solver="lbfgs", max_iter=1000)
        try:
            clf.fit(x, y)
            intercept, slope = clf.coef_.ravel().tolist()
        except Exception:
            pass
    return {"split": split, "ece": ece, "mce": mce, "calibration_intercept": intercept, "calibration_slope": slope}


def regression_metrics(y, pred, split, target):
    y = np.asarray(y, float)
    pred = np.asarray(pred, float)
    err = pred - y
    r, pv = safe_spearman(y, pred)
    return {
        "split": split,
        "target": target,
        "n": int(len(y)),
        "mean_observed_hours": float(np.mean(y)),
        "mean_predicted_hours": float(np.mean(pred)),
        "mean_prediction_ratio": float(np.mean(pred) / np.mean(y)) if np.mean(y) > 0 else np.nan,
        "mae_hours": float(np.mean(np.abs(err))),
        "rmse_hours": float(np.sqrt(np.mean(err ** 2))),
        "median_absolute_error_hours": float(np.median(np.abs(err))),
        "spearman_r": r,
        "spearman_p": pv,
    }


def expected_burden_calibration(frame, score_col, split, groups=DEFAULT_CALIBRATION_GROUPS):
    d = frame[["delivery_burden_hours", score_col]].copy()
    d[score_col] = np.maximum(pd.to_numeric(d[score_col], errors="coerce"), 0)
    q = min(groups, len(d))
    d["bin"] = pd.qcut(d[score_col].rank(method="first"), q=q, labels=False, duplicates="drop") + 1
    out = d.groupby("bin", as_index=False).agg(n=("delivery_burden_hours", "size"), mean_observed_hours=("delivery_burden_hours", "mean"), mean_predicted_hours=(score_col, "mean"), median_observed_hours=("delivery_burden_hours", "median"), pred_min=(score_col, "min"), pred_max=(score_col, "max"))
    out.insert(0, "split", split)
    return out


def topk_table(frame, score_col, split, rates):
    d = frame[["problem_delivery", "delivery_burden_hours", score_col]].copy().sort_values(score_col, ascending=False)
    total_burden = float(d["delivery_burden_hours"].sum())
    total_problems = float(d["problem_delivery"].sum())
    mean_burden = float(d["delivery_burden_hours"].mean())
    rows = []
    for k in rates:
        n = max(1, int(np.ceil(k * len(d))))
        h = d.head(n)
        rows.append({
            "split": split,
            "score": score_col,
            "top_k_rate": float(k),
            "orders_flagged": n,
            "problem_rate": float(h["problem_delivery"].mean()),
            "mean_observed_burden_hours": float(h["delivery_burden_hours"].mean()),
            "captured_problem_share": float(h["problem_delivery"].sum() / total_problems) if total_problems > 0 else np.nan,
            "captured_burden_share": float(h["delivery_burden_hours"].sum() / total_burden) if total_burden > 0 else np.nan,
            "lift_over_mean_burden": float(h["delivery_burden_hours"].mean() / mean_burden) if mean_burden > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def bootstrap_topk_difference(frame, score_a, score_b, rates, n_boot=1000, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    n = len(frame)
    rows = []
    base_a = topk_table(frame, score_a, "test", rates).set_index("top_k_rate")
    base_b = topk_table(frame, score_b, "test", rates).set_index("top_k_rate")
    draws = {k: [] for k in rates}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        b = frame.iloc[idx]
        ta = topk_table(b, score_a, "boot", rates).set_index("top_k_rate")
        tb = topk_table(b, score_b, "boot", rates).set_index("top_k_rate")
        for k in rates:
            draws[k].append(float(ta.loc[k, "captured_burden_share"] - tb.loc[k, "captured_burden_share"]))
    for k in rates:
        vals = np.asarray(draws[k], float)
        rows.append({
            "top_k_rate": float(k),
            "score_a": score_a,
            "score_b": score_b,
            "difference_captured_burden_share": float(base_a.loc[k, "captured_burden_share"] - base_b.loc[k, "captured_burden_share"]),
            "bootstrap_ci_low": float(np.quantile(vals, 0.025)),
            "bootstrap_ci_high": float(np.quantile(vals, 0.975)),
            "bootstrap_n": int(n_boot),
        })
    return pd.DataFrame(rows)


def fit_scenario(train, validation, test, xtr, xva, xte, c_value, alpha, top_k_rates, threshold_reference=np.nan):
    beta_occ, p_train = logistic_fit(train["problem_delivery"], xtr, c_value)
    p_val = logistic_predict(beta_occ, xva)
    p_test = logistic_predict(beta_occ, xte)
    threshold, threshold_tab = select_threshold(validation["problem_delivery"], p_val)
    severity_mask_train = train["problem_delivery"].eq(1).to_numpy()
    beta_sev, smearing, train_log_resid = linear_fit_log(train.loc[train["problem_delivery"].eq(1), "severity_hours_if_problem"], xtr.loc[train["problem_delivery"].eq(1)])
    sev_train = linear_predict_log(beta_sev, smearing, xtr)
    sev_val = linear_predict_log(beta_sev, smearing, xva)
    sev_test = linear_predict_log(beta_sev, smearing, xte)
    beta_direct, direct_smear = linear_fit_log1p(train["delivery_burden_hours"], xtr)
    direct_train = linear_predict_log1p(beta_direct, direct_smear, xtr)
    direct_val = linear_predict_log1p(beta_direct, direct_smear, xva)
    direct_test = linear_predict_log1p(beta_direct, direct_smear, xte)
    for f, p, s, d in [(train, p_train, sev_train, direct_train), (validation, p_val, sev_val, direct_val), (test, p_test, sev_test, direct_test)]:
        f["p_problem"] = p
        f["predicted_severity_hours_if_problem"] = s
        f["expected_burden_hours"] = p * s
        f["direct_burden_hours"] = d
        f["occurrence_only_score"] = p
        f["severity_only_score"] = s
    val_problem = validation["problem_delivery"].eq(1)
    val_log_obs = np.log(np.maximum(validation.loc[val_problem, "severity_hours_if_problem"].to_numpy(float), 1e-8))
    val_log_pred = np.log(np.maximum(validation.loc[val_problem, "predicted_severity_hours_if_problem"].to_numpy(float), 1e-8))
    q_conf, conf_rank = finite_sample_quantile(np.abs(val_log_obs - val_log_pred), alpha)
    test["severity_pi_lower"] = np.maximum(test["predicted_severity_hours_if_problem"] * np.exp(-q_conf), 0)
    test["severity_pi_upper"] = test["predicted_severity_hours_if_problem"] * np.exp(q_conf)
    test_problem = test["problem_delivery"].eq(1)
    coverage = np.mean((test.loc[test_problem, "severity_hours_if_problem"] >= test.loc[test_problem, "severity_pi_lower"]) & (test.loc[test_problem, "severity_hours_if_problem"] <= test.loc[test_problem, "severity_pi_upper"])) if test_problem.any() else np.nan
    occ_metrics = pd.DataFrame([
        metrics_binary(train["problem_delivery"], p_train, threshold, "train", "olist_validation_mcc"),
        metrics_binary(validation["problem_delivery"], p_val, threshold, "validation", "olist_validation_mcc"),
        metrics_binary(test["problem_delivery"], p_test, threshold, "test", "olist_validation_mcc"),
    ])
    frozen_threshold_metrics = pd.DataFrame()
    if np.isfinite(threshold_reference):
        frozen_threshold_metrics = pd.DataFrame([metrics_binary(test["problem_delivery"], p_test, threshold_reference, "test", "dataco_frozen_threshold")])
    occ_cal = pd.concat([calibration_table(train["problem_delivery"], p_train, "train"), calibration_table(validation["problem_delivery"], p_val, "validation"), calibration_table(test["problem_delivery"], p_test, "test")], ignore_index=True)
    occ_cal_sum = pd.DataFrame([calibration_summary(train["problem_delivery"], p_train, "train"), calibration_summary(validation["problem_delivery"], p_val, "validation"), calibration_summary(test["problem_delivery"], p_test, "test")])
    sev_metrics = []
    exp_metrics = []
    direct_metrics = []
    for split, f in [("train", train), ("validation", validation), ("test", test)]:
        m = f["problem_delivery"].eq(1)
        sev_metrics.append(regression_metrics(f.loc[m, "severity_hours_if_problem"], f.loc[m, "predicted_severity_hours_if_problem"], split, "conditional_severity"))
        exp_metrics.append(regression_metrics(f["delivery_burden_hours"], f["expected_burden_hours"], split, "expected_burden"))
        direct_metrics.append(regression_metrics(f["delivery_burden_hours"], f["direct_burden_hours"], split, "direct_burden"))
    topk_primary = pd.concat([topk_table(train, "expected_burden_hours", "train", top_k_rates), topk_table(validation, "expected_burden_hours", "validation", top_k_rates), topk_table(test, "expected_burden_hours", "test", top_k_rates)], ignore_index=True)
    topk_baselines = pd.concat([
        topk_table(test, "expected_burden_hours", "test", top_k_rates),
        topk_table(test, "occurrence_only_score", "test", top_k_rates),
        topk_table(test, "severity_only_score", "test", top_k_rates),
        topk_table(test, "direct_burden_hours", "test", top_k_rates),
    ], ignore_index=True)
    rng = np.random.default_rng(RANDOM_STATE)
    test["random_score"] = rng.random(len(test))
    topk_baselines = pd.concat([topk_baselines, topk_table(test, "random_score", "test", top_k_rates)], ignore_index=True)
    exp_cal_test = expected_burden_calibration(test, "expected_burden_hours", "test")
    conf = pd.DataFrame([{
        "split": "test", "alpha": alpha, "nominal_coverage": 1 - alpha,
        "calibration_n": int(val_problem.sum()), "finite_sample_rank": conf_rank,
        "q_abs_log_residual": q_conf, "observed_coverage": float(coverage),
        "mean_interval_width_hours_problematic_test": float((test.loc[test_problem, "severity_pi_upper"] - test.loc[test_problem, "severity_pi_lower"]).mean()) if test_problem.any() else np.nan,
    }])
    coef_occ = pd.DataFrame({"term": beta_occ.index, "penalized_beta": beta_occ.values, "penalized_odds_ratio": np.exp(np.clip(beta_occ.values, -700, 700))})
    coef_sev = pd.DataFrame({"term": beta_sev.index, "beta": beta_sev.values, "severity_ratio": np.exp(np.clip(beta_sev.values, -700, 700))})
    return {
        "train": train, "validation": validation, "test": test,
        "occurrence_metrics": occ_metrics, "frozen_threshold_metrics": frozen_threshold_metrics,
        "occurrence_calibration": occ_cal, "occurrence_calibration_summary": occ_cal_sum,
        "severity_metrics": pd.DataFrame(sev_metrics), "expected_metrics": pd.DataFrame(exp_metrics), "direct_metrics": pd.DataFrame(direct_metrics),
        "topk_primary": topk_primary, "topk_baselines": topk_baselines, "expected_calibration_test": exp_cal_test,
        "conformal": conf, "threshold_candidates": threshold_tab, "threshold": threshold,
        "occurrence_coef": coef_occ, "severity_coef": coef_sev, "smearing": smearing,
    }


def transportability_audit(prior_features, external_feature_audit):
    if len(prior_features) == 0:
        return pd.DataFrame([{"item": "strict_frozen_dataco_model_transport", "value": "not_assessed_prior_feature_list_missing"}])
    prior_col = "encoded_feature_post_qr" if "encoded_feature_post_qr" in prior_features.columns else prior_features.columns[-1]
    prior = prior_features[prior_col].dropna().astype(str).tolist()
    external = set(external_feature_audit.loc[external_feature_audit["retained_post_qr"].eq(1), "encoded_term"].astype(str).tolist())
    exact = [x for x in prior if x in external]
    rows = [{"item": "prior_dataco_post_qr_features", "value": len(prior)}, {"item": "external_olist_post_qr_features", "value": len(external)}, {"item": "exact_name_overlap", "value": len(exact)}, {"item": "exact_name_overlap_features", "value": "|".join(exact)}, {"item": "strict_frozen_dataco_model_transport", "value": "not_performed_feature_spaces_and_predictor_semantics_are_not_equivalent"}, {"item": "external_analysis_interpretation", "value": "external_replication_of_the_prespecified_two_part_framework_with_frozen_methodological_rules_not_direct_transport_of_dataco_coefficients"}]
    return pd.DataFrame(rows)


def make_sensitivity_scenarios(settings, train_base):
    q = float(settings["cancellation_penalty_quantile"])
    primary_penalty = compute_training_penalty(train_base, "end_of_estimated_day", q)
    q90 = compute_training_penalty(train_base, "end_of_estimated_day", 0.90)
    q99 = compute_training_penalty(train_base, "end_of_estimated_day", 0.99)
    midnight_q95 = compute_training_penalty(train_base, "estimated_timestamp_midnight", q)
    return [
        {"scenario": "primary_olist_train_q95_end_of_day", "deadline_mode": "end_of_estimated_day", "penalty_hours": primary_penalty, "penalty_source": f"olist_training_quantile_{q:.2f}"},
        {"scenario": "olist_train_q90_end_of_day", "deadline_mode": "end_of_estimated_day", "penalty_hours": q90, "penalty_source": "olist_training_quantile_0.90"},
        {"scenario": "olist_train_q99_end_of_day", "deadline_mode": "end_of_estimated_day", "penalty_hours": q99, "penalty_source": "olist_training_quantile_0.99"},
        {"scenario": "dataco_fixed_penalty_end_of_day", "deadline_mode": "end_of_estimated_day", "penalty_hours": float(settings["dataco_primary_penalty_hours"]), "penalty_source": "frozen_dataco_primary_penalty_hours"},
        {"scenario": "midnight_deadline_olist_q95", "deadline_mode": "estimated_timestamp_midnight", "penalty_hours": midnight_q95, "penalty_source": f"olist_training_quantile_{q:.2f}"},
    ]


def outcome_distribution(frame, split):
    out = frame.groupby("clean_external_outcome", as_index=False).agg(n=("order_id", "size"), mean_burden_hours=("delivery_burden_hours", "mean"))
    out.insert(0, "split", split)
    out["share"] = out["n"] / out["n"].sum()
    return out


def sensitivity_summary_row(scenario, fit, primary_test_score=None):
    occ = fit["occurrence_metrics"].query("split == 'test'").iloc[0]
    sev = fit["severity_metrics"].query("split == 'test'").iloc[0]
    expm = fit["expected_metrics"].query("split == 'test'").iloc[0]
    top30 = fit["topk_primary"].query("split == 'test' and abs(top_k_rate - 0.30) < 1e-12")
    top10 = fit["topk_primary"].query("split == 'test' and abs(top_k_rate - 0.10) < 1e-12")
    test_score = fit["test"]["expected_burden_hours"].to_numpy(float)
    rank_vs_primary = np.nan
    if primary_test_score is not None and len(primary_test_score) == len(test_score):
        rank_vs_primary = safe_spearman(primary_test_score, test_score)[0]
    return {
        "scenario": scenario["scenario"], "deadline_mode": scenario["deadline_mode"], "penalty_hours": scenario["penalty_hours"], "penalty_source": scenario["penalty_source"],
        "test_problem_prevalence": float(occ["prevalence"]), "test_auc": float(occ["auc"]), "test_average_precision": float(occ["average_precision"]), "test_brier": float(occ["brier_score"]),
        "test_severity_spearman": float(sev["spearman_r"]), "test_expected_burden_spearman": float(expm["spearman_r"]), "test_expected_mean_ratio": float(expm["mean_prediction_ratio"]),
        "top10_captured_burden_share": float(top10.iloc[0]["captured_burden_share"]) if len(top10) else np.nan,
        "top30_captured_burden_share": float(top30.iloc[0]["captured_burden_share"]) if len(top30) else np.nan,
        "expected_score_spearman_vs_primary": rank_vs_primary,
    }


def plot_roc_like(test, path):
    y = test["problem_delivery"].to_numpy(int)
    p = test["p_problem"].to_numpy(float)
    order = np.argsort(-p)
    ys = y[order]
    tpr = np.cumsum(ys) / max(ys.sum(), 1)
    fpr = np.cumsum(1 - ys) / max((1 - ys).sum(), 1)
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.plot(np.r_[0, fpr], np.r_[0, tpr])
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Olist external test: occurrence ROC")
    fig.tight_layout()
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_calibration(cal, path):
    d = cal[cal["split"].eq("test")]
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.plot(d["mean_p"], d["observed_rate"], marker="o")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed problem rate")
    ax.set_title("Olist external test: occurrence calibration")
    fig.tight_layout()
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_topk(topk, path):
    d = topk[topk["split"].eq("test")]
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.plot(d["top_k_rate"] * 100, d["captured_burden_share"] * 100, marker="o")
    ax.plot([0, 100], [0, 100], linestyle="--")
    ax.set_xlabel("Orders monitored (%)")
    ax.set_ylabel("Observed burden captured (%)")
    ax.set_title("Olist external test: burden concentration")
    fig.tight_layout()
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def write_excel(path, sheets):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            if df is None:
                continue
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)
            safe_name = name[:31]
            df.to_excel(writer, sheet_name=safe_name, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=None)
    parser.add_argument("--archive", default=None)
    parser.add_argument("--output-dir", default="external_olist_results")
    parser.add_argument("--no-geolocation", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    results_path = find_results_file(base, args.results)
    archive_path = find_archive(base, args.archive)
    settings, prior_features = read_prior_settings(results_path)
    read_csv, data_source_mode = make_reader(base, archive_path)
    orders, customers, items, payments, products, sellers, translation, geolocation = load_olist_tables(read_csv, use_geolocation=not args.no_geolocation)
    order_level, raw_status_counts, eligibility_audit = build_order_level(orders, customers, items, payments, products, sellers, translation, geolocation)
    train_base, val_base, test_base = split_chronological(order_level, settings["split_train"], settings["split_validation"])
    xtr, xva, xte, scaler, refs, unseen, feature_audit, qr_audit, missingness = prepare_features(train_base, val_base, test_base, DEFAULT_RARE_MIN_COUNT)
    transport = transportability_audit(prior_features, feature_audit)
    scenarios = make_sensitivity_scenarios(settings, train_base)
    primary = scenarios[0]
    train_primary = apply_outcome(train_base, primary["penalty_hours"], primary["deadline_mode"])
    val_primary = apply_outcome(val_base, primary["penalty_hours"], primary["deadline_mode"])
    test_primary = apply_outcome(test_base, primary["penalty_hours"], primary["deadline_mode"])
    primary_fit = fit_scenario(train_primary, val_primary, test_primary, xtr, xva, xte, settings["occurrence_C"], settings["conformal_alpha"], settings["top_k_rates"], settings["dataco_threshold"])
    sensitivity_rows = [sensitivity_summary_row(primary, primary_fit, primary_fit["test"]["expected_burden_hours"].to_numpy(float))]
    primary_score = primary_fit["test"]["expected_burden_hours"].to_numpy(float)
    for sc in scenarios[1:]:
        tr = apply_outcome(train_base, sc["penalty_hours"], sc["deadline_mode"])
        va = apply_outcome(val_base, sc["penalty_hours"], sc["deadline_mode"])
        te = apply_outcome(test_base, sc["penalty_hours"], sc["deadline_mode"])
        fit = fit_scenario(tr, va, te, xtr, xva, xte, settings["occurrence_C"], settings["conformal_alpha"], settings["top_k_rates"], settings["dataco_threshold"])
        sensitivity_rows.append(sensitivity_summary_row(sc, fit, primary_score))
    sensitivity = pd.DataFrame(sensitivity_rows)
    test_out = primary_fit["test"].copy()
    boot = bootstrap_topk_difference(test_out, "expected_burden_hours", "occurrence_only_score", settings["top_k_rates"], n_boot=max(args.bootstrap, 0)) if args.bootstrap > 0 else pd.DataFrame()
    dist = pd.concat([outcome_distribution(primary_fit["train"], "train"), outcome_distribution(primary_fit["validation"], "validation"), outcome_distribution(primary_fit["test"], "test")], ignore_index=True)
    settings_table = pd.DataFrame([{"setting": k, "value": json.dumps(v) if isinstance(v, (list, tuple, dict)) else v} for k, v in settings.items()])
    external_audit = pd.DataFrame([
        {"item": "data_source_mode", "value": data_source_mode},
        {"item": "archive_path", "value": str(archive_path) if archive_path else "extracted_csvs"},
        {"item": "archive_sha256", "value": sha256_file(archive_path) if archive_path else np.nan},
        {"item": "prior_results_path", "value": str(results_path)},
        {"item": "prior_results_sha256", "value": settings["results_sha256"]},
        {"item": "external_interpretation", "value": "external_replication_of_framework_not_direct_transport_of_dataco_coefficients"},
        {"item": "primary_deadline_definition", "value": "end_of_estimated_delivery_calendar_day"},
        {"item": "primary_cancellation_rule", "value": f"Olist training-derived quantile {settings['cancellation_penalty_quantile']:.2f} copied from DataCo framework"},
        {"item": "primary_penalty_hours", "value": primary["penalty_hours"]},
        {"item": "post_purchase_fields_excluded_from_predictors", "value": "order_approved_at|order_delivered_carrier_date|order_delivered_customer_date|reviews|shipping_limit_date"},
        {"item": "outcome_fields_not_predictors", "value": "order_status|order_delivered_customer_date|order_estimated_delivery_date_as_timestamp; only promise_horizon_days is used as ex_ante predictor"},
        {"item": "train_rows", "value": len(train_base)},
        {"item": "validation_rows", "value": len(val_base)},
        {"item": "test_rows", "value": len(test_base)},
        {"item": "features_post_qr_excluding_const", "value": int(feature_audit["retained_post_qr"].sum())},
        {"item": "python", "value": sys.version.split()[0]},
        {"item": "platform", "value": platform.platform()},
        {"item": "numpy", "value": np.__version__},
        {"item": "pandas", "value": pd.__version__},
    ])
    outdir = (base / args.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    workbook = outdir / "olist_external_validation_results.xlsx"
    sheets = {
        "README": pd.DataFrame([
            {"section": "Interpretation", "description": "External replication of the prespecified two-part burden framework on an independent Olist e-commerce dataset. It is not direct transport of the fitted DataCo coefficients because the predictor spaces are not equivalent."},
            {"section": "Primary burden rule", "description": "Delivered orders are late only after the end of the estimated delivery calendar day. Canceled and unavailable orders receive a penalty equal to the same training-derived quantile rule used in DataCo, recalculated from Olist training data and frozen for validation/test."},
            {"section": "Predictor timing", "description": "Only purchase-time or order-content information is used as predictors. Post-purchase fulfillment fields and reviews are excluded."},
            {"section": "Chronology", "description": "The DataCo split proportions are read from the prior results workbook and applied chronologically to Olist."},
            {"section": "Primary model", "description": "L2 logistic occurrence model and log-linear positive severity model with Duan smearing, using the occurrence C and conformal alpha read from the prior results workbook."},
        ]),
        "Prior_Settings": settings_table,
        "External_Audit": external_audit,
        "Raw_Status_Counts": raw_status_counts,
        "Eligibility_Audit": eligibility_audit,
        "Transportability_Audit": transport,
        "Scaler_Train": scaler,
        "Reference_Levels": refs,
        "Unseen_Levels": unseen,
        "Feature_Audit": feature_audit,
        "QR_Audit": qr_audit,
        "Missingness": missingness,
        "Sensitivity_Settings": pd.DataFrame(scenarios),
        "Sensitivity_Summary": sensitivity,
        "Outcome_Distribution": dist,
        "Occurrence_Metrics": primary_fit["occurrence_metrics"],
        "Frozen_Threshold_Test": primary_fit["frozen_threshold_metrics"],
        "Occurrence_Calibration": primary_fit["occurrence_calibration"],
        "Occurrence_Cal_Summary": primary_fit["occurrence_calibration_summary"],
        "Severity_Metrics": primary_fit["severity_metrics"],
        "Expected_Burden_Metrics": primary_fit["expected_metrics"],
        "Direct_Burden_Metrics": primary_fit["direct_metrics"],
        "Expected_Burden_Cal": primary_fit["expected_calibration_test"],
        "Conformal_Intervals": primary_fit["conformal"],
        "Threshold_Candidates": primary_fit["threshold_candidates"],
        "TopK_Primary": primary_fit["topk_primary"],
        "TopK_SameData_Baselines": primary_fit["topk_baselines"],
        "TopK_Bootstrap_Difference": boot,
        "Occurrence_Coef": primary_fit["occurrence_coef"],
        "Severity_Coef": primary_fit["severity_coef"],
        "Pred_Test_Sample": test_out.head(5000),
    }
    write_excel(workbook, sheets)
    test_out.to_csv(outdir / "olist_external_test_predictions.csv", index=False)
    sensitivity.to_csv(outdir / "olist_external_sensitivity_summary.csv", index=False)
    feature_audit.to_csv(outdir / "olist_external_feature_audit.csv", index=False)
    transport.to_csv(outdir / "olist_external_transportability_audit.csv", index=False)
    plot_roc_like(test_out, outdir / "olist_external_occurrence_roc.png")
    plot_calibration(primary_fit["occurrence_calibration"], outdir / "olist_external_occurrence_calibration.png")
    plot_topk(primary_fit["topk_primary"], outdir / "olist_external_topk_burden_capture.png")
    manifest_rows = []
    for p in sorted(outdir.iterdir()):
        if p.is_file():
            manifest_rows.append({"file": p.name, "sha256": sha256_file(p), "bytes": p.stat().st_size})
    pd.DataFrame(manifest_rows).to_csv(outdir / "file_manifest_sha256.csv", index=False)
    req = "numpy\npandas\nscipy\nscikit-learn\nmatplotlib\nopenpyxl\n"
    (outdir / "requirements_external.txt").write_text(req, encoding="utf-8")
    summary = sensitivity.iloc[0].to_dict()
    print(f"Completed Olist external framework replication with {len(order_level):,} eligible orders.")
    print(f"Primary Olist training-derived cancellation penalty: {primary['penalty_hours']:.3f} hours.")
    print(f"External test AUC: {summary['test_auc']:.4f}")
    print(f"External test expected-burden Spearman: {summary['test_expected_burden_spearman']:.4f}")
    print(f"External test top-30% burden capture: {summary['top30_captured_burden_share']:.4f}")
    print(f"Results workbook: {workbook}")


if __name__ == "__main__":
    main()

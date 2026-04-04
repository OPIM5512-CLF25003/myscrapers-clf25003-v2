import os
import io
import json
import logging
import traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from google.cloud import storage

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor


# =========================================================
# ENV
# =========================================================
PROJECT_ID = os.getenv("PROJECT_ID", "")
GCS_BUCKET = os.getenv("GCS_BUCKET", "")
DATA_KEY = os.getenv("DATA_KEY", "structured/datasets/listings_master_llm.csv")
OUTPUT_PREFIX = os.getenv("OUTPUT_PREFIX", "preds")
TIMEZONE = os.getenv("TIMEZONE", "America/New_York")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")


# =========================================================
# GCS HELPERS
# =========================================================
def _read_csv_from_gcs(client: storage.Client, bucket: str, key: str) -> pd.DataFrame:
    b = client.bucket(bucket)
    blob = b.blob(key)
    if not blob.exists():
        raise FileNotFoundError(f"gs://{bucket}/{key} not found")
    return pd.read_csv(io.BytesIO(blob.download_as_bytes()))


def _write_csv_to_gcs(client: storage.Client, bucket: str, key: str, df: pd.DataFrame):
    b = client.bucket(bucket)
    blob = b.blob(key)
    blob.upload_from_string(df.to_csv(index=False), content_type="text/csv")


def _write_json_to_gcs(client: storage.Client, bucket: str, key: str, payload: dict):
    b = client.bucket(bucket)
    blob = b.blob(key)
    blob.upload_from_string(json.dumps(payload, indent=2), content_type="application/json")


def _write_bytes_to_gcs(client: storage.Client, bucket: str, key: str, data: bytes, content_type: str):
    b = client.bucket(bucket)
    blob = b.blob(key)
    blob.upload_from_string(data, content_type=content_type)


# =========================================================
# CLEANING / FEATURE HELPERS
# =========================================================
def _clean_numeric(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace(r"[^\d.]+", "", regex=True).str.strip()
    return pd.to_numeric(s, errors="coerce")


def _extract_engine_liters(s: pd.Series) -> pd.Series:
    vals = s.astype(str).str.extract(r"(\d{1,2}\.\d)")
    return pd.to_numeric(vals[0], errors="coerce")


def _normalize_text_col(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .replace({"nan": np.nan, "none": np.nan, "": np.nan})
    )


def _reduce_rare_categories(df: pd.DataFrame, cols, min_count=10):
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            continue
        vc = df[c].value_counts(dropna=True)
        keep = set(vc[vc >= min_count].index)
        df[c] = df[c].where(df[c].isin(keep), "OTHER")
    return df


def _apply_allowed_categories(series: pd.Series, allowed_values: set) -> pd.Series:
    return series.where(series.isin(allowed_values), "OTHER")


def _cap_series(train_s: pd.Series, apply_s: pd.Series, low_q=0.01, high_q=0.99):
    train_non_null = train_s.dropna()
    if train_non_null.empty:
        return apply_s, np.nan, np.nan
    lo = train_non_null.quantile(low_q)
    hi = train_non_null.quantile(high_q)
    return apply_s.clip(lower=lo, upper=hi), float(lo), float(hi)


def _feature_engineering(df: pd.DataFrame, timezone: str) -> pd.DataFrame:
    df = df.copy()

    # Datetime parsing
    df["scraped_at_dt_utc"] = pd.to_datetime(df["scraped_at"], errors="coerce", utc=True)
    try:
        df["scraped_at_local"] = df["scraped_at_dt_utc"].dt.tz_convert(timezone)
    except Exception:
        df["scraped_at_local"] = df["scraped_at_dt_utc"]

    df["date_local"] = df["scraped_at_local"].dt.date
    df["scrape_hour"] = df["scraped_at_local"].dt.hour
    df["scrape_dow"] = df["scraped_at_local"].dt.dayofweek
    df["scrape_month"] = df["scraped_at_local"].dt.month

    if "posted_date" in df.columns:
        df["posted_date_dt"] = pd.to_datetime(df["posted_date"], errors="coerce")
        df["posted_month"] = df["posted_date_dt"].dt.month
        df["posted_dow"] = df["posted_date_dt"].dt.dayofweek
        posted_year = df["posted_date_dt"].dt.year
    else:
        df["posted_date_dt"] = pd.NaT
        df["posted_month"] = np.nan
        df["posted_dow"] = np.nan
        posted_year = np.nan

    # Numeric cleaning
    for c in ["price", "year", "mileage", "mpg_city", "mpg_highway", "location_zip"]:
        if c in df.columns:
            df[f"{c}_num"] = _clean_numeric(df[c])

    # Text normalization
    text_cols = [
        "make", "model", "series", "transmission", "fuel", "body_type", "color",
        "title_status", "condition", "drivetrain", "engine", "location_city",
        "location_state", "dealer_name"
    ]
    for c in text_cols:
        if c in df.columns:
            df[c] = _normalize_text_col(df[c])

    # Engineered features
    current_year = pd.Timestamp.now(tz=timezone).year if timezone else pd.Timestamp.now().year

    if "year_num" in df.columns:
        df["vehicle_age"] = current_year - df["year_num"]
        df["vehicle_age"] = df["vehicle_age"].where(df["vehicle_age"] >= 0, np.nan)

        if isinstance(posted_year, pd.Series):
            df["car_age_at_listing"] = posted_year - df["year_num"]
            df["car_age_at_listing"] = df["car_age_at_listing"].where(df["car_age_at_listing"] >= 0, np.nan)
        else:
            df["car_age_at_listing"] = np.nan
    else:
        df["vehicle_age"] = np.nan
        df["car_age_at_listing"] = np.nan

    if "mileage_num" in df.columns:
        df["mileage_per_year"] = df["mileage_num"] / df["vehicle_age"].replace(0, np.nan)
    else:
        df["mileage_per_year"] = np.nan

    if "mpg_city_num" in df.columns or "mpg_highway_num" in df.columns:
        mpg_cols = [c for c in ["mpg_city_num", "mpg_highway_num"] if c in df.columns]
        df["mpg_avg"] = df[mpg_cols].mean(axis=1)
    else:
        df["mpg_avg"] = np.nan

    if "engine" in df.columns:
        df["engine_liters"] = _extract_engine_liters(df["engine"])
    else:
        df["engine_liters"] = np.nan

    if "transmission" in df.columns:
        df["is_automatic"] = df["transmission"].fillna("").str.contains("auto").astype(int)
    else:
        df["is_automatic"] = 0

    if "title_status" in df.columns:
        df["is_clean_title"] = df["title_status"].fillna("").str.contains("clean").astype(int)
    else:
        df["is_clean_title"] = 0

    if "dealer_name" in df.columns:
        df["has_dealer"] = df["dealer_name"].notna().astype(int)
    else:
        df["has_dealer"] = 0

    return df


# =========================================================
# MAIN TRAINING LOGIC
# =========================================================
def run_once(dry_run: bool = False):
    client = storage.Client(project=PROJECT_ID)
    df = _read_csv_from_gcs(client, GCS_BUCKET, DATA_KEY)

    required = {"scraped_at", "price", "make", "model", "year", "mileage"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = _feature_engineering(df, TIMEZONE)

    unique_dates = sorted(d for d in df["date_local"].dropna().unique())
    if len(unique_dates) < 2:
        return {
            "status": "noop",
            "reason": "need at least two distinct local dates",
            "dates_found": [str(d) for d in unique_dates]
        }

    today_local = unique_dates[-1]
    train_df = df[df["date_local"] < today_local].copy()
    holdout_df = df[df["date_local"] == today_local].copy()

    # Keep only rows with target
    train_df = train_df[train_df["price_num"].notna()].copy()
    holdout_df = holdout_df[holdout_df["price_num"].notna()].copy()

    # Basic sanity filters
    if "year_num" in train_df.columns:
        train_df = train_df[(train_df["price_num"] >= 500) & (train_df["year_num"] >= 1980)].copy()
        holdout_df = holdout_df[(holdout_df["price_num"] >= 500) & (holdout_df["year_num"] >= 1980)].copy()
    else:
        train_df = train_df[train_df["price_num"] >= 500].copy()
        holdout_df = holdout_df[holdout_df["price_num"] >= 500].copy()

    if len(train_df) < 50:
        return {
            "status": "noop",
            "reason": "too few training rows after filtering",
            "train_rows": int(len(train_df))
        }

    # Sort before TimeSeriesSplit
    train_df = train_df.sort_values("scraped_at_local").reset_index(drop=True)
    holdout_df = holdout_df.sort_values("scraped_at_local").reset_index(drop=True)

    # -----------------------------
    # Outlier handling
    # -----------------------------
    # Cap target only on train; do NOT change holdout target
    train_df["price_num"], price_lo, price_hi = _cap_series(train_df["price_num"], train_df["price_num"], 0.01, 0.99)

    # Cap feature outliers using train thresholds, then apply to both train and holdout
    feature_outlier_cols = ["mileage_num", "vehicle_age", "mileage_per_year", "mpg_avg", "engine_liters"]
    outlier_caps = {}

    for c in feature_outlier_cols:
        if c in train_df.columns:
            train_df[c], lo, hi = _cap_series(train_df[c], train_df[c], 0.01, 0.99)
            outlier_caps[c] = {"low": lo, "high": hi}
            if c in holdout_df.columns and not np.isnan(lo) and not np.isnan(hi):
                holdout_df[c] = holdout_df[c].clip(lo, hi)

    # -----------------------------
    # Rare category handling
    # -----------------------------
    rare_cols = ["model", "series", "engine", "dealer_name", "location_city", "color"]
    train_df = _reduce_rare_categories(train_df, rare_cols, min_count=8)

    for c in rare_cols:
        if c in train_df.columns and c in holdout_df.columns:
            allowed = set(train_df[c].dropna().unique())
            holdout_df[c] = _apply_allowed_categories(holdout_df[c], allowed)

    target = "price_num"

    numeric_features = [
        "year_num",
        "mileage_num",
        "mpg_city_num",
        "mpg_highway_num",
        "location_zip_num",
        "vehicle_age",
        "car_age_at_listing",
        "mileage_per_year",
        "mpg_avg",
        "engine_liters",
        "is_automatic",
        "is_clean_title",
        "has_dealer",
        "scrape_hour",
        "scrape_dow",
        "scrape_month",
        "posted_month",
        "posted_dow",
    ]

    categorical_features = [
        "make",
        "model",
        "series",
        "transmission",
        "fuel",
        "body_type",
        "color",
        "title_status",
        "condition",
        "drivetrain",
        "engine",
        "location_city",
        "location_state",
        "dealer_name",
    ]

    numeric_features = [c for c in numeric_features if c in train_df.columns]
    categorical_features = [c for c in categorical_features if c in train_df.columns]
    features = numeric_features + categorical_features

    if not features:
        return {"status": "noop", "reason": "no valid features available after preprocessing"}

    X_train = train_df[features].copy()
    y_train = train_df[target].copy()

    X_holdout = holdout_df[features].copy() if len(holdout_df) > 0 else pd.DataFrame(columns=features)
    y_holdout = holdout_df[target].copy() if len(holdout_df) > 0 else pd.Series(dtype=float)

    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imp", SimpleImputer(strategy="median"))
                ]),
                numeric_features
            ),
            (
                "cat",
                Pipeline([
                    ("imp", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore"))
                ]),
                categorical_features
            ),
        ],
        remainder="drop"
    )

    base_model = RandomForestRegressor(
        random_state=42,
        n_jobs=-1
    )

    pipe = Pipeline([
        ("pre", pre),
        ("model", base_model)
    ])

    # Smaller grid for cloud/runtime safety
    param_grid = {
        "model__n_estimators": [200, 300],
        "model__max_depth": [10, None],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2],
        "model__max_features": ["sqrt"]
    }

    tscv = TimeSeriesSplit(n_splits=4)

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        cv=tscv,
        n_jobs=-1,
        verbose=1,
        refit=True
    )

    grid.fit(X_train, y_train)
    best_pipe = grid.best_estimator_

    # Run folder
    now_utc = pd.Timestamp.now(tz="UTC")
    base_out = f"{OUTPUT_PREFIX}/{now_utc.strftime('%Y%m%d%H')}"

    # -----------------------------
    # Predict today's listings
    # -----------------------------
    mae_today = None
    preds_df = pd.DataFrame()

    if len(X_holdout) > 0:
        y_hat = best_pipe.predict(X_holdout)

        keep_cols = [c for c in ["post_id", "scraped_at", "make", "model", "price"] if c in holdout_df.columns]
        preds_df = holdout_df[keep_cols].copy()
        preds_df["actual_price"] = y_holdout.values
        preds_df["pred_price"] = np.round(y_hat, 2)

        mae_today = float(mean_absolute_error(y_holdout, y_hat))

    # -----------------------------
    # Permutation importance
    # -----------------------------
    perm_X = X_holdout if len(X_holdout) > 0 else X_train
    perm_y = y_holdout if len(X_holdout) > 0 else y_train

    perm = permutation_importance(
        best_pipe,
        perm_X,
        perm_y,
        n_repeats=5,
        random_state=42,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )

    importance_df = pd.DataFrame({
        "feature": features,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std
    }).sort_values("importance_mean", ascending=False)

    # -----------------------------
    # PDP for top 3 numeric features
    # -----------------------------
    top_numeric = [f for f in importance_df["feature"].tolist() if f in numeric_features][:3]
    pdp_files = []

    for f in top_numeric:
        fig, ax = plt.subplots(figsize=(7, 4))
        PartialDependenceDisplay.from_estimator(
            best_pipe,
            X_train,
            [f],
            ax=ax
        )
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        pdp_key = f"{base_out}/pdp_{f}.png"
        _write_bytes_to_gcs(client, GCS_BUCKET, pdp_key, buf.getvalue(), "image/png")
        pdp_files.append(pdp_key)

    metrics = {
        "status": "ok",
        "today_local": str(today_local),
        "train_rows": int(len(train_df)),
        "holdout_rows": int(len(holdout_df)),
        "mae_today": mae_today,
        "best_params": grid.best_params_,
        "best_cv_mae": float(-grid.best_score_),
        "top_pdp_features": top_numeric,
        "pdp_files": pdp_files,
        "feature_count": len(features),
        "numeric_feature_count": len(numeric_features),
        "categorical_feature_count": len(categorical_features),
        "target_train_cap": {"low": price_lo, "high": price_hi},
        "feature_caps": outlier_caps,
    }

    if not dry_run:
        if len(preds_df) > 0:
            _write_csv_to_gcs(client, GCS_BUCKET, f"{base_out}/preds-llm.csv", preds_df)

        _write_csv_to_gcs(client, GCS_BUCKET, f"{base_out}/permutation_importance.csv", importance_df)
        _write_json_to_gcs(client, GCS_BUCKET, f"{base_out}/metrics.json", metrics)
        logging.info("Artifacts written to gs://%s/%s/", GCS_BUCKET, base_out)
    else:
        logging.info("Dry run enabled. No CSV/JSON outputs written.")

    return {
        "status": "ok",
        "today_local": str(today_local),
        "train_rows": int(len(train_df)),
        "holdout_rows": int(len(holdout_df)),
        "mae_today": mae_today,
        "best_params": grid.best_params_,
        "best_cv_mae": float(-grid.best_score_),
        "top_3_numeric_pdp_features": top_numeric,
        "dry_run": dry_run
    }


# =========================================================
# HTTP ENTRYPOINT
# =========================================================
def train_rf_tuned_http(request):
    try:
        body = request.get_json(silent=True) or {}
        result = run_once(
            dry_run=bool(body.get("dry_run", False))
        )
        code = 200 if result.get("status") == "ok" else 204
        return (json.dumps(result), code, {"Content-Type": "application/json"})
    except Exception as e:
        logging.error("Error: %s", e)
        logging.error("Trace:\n%s", traceback.format_exc())
        return (
            json.dumps({"status": "error", "error": str(e)}),
            500,
            {"Content-Type": "application/json"}
        )

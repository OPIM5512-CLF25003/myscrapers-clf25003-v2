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
OUTPUT_PREFIX = os.getenv("OUTPUT_PREFIX", "structured/preds-llm")
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


def _normalize_text_col(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .replace({"nan": np.nan, "none": np.nan, "": np.nan})
    )


def _standardize_make(s: pd.Series) -> pd.Series:
    s = _normalize_text_col(s)
    replacements = {
        "chevy": "chevrolet",
        "vw": "volkswagen",
        "mercedes benz": "mercedes-benz",
        "mercedez-benz": "mercedes-benz",
        "mercedes": "mercedes-benz",
        "infinity": "infiniti",
        "hyandia": "hyundai",
        "hyundia": "hyundai",
    }
    return s.replace(replacements)


def _first_token_clean(s: pd.Series) -> pd.Series:
    s = _normalize_text_col(s)
    return s.str.extract(r"([a-z0-9]+(?:-[a-z0-9]+)?)")[0]


def _extract_engine_liters(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.lower()
    liters = s.str.extract(r"(\d+(?:\.\d+)?)\s*(?:l|liter|litre)?")[0]
    vals = pd.to_numeric(liters, errors="coerce")
    vals = vals.where((vals >= 0.6) & (vals <= 8.5), np.nan)
    return vals


def _extract_engine_cylinders(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.lower()

    cyl_1 = s.str.extract(r"(\d+)\s*[- ]?(?:cyl|cylinder|cylinders)")[0]
    cyl_2 = s.str.extract(r"\bv[- ]?(\d+)\b")[0]
    cyl_3 = s.str.extract(r"\bi[- ]?(\d+)\b")[0]
    cyl_4 = s.str.extract(r"\bh[- ]?(\d+)\b")[0]

    vals = pd.to_numeric(cyl_1, errors="coerce")
    vals = vals.fillna(pd.to_numeric(cyl_2, errors="coerce"))
    vals = vals.fillna(pd.to_numeric(cyl_3, errors="coerce"))
    vals = vals.fillna(pd.to_numeric(cyl_4, errors="coerce"))
    vals = vals.where((vals >= 2) & (vals <= 16), np.nan)
    return vals


def _normalize_transmission(s: pd.Series) -> pd.Series:
    s = _normalize_text_col(s)
    if s is None:
        return s
    s = s.replace({
        "cvt": "automatic",
        "a/t": "automatic",
        "m/t": "manual",
    })
    s = s.where(~s.fillna("").str.contains("cvt|auto"), "automatic")
    s = s.where(~s.fillna("").str.contains("manual|stick"), "manual")
    return s


def _normalize_fuel(s: pd.Series) -> pd.Series:
    s = _normalize_text_col(s)
    return s.replace({
        "gasoline": "gas",
        "flex-fuel": "flex fuel",
        "plugin hybrid": "plug-in hybrid",
    })


def _normalize_body_type(s: pd.Series) -> pd.Series:
    s = _normalize_text_col(s)
    return s.replace({
        "sport utility": "suv",
        "sport utility vehicle": "suv",
        "4dr car": "sedan",
        "2dr car": "coupe",
        "pickup": "truck",
        "pickup truck": "truck",
    })


def _normalize_condition(s: pd.Series) -> pd.Series:
    return _normalize_text_col(s)


def _normalize_state(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip().str.upper().replace({"NAN": np.nan, "NONE": np.nan, "": np.nan})
    return s


def _cap_series(train_s: pd.Series, apply_s: pd.Series, low_q=0.01, high_q=0.99):
    train_non_null = train_s.dropna()
    if train_non_null.empty:
        return apply_s, np.nan, np.nan
    lo = train_non_null.quantile(low_q)
    hi = train_non_null.quantile(high_q)
    return apply_s.clip(lower=lo, upper=hi), float(lo), float(hi)


# =========================================================
# FEATURE ENGINEERING
# =========================================================
def _feature_engineering(df: pd.DataFrame, timezone: str) -> pd.DataFrame:
    df = df.copy()

    # -----------------------------
    # Datetime parsing
    # -----------------------------
    df["scraped_at_dt_utc"] = pd.to_datetime(df["scraped_at"], errors="coerce", utc=True)
    try:
        df["scraped_at_local"] = df["scraped_at_dt_utc"].dt.tz_convert(timezone)
    except Exception:
        df["scraped_at_local"] = df["scraped_at_dt_utc"]

    df["date_local"] = df["scraped_at_local"].dt.date

    if "posted_date" in df.columns:
        df["posted_date_dt"] = pd.to_datetime(df["posted_date"], errors="coerce")
        scraped_naive = df["scraped_at_local"].dt.tz_localize(None, ambiguous="NaT", nonexistent="NaT")
        df["days_since_posted"] = (scraped_naive - df["posted_date_dt"]).dt.days
        df["days_since_posted"] = df["days_since_posted"].where(df["days_since_posted"] >= 0, np.nan)
    else:
        df["posted_date_dt"] = pd.NaT
        df["days_since_posted"] = np.nan

    # -----------------------------
    # Numeric cleaning
    # -----------------------------
    for c in ["price", "year", "mileage"]:
        if c in df.columns:
            df[f"{c}_num"] = _clean_numeric(df[c])

    # -----------------------------
    # Text normalization
    # -----------------------------
    if "make" in df.columns:
        df["make"] = _standardize_make(df["make"])

    if "model" in df.columns:
        df["model"] = _normalize_text_col(df["model"])
        df["model_base"] = _first_token_clean(df["model"])
    else:
        df["model_base"] = np.nan

    if "transmission" in df.columns:
        df["transmission"] = _normalize_transmission(df["transmission"])
    else:
        df["transmission"] = np.nan

    if "fuel" in df.columns:
        df["fuel"] = _normalize_fuel(df["fuel"])
    else:
        df["fuel"] = np.nan

    if "condition" in df.columns:
        df["condition"] = _normalize_condition(df["condition"])
    else:
        df["condition"] = np.nan

    if "body_type" in df.columns:
        df["body_type"] = _normalize_body_type(df["body_type"])
    else:
        df["body_type"] = np.nan

    if "location_state" in df.columns:
        df["location_state"] = _normalize_state(df["location_state"])
    else:
        df["location_state"] = np.nan

    if "engine" in df.columns:
        df["engine"] = _normalize_text_col(df["engine"])
        df["engine_liters"] = _extract_engine_liters(df["engine"])
        df["engine_cylinders"] = _extract_engine_cylinders(df["engine"])
    else:
        df["engine_liters"] = np.nan
        df["engine_cylinders"] = np.nan

    # -----------------------------
    # Engineered numeric features
    # -----------------------------
    current_year = pd.Timestamp.now(tz=timezone).year if timezone else pd.Timestamp.now().year

    if "mileage_num" in df.columns and "year_num" in df.columns:
        vehicle_age_tmp = current_year - df["year_num"]
        vehicle_age_tmp = vehicle_age_tmp.where(vehicle_age_tmp > 0, np.nan)
        df["mileage_per_year"] = df["mileage_num"] / vehicle_age_tmp
    else:
        df["mileage_per_year"] = np.nan

    # -----------------------------
    # Missingness flags
    # -----------------------------
    df["engine_liters_missing"] = df["engine_liters"].isna().astype(int)
    df["engine_cylinders_missing"] = df["engine_cylinders"].isna().astype(int)
    df["days_since_posted_missing"] = df["days_since_posted"].isna().astype(int)

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
    train_df["price_num"], price_lo, price_hi = _cap_series(
        train_df["price_num"], train_df["price_num"], 0.01, 0.99
    )

    feature_outlier_cols = [
        "mileage_num",
        "mileage_per_year",
        "engine_liters",
        "engine_cylinders",
        "days_since_posted",
    ]
    outlier_caps = {}

    for c in feature_outlier_cols:
        if c in train_df.columns:
            train_df[c], lo, hi = _cap_series(train_df[c], train_df[c], 0.01, 0.99)
            outlier_caps[c] = {"low": lo, "high": hi}
            if c in holdout_df.columns and not np.isnan(lo) and not np.isnan(hi):
                holdout_df[c] = holdout_df[c].clip(lo, hi)

    target = "price_num"

    # -----------------------------
    # HIGH IMPACT FEATURES ONLY
    # -----------------------------
    numeric_features = [
        "year_num",
        "mileage_num",
        "mileage_per_year",
        "engine_liters",
        "engine_cylinders",
        "days_since_posted",
        "engine_liters_missing",
        "engine_cylinders_missing",
        "days_since_posted_missing",
    ]

    categorical_features = [
        "make",
        "model_base",
        "transmission",
        "fuel",
        "condition",
        "body_type",
        "location_state",
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

    param_grid = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [10, 20, None],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", 0.7]
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

    logging.info("Holdout rows available for prediction: %d", len(X_holdout))
    logging.info("Training features used: %s", features)

    if len(X_holdout) > 0:
        y_hat = best_pipe.predict(X_holdout)

        # Fixed output schema: identifiers + final model columns only
        output_cols = [
            "post_id",
            "scraped_at",
            "year_num",
            "mileage_num",
            "mileage_per_year",
            "engine_liters",
            "engine_cylinders",
            "days_since_posted",
            "engine_liters_missing",
            "engine_cylinders_missing",
            "days_since_posted_missing",
            "make",
            "model_base",
            "transmission",
            "fuel",
            "condition",
            "body_type",
            "location_state",
        ]

        for c in output_cols:
            if c not in holdout_df.columns:
                holdout_df[c] = np.nan

        preds_df = holdout_df[output_cols].copy()
        preds_df["actual_price"] = y_holdout.values
        preds_df["pred_price"] = np.round(y_hat, 2)

        mae_today = float(mean_absolute_error(y_holdout, y_hat))
        logging.info("Prediction rows written to preds_df: %d", len(preds_df))
    else:
        logging.warning("Holdout set is empty; preds-llm.csv will not be created.")

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
        "features_used": features,
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

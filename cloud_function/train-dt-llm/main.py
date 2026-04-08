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
# FEATURE NAME MAP (short -> full)
# =========================================================
feature_name_map = {
    "yr": "year",
    "mi": "mileage",
    "mi_py": "mileage_per_year",
    "mk": "make",
    "mdl": "model",
    "trn": "transmission",
    "ful": "fuel",
    "cnd": "condition",
    "bdy": "body_type",
    "st": "state",
    "eng_l": "engine_liters",
    "eng_c": "engine_cylinders",
    "days_post": "days_since_posted",
    "eng_l_miss": "engine_liters_missing",
    "eng_c_miss": "engine_cylinders_missing",
    "days_post_miss": "days_since_posted_missing"
}


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
# BASIC HELPERS
# =========================================================
def _clean_numeric(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace(r"[^\d.]+", "", regex=True).str.strip()
    return pd.to_numeric(s, errors="coerce")


def _norm_text(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .replace({"nan": np.nan, "none": np.nan, "": np.nan})
    )


def _std_make(s: pd.Series) -> pd.Series:
    s = _norm_text(s)
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


def _model_base(s: pd.Series) -> pd.Series:
    s = _norm_text(s)
    return s.str.extract(r"([a-z0-9]+(?:-[a-z0-9]+)?)")[0]


def _eng_l(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.lower()
    vals = s.str.extract(r"(\d+(?:\.\d+)?)\s*(?:l|liter|litre)?")[0]
    vals = pd.to_numeric(vals, errors="coerce")
    vals = vals.where((vals >= 0.6) & (vals <= 8.5), np.nan)
    return vals


def _eng_cyl(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.lower()

    cyl1 = s.str.extract(r"(\d+)\s*[- ]?(?:cyl|cylinder|cylinders)")[0]
    cyl2 = s.str.extract(r"\bv[- ]?(\d+)\b")[0]
    cyl3 = s.str.extract(r"\bi[- ]?(\d+)\b")[0]
    cyl4 = s.str.extract(r"\bh[- ]?(\d+)\b")[0]

    vals = pd.to_numeric(cyl1, errors="coerce")
    vals = vals.fillna(pd.to_numeric(cyl2, errors="coerce"))
    vals = vals.fillna(pd.to_numeric(cyl3, errors="coerce"))
    vals = vals.fillna(pd.to_numeric(cyl4, errors="coerce"))
    vals = vals.where((vals >= 2) & (vals <= 16), np.nan)
    return vals


def _norm_trans(s: pd.Series) -> pd.Series:
    s = _norm_text(s)
    s = s.replace({"cvt": "automatic", "a/t": "automatic", "m/t": "manual"})
    s = s.where(~s.fillna("").str.contains("cvt|auto"), "automatic")
    s = s.where(~s.fillna("").str.contains("manual|stick"), "manual")
    return s


def _norm_fuel(s: pd.Series) -> pd.Series:
    s = _norm_text(s)
    return s.replace({
        "gasoline": "gas",
        "flex-fuel": "flex fuel",
        "plugin hybrid": "plug-in hybrid",
    })


def _norm_body(s: pd.Series) -> pd.Series:
    s = _norm_text(s)
    return s.replace({
        "sport utility": "suv",
        "sport utility vehicle": "suv",
        "4dr car": "sedan",
        "2dr car": "coupe",
        "pickup": "truck",
        "pickup truck": "truck",
    })


def _norm_cond(s: pd.Series) -> pd.Series:
    return _norm_text(s)


def _norm_state(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .str.upper()
        .replace({"NAN": np.nan, "NONE": np.nan, "": np.nan})
    )


def _cap_series(train_s: pd.Series, apply_s: pd.Series, low_q=0.01, high_q=0.99):
    train_non_null = train_s.dropna()
    if train_non_null.empty:
        return apply_s, np.nan, np.nan
    lo = train_non_null.quantile(low_q)
    hi = train_non_null.quantile(high_q)
    return apply_s.clip(lower=lo, upper=hi), float(lo), float(hi)


def _ensure_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df


# =========================================================
# TYPE CHECK + FORMAT CONVERSION
# =========================================================
def _prepare_base_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "scraped_at" in df.columns:
        df["scraped_at"] = pd.to_datetime(df["scraped_at"], errors="coerce", utc=True)

    if "posted_date" in df.columns:
        df["posted_date"] = pd.to_datetime(df["posted_date"], errors="coerce")

    for c in ["price", "year", "mileage", "mpg_city", "mpg_highway", "location_zip"]:
        if c in df.columns:
            df[c] = _clean_numeric(df[c])

    text_cols = [
        "make", "model", "series", "transmission", "fuel", "body_type",
        "color", "title_status", "condition", "drivetrain", "engine",
        "location_city", "location_state", "full_address", "dealer_name",
        "phone", "website", "source_txt"
    ]
    for c in text_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).replace({"nan": np.nan, "None": np.nan})

    return df


# =========================================================
# FEATURE ENGINEERING
# =========================================================
def _feature_engineering(df: pd.DataFrame, timezone: str) -> pd.DataFrame:
    df = df.copy()

    df["scraped_at_local"] = df["scraped_at"]
    try:
        df["scraped_at_local"] = df["scraped_at"].dt.tz_convert(timezone)
    except Exception:
        pass

    df["date_local"] = df["scraped_at_local"].dt.date

    if "make" in df.columns:
        df["make"] = _std_make(df["make"])
    if "model" in df.columns:
        df["model"] = _norm_text(df["model"])
    if "transmission" in df.columns:
        df["transmission"] = _norm_trans(df["transmission"])
    if "fuel" in df.columns:
        df["fuel"] = _norm_fuel(df["fuel"])
    if "condition" in df.columns:
        df["condition"] = _norm_cond(df["condition"])
    if "body_type" in df.columns:
        df["body_type"] = _norm_body(df["body_type"])
    if "location_state" in df.columns:
        df["location_state"] = _norm_state(df["location_state"])
    if "engine" in df.columns:
        df["engine"] = _norm_text(df["engine"])

    # short, easy feature names
    df["p"] = df["price"] if "price" in df.columns else np.nan
    df["yr"] = df["year"] if "year" in df.columns else np.nan
    df["mi"] = df["mileage"] if "mileage" in df.columns else np.nan

    df["mk"] = df["make"] if "make" in df.columns else np.nan
    df["mdl"] = _model_base(df["model"]) if "model" in df.columns else np.nan
    df["trn"] = df["transmission"] if "transmission" in df.columns else np.nan
    df["ful"] = df["fuel"] if "fuel" in df.columns else np.nan
    df["cnd"] = df["condition"] if "condition" in df.columns else np.nan
    df["bdy"] = df["body_type"] if "body_type" in df.columns else np.nan
    df["st"] = df["location_state"] if "location_state" in df.columns else np.nan

    df["eng_l"] = _eng_l(df["engine"]) if "engine" in df.columns else np.nan
    df["eng_c"] = _eng_cyl(df["engine"]) if "engine" in df.columns else np.nan

    if "posted_date" in df.columns:
        scraped_naive = df["scraped_at_local"].dt.tz_localize(None, ambiguous="NaT", nonexistent="NaT")
        df["days_post"] = (scraped_naive - df["posted_date"]).dt.days
        df["days_post"] = df["days_post"].where(df["days_post"] >= 0, np.nan)
    else:
        df["days_post"] = np.nan

    current_year = pd.Timestamp.now(tz=timezone).year if timezone else pd.Timestamp.now().year
    vehicle_age_tmp = current_year - df["yr"]
    vehicle_age_tmp = vehicle_age_tmp.where(vehicle_age_tmp > 0, np.nan)
    df["mi_py"] = df["mi"] / vehicle_age_tmp

    # missing flags
    df["eng_l_miss"] = df["eng_l"].isna().astype(int)
    df["eng_c_miss"] = df["eng_c"].isna().astype(int)
    df["days_post_miss"] = df["days_post"].isna().astype(int)

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

    # 1. datatype check + conversion
    df = _prepare_base_types(df)

    # 2. feature engineering
    df = _feature_engineering(df, TIMEZONE)

    # 3. split past vs today
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

    # 4. keep rows with target
    train_df = train_df[train_df["p"].notna()].copy()
    holdout_df = holdout_df[holdout_df["p"].notna()].copy()

    # 5. sanity filters
    train_df = train_df[(train_df["p"] >= 500) & (train_df["yr"] >= 1980)].copy()
    holdout_df = holdout_df[(holdout_df["p"] >= 500) & (holdout_df["yr"] >= 1980)].copy()

    if len(train_df) < 50:
        return {
            "status": "noop",
            "reason": "too few training rows after filtering",
            "train_rows": int(len(train_df))
        }

    train_df = train_df.sort_values("scraped_at_local").reset_index(drop=True)
    holdout_df = holdout_df.sort_values("scraped_at_local").reset_index(drop=True)

    # 6. outlier handling
    train_df["p"], p_lo, p_hi = _cap_series(train_df["p"], train_df["p"], 0.01, 0.99)

    outlier_cols = ["mi", "mi_py", "eng_l", "eng_c", "days_post"]
    outlier_caps = {}
    for c in outlier_cols:
        if c in train_df.columns:
            train_df[c], lo, hi = _cap_series(train_df[c], train_df[c], 0.01, 0.99)
            outlier_caps[c] = {"low": lo, "high": hi}
            if c in holdout_df.columns and not np.isnan(lo) and not np.isnan(hi):
                holdout_df[c] = holdout_df[c].clip(lo, hi)

    target = "p"

    # 7. final model columns
    num_feats = [
        "yr",
        "mi",
        "mi_py",
        "eng_l",
        "eng_c",
        "days_post",
        "eng_l_miss",
        "eng_c_miss",
        "days_post_miss",
    ]

    cat_feats = [
        "mk",
        "mdl",
        "trn",
        "ful",
        "cnd",
        "bdy",
        "st",
    ]

    num_feats = [c for c in num_feats if c in train_df.columns]
    cat_feats = [c for c in cat_feats if c in train_df.columns]
    feats = num_feats + cat_feats

    if not feats:
        return {"status": "noop", "reason": "no valid features available after preprocessing"}

    X_train = train_df[feats].copy()
    y_train = train_df[target].copy()

    X_holdout = holdout_df[feats].copy() if len(holdout_df) > 0 else pd.DataFrame(columns=feats)
    y_holdout = holdout_df[target].copy() if len(holdout_df) > 0 else pd.Series(dtype=float)

    # 8. missing value handling
    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imp", SimpleImputer(strategy="median"))
                ]),
                num_feats
            ),
            (
                "cat",
                Pipeline([
                    ("imp", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore"))
                ]),
                cat_feats
            ),
        ],
        remainder="drop"
    )

    model = RandomForestRegressor(random_state=42, n_jobs=-1)

    pipe = Pipeline([
        ("pre", pre),
        ("model", model)
    ])

    # 9. hyperparameter tuning
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

    now_utc = pd.Timestamp.now(tz="UTC")
    base_out = f"{OUTPUT_PREFIX}/{now_utc.strftime('%Y%m%d%H')}"

    # 10. predict only today's listings
    mae_today = None
    preds_df = pd.DataFrame()

    if len(X_holdout) > 0:
        y_hat = best_pipe.predict(X_holdout)

        output_cols = [
            "post_id",
            "scraped_at",
            "yr",
            "mi",
            "mi_py",
            "eng_l",
            "eng_c",
            "days_post",
            "eng_l_miss",
            "eng_c_miss",
            "days_post_miss",
            "mk",
            "mdl",
            "trn",
            "ful",
            "cnd",
            "bdy",
            "st",
        ]

        holdout_df = _ensure_cols(holdout_df, output_cols)

        preds_df = holdout_df[output_cols].copy()
        preds_df["actual_price"] = y_holdout.values
        preds_df["pred_price"] = np.round(y_hat, 2)

        mae_today = float(mean_absolute_error(y_holdout, y_hat))

    # 11. permutation importance
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

    importance_short_df = pd.DataFrame({
        "feature": feats,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std
    }).sort_values("importance_mean", ascending=False)

    importance_df = importance_short_df.copy()
    importance_df["feature"] = importance_df["feature"].map(lambda x: feature_name_map.get(x, x))

    # 12. PDP for top 3 numeric features
    top_num = [f for f in importance_short_df["feature"].tolist() if f in num_feats][:3]
    pdp_files = []

    for f in top_num:
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

        readable_name = feature_name_map.get(f, f)
        pdp_key = f"{base_out}/pdp_{readable_name}.png"
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
        "features_used_short": feats,
        "features_used_full": [feature_name_map.get(f, f) for f in feats],
        "top_pdp_features_short": top_num,
        "top_pdp_features_full": [feature_name_map.get(f, f) for f in top_num],
        "pdp_files": pdp_files,
        "feature_count": len(feats),
        "numeric_feature_count": len(num_feats),
        "categorical_feature_count": len(cat_feats),
        "target_train_cap": {"low": p_lo, "high": p_hi},
        "feature_caps": outlier_caps,
    }

    if not dry_run:
        if len(preds_df) > 0:
            _write_csv_to_gcs(client, GCS_BUCKET, f"{base_out}/preds-llm.csv", preds_df)

        _write_csv_to_gcs(client, GCS_BUCKET, f"{base_out}/permutation_importance.csv", importance_df)
        _write_json_to_gcs(client, GCS_BUCKET, f"{base_out}/metrics.json", metrics)

    return {
        "status": "ok",
        "today_local": str(today_local),
        "train_rows": int(len(train_df)),
        "holdout_rows": int(len(holdout_df)),
        "mae_today": mae_today,
        "best_params": grid.best_params_,
        "best_cv_mae": float(-grid.best_score_),
        "top_3_numeric_pdp_features": [feature_name_map.get(f, f) for f in top_num],
        "dry_run": dry_run
    }


# =========================================================
# HTTP ENTRYPOINT
# =========================================================
def train_rf_tuned_http(request):
    try:
        body = request.get_json(silent=True) or {}
        result = run_once(dry_run=bool(body.get("dry_run", False)))
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

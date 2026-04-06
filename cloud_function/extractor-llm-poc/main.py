# main.py
# Purpose: PoC LLM extractor that reads your existing per-listing JSONL records,
# fetches the original TXT, asks an LLM (Vertex AI) to extract fields, and writes
# a sibling "<post_id>_llm.jsonl" to the NEW 'jsonl_llm/' sub-directory.
#
# FINAL FIXES INCLUDED:
# 1. Schema updated to use "type": "string" + "nullable": True.
# 2. system_instruction removed from GenerationConfig and merged into prompt.
# 3. LLM_MODEL set to 'gemini-2.5-flash' (Fixes 404/NotFound error).
# 4. "additionalProperties": False removed from schema (Fixes internal ParseError).
# 5. Non-breaking spaces (U+00A0) replaced with standard spaces (U+0020). <--- FIX FOR THIS ERROR

import os
import re
import json
import logging
import traceback
import time
from datetime import datetime, timezone

from flask import Request, jsonify
from google.api_core import retry as gax_retry
from google.cloud import storage

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
from google.api_core.exceptions import ResourceExhausted, InternalServerError, Aborted, DeadlineExceeded


# -------------------- ENV --------------------
PROJECT_ID           = os.getenv("PROJECT_ID", "")
REGION               = os.getenv("REGION", "us-central1")
BUCKET_NAME          = os.getenv("GCS_BUCKET", "")
STRUCTURED_PREFIX    = os.getenv("STRUCTURED_PREFIX", "structured")
LLM_PROVIDER         = os.getenv("LLM_PROVIDER", "vertex").lower()
LLM_MODEL            = os.getenv("LLM_MODEL", "gemini-2.5-flash")
OVERWRITE_DEFAULT    = os.getenv("OVERWRITE", "false").lower() == "true"
MAX_FILES_DEFAULT    = int(os.getenv("MAX_FILES", "0") or 0)

READ_RETRY = gax_retry.Retry(
    predicate=gax_retry.if_transient_error,
    initial=1.0, maximum=10.0, multiplier=2.0, deadline=120.0
)

def _if_llm_retryable(exception):
    return isinstance(exception, (ResourceExhausted, InternalServerError, Aborted, DeadlineExceeded))

LLM_RETRY = gax_retry.Retry(
    predicate=_if_llm_retryable,
    initial=5.0, maximum=30.0, multiplier=2.0, deadline=180.0,
)

storage_client = storage.Client()
_CACHED_MODEL_OBJ = None

RUN_ID_ISO_RE = re.compile(r"^\d{8}T\d{6}Z$")
RUN_ID_PLAIN_RE = re.compile(r"^\d{14}$")


# -------------------- HELPERS --------------------
def _get_vertex_model() -> GenerativeModel:
    global _CACHED_MODEL_OBJ
    if _CACHED_MODEL_OBJ is None:
        if not PROJECT_ID:
            raise RuntimeError("PROJECT_ID environment variable is missing.")
        vertexai.init(project=PROJECT_ID, location=REGION)
        _CACHED_MODEL_OBJ = GenerativeModel(LLM_MODEL)
        logging.info(f"Initialized Vertex AI model: {LLM_MODEL} in {REGION}")
    return _CACHED_MODEL_OBJ


def _list_structured_run_ids(bucket: str, structured_prefix: str) -> list[str]:
    it = storage_client.list_blobs(bucket, prefix=f"{structured_prefix}/", delimiter="/")
    for _ in it:
        pass

    runs = []
    for pref in getattr(it, "prefixes", []):
        tail = pref.rstrip("/").split("/")[-1]
        if tail.startswith("run_id="):
            cand = tail.split("run_id=", 1)[1]
            if RUN_ID_ISO_RE.match(cand) or RUN_ID_PLAIN_RE.match(cand):
                runs.append(cand)
    return sorted(runs)


def _normalize_run_id_iso(run_id: str) -> str:
    try:
        if RUN_ID_ISO_RE.match(run_id):
            dt = datetime.strptime(run_id, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        elif RUN_ID_PLAIN_RE.match(run_id):
            dt = datetime.strptime(run_id, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        else:
            raise ValueError("unsupported run_id")
        return dt.isoformat().replace("+00:00", "Z")
    except Exception:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _list_per_listing_jsonl_for_run(bucket: str, run_id: str) -> list[str]:
    prefix = f"{STRUCTURED_PREFIX}/run_id={run_id}/jsonl/"
    bucket_obj = storage_client.bucket(bucket)
    names = []
    for b in bucket_obj.list_blobs(prefix=prefix):
        if b.name.endswith(".jsonl"):
            names.append(b.name)
    return names


def _download_text(blob_name: str) -> str:
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)
    return blob.download_as_text(retry=READ_RETRY, timeout=120)


def _upload_jsonl_line(blob_name: str, record: dict):
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
    blob.upload_from_string(line, content_type="application/x-ndjson")


def _blob_exists(blob_name: str) -> bool:
    bucket = storage_client.bucket(BUCKET_NAME)
    return bucket.blob(blob_name).exists()


def _safe_int(x):
    try:
        if x is None or x == "":
            return None
        return int(str(x).replace(",", "").replace("$", "").strip())
    except Exception:
        return None


def _norm_str(x):
    if x is None:
        return None
    x = str(x).strip()
    return x if x else None


def _norm_lower_str(x):
    x = _norm_str(x)
    return x.lower() if x else None


def _normalize_transmission(x):
    x = _norm_lower_str(x)
    if not x:
        return None
    if "cvt" in x:
        return "automatic"
    if "auto" in x or "a/t" in x:
        return "automatic"
    if "manual" in x or "m/t" in x or "stick" in x:
        return "manual"
    return x


def _normalize_fuel(x):
    x = _norm_lower_str(x)
    if not x:
        return None
    mapping = {
        "gasoline": "gas",
        "gas": "gas",
        "diesel": "diesel",
        "hybrid": "hybrid",
        "electric": "electric",
        "flex fuel": "flex fuel",
        "flex-fuel": "flex fuel",
        "plugin hybrid": "plug-in hybrid",
        "plug-in hybrid": "plug-in hybrid",
    }
    return mapping.get(x, x)


def _normalize_body_type(x):
    x = _norm_lower_str(x)
    if not x:
        return None

    body_map = {
        "sport utility": "suv",
        "sport utility vehicle": "suv",
        "suv": "suv",
        "wagon": "wagon",
        "sedan": "sedan",
        "coupe": "coupe",
        "hatchback": "hatchback",
        "convertible": "convertible",
        "pickup": "truck",
        "pickup truck": "truck",
        "truck": "truck",
        "van": "van",
        "minivan": "minivan",
        "4dr car": "sedan",
        "2dr car": "coupe",
        "crew cab pickup": "truck",
    }
    return body_map.get(x, x)


def _normalize_drivetrain(x):
    x = _norm_lower_str(x)
    if not x:
        return None

    if x in {"awd", "all wheel drive", "all-wheel drive", "4matic", "xdrive", "quattro"}:
        return "awd"
    if x in {"4wd", "four wheel drive", "4 wheel drive"}:
        return "4wd"
    if x in {"fwd", "front wheel drive", "front-wheel drive"}:
        return "fwd"
    if x in {"rwd", "rear wheel drive", "rear-wheel drive"}:
        return "rwd"
    return x


def _normalize_condition(x):
    x = _norm_lower_str(x)
    if not x:
        return None
    return x


def _normalize_title_status(x):
    x = _norm_lower_str(x)
    if not x:
        return None
    return x


# -------------------- VERTEX AI CALL --------------------
def _vertex_extract_fields(raw_text: str) -> dict:
    model = _get_vertex_model()

    schema = {
        "type": "object",
        "properties": {
            "price": {"type": "integer", "nullable": True},
            "year": {"type": "integer", "nullable": True},
            "make": {"type": "string", "nullable": True},
            "model": {"type": "string", "nullable": True},
            "series": {"type": "string", "nullable": True},
            "trim": {"type": "string", "nullable": True},
            "mileage": {"type": "integer", "nullable": True},
            "vin": {"type": "string", "nullable": True},
            "stock_number": {"type": "string", "nullable": True},
            "transmission": {"type": "string", "nullable": True},
            "body_type": {"type": "string", "nullable": True},
            "fuel": {"type": "string", "nullable": True},
            "color": {"type": "string", "nullable": True},
            "title_status": {"type": "string", "nullable": True},
            "condition": {"type": "string", "nullable": True},
            "drivetrain": {"type": "string", "nullable": True},
            "engine": {"type": "string", "nullable": True},
            "mpg_city": {"type": "integer", "nullable": True},
            "mpg_highway": {"type": "integer", "nullable": True},
            "location_city": {"type": "string", "nullable": True},
            "location_state": {"type": "string", "nullable": True},
            "location_zip": {"type": "string", "nullable": True},
            "full_address": {"type": "string", "nullable": True},
            "dealer_name": {"type": "string", "nullable": True},
            "phone": {"type": "string", "nullable": True},
            "website": {"type": "string", "nullable": True},
            "posted_date": {"type": "string", "nullable": True},
            "post_id": {"type": "string", "nullable": True}
        },
        "required": [
            "price", "year", "make", "model", "series", "trim", "mileage",
            "vin", "stock_number", "transmission", "body_type", "fuel", "color",
            "title_status", "condition", "drivetrain", "engine",
            "mpg_city", "mpg_highway",
            "location_city", "location_state", "location_zip", "full_address",
            "dealer_name", "phone", "website", "posted_date", "post_id"
        ]
    }

    sys_instr = """
You are extracting structured vehicle listing data from messy Craigslist-style text.

Return one JSON object only. No markdown. No explanation.

GENERAL RULES
- Extract values even when formatting is inconsistent.
- Prefer labeled fields first, then fallback to title/body text.
- If the same field appears multiple times, choose the most specific or reliable value.
- If a value clearly exists, do not return null just because formatting is unusual.
- If a value truly does not exist, return null.
- Never invent values.

IMPORTANT FIELD RULES

price:
- Extract the listing sale price in USD as an integer.
- Example: "$14,800" -> 14800

year:
- Extract the vehicle year as a 4-digit integer.

make:
- Extract brand name such as BMW, Subaru, Toyota, Chevrolet.
- Standardize obvious aliases:
  Chevy -> Chevrolet
  VW -> Volkswagen
  Mercedes Benz -> Mercedes-Benz
  Infinity -> Infiniti

model:
- Extract the main model name only when possible.
- Examples:
  "BMW 5 Series" -> "5 Series"
  "Mazda CX-5" -> "CX-5"
  "Subaru XV Crosstrek" -> "XV Crosstrek"

series:
- Extract trim/series details if present beyond make/model.
- Examples:
  "2.0i Premium AWD 4dr Crossover CVT"
  "535i xDrive AWD"
  If absent, return null.

trim:
- Extract trim if clearly stated, otherwise null.

mileage:
- Extract odometer/miles as integer.
- Examples:
  "100,300" -> 100300
  "odometer: 154,359" -> 154359

vin:
- Extract VIN exactly.

stock_number:
- Extract stock number exactly.

transmission:
- Look for labels like:
  "transmission:", "Transmission:", "A/T", "M/T", "CVT"
- Normalize:
  CVT -> automatic
  Automatic / A/T -> automatic
  Manual / M/T / stick -> manual

body_type:
- Use labels such as "type:" or "Body:".
- Normalize common values:
  Sport Utility / SUV -> suv
  4dr Car -> sedan
  Wagon -> wagon
  Sedan -> sedan
  Coupe -> coupe
  Hatchback -> hatchback
  Sport Utility -> suv
  Pickup / Truck -> truck

fuel:
- Normalize common values:
  gasoline -> gas
  gas -> gas
  diesel -> diesel
  hybrid -> hybrid
  electric -> electric
  flex fuel -> flex fuel

color:
- Use exterior paint color if present.
- Prefer explicit labels like "paint color:" or "Color:".
- If two colors are mentioned like "Black on Black Leather", use exterior color.

title_status:
- Extract values like clean, rebuilt, salvage.

condition:
- Extract labeled condition such as excellent, good, like new, fair.
- If not labeled and clearly described in text, use that only if obvious.

drivetrain:
- Look for "drive:" or terms like AWD, FWD, RWD, 4WD, xDrive, 4MATIC, quattro.
- Normalize:
  xDrive / quattro / 4MATIC / AWD -> awd
  4WD -> 4wd
  FWD -> fwd
  RWD -> rwd

engine:
- Extract the best full engine description, not just cylinders.
- Prefer specific engine lines like:
  "3L Straight 6 Cylinder Engine"
  "2.0L H4 148hp 145ft. lbs."
  "3.5L V6 Direct Injection"
- If only cylinders are present, use that rather than null.

mpg_city and mpg_highway:
- Extract integers from patterns like:
  "26 city / 33 highway"
  "29 M.P.G." only if clearly attributable.
- If only one MPG value exists and city/highway split is unclear, return null for both.

location_city, location_state, location_zip:
- Extract separately when address or location appears.
- Example:
  "Stamford, CT 06902" -> city=Stamford, state=CT, zip=06902

full_address:
- Extract full street address if present.
- Example:
  "115 Jefferson St Stamford, CT 06902"

dealer_name:
- Extract from "Offered by:" or dealer block if seller is a dealer.
- For owner listings with no dealer, return null.

phone:
- Extract best phone number.

website:
- Extract website URL/domain if present.

posted_date:
- Extract the posted timestamp exactly as shown when possible.
- Examples:
  "2026-03-24 14:30"
  "2026-03-24 10:01"

post_id:
- Extract post ID exactly.

Return JSON only.
"""

    prompt = f"{sys_instr}\n\nLISTING TEXT:\n{raw_text}"

    gen_cfg = GenerationConfig(
        temperature=0.0,
        top_p=1.0,
        top_k=40,
        candidate_count=1,
        response_mime_type="application/json",
        response_schema=schema,
    )

    max_attempts = 3
    resp = None
    for attempt in range(max_attempts):
        try:
            resp = model.generate_content(prompt, generation_config=gen_cfg)
            break
        except Exception as e:
            if not _if_llm_retryable(e) or attempt == max_attempts - 1:
                logging.error(f"Fatal/non-retryable LLM error or max retries reached: {e}")
                raise
            sleep_time = 5 * (2 ** attempt)
            logging.warning(f"Transient LLM error on attempt {attempt+1}/{max_attempts}. Retrying in {sleep_time:.2f}s...")
            time.sleep(sleep_time)

    if resp is None:
        raise RuntimeError("LLM call failed after all retries.")

    parsed = json.loads(resp.text)

    parsed["price"] = _safe_int(parsed.get("price"))
    parsed["year"] = _safe_int(parsed.get("year"))
    parsed["mileage"] = _safe_int(parsed.get("mileage"))
    parsed["mpg_city"] = _safe_int(parsed.get("mpg_city"))
    parsed["mpg_highway"] = _safe_int(parsed.get("mpg_highway"))

    parsed["make"] = _norm_str(parsed.get("make"))
    parsed["model"] = _norm_str(parsed.get("model"))
    parsed["series"] = _norm_str(parsed.get("series"))
    parsed["trim"] = _norm_str(parsed.get("trim"))
    parsed["vin"] = _norm_str(parsed.get("vin"))
    parsed["stock_number"] = _norm_str(parsed.get("stock_number"))
    parsed["color"] = _norm_str(parsed.get("color"))
    parsed["engine"] = _norm_str(parsed.get("engine"))
    parsed["dealer_name"] = _norm_str(parsed.get("dealer_name"))
    parsed["phone"] = _norm_str(parsed.get("phone"))
    parsed["website"] = _norm_str(parsed.get("website"))
    parsed["posted_date"] = _norm_str(parsed.get("posted_date"))
    parsed["post_id"] = _norm_str(parsed.get("post_id"))
    parsed["full_address"] = _norm_str(parsed.get("full_address"))
    parsed["location_city"] = _norm_str(parsed.get("location_city"))
    parsed["location_state"] = _norm_str(parsed.get("location_state"))
    parsed["location_zip"] = _norm_str(parsed.get("location_zip"))

    parsed["transmission"] = _normalize_transmission(parsed.get("transmission"))
    parsed["fuel"] = _normalize_fuel(parsed.get("fuel"))
    parsed["body_type"] = _normalize_body_type(parsed.get("body_type"))
    parsed["drivetrain"] = _normalize_drivetrain(parsed.get("drivetrain"))
    parsed["condition"] = _normalize_condition(parsed.get("condition"))
    parsed["title_status"] = _normalize_title_status(parsed.get("title_status"))

    return parsed


# -------------------- HTTP ENTRY --------------------
def llm_extract_http(request: Request):
    logging.getLogger().setLevel(logging.INFO)

    if not BUCKET_NAME:
        return jsonify({"ok": False, "error": "missing GCS_BUCKET env"}), 500
    if not PROJECT_ID:
        return jsonify({"ok": False, "error": "missing PROJECT_ID env"}), 500
    if LLM_PROVIDER != "vertex":
        return jsonify({"ok": False, "error": "PoC supports LLM_PROVIDER='vertex' only"}), 400

    try:
        body = request.get_json(silent=True) or {}
    except Exception:
        body = {}

    run_id = body.get("run_id")
    max_files = int(body.get("max_files") or MAX_FILES_DEFAULT or 0)
    overwrite = bool(body.get("overwrite")) if "overwrite" in body else OVERWRITE_DEFAULT

    if not run_id:
        runs = _list_structured_run_ids(BUCKET_NAME, STRUCTURED_PREFIX)
        if not runs:
            return jsonify({"ok": False, "error": f"no run_ids found under {STRUCTURED_PREFIX}/"}), 200
        run_id = runs[-1]

    structured_iso = _normalize_run_id_iso(run_id)

    inputs = _list_per_listing_jsonl_for_run(BUCKET_NAME, run_id)
    if not inputs:
        return jsonify({"ok": True, "run_id": run_id, "processed": 0, "written": 0, "skipped": 0, "errors": 0}), 200
    if max_files > 0:
        inputs = inputs[:max_files]

    logging.info(f"Starting LLM extraction for run_id={run_id} ({len(inputs)} files to process)")

    processed = written = skipped = errors = 0

    for in_key in inputs:
        processed += 1
        try:
            raw_line = _download_text(in_key).strip()
            if not raw_line:
                raise ValueError("empty input jsonl")
            base_rec = json.loads(raw_line)

            post_id = base_rec.get("post_id")
            if not post_id:
                raise ValueError("missing post_id in input record")

            source_txt_key = base_rec.get("source_txt")
            if not source_txt_key:
                raise ValueError("missing source_txt in input record")

            out_prefix = in_key.rsplit("/", 2)[0] + "/jsonl_llm"
            out_key = out_prefix + f"/{post_id}_llm.jsonl"

            if not overwrite and _blob_exists(out_key):
                skipped += 1
                continue

            raw_listing = _download_text(source_txt_key)
            parsed = _vertex_extract_fields(raw_listing)

            out_record = {
                "post_id": post_id,
                "run_id": base_rec.get("run_id", run_id),
                "scraped_at": base_rec.get("scraped_at", structured_iso),
                "source_txt": source_txt_key,
                "price": parsed.get("price"),
                "year": parsed.get("year"),
                "make": parsed.get("make"),
                "model": parsed.get("model"),
                "series": parsed.get("series"),
                "mileage": parsed.get("mileage"),
                "transmission": parsed.get("transmission"),
                "fuel": parsed.get("fuel"),
                "body_type": parsed.get("body_type"),
                "color": parsed.get("color"),
                "title_status": parsed.get("title_status"),
                "condition": parsed.get("condition"),
                "vin": parsed.get("vin"),
                "stock_number": parsed.get("stock_number"),
                "drivetrain": parsed.get("drivetrain"),
                "engine": parsed.get("engine"),
                "mpg_city": parsed.get("mpg_city"),
                "mpg_highway": parsed.get("mpg_highway"),
                "location_city": parsed.get("location_city"),
                "location_state": parsed.get("location_state"),
                "location_zip": parsed.get("location_zip"),
                "full_address": parsed.get("full_address"),
                "dealer_name": parsed.get("dealer_name"),
                "phone": parsed.get("phone"),
                "website": parsed.get("website"),
                "posted_date": parsed.get("posted_date"),
                "llm_provider": "vertex",
                "llm_model": LLM_MODEL,
                "llm_ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            }

            _upload_jsonl_line(out_key, out_record)
            written += 1

        except Exception as e:
            errors += 1
            logging.error(f"LLM extraction failed for {in_key}: {e}\n{traceback.format_exc()}")

    result = {
        "ok": True,
        "version": "extractor-llm-poc",
        "run_id": run_id,
        "processed": processed,
        "written": written,
        "skipped": skipped,
        "errors": errors,
    }
    logging.info(json.dumps(result))
    return jsonify(result), 200

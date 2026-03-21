# main.py
# Purpose: Convert raw TXT -> one-line JSON records (.jsonl) in GCS.
# Compatible input layouts:
#   gs://<bucket>/<SCRAPES_PREFIX>/<RUN>/*.txt
#   gs://<bucket>/<SCRAPES_PREFIX>/<RUN>/txt/*.txt
# where <RUN> is either 20251026T170002Z or 20251026170002.
# Output:
#   gs://<bucket>/<STRUCTURED_PREFIX>/run_id=<RUN>/jsonl/<post_id>.jsonl

import os
import re
import json
import logging
import traceback
from datetime import datetime, timezone

from flask import Request, jsonify
from google.api_core import retry as gax_retry
from google.cloud import storage

# -------------------- ENV --------------------
PROJECT_ID         = os.getenv("PROJECT_ID")
BUCKET_NAME        = os.getenv("GCS_BUCKET")                        # REQUIRED
SCRAPES_PREFIX     = os.getenv("SCRAPES_PREFIX", "scrapes")         # input
STRUCTURED_PREFIX  = os.getenv("STRUCTURED_PREFIX", "structured")   # output

# Accept BOTH run id styles:
RUN_ID_ISO_RE   = re.compile(r"^\d{8}T\d{6}Z$")  # 20251026T170002Z
RUN_ID_PLAIN_RE = re.compile(r"^\d{14}$")        # 20251026170002

READ_RETRY = gax_retry.Retry(
    predicate=gax_retry.if_transient_error,
    initial=1.0, maximum=10.0, multiplier=2.0, deadline=120.0
)

storage_client = storage.Client()

# -------------------- SIMPLE REGEX EXTRACTORS --------------------
PRICE_RE = re.compile(r"\$\s?([0-9,]+)")
YEAR_RE  = re.compile(r"\b(19|20)\d{2}\b")
DOOR_RE  = re.compile(r'\b([2-9])\s*-?\s*door\b', re.I)
COLOR_RE = re.compile(r'\b(red|blue|black|white|silver|gray|grey|green|yellow|orange|brown|gold|purple|beige|maroon|navy|pearl|charcoal)\b', re.I)

# -------------------- HELPERS --------------------
def _list_run_ids(bucket: str, scrapes_prefix: str) -> list[str]:
    """
    List run folders under gs://bucket/<scrapes_prefix>/ and return normalized run_ids.
    Accept:
      - <scrapes_prefix>/run_id=20251026T170002Z/
      - <scrapes_prefix>/20251026170002/
    """
    it = storage_client.list_blobs(bucket, prefix=f"{scrapes_prefix}/", delimiter="/")
    for _ in it:
        pass  # populate it.prefixes

    run_ids: list[str] = []
    for pref in getattr(it, "prefixes", []):
        # e.g., 'scrapes/run_id=20251026T170002Z/' OR 'scrapes/20251026170002/'
        tail = pref.rstrip("/").split("/")[-1]
        cand = tail.split("run_id=", 1)[1] if tail.startswith("run_id=") else tail
        if RUN_ID_ISO_RE.match(cand) or RUN_ID_PLAIN_RE.match(cand):
            run_ids.append(cand)
    return sorted(run_ids)

def _txt_objects_for_run(run_id: str) -> list[str]:
    """
    Return .txt object names for a given run_id.
    Tries (in order) and returns the first non-empty list:
      scrapes/run_id=<run_id>/txt/
      scrapes/run_id=<run_id>/
      scrapes/<run_id>/txt/
      scrapes/<run_id>/
    """
    bucket = storage_client.bucket(BUCKET_NAME)
    candidates = [
        f"{SCRAPES_PREFIX}/run_id={run_id}/txt/",
        f"{SCRAPES_PREFIX}/run_id={run_id}/",
        f"{SCRAPES_PREFIX}/{run_id}/txt/",
        f"{SCRAPES_PREFIX}/{run_id}/",
    ]
    for pref in candidates:
        names = [b.name for b in bucket.list_blobs(prefix=pref) if b.name.endswith(".txt")]
        if names:
            return names
    return []

def _download_text(blob_name: str) -> str:
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)
    return blob.download_as_text(retry=READ_RETRY, timeout=120)

def _upload_jsonl_line(blob_name: str, record: dict):
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
    blob.upload_from_string(line, content_type="application/x-ndjson")

def _parse_run_id_as_iso(run_id: str) -> str:
    """Normalize either run_id style to ISO8601 Z (fallback = now UTC)."""
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

# -------------------- PARSE A LISTING --------------------
def parse_listing(text: str) -> dict:
    import re

    d = {}
    raw = text or ""
    text_l = raw.lower()

    # ---------------- price ----------------
    m = re.search(r"\$\s*([0-9,]+)", raw)
    if m:
        try:
            d["price"] = int(m.group(1).replace(",", ""))
        except ValueError:
            pass

    # ---------------- year ----------------
    year_val = None

    # Prefer standalone year line in main details block
    m = re.search(r"(?m)^\s*((?:19|20)\d{2})\s*$", raw)
    if m:
        try:
            year_val = int(m.group(1))
            d["year"] = year_val
        except ValueError:
            pass

    # Fallback: labeled Year field
    if year_val is None:
        m = re.search(r"(?im)^\s*year\s*[:\-]?\s*((?:19|20)\d{2})\s*$", raw)
        if m:
            try:
                year_val = int(m.group(1))
                d["year"] = year_val
            except ValueError:
                pass

    # ---------------- make / model ----------------
    KNOWN_MAKES = {
        "acura","alfa","audi","bmw","buick","cadillac","chevrolet","chevy","chrysler",
        "dodge","fiat","ford","gmc","genesis","honda","hyundai","infiniti","isuzu",
        "jaguar","jeep","kia","land","lexus","lincoln","mazda","mercedes","mercury",
        "mini","mitsubishi","nissan","pontiac","porsche","ram","rivian","saab","saturn",
        "scion","subaru","suzuki","tesla","toyota","volkswagen","volvo","vw","benz",
        "maserati","lucid","oldsmobile"
    }

    BAD_FIRST_WORDS = {
        "contact","information","new","north","south","east","west","buy","here","print",
        "posted","updated","reply","favorite","hide","flag","delivery","google","map",
        "condition","fuel","drive","odometer","transmission","type","title","vin","stock",
        "conversion","color","interior","engine","miles","price","brand","vehicle"
    }

    def clean_spaces(s: str) -> str:
        return re.sub(r"\s+", " ", s or "").strip()

    def normalize_make(make: str) -> str:
        make = clean_spaces(make).lower()
        mapping = {
            "chevy": "Chevrolet",
            "vw": "Volkswagen",
            "benz": "Mercedes-Benz",
            "land": "Land Rover",
        }
        return mapping.get(make, make.title())

    def looks_like_boundary(line: str) -> bool:
        x = clean_spaces(line).lower()
        if not x:
            return True
        boundary_starts = (
            "condition", "cylinders", "drive", "fuel", "odometer", "paint color",
            "title status", "transmission", "type", "vin", "stock", "price", "miles",
            "engine", "color", "interior color", "vehicle description", "post id",
            "posted", "updated", "qr code", "more ads", "delivery available"
        )
        return any(x.startswith(b) for b in boundary_starts)

    def try_labeled_make_model(text: str):
        make = None
        model = None

        m1 = re.search(r"(?im)^\s*make\s*[:\-]?\s*([A-Za-z][A-Za-z&\-\s]+?)\s*$", text)
        if m1:
            make = clean_spaces(m1.group(1))

        m2 = re.search(r"(?im)^\s*model\s*[:\-]?\s*([A-Za-z0-9][A-Za-z0-9&\-\.\s/]+?)\s*$", text)
        if m2:
            model = clean_spaces(m2.group(1))

        if make and model:
            first = make.split()[0].lower()
            if first not in BAD_FIRST_WORDS:
                return normalize_make(make), model
        return None

    def try_year_followed_by_vehicle_line(text: str):
        lines = text.splitlines()

        for i, line in enumerate(lines[:-1]):
            cur = clean_spaces(line)
            if re.fullmatch(r"(19|20)\d{2}", cur):
                nxt = clean_spaces(lines[i + 1])
                nxt_l = nxt.lower()

                if not nxt or looks_like_boundary(nxt):
                    continue

                parts = nxt.split()
                if len(parts) < 2:
                    continue

                first = parts[0].lower()
                if first in BAD_FIRST_WORDS:
                    continue

                # best case: first token is known make
                if first in KNOWN_MAKES:
                    make = normalize_make(parts[0])

                    # take model until boundary-ish token if it appears
                    model_tokens = []
                    for tok in parts[1:]:
                        tl = tok.lower()
                        if tl in BAD_FIRST_WORDS:
                            break
                        model_tokens.append(tok)

                    if model_tokens:
                        return make, clean_spaces(" ".join(model_tokens))

        return None

    def try_title_line(text: str):
        # first line often looks like:
        # "2013 ford escape for sale - Derby, CT - craigslist"
        first_line = clean_spaces(text.splitlines()[0] if text.splitlines() else "")
        fl = first_line.lower()

        m = re.match(
            r"^\s*((?:19|20)\d{2})\s+([a-z]+)\s+(.+?)\s+for\s+sale\b",
            fl,
            re.I
        )
        if not m:
            return None

        make = m.group(2).strip()
        model_blob = clean_spaces(m.group(3))

        if make.lower() in BAD_FIRST_WORDS:
            return None

        # trim trailing location-ish noise if any
        model_blob = re.split(r"\s+-\s+", model_blob)[0].strip()

        return normalize_make(make), model_blob.title()

    mm = (
        try_labeled_make_model(raw)
        or try_year_followed_by_vehicle_line(raw)
        or try_title_line(raw)
    )

    if mm:
        d["make"] = mm[0]
        d["model"] = mm[1]

    # ---------------- mileage ----------------
    mi = None

    m = re.search(r"(?is)(?:odometer|mileage)\s*:\s*([0-9,]+)", raw)
    if m:
        try:
            mi = int(m.group(1).replace(",", ""))
        except ValueError:
            mi = None

    if mi is None:
        m = re.search(r"(?im)^\s*miles\s*[:\-]?\s*([0-9,]+)\s*$", raw)
        if m:
            try:
                mi = int(m.group(1).replace(",", ""))
            except ValueError:
                mi = None

    if mi is None:
        m = re.search(r"\b([0-9]{1,3}(?:,[0-9]{3})+)\s*miles?\b", raw, re.I)
        if m:
            try:
                mi = int(m.group(1).replace(",", ""))
            except ValueError:
                mi = None

    if mi is not None:
        d["mileage"] = mi

    # ---------------- fuel ----------------
    m = re.search(r"(?is)fuel\s*:\s*(gas|gasoline|diesel|electric|hybrid|phev|bev)", raw)
    if not m:
        m = re.search(r"(?im)^\s*fuel\s+type\s*[:\-]?\s*(gas|gasoline|diesel|electric|hybrid|phev|bev)\s*$", raw)
    if m:
        fuel = m.group(1).lower()
        fuel_map = {"gasoline": "gas"}
        d["fuel_type"] = fuel_map.get(fuel, fuel)

    # ---------------- transmission ----------------
    m = re.search(r"(?is)transmission\s*:\s*([A-Za-z0-9\-\s]+)", raw)
    if m:
        val = clean_spaces(m.group(1)).lower()
        if "automatic" in val:
            d["transmission"] = "automatic"
        elif "manual" in val:
            d["transmission"] = "manual"
        elif "cvt" in val:
            d["transmission"] = "cvt"

    # ---------------- color ----------------
    m = re.search(r"(?is)paint\s*color\s*:\s*([A-Za-z]+)", raw)
    if not m:
        m = re.search(r"(?im)^\s*color\s*[:\-]?\s*([A-Za-z]+)\s*$", raw)
    if m:
        d["color"] = m.group(1).lower()

    # ---------------- cylinders ----------------
    m = re.search(r"(?is)cylinders\s*:\s*([0-9]+)\s*cylinders?", raw)
    if not m:
        m = re.search(r"\b([0-9]+)\s*cylinders?\b", raw, re.I)
    if m:
        try:
            d["cylinders"] = int(m.group(1))
        except ValueError:
            pass

    # ---------------- drive type ----------------
    m = re.search(r"(?is)drive\s*:\s*(4wd|awd|fwd|rwd)", raw)
    if m:
        d["drive_type"] = m.group(1).lower()

    # ---------------- condition ----------------
    m = re.search(r"(?is)condition\s*:\s*(new|like new|excellent|good|fair|salvage)", raw)
    if m:
        d["condition"] = m.group(1).lower()

    # ---------------- num_doors ----------------
    # First try explicit "4 door" / "4dr"
    m = re.search(r"\b([2-6])\s*(?:door|dr)\b", raw, re.I)
    if m:
        try:
            d["num_doors"] = int(m.group(1))
        except ValueError:
            pass
    else:
        # common body-type defaults when explicit door count missing
        vehicle_type = None
        m_type = re.search(r"(?is)type\s*:\s*([A-Za-z]+)", raw)
        if m_type:
            vehicle_type = m_type.group(1).lower()

        model_text = (d.get("model") or "").lower()

        if "coupe" in model_text:
            d["num_doors"] = 2
        elif vehicle_type in {"sedan", "suv", "wagon", "hatchback", "minivan"}:
            d["num_doors"] = 4

    return d
# -------------------- HTTP ENTRY --------------------
def extract_http(request: Request):
    """
    Reads latest (or requested) run's TXT listings and writes ONE-LINE JSON records to:
      gs://<bucket>/<STRUCTURED_PREFIX>/run_id=<run_id>/jsonl/<post_id>.jsonl
    Request JSON (optional):
      { "run_id": "<...>", "max_files": 0, "overwrite": false }
    """
    logging.getLogger().setLevel(logging.INFO)

    if not BUCKET_NAME:
        return jsonify({"ok": False, "error": "missing GCS_BUCKET env"}), 500

    try:
        body = request.get_json(silent=True) or {}
    except Exception:
        body = {}

    run_id    = body.get("run_id")
    max_files = int(body.get("max_files") or 0)        # 0 = unlimited
    overwrite = bool(body.get("overwrite") or False)

    # Pick newest run if not provided
    if not run_id:
        runs = _list_run_ids(BUCKET_NAME, SCRAPES_PREFIX)
        if not runs:
            return jsonify({"ok": False, "error": f"no run_ids found under {SCRAPES_PREFIX}/"}), 200
        run_id = runs[-1]

    scraped_at_iso = _parse_run_id_as_iso(run_id)

    txt_blobs = _txt_objects_for_run(run_id)
    if not txt_blobs:
        return jsonify({"ok": False, "run_id": run_id, "error": "no .txt files found for run"}), 200
    if max_files > 0:
        txt_blobs = txt_blobs[:max_files]

    processed = written = skipped = errors = 0
    bucket = storage_client.bucket(BUCKET_NAME)

    for name in txt_blobs:
        try:
            text = _download_text(name)
            fields = parse_listing(text)

            post_id = os.path.splitext(os.path.basename(name))[0]
            record = {
                "post_id": post_id,
                "run_id": run_id,
                "scraped_at": scraped_at_iso,
                "source_txt": name,
                **fields,
            }

            out_key = f"{STRUCTURED_PREFIX}/run_id={run_id}/jsonl/{post_id}.jsonl"

            if not overwrite and bucket.blob(out_key).exists():
                skipped += 1
            else:
                _upload_jsonl_line(out_key, record)
                written += 1

        except Exception as e:
            errors += 1
            logging.error(f"Failed {name}: {e}\n{traceback.format_exc()}")

        processed += 1

    result = {
        "ok": True,
        "version": "extractor-v3-jsonl-flex",
        "run_id": run_id,
        "processed_txt": processed,
        "written_jsonl": written,
        "skipped_existing": skipped,
        "errors": errors
    }
    logging.info(json.dumps(result))
    return jsonify(result), 200

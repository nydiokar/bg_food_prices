import os, time, json, pathlib, requests, random, datetime
from dotenv import load_dotenv
load_dotenv()

BASE = "https://foodprice.bg/bg/1"
RAW_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw"
FAIL_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw_failed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
FAIL_DIR.mkdir(parents=True, exist_ok=True)

UA = os.getenv("UA", "Mozilla/5.0")
COOKIE = os.getenv("COOKIE", "")

class NonJSONResponseError(RuntimeError):
    pass

HEADERS = {
    "User-Agent": UA,
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Origin": "https://foodprice.bg",
    "X-Requested-With": "XMLHttpRequest",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-Dest": "empty",
    # Client hints (as seen in browser)
    "sec-ch-ua": '"Not;A=Brand";v="99", "Chromium";v="139"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "sec-gpc": "1",
    "priority": "u=1, i",
}
if COOKIE:
    HEADERS["Cookie"] = COOKIE

RATE = float(os.getenv("RATE_PER_SEC", "1"))

# Persistent sessions
_requests_session = requests.Session()
_requests_session.headers.update(HEADERS)
try:
    from curl_cffi import requests as curl_requests  # type: ignore
    _curl_available = True
    _curl_session = curl_requests.Session(impersonate="chrome")
    _curl_session.headers.update(HEADERS)
except Exception:
    _curl_available = False
    _curl_session = None

def _cache_path(product_id: int) -> pathlib.Path:
    return RAW_DIR / f"product_{product_id}.json"

def _fail_path(product_id: int) -> pathlib.Path:
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    return FAIL_DIR / f"product_{product_id}_{ts}.txt"

def _try_load_json(text: str) -> dict:
    text = (text or "").strip()
    if not text:
        raise NonJSONResponseError("Empty body")
    return json.loads(text)

def _decode_response_to_json(r) -> dict:
    ctype = (getattr(r, "headers", {}).get("content-type") or "").lower()
    if "application/json" not in ctype:
        raise NonJSONResponseError(f"Non-JSON content-type: {ctype}")
    # Try native first
    try:
        return r.json()
    except Exception:
        pass
    # Fallback to text
    return _try_load_json(getattr(r, "text", ""))

def _fetch_once(product_id: int):
    params = {"product": product_id}
    ref = {"Referer": f"https://foodprice.bg/bg/1?product={product_id}"}
    if _curl_available and _curl_session is not None:
        r = _curl_session.get(BASE, params=params, headers=ref, timeout=30)
        r.raise_for_status()
        return r
    r = _requests_session.get(BASE, params=params, headers=ref, timeout=30)
    r.raise_for_status()
    return r

def fetch_product_json(product_id: int, refresh: bool=False) -> dict:
    cache = _cache_path(product_id)
    if cache.exists() and not refresh:
        return json.loads(cache.read_text(encoding="utf-8"))
    # One-time warm-up for requests
    if getattr(fetch_product_json, "_warmed", False) is False:
        try:
            _requests_session.get("https://foodprice.bg/", timeout=30)
        except Exception:
            pass
        fetch_product_json._warmed = True  # type: ignore[attr-defined]
    # Retry with backoff and jitter
    last_err: Exception | None = None
    for attempt in range(1, 6):
        try:
            r = _fetch_once(product_id)
            js = _decode_response_to_json(r)
            cache.write_text(json.dumps(js, ensure_ascii=False), encoding="utf-8")
            time.sleep(max(0.0, 1.0 / RATE))
            return js
        except Exception as e:
            last_err = e
            # Save non-JSON body for diagnostics
            if isinstance(e, NonJSONResponseError):
                body = getattr(r, "text", "") if 'r' in locals() else ""
                try:
                    _fail_path(product_id).write_text(str(body)[:2000], encoding="utf-8")
                except Exception:
                    pass
            # Backoff with jitter
            time.sleep(attempt + random.uniform(0.25, 0.75))
            continue
    raise last_err if last_err else RuntimeError("Unknown fetch error")

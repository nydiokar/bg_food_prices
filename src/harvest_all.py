from catalog import PRODUCTS, PRODUCT_IDS
from fetch import fetch_product_json, NonJSONResponseError
from normalize import normalize
from ingest import get_conn, upsert_frame, upsert_product_names, save_raw
import pathlib, datetime, json

EXP = pathlib.Path(__file__).resolve().parent.parent / "data" / "exports"
EXP.mkdir(parents=True, exist_ok=True)
FAIL_LOG = EXP / "_failures.log"


def _is_valid_cached_json(product_id: int) -> bool:
    """Check if cached JSON is valid and contains expected structure."""
    cache_path = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw" / f"product_{product_id}.json"
    if not cache_path.exists():
        return False
    try:
        with cache_path.open("r", encoding="utf-8") as fh:
            js = json.load(fh)
        # Basic validation: should have dates and filters
        return isinstance(js, dict) and "dates" in js and "filters" in js
    except Exception:
        return False


def run():
    conn = get_conn()
    upsert_product_names(conn, PRODUCTS)
    for pid in PRODUCT_IDS:
        try:
            # Skip if we have corrupted/non-JSON cache
            if not _is_valid_cached_json(pid):
                # Force refresh for corrupted cache
                js = fetch_product_json(pid, refresh=True)
            else:
                js = fetch_product_json(pid, refresh=False)
            df = normalize(js, pid)
            upsert_frame(conn, df)
            save_raw(conn, pid, js)
            df.to_csv(EXP / f"product_{pid}.csv", index=False, encoding="utf-8-sig")
            try:
                df.to_parquet(EXP / f"product_{pid}.parquet", index=False)
            except Exception:
                pass
        except Exception as e:
            # Minimal, polite logging: record and continue
            ts = datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z"
            try:
                FAIL_LOG.write_text("", encoding="utf-8", append=False)  # ensure file exists
            except Exception:
                pass
            with FAIL_LOG.open("a", encoding="utf-8") as fh:
                fh.write(f"{ts} skip product {pid}: {e}\n")
            continue


if __name__ == "__main__":
    run()

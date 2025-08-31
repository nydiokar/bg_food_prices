from catalog import PRODUCT_IDS, PRODUCTS
from fetch import fetch_product_json
from normalize import normalize
from ingest import get_conn, upsert_frame, upsert_product_names, save_raw

def run(product_ids=None):
    conn = get_conn()
    upsert_product_names(conn, PRODUCTS)
    ids = PRODUCT_IDS if product_ids is None else list(product_ids)
    for pid in ids:
        js = fetch_product_json(pid, refresh=True)
        df = normalize(js, pid)
        upsert_frame(conn, df)
        save_raw(conn, pid, js)

if __name__ == "__main__":
    run()

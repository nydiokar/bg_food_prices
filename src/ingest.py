import sqlite3, json, datetime, pathlib, pandas as pd

DB = pathlib.Path(__file__).resolve().parent.parent / "data" / "foodprice.sqlite"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS products(
  product_id INTEGER PRIMARY KEY,
  name TEXT,
  unit TEXT
);
CREATE TABLE IF NOT EXISTS cities(
  city TEXT PRIMARY KEY
);
CREATE TABLE IF NOT EXISTS price_series(
  product_id INTEGER,
  date TEXT,
  city TEXT,
  market_type TEXT CHECK(market_type IN ('retail','wholesale')),
  price REAL,
  unit TEXT,
  PRIMARY KEY(product_id, date, city, market_type)
);
CREATE TABLE IF NOT EXISTS raw_cache(
  product_id INTEGER PRIMARY KEY,
  fetched_at TEXT,
  payload TEXT
);
CREATE INDEX IF NOT EXISTS idx_ps_date ON price_series(date);
CREATE INDEX IF NOT EXISTS idx_ps_city ON price_series(city);
"""

def get_conn():
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.executescript(SCHEMA_SQL)
    return conn

def upsert_frame(conn, df: pd.DataFrame):
    rows = df[["product_id","date","city","market_type","price","unit"]].values.tolist()
    conn.executemany(
      "INSERT OR IGNORE INTO price_series(product_id,date,city,market_type,price,unit) VALUES (?,?,?,?,?,?)",
      rows
    )
    conn.executemany("INSERT OR IGNORE INTO cities(city) VALUES (?)", [(c,) for c in df["city"].unique()])
    prod = df[["product_id","unit"]].drop_duplicates()
    conn.executemany("INSERT OR IGNORE INTO products(product_id,unit) VALUES (?,?)", prod.values.tolist())
    conn.commit()

def upsert_product_names(conn, mapping: dict):
    rows = [(int(pid), name) for pid, name in mapping.items()]
    conn.executemany(
      "INSERT INTO products(product_id, name) VALUES (?,?) "
      "ON CONFLICT(product_id) DO UPDATE SET name=excluded.name",
      rows
    )
    conn.commit()

def save_raw(conn, product_id: int, payload: dict):
    conn.execute("INSERT OR REPLACE INTO raw_cache(product_id,fetched_at,payload) VALUES (?,?,?)",
                 (product_id, datetime.datetime.utcnow().isoformat()+"Z", json.dumps(payload, ensure_ascii=False)))
    conn.commit()

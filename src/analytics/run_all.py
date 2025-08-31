from pathlib import Path
from .metrics import generate_all
from .signals import product_watchlist, city_watchlist, load

ROOT = Path(__file__).resolve().parents[2]
DB   = ROOT / "data" / "foodprice.sqlite"
OUT  = ROOT / "data" / "exports" / "analytics"

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    generate_all(str(DB), str(OUT))
    df = load(str(DB))
    product_watchlist(df).to_csv(OUT/"product_watchlist.csv", index=False, encoding="utf-8-sig")
    city_watchlist(df).to_csv(OUT/"city_watchlist.csv", index=False, encoding="utf-8-sig")
    print("Analytics + Signals written to", OUT)

# foodprice suite (one call per product, full history per payload)

Windows quickstart:
  python -m venv .venv
  .venv\Scripts\activate
  pip install -r requirements.txt
  python -m src.harvest_all
  # later:
  python -m src.update_incremental

Data layout:
  data/foodprice.sqlite     # facts + dims
  data/raw/product_<id>.json
  data/exports/product_<id>.csv / .parquet

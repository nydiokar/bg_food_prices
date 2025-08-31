import pandas as pd


def _melt(dates, block: dict, mtype: str) -> pd.DataFrame:
    rows = []
    for loc, series in block.items():
        for d, v in zip(dates, series):
            if v in (None, "NAN", "NaN", "nan"):
                continue
            rows.append((d, loc, mtype, float(v)))
    return pd.DataFrame(rows, columns=["date","location","market_type","price"])


def normalize(js: dict, product_id: int) -> pd.DataFrame:
    dates = js["dates"]
    unit  = (js.get("unit") or "").strip()
    filt  = js["filters"]
    parts = []
    if "filterByLocation" in filt:
        parts.append(_melt(dates, filt["filterByLocation"], "retail"))
    if "filterByLocationBulk" in filt:
        parts.append(_melt(dates, filt["filterByLocationBulk"], "wholesale"))
    if not parts:
        return pd.DataFrame(columns=["product_id","date","city","market_type","price","unit"])
    df = pd.concat(parts, ignore_index=True)
    # city = text before the parenthesis, trimmed
    df["city"] = df["location"].str.extract(r"^(.*?)(?:\s+\(.*\))?$")[0].str.strip()
    df = df.drop(columns=["location"])
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    df["product_id"] = product_id
    df["unit"] = unit
    df = df[["product_id","date","city","market_type","price","unit"]].sort_values(["product_id","city","market_type","date"])
    return df

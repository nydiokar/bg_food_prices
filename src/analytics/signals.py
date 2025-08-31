import sqlite3, pandas as pd, numpy as np
from datetime import timedelta

# thresholds
YOY_HEAT = 0.08
MAD_MULT = 1.5
VOL_MULT = 1.75
MARGIN_Z = 2.0
PREMIUM = 0.05
AFF_PREM = 0.07
AFF_LEVEL = 110.0
BREADTH = 0.60

def _pct_change(s): 
    return s.pct_change(fill_method=None)

def load(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    facts = pd.read_sql_query("SELECT * FROM price_series", conn)
    prods = pd.read_sql_query("SELECT product_id, name FROM products", conn).drop_duplicates()
    conn.close()
    facts["date"] = pd.to_datetime(facts["date"])
    facts["price"] = pd.to_numeric(facts["price"], errors="coerce")
    facts["market_type"] = facts["market_type"].fillna("retail")
    return facts.merge(prods, on="product_id", how="left")

def _mad(x):
    x = np.asarray(x); x = x[~np.isnan(x)]
    if x.size == 0: return 0.0
    med = np.median(x)
    return np.median(np.abs(x - med)) or 1e-9

def product_watchlist(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate product watchlist with comprehensive error handling and risk scoring.
    
    Args:
        df: DataFrame with columns [product_id, name, city, date, price, market_type]
        
    Returns:
        DataFrame with product risk scores and analysis
        
    Raises:
        ValueError: If input data is invalid or processing fails
    """
    try:
        # Input validation
        if df is None or df.empty:
            raise ValueError("Input DataFrame is empty or None")
        
        required_cols = ["product_id", "name", "city", "date", "price", "market_type"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for sufficient data
        if len(df) < 2:
            raise ValueError("Insufficient data for product watchlist (need at least 2 records)")
        
        print(f"🔍 Generating product watchlist for {df['product_id'].nunique()} products")
        
        # Data preparation with error handling
        try:
            # National weekly medians
            nat = (df.groupby(["product_id","name","date"])["price"].median()
                     .reset_index().sort_values(["product_id","date"]))
            
            if nat.empty:
                raise ValueError("No data after national median calculation")
            
            # Calculate returns with validation
            nat["ret"] = nat.groupby("product_id")["price"].pct_change(fill_method=None)
            valid_returns = nat["ret"].notna().sum()
            print(f"📊 Return calculation: {valid_returns}/{len(nat)} valid returns")
            
        except Exception as e:
            raise ValueError(f"Data preparation failed: {e}")

        def feats(g: pd.DataFrame) -> pd.Series:
            """Calculate product features with comprehensive error handling"""
            try:
                if g is None or g.empty:
                    return pd.Series({
                        "last": np.nan, "yoy": np.nan, "slope_8w": np.nan,
                        "HEAT_YOY": False, "HEAT_MOM_PERSIST": False,
                        "VOL_REGIME": False, "ELEVATED_LEVEL": False, "REVERSAL_UP": False
                    })
                
                g = g.sort_values("date")
                end = g["date"].max()
                
                # Safe last price extraction
                try:
                    last = g["price"].iloc[-1]
                except IndexError:
                    last = np.nan
                
                # Safe YoY calculation
                try:
                    if len(g) >= 2:
                        target = end - pd.DateOffset(years=1)
                        idx = (g["date"] - target).abs().values.argmin()
                        yoy = last / g["price"].iloc[idx] - 1
                    else:
                        yoy = np.nan
                except (IndexError, ZeroDivisionError):
                    yoy = np.nan
                
                # Safe rolling window calculations
                try:
                    r12m = g[g["date"] >= end - pd.DateOffset(weeks=52)]
                    r12w = g[g["date"] >= end - pd.DateOffset(weeks=12)]
                except Exception:
                    r12m = pd.DataFrame()
                    r12w = pd.DataFrame()
                
                # Safe MAD calculation
                try:
                    mad = _mad(r12m["ret"]) if not r12m.empty else 0.0
                except Exception:
                    mad = 0.0
                
                # Safe heat indicators
                try:
                    heat_mom_persist = (r12w["ret"] > MAD_MULT * mad).sum() >= 2 if not r12w.empty else False
                except Exception:
                    heat_mom_persist = False
                
                # Safe volatility regime
                try:
                    vol_regime = r12w["ret"].std(skipna=True) > VOL_MULT * r12m["ret"].std(skipna=True)
                except Exception:
                    vol_regime = False
                
                # Safe slope calculation
                def safe_slope(h: pd.DataFrame) -> float:
                    try:
                        if len(h) < 2:
                            return 0.0
                        x = (h["date"] - h["date"].min()).dt.days.values
                        return np.polyfit(x, h["price"].values, 1)[0]
                    except (np.linalg.LinAlgError, ValueError):
                        return 0.0
                
                slope_8w = safe_slope(g[g["date"] >= end - pd.DateOffset(weeks=8)])
                slope_8w_prev = safe_slope(g[(g["date"] < end - pd.DateOffset(weeks=8)) & 
                                            (g["date"] >= end - pd.DateOffset(weeks=16))])
                
                # Safe reversal detection
                try:
                    reversal_up = (slope_8w > 0) and (slope_8w_prev < 0)
                except Exception:
                    reversal_up = False
                
                # Safe elevated level detection
                try:
                    elevated = last >= np.nanpercentile(r12m["price"], 95) if len(r12m) >= 5 else False
                except Exception:
                    elevated = False
                
                return pd.Series({
                    "last": last, "yoy": yoy, "slope_8w": slope_8w,
                    "HEAT_YOY": bool((yoy >= YOY_HEAT) and (slope_8w > 0)),
                    "HEAT_MOM_PERSIST": bool(heat_mom_persist),
                    "VOL_REGIME": bool(vol_regime),
                    "ELEVATED_LEVEL": bool(elevated),
                    "REVERSAL_UP": bool(reversal_up),
                })
                
            except Exception as e:
                print(f"Warning: Feature calculation failed for product group: {e}")
                return pd.Series({
                    "last": np.nan, "yoy": np.nan, "slope_8w": np.nan,
                    "HEAT_YOY": False, "HEAT_MOM_PERSIST": False,
                    "VOL_REGIME": False, "ELEVATED_LEVEL": False, "REVERSAL_UP": False
                })

        # Apply feature calculation with error handling
        try:
            base = nat.groupby(["product_id", "name"]).apply(feats, include_groups=False).reset_index()
            
            if base.empty:
                raise ValueError("Feature calculation produced empty result")
            
            print(f"✅ Features calculated for {len(base)} products")
            
        except Exception as e:
            raise ValueError(f"Feature calculation failed: {e}")

        # Margin stress calculation with error handling
        try:
            # Create pivot table for margin analysis
            piv = df.pivot_table(index=["product_id","city","date"], columns="market_type", values="price")
            
            if piv.empty:
                print("⚠️  No data for margin analysis")
                piv["margin"] = np.nan
            else:
                # Safe margin calculation
                retail_prices = piv.get("retail", pd.Series([np.nan] * len(piv)))
                wholesale_prices = piv.get("wholesale", pd.Series([np.nan] * len(piv)))
                piv["margin"] = retail_prices - wholesale_prices
            
            # Aggregate margins by product and date
            stress = piv.reset_index().groupby(["product_id","date"])["margin"].median().reset_index()
            
        except Exception as e:
            print(f"Warning: Margin analysis failed: {e}")
            stress = pd.DataFrame(columns=["product_id", "date", "margin"])

        def margin_flag(g: pd.DataFrame) -> pd.Series:
            """Calculate margin stress with error handling"""
            try:
                if g is None or g.empty:
                    return pd.Series({"MARGIN_STRESS": False})
                
                end = g["date"].max()
                r26 = g[g["date"] >= end - pd.DateOffset(weeks=26)]["margin"]
                
                if len(r26) == 0:
                    return pd.Series({"MARGIN_STRESS": False})
                
                # Safe statistics calculation
                try:
                    mu, sd = r26.mean(skipna=True), r26.std(skipna=True)
                    last = g[g["date"]==end]["margin"].iloc[-1] if not g[g["date"]==end].empty else np.nan
                    
                    # Safe Z-score calculation
                    if pd.isna(sd) or sd == 0 or pd.isna(last) or pd.isna(mu):
                        return pd.Series({"MARGIN_STRESS": False})
                    
                    z = (last - mu) / sd
                    return pd.Series({"MARGIN_STRESS": bool(z > MARGIN_Z)})
                    
                except Exception:
                    return pd.Series({"MARGIN_STRESS": False})
                    
            except Exception as e:
                print(f"Warning: Margin flag calculation failed: {e}")
                return pd.Series({"MARGIN_STRESS": False})

        # Apply margin analysis with error handling
        try:
            if not stress.empty:
                mflag = stress.groupby("product_id").apply(margin_flag, include_groups=False).reset_index()
            else:
                mflag = pd.DataFrame(columns=["product_id", "MARGIN_STRESS"])
                mflag["MARGIN_STRESS"] = False
            
            # Merge results
            out = base.merge(mflag, on="product_id", how="left").fillna({"MARGIN_STRESS": False})
            
        except Exception as e:
            print(f"Warning: Margin analysis failed: {e}")
            out = base.copy()
            out["MARGIN_STRESS"] = False

        # Calculate risk score with validation
        try:
            out["score"] = (
                2*out["HEAT_YOY"].astype(int) +
                2*out["HEAT_MOM_PERSIST"].astype(int) +
                1*out["VOL_REGIME"].astype(int) +
                2*out["MARGIN_STRESS"].astype(int) +
                1*out["ELEVATED_LEVEL"].astype(int) +
                1*out["REVERSAL_UP"].astype(int)
            )
            
            # Generate reasons with error handling
            def safe_reasons(r):
                try:
                    keys = ["HEAT_YOY","HEAT_MOM_PERSIST","VOL_REGIME","MARGIN_STRESS","ELEVATED_LEVEL","REVERSAL_UP"]
                    valid_keys = [k for k in keys if k in r.index and r[k]]
                    return ", ".join(valid_keys) if valid_keys else "No risk factors"
                except Exception:
                    return "Error in reason generation"
            
            out["reasons"] = out.apply(safe_reasons, axis=1)
            
            # Sort and select columns
            result = out.sort_values(["score","yoy","slope_8w"], ascending=[False, False, False])
            
            # Ensure all required columns exist
            required_output_cols = ["product_id","name","score","reasons","last","yoy","slope_8w"]
            existing_cols = [col for col in required_output_cols if col in result.columns]
            
            final_result = result[existing_cols]
            
            print(f"✅ Successfully generated product watchlist with {len(final_result)} products")
            print(f"📊 Risk score distribution: {final_result['score'].value_counts().sort_index().to_dict()}")
            
            return final_result
            
        except Exception as e:
            raise ValueError(f"Risk scoring failed: {e}")
            
    except Exception as e:
        print(f"❌ Product watchlist generation failed: {e}")
        raise

def city_watchlist(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy().sort_values(["product_id","city","market_type","date"])
    # equal-weight basket, weekly
    x["base_price"] = x.groupby(["product_id","city","market_type"])["price"].transform(lambda s: s.ffill().bfill().iloc[0])
    x["rel"] = 100.0 * x["price"] / x["base_price"]
    basket = (x.groupby(["city","market_type","date"])["rel"].mean().reset_index()
                .rename(columns={"rel":"basket"}))
    
    # Retail analysis (existing logic)
    ret = basket[basket["market_type"]=="retail"]
    nat = ret.groupby("date")["basket"].median().rename("nat").reset_index()
    joined = ret.merge(nat, on="date", how="inner")

    def feats(g):
        g = g.sort_values("date")
        end = g["date"].max()
        last = g[g["date"]==end].iloc[-1]
        prem = last["basket"]/last["nat"] - 1
        win = g[g["date"]>= end - pd.DateOffset(weeks=4)]
        if len(win) < 2:
            slope_4w = 0.0
        else:
            x = (win["date"] - win["date"].min()).dt.days.values
            slope_4w = np.polyfit(x, win["basket"].values, 1)[0]
        return pd.Series({"premium": prem, "slope_4w": slope_4w})

    feats_city = joined.groupby("city").apply(feats, include_groups=False).reset_index()
    
    # NEW: Wholesale analysis for comprehensive city insights
    wholesale = basket[basket["market_type"]=="wholesale"]
    if not wholesale.empty:
        # Calculate wholesale trends relative to national wholesale median
        wholesale_nat = wholesale.groupby("date")["basket"].median().rename("wholesale_nat").reset_index()
        wholesale_joined = wholesale.merge(wholesale_nat, on="date", how="inner")
        
        def wholesale_feats(g):
            g = g.sort_values("date")
            end = g["date"].max()
            last = g[g["date"]==end].iloc[-1]
            wholesale_prem = last["basket"]/last["wholesale_nat"] - 1
            win = g[g["date"]>= end - pd.DateOffset(weeks=4)]
            if len(win) < 2:
                wholesale_slope_4w = 0.0
            else:
                x = (win["date"] - win["date"].min()).dt.days.values
                wholesale_slope_4w = np.polyfit(x, win["basket"].values, 1)[0]
            return pd.Series({
                "wholesale_premium": wholesale_prem, 
                "wholesale_slope_4w": wholesale_slope_4w
            })
        
        wholesale_feats_city = wholesale_joined.groupby("city").apply(wholesale_feats, include_groups=False).reset_index()
        
        # Merge wholesale insights with retail insights
        feats_city = feats_city.merge(wholesale_feats_city, on="city", how="left")
        
        # Add wholesale-specific risk indicators
        feats_city["WHOLESALE_PREMIUM_RISING"] = (
            (feats_city["wholesale_premium"] >= PREMIUM) & 
            (feats_city["wholesale_slope_4w"] > 0)
        )
        feats_city["WHOLESALE_STRESS"] = feats_city["wholesale_premium"] >= AFF_PREM
        
        # Fill NaN values for cities without wholesale data
        feats_city = feats_city.fillna({
            "wholesale_premium": 0.0,
            "wholesale_slope_4w": 0.0,
            "WHOLESALE_PREMIUM_RISING": False,
            "WHOLESALE_STRESS": False
        })
    else:
        # No wholesale data available
        feats_city["wholesale_premium"] = 0.0
        feats_city["wholesale_slope_4w"] = 0.0
        feats_city["WHOLESALE_PREMIUM_RISING"] = False
        feats_city["WHOLESALE_STRESS"] = False

    # breadth up
    df_sorted = df.sort_values(["city","product_id","date"])
    df_sorted["up"] = df_sorted.groupby(["city","product_id"])["price"].diff() > 0
    def breadth(g):
        end = g["date"].max()
        last4 = g[g["date"]>= end - pd.DateOffset(weeks=4)]
        numer = last4.groupby("product_id")["up"].last().sum()
        denom = last4["product_id"].nunique()
        return numer/denom if denom else np.nan
    breadth_city = df_sorted.groupby("city").apply(breadth, include_groups=False).rename("breadth").reset_index()

    # margin stress - FIXED: Based on test results
    piv = df.pivot_table(index=["city","date"], columns="market_type", values="price")
    piv["margin"] = piv.get("retail", pd.Series([np.nan] * len(piv))) - piv.get("wholesale", pd.Series([np.nan] * len(piv)))
    
    def c_margin_flag(g):
        # g is a Series with margin values, indexed by MultiIndex (city, date)
        if len(g) == 0:
            return pd.Series({"CITY_MARGIN_STRESS": False})
        
        # Extract dates from MultiIndex level 1 (date level)
        dates = g.index.get_level_values(1)
        end = dates.max()
        
        # Filter data for last 26 weeks
        r26 = g[dates >= end - pd.DateOffset(weeks=26)]
        
        if len(r26) == 0:
            return pd.Series({"CITY_MARGIN_STRESS": False})
        
        mu, sd = r26.mean(skipna=True), r26.std(skipna=True)
        last = g.iloc[-1] if len(g) > 0 else np.nan
        
        # Proper boolean logic for pandas
        if pd.isna(sd) or sd == 0 or pd.isna(last) or pd.isna(mu):
            return pd.Series({"CITY_MARGIN_STRESS": False})
        
        z = (last - mu) / sd
        return pd.Series({"CITY_MARGIN_STRESS": bool(abs(z) > MARGIN_Z)})
    
    mflag = piv.groupby("city")["margin"].apply(c_margin_flag).reset_index()
    
    # FIX: The apply function created a MultiIndex, we need to flatten it properly
    if 'level_1' in mflag.columns:
        # Rename the columns to what we actually want
        mflag = mflag.rename(columns={'level_1': 'CITY_MARGIN_STRESS'})
        # The 'margin' column contains the actual boolean values
        mflag['CITY_MARGIN_STRESS'] = mflag['margin']
        # Drop the old margin column
        mflag = mflag[['city', 'CITY_MARGIN_STRESS']]
    
    out = feats_city.merge(breadth_city, on="city", how="left").merge(mflag, on="city", how="left")
    
    out["PREMIUM_RISING"]    = (out["premium"] >= PREMIUM) & (out["slope_4w"] > 0)
    out["BREADTH_UP"]        = out["breadth"] >= BREADTH
    out["AFFORDABILITY_ALERT"]= (out["premium"] >= AFF_PREM) & ((out["premium"] + 1)*100 >= AFF_LEVEL)

    # Enhanced scoring including wholesale indicators
    out["score"] = (
        2*out["PREMIUM_RISING"].astype(int) +
        2*out["BREADTH_UP"].astype(int) +
        2*out["CITY_MARGIN_STRESS"].astype(int) +
        1*out["AFFORDABILITY_ALERT"].astype(int) +
        1*out["WHOLESALE_PREMIUM_RISING"].astype(int) +      # NEW: Wholesale premium rising
        1*out["WHOLESALE_STRESS"].astype(int)                 # NEW: Wholesale stress
    )
    
    def reasons(r):
        keys = ["PREMIUM_RISING","BREADTH_UP","CITY_MARGIN_STRESS","AFFORDABILITY_ALERT",
                "WHOLESALE_PREMIUM_RISING","WHOLESALE_STRESS"]
        return ", ".join([k for k in keys if r[k]])
    
    out["reasons"] = out.apply(reasons, axis=1)
    
    # Return enhanced results with wholesale insights
    return out.sort_values(["score","premium"], ascending=[False, False])[
        ["city","score","reasons","premium","breadth","CITY_MARGIN_STRESS",
         "wholesale_premium","WHOLESALE_PREMIUM_RISING","WHOLESALE_STRESS"]
    ]

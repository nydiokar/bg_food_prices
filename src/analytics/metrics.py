import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path

def _pct_change(s: pd.Series) -> pd.Series:
    return s.pct_change(fill_method=None)

# ---------- load, weeklyize (numeric-only), return tidy facts ----------
def _to_weekly_numeric(f: pd.DataFrame) -> pd.DataFrame:
    # expects columns: product_id, city, market_type, date, price
    f = f.sort_values(["product_id", "city", "market_type", "date"]).set_index("date")

    def resample_group(g: pd.DataFrame) -> pd.DataFrame:
        # resample ONLY the numeric price
        s = g["price"].resample("W-FRI").median()
        s = s.rolling(4, min_periods=1).median()         # smooth noise, keep level
        out = s.to_frame("price")
        # Get the grouping values from the group name
        out["product_id"]  = g.name[0] if hasattr(g, 'name') and g.name else g["product_id"].iloc[0]
        out["city"]        = g.name[1] if hasattr(g, 'name') and g.name else g["city"].iloc[0]
        out["market_type"] = g.name[2] if hasattr(g, 'name') and g.name else g["market_type"].iloc[0]
        return out.reset_index()

    return (
        f.groupby(["product_id", "city", "market_type"], group_keys=False)
         .apply(resample_group, include_groups=False)
         .reset_index(drop=True)
    )

def load_facts(db_path: str) -> pd.DataFrame:
    """
    Load and process facts from database with comprehensive error handling.
    
    Args:
        db_path: Path to SQLite database
        
    Returns:
        Processed DataFrame with weekly numeric data
        
    Raises:
        FileNotFoundError: If database file doesn't exist
        sqlite3.Error: If database operations fail
        ValueError: If data processing fails
    """
    try:
        # Validate database path
        if not Path(db_path).exists():
            raise FileNotFoundError(f"Database file not found: {db_path}")
        
        # Database connection with error handling
        try:
            conn = sqlite3.connect(db_path)
        except sqlite3.Error as e:
            raise sqlite3.Error(f"Failed to connect to database: {e}")
        
        try:
            # Load price series with validation
            facts = pd.read_sql_query(
                "SELECT product_id, city, market_type, date, price FROM price_series", conn
            )
            if facts.empty:
                raise ValueError("No price data found in database")
            
            # Load products with validation
            prods = pd.read_sql_query(
                "SELECT product_id, name FROM products", conn
            ).drop_duplicates()
            if prods.empty:
                raise ValueError("No product data found in database")
                
        except sqlite3.Error as e:
            raise sqlite3.Error(f"Failed to query database: {e}")
        finally:
            conn.close()
        
        # Data validation and processing with error handling
        try:
            # Validate required columns
            required_cols = ["product_id", "city", "market_type", "date", "price"]
            missing_cols = [col for col in required_cols if col not in facts.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Convert date with validation
            facts["date"] = pd.to_datetime(facts["date"], errors="coerce")
            invalid_dates = facts["date"].isna().sum()
            if invalid_dates > 0:
                print(f"Warning: {invalid_dates} invalid dates found and converted to NaT")
            
            # Convert price with validation
            facts["price"] = pd.to_numeric(facts["price"], errors="coerce")
            invalid_prices = facts["price"].isna().sum()
            if invalid_prices > 0:
                print(f"Warning: {invalid_prices} invalid prices found and converted to NaN")
            
            # Handle market type defaults
            facts["market_type"] = facts["market_type"].fillna("retail")
            
            # Weekly processing
            wk = _to_weekly_numeric(facts)
            if wk.empty:
                raise ValueError("Weekly processing produced empty dataset")
            
            # Merge with products
            wk = wk.merge(prods, on="product_id", how="left")
            missing_names = wk["name"].isna().sum()
            if missing_names > 0:
                print(f"Warning: {missing_names} records missing product names")
            
            print(f"✅ Successfully loaded {len(wk)} records from database")
            return wk
            
        except Exception as e:
            raise ValueError(f"Data processing failed: {e}")
            
    except Exception as e:
        print(f"❌ Failed to load facts: {e}")
        raise

# ---------- summaries on weekly facts ----------
def product_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate product summary statistics with comprehensive error handling.
    
    Args:
        df: DataFrame with columns [product_id, name, date, price]
        
    Returns:
        DataFrame with product summary statistics
        
    Raises:
        ValueError: If input data is invalid or processing fails
    """
    try:
        # Input validation
        if df is None or df.empty:
            raise ValueError("Input DataFrame is empty or None")
        
        required_cols = ["product_id", "name", "date", "price"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for sufficient data
        if len(df) < 2:
            raise ValueError("Insufficient data for analysis (need at least 2 records)")
        
        # Data preparation with error handling
        try:
            nat = (
                df.groupby(["product_id", "name", "date"])["price"]
                  .median()
                  .reset_index()
                  .sort_values(["product_id", "date"])
            )
            
            if nat.empty:
                raise ValueError("No data after grouping and median calculation")
            
            # Calculate returns with validation
            nat["ret"] = nat.groupby("product_id")["price"].pct_change(fill_method=None)
            
        except Exception as e:
            raise ValueError(f"Data preparation failed: {e}")

        def agg(g: pd.DataFrame) -> pd.Series:
            """Aggregate function with error handling for each product group"""
            try:
                if g is None or g.empty:
                    return pd.Series({
                        "start": pd.NaT, "end": pd.NaT, "n_obs": 0,
                        "median_price": np.nan, "avg_mom": np.nan,
                        "vol_weekly": np.nan, "trend_slope_per_day": np.nan,
                        "last_price": np.nan, "yoy_last": np.nan
                    })
                
                g = g.sort_values("date")
                
                # Safe slope calculation
                try:
                    if len(g) >= 2:
                        x = (g["date"] - g["date"].min()).dt.days.values
                        slope = np.polyfit(x, g["price"].values, 1)[0]
                    else:
                        slope = np.nan
                except (np.linalg.LinAlgError, ValueError):
                    slope = np.nan
                
                end = g["date"].max()
                
                # Safe YoY calculation
                try:
                    if len(g) >= 2:
                        target = end - pd.DateOffset(years=1)
                        idx = (g["date"] - target).abs().values.argmin()
                        yoy = (g["price"].iloc[-1] / g["price"].iloc[idx] - 1)
                    else:
                        yoy = np.nan
                except (IndexError, ZeroDivisionError):
                    yoy = np.nan
                
                # Safe statistics calculation
                try:
                    median_price = g["price"].median()
                    avg_mom = g["ret"].mean(skipna=True)
                    vol_weekly = g["ret"].std(skipna=True)
                    last_price = g["price"].iloc[-1]
                except Exception:
                    median_price = np.nan
                    avg_mom = np.nan
                    vol_weekly = np.nan
                    last_price = np.nan
                
                return pd.Series({
                    "start": g["date"].min().date() if not g.empty else pd.NaT,
                    "end": end.date() if not g.empty else pd.NaT,
                    "n_obs": len(g),
                    "median_price": median_price,
                    "avg_mom": avg_mom,
                    "vol_weekly": vol_weekly,
                    "trend_slope_per_day": slope,
                    "last_price": last_price,
                    "yoy_last": yoy
                })
                
            except Exception as e:
                print(f"Warning: Failed to process product group: {e}")
                return pd.Series({
                    "start": pd.NaT, "end": pd.NaT, "n_obs": 0,
                    "median_price": np.nan, "avg_mom": np.nan,
                    "vol_weekly": np.nan, "trend_slope_per_day": np.nan,
                    "last_price": np.nan, "yoy_last": np.nan
                })

        # Apply aggregation with error handling
        try:
            result = nat.groupby(["product_id", "name"]).apply(agg, include_groups=False).reset_index()
            
            if result.empty:
                raise ValueError("Product summary calculation produced empty result")
            
            print(f"✅ Successfully generated product summary for {len(result)} products")
            return result
            
        except Exception as e:
            raise ValueError(f"Product summary aggregation failed: {e}")
            
    except Exception as e:
        print(f"❌ Product summary failed: {e}")
        raise

def city_spread(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["product_id", "name", "date"])["price"]
    s = g.agg(
        p10=lambda x: np.nanpercentile(x, 10),
        median="median",
        p90=lambda x: np.nanpercentile(x, 90),
    ).reset_index()
    s["spread_abs"] = s["p90"] - s["p10"]
    s["spread_rel"] = s["spread_abs"] / s["median"]
    return s

def margins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate retail-wholesale margins with comprehensive error handling and data imputation.
    
    Args:
        df: DataFrame with columns [product_id, name, city, date, price, market_type]
        
    Returns:
        DataFrame with margin calculations and data quality indicators
        
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
            raise ValueError("Insufficient data for margin analysis (need at least 2 records)")
        
        # Validate market types
        valid_market_types = df["market_type"].unique()
        if len(valid_market_types) == 0:
            raise ValueError("No valid market types found")
        
        print(f"📊 Processing margins for {len(valid_market_types)} market types: {valid_market_types}")
        
        # Data preparation with error handling
        try:
            # Create pivot table with better handling of missing data
            piv = df.pivot_table(
                index=["product_id", "name", "city", "date"],
                columns="market_type",
                values="price",
                aggfunc="first",  # Use first value if duplicates exist
                fill_value=np.nan
            )
            
            if piv.empty:
                raise ValueError("Pivot table creation produced empty result")
            
            # Handle missing market types gracefully
            available_market_types = piv.columns.tolist()
            print(f"📈 Available market types in pivot: {available_market_types}")
            
            # Initialize price series with NaN
            retail_prices = pd.Series([np.nan] * len(piv), index=piv.index)
            wholesale_prices = pd.Series([np.nan] * len(piv), index=piv.index)
            
            # Safely extract prices for each market type
            if "retail" in available_market_types:
                retail_prices = piv["retail"].fillna(np.nan)
                print(f"✅ Retail prices: {retail_prices.notna().sum()}/{len(retail_prices)} valid")
            else:
                print("⚠️  No retail data available")
                
            if "wholesale" in available_market_types:
                wholesale_prices = piv["wholesale"].fillna(np.nan)
                print(f"✅ Wholesale prices: {wholesale_prices.notna().sum()}/{len(wholesale_prices)} valid")
            else:
                print("⚠️  No wholesale data available")
            
            # Calculate margins with validation
            margins = retail_prices - wholesale_prices
            
            # Data quality analysis
            total_records = len(margins)
            valid_margins = margins.notna().sum()
            missing_margins = total_records - valid_margins
            
            print(f"📊 Margin calculation results:")
            print(f"   Total records: {total_records}")
            print(f"   Valid margins: {valid_margins}")
            print(f"   Missing margins: {missing_margins}")
            print(f"   Data completeness: {valid_margins/total_records*100:.1f}%")
            
            # Add margin to pivot table
            piv["margin"] = margins
            
            # Create result with additional data quality columns
            result = piv.reset_index()
            
            # Add data quality indicators
            result["has_retail"] = result["retail"].notna()
            result["has_wholesale"] = result["wholesale"].notna()
            result["margin_quality"] = result.apply(
                lambda row: "complete" if row["has_retail"] and row["has_wholesale"] 
                else "retail_only" if row["has_retail"] 
                else "wholesale_only" if row["has_wholesale"] 
                else "missing_both", axis=1
            )
            
            # Select and reorder columns
            final_columns = ["product_id", "name", "city", "date", "margin", 
                           "retail", "wholesale", "margin_quality"]
            available_final_columns = [col for col in final_columns if col in result.columns]
            
            result = result[available_final_columns]
            
            print(f"✅ Successfully calculated margins for {len(result)} records")
            return result
            
        except Exception as e:
            raise ValueError(f"Margin calculation failed: {e}")
            
    except Exception as e:
        print(f"❌ Margin analysis failed: {e}")
        raise

def anomalies(df: pd.DataFrame, thresh: float = 3.5) -> pd.DataFrame:
    """
    Detect price anomalies using MAD-based methodology with comprehensive error handling.
    
    Args:
        df: DataFrame with columns [product_id, city, market_type, date, price]
        thresh: MAD multiplier threshold for anomaly detection (default: 3.5)
        
    Returns:
        DataFrame with detected anomalies and their characteristics
        
    Raises:
        ValueError: If input data is invalid or processing fails
    """
    try:
        # Input validation
        if df is None or df.empty:
            raise ValueError("Input DataFrame is empty or None")
        
        required_cols = ["product_id", "city", "market_type", "date", "price"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Validate threshold
        if not isinstance(thresh, (int, float)) or thresh <= 0:
            raise ValueError(f"Invalid threshold: {thresh}. Must be positive number.")
        
        print(f"🔍 Detecting anomalies with threshold: {thresh}")
        
        # Data preparation with error handling
        try:
            # Sort data for proper grouping
            df = df.sort_values(["product_id", "city", "market_type", "date"])
            
            # Calculate returns with validation
            df["ret"] = df.groupby(["product_id", "city", "market_type"])["price"].transform(_pct_change)
            
            # Check for valid returns
            valid_returns = df["ret"].notna().sum()
            total_records = len(df)
            print(f"📊 Return calculation: {valid_returns}/{total_records} valid returns")
            
            if valid_returns == 0:
                print("⚠️  No valid returns calculated - cannot detect anomalies")
                return pd.DataFrame(columns=["product_id", "city", "market_type", "date", "price", "ret"])
            
        except Exception as e:
            raise ValueError(f"Data preparation failed: {e}")

        def mad_flag(s: pd.Series) -> pd.Series:
            """Detect anomalies using MAD-based methodology with error handling"""
            try:
                if s is None or s.empty:
                    return pd.Series([False] * len(s), index=s.index)
                
                # Convert to numeric and remove NaN values
                x = s.values.astype(float)
                x = x[~np.isnan(x)]
                
                if x.size == 0:
                    return pd.Series([False] * len(s), index=s.index)
                
                # Calculate median and MAD with validation
                try:
                    med = np.median(x)
                    mad = np.median(np.abs(x - med))
                    
                    # Handle edge case where MAD is zero
                    if mad == 0:
                        mad = 1e-9
                    
                    # Calculate robust Z-scores
                    rz = 0.6745 * (s.fillna(0) - med) / mad
                    
                    # Flag anomalies
                    return rz.abs() > thresh
                    
                except (ValueError, RuntimeWarning) as e:
                    print(f"Warning: MAD calculation failed for group: {e}")
                    return pd.Series([False] * len(s), index=s.index)
                    
            except Exception as e:
                print(f"Warning: Anomaly detection failed for group: {e}")
                return pd.Series([False] * len(s), index=s.index)

        # Apply anomaly detection with error handling
        try:
            df["is_anomaly"] = df.groupby(["product_id", "city", "market_type"])["ret"].transform(mad_flag)
            
            # Count anomalies
            anomaly_count = df["is_anomaly"].sum()
            print(f"🔍 Anomaly detection complete: {anomaly_count} anomalies found")
            
            # Filter anomalies
            anomalies_df = df.loc[df["is_anomaly"]].copy()
            
            if anomalies_df.empty:
                print("✅ No anomalies detected")
                return pd.DataFrame(columns=["product_id", "city", "market_type", "date", "price", "ret"])
            
            # Select columns that exist
            available_cols = ["product_id", "name", "city", "market_type", "date", "price", "ret"]
            existing_cols = [col for col in available_cols if col in anomalies_df.columns]
            
            result = anomalies_df[existing_cols]
            
            # Add anomaly characteristics
            if "ret" in result.columns:
                result["anomaly_magnitude"] = result["ret"].abs()
                result["anomaly_direction"] = result["ret"].apply(lambda x: "up" if x > 0 else "down")
            
            print(f"✅ Successfully detected {len(result)} anomalies")
            return result
            
        except Exception as e:
            raise ValueError(f"Anomaly detection failed: {e}")
            
    except Exception as e:
        print(f"❌ Anomaly detection failed: {e}")
        raise

def basket_index(df: pd.DataFrame, weights: dict | None = None) -> pd.DataFrame:
    """
    Calculate basket index with configurable weighting.
    
    Args:
        df: DataFrame with columns [product_id, city, market_type, date, price]
        weights: Optional dict mapping product_id to weight. If None, uses equal weights per product.
    
    Returns:
        DataFrame with columns [city, market_type, date, basket_index]
    """
    if weights is None:
        weights = {pid: 1.0 for pid in df["product_id"].unique()}
    x = df.copy().sort_values(["product_id", "city", "market_type", "date"])
    x["w"] = x["product_id"].map(weights).fillna(0.0)
    # base = first valid price per product-city-market
    x["base_price"] = x.groupby(["product_id", "city", "market_type"])["price"].transform(
        lambda s: s.ffill().bfill().iloc[0]
    )
    x["rel"] = 100.0 * x["price"] / x["base_price"]
    
    def weighted_avg(group):
        if group["w"].sum() > 0:
            return np.average(group["rel"], weights=group["w"])
        return np.nan
    
    basket = (
        x.groupby(["city", "market_type", "date"])
         .apply(weighted_avg, include_groups=False)
         .reset_index(name="basket_index")
    )
    return basket

def basket_index_true_equal_weight(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate basket index with true equal weighting across all products.
    
    This function ensures that each product contributes equally to the basket,
    regardless of how many cities or market types have data for that product.
    
    Args:
        df: DataFrame with columns [product_id, city, market_type, date, price]
    
    Returns:
        DataFrame with columns [city, market_type, date, basket_index]
    """
    x = df.copy().sort_values(["product_id", "city", "market_type", "date"])
    
    # Calculate relative prices for each product-city-market combination
    x["base_price"] = x.groupby(["product_id", "city", "market_type"])["price"].transform(
        lambda s: s.ffill().bfill().iloc[0]
    )
    x["rel"] = 100.0 * x["price"] / x["base_price"]
    
    # Group by city, market_type, date and calculate equal-weighted average
    # Each product gets equal weight (1/N) regardless of data availability
    def equal_weighted_avg(group):
        # Get unique products in this group
        products = group["product_id"].unique()
        if len(products) == 0:
            return np.nan
        
        # Calculate average relative price across all products
        # This ensures equal weight per product, not per data point
        product_means = []
        for pid in products:
            product_data = group[group["product_id"] == pid]["rel"]
            if not product_data.empty and not product_data.isna().all():
                product_means.append(product_data.mean())
        
        if product_means:
            return np.mean(product_means)  # Equal weight per product
        return np.nan
    
    basket = (
        x.groupby(["city", "market_type", "date"])
         .apply(equal_weighted_avg, include_groups=False)
         .reset_index(name="basket_index")
    )
    return basket

def data_quality_report(df: pd.DataFrame) -> dict:
    """
    Generate comprehensive data quality report for the dataset.
    
    Args:
        df: DataFrame with price data
        
    Returns:
        Dictionary containing data quality metrics and insights
    """
    try:
        if df is None or df.empty:
            return {"error": "Empty or None DataFrame"}
        
        report = {
            "overview": {
                "total_records": len(df),
                "date_range": {
                    "start": df["date"].min().strftime("%Y-%m-%d") if not df.empty else None,
                    "end": df["date"].max().strftime("%Y-%m-%d") if not df.empty else None
                },
                "unique_products": df["product_id"].nunique(),
                "unique_cities": df["city"].nunique(),
                "unique_market_types": df["market_type"].nunique()
            },
            "data_completeness": {
                "valid_prices": df["price"].notna().sum(),
                "valid_dates": df["date"].notna().sum(),
                "valid_product_ids": df["product_id"].notna().sum(),
                "valid_cities": df["city"].notna().sum(),
                "valid_market_types": df["market_type"].notna().sum(),
                "valid_names": df["name"].notna().sum() if "name" in df.columns else 0
            },
            "market_type_distribution": df["market_type"].value_counts().to_dict(),
            "city_coverage": {
                "cities_with_retail": len(df[df["market_type"] == "retail"]["city"].unique()),
                "cities_with_wholesale": len(df[df["market_type"] == "wholesale"]["city"].unique()),
                "cities_with_both": len(
                    set(df[df["market_type"] == "retail"]["city"].unique()) & 
                    set(df[df["market_type"] == "wholesale"]["city"].unique())
                )
            },
            "product_coverage": {
                "products_with_retail": len(df[df["market_type"] == "retail"]["product_id"].unique()),
                "products_with_wholesale": len(df[df["market_type"] == "wholesale"]["product_id"].unique()),
                "products_with_both": len(
                    set(df[df["market_type"] == "retail"]["product_id"].unique()) & 
                    set(df[df["market_type"] == "wholesale"]["product_id"].unique())
                )
            },
            "temporal_coverage": {
                "weekly_records": len(df),
                "date_gaps": _identify_date_gaps(df),
                "seasonal_coverage": _analyze_seasonal_coverage(df)
            },
            "data_quality_score": _calculate_data_quality_score(df)
        }
        
        # Calculate percentages
        total = len(df)
        report["data_completeness"]["price_completeness_pct"] = (report["data_completeness"]["valid_prices"] / total) * 100
        report["data_completeness"]["date_completeness_pct"] = (report["data_completeness"]["valid_dates"] / total) * 100
        
        return report
        
    except Exception as e:
        return {"error": f"Data quality report generation failed: {e}"}

def _identify_date_gaps(df: pd.DataFrame) -> dict:
    """Identify gaps in temporal coverage"""
    try:
        df_sorted = df.sort_values("date")
        date_diff = df_sorted["date"].diff().dt.days
        gaps = date_diff[date_diff > 7]  # Gaps larger than a week
        
        return {
            "total_gaps": len(gaps),
            "max_gap_days": int(gaps.max()) if len(gaps) > 0 else 0,
            "avg_gap_days": float(gaps.mean()) if len(gaps) > 0 else 0,
            "gap_distribution": gaps.value_counts().head(10).to_dict()
        }
    except Exception:
        return {"error": "Date gap analysis failed"}

def _analyze_seasonal_coverage(df: pd.DataFrame) -> dict:
    """Analyze seasonal data coverage"""
    try:
        # Create a copy to avoid modifying original dataframe
        df_copy = df.copy()
        df_copy["month"] = df_copy["date"].dt.month
        
        # Handle case where month column might be empty
        if df_copy["month"].isna().all():
            return {"error": "No valid dates for seasonal analysis"}
        
        monthly_counts = df_copy["month"].value_counts().sort_index()
        
        # Ensure we have data for each season
        spring_data = monthly_counts[monthly_counts.index.isin([3, 4, 5])]
        summer_data = monthly_counts[monthly_counts.index.isin([6, 7, 8])]
        autumn_data = monthly_counts[monthly_counts.index.isin([9, 10, 11])]
        winter_data = monthly_counts[monthly_counts.index.isin([12, 1, 2])]
        
        return {
            "monthly_distribution": monthly_counts.to_dict(),
            "seasonal_balance": {
                "spring": int(spring_data.sum()) if not spring_data.empty else 0,
                "summer": int(summer_data.sum()) if not summer_data.empty else 0,
                "autumn": int(autumn_data.sum()) if not autumn_data.empty else 0,
                "winter": int(winter_data.sum()) if not winter_data.empty else 0
            }
        }
    except Exception as e:
        return {"error": f"Seasonal analysis failed: {str(e)}"}

def _calculate_data_quality_score(df: pd.DataFrame) -> dict:
    """Calculate overall data quality score"""
    try:
        total = len(df)
        if total == 0:
            return {"score": 0, "grade": "F", "issues": ["No data"]}
        
        # Calculate completeness scores
        price_score = df["price"].notna().sum() / total
        date_score = df["date"].notna().sum() / total
        city_score = df["city"].notna().sum() / total
        market_score = df["market_type"].notna().sum() / total
        
        # Calculate coverage scores
        product_coverage = df["product_id"].nunique() / max(df["product_id"].nunique(), 1)
        city_coverage = df["city"].nunique() / max(df["city"].nunique(), 1)
        market_coverage = df["market_type"].nunique() / max(df["market_type"].nunique(), 1)
        
        # Overall score (weighted average)
        overall_score = (
            price_score * 0.25 +      # Price data is critical
            date_score * 0.20 +       # Date data is critical
            city_score * 0.15 +       # City data is important
            market_score * 0.15 +     # Market type is important
            product_coverage * 0.10 + # Product diversity
            city_coverage * 0.10 +    # Geographic diversity
            market_coverage * 0.05    # Market type diversity
        ) * 100
        
        # Grade assignment
        if overall_score >= 90:
            grade = "A"
        elif overall_score >= 80:
            grade = "B"
        elif overall_score >= 70:
            grade = "C"
        elif overall_score >= 60:
            grade = "D"
        else:
            grade = "F"
        
        # Identify issues
        issues = []
        if price_score < 0.95:
            issues.append(f"Price completeness: {price_score*100:.1f}%")
        if date_score < 0.95:
            issues.append(f"Date completeness: {date_score*100:.1f}%")
        if city_score < 0.95:
            issues.append(f"City completeness: {city_score*100:.1f}%")
        if market_score < 0.95:
            issues.append(f"Market type completeness: {market_score*100:.1f}%")
        
        return {
            "score": round(overall_score, 1),
            "grade": grade,
            "issues": issues,
            "component_scores": {
                "price_completeness": round(price_score * 100, 1),
                "date_completeness": round(date_score * 100, 1),
                "city_completeness": round(city_score * 100, 1),
                "market_completeness": round(market_score * 100, 1),
                "product_coverage": round(product_coverage * 100, 1),
                "city_coverage": round(city_coverage * 100, 1),
                "market_coverage": round(market_coverage * 100, 1)
            }
        }
        
    except Exception as e:
        return {"error": f"Quality score calculation failed: {e}"}

def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    nat = (
        df.groupby(["product_id", "name", "date"])["price"]
          .median()
          .reset_index()
          .sort_values(["product_id", "date"])
    )
    nat["ret"] = nat.groupby("product_id")["price"].pct_change(fill_method=None)
    piv = nat.pivot(index="date", columns="product_id", values="ret")
    corr = piv.corr()
    corr.index.name = "product_id"
    return corr.reset_index()

# ---------- batch writer ----------
def generate_all(db_path: str, out_dir: str):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    df = load_facts(db_path)
    product_summary(df).to_csv(out / "product_summary.csv", index=False, encoding="utf-8-sig")
    city_spread(df).to_csv(out / "city_spread.csv", index=False, encoding="utf-8-sig")
    margins(df).to_csv(out / "margins.csv", index=False, encoding="utf-8-sig")
    anomalies(df).to_csv(out / "anomalies.csv", index=False, encoding="utf-8-sig")
    basket_index(df).to_csv(out / "basket_index.csv", index=False, encoding="utf-8-sig")
    correlation_matrix(df).to_csv(out / "correlation.csv", index=False, encoding="utf-8-sig")

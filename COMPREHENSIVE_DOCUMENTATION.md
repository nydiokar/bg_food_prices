# Bulgarian Food Price Intelligence - Comprehensive Documentation

## Project Overview

The **Bulgarian Food Price Intelligence** system is a comprehensive food price monitoring and analysis platform that tracks retail and wholesale food prices across Bulgaria. The system gathers data from multiple sources, processes it through various analytical pipelines, and provides insights into price trends, market signals, and anomalies.

### What the Project Does

The system monitors **36 essential food products** across **29 cities** in Bulgaria, collecting weekly price data for both retail and wholesale markets. It provides:

- **Real-time price monitoring** across different market types
- **Market signal detection** for unusual price movements
- **City-level analysis** showing regional price variations
- **Margin analysis** between retail and wholesale prices
- **Anomaly detection** for price spikes and drops
- **Basket index calculations** for overall food price trends

## Data Gathering and Sources

### Data Collection Process

The system collects data through the following pipeline:

1. **Web Scraping**: Data is harvested from `https://foodprice.bg` using automated HTTP requests
2. **JSON API**: Each product has a unique endpoint that returns structured price data
3. **Caching**: Raw JSON responses are cached locally to minimize API calls
4. **Normalization**: Raw data is processed into standardized database format
5. **Storage**: Processed data is stored in SQLite database with proper indexing

### Data Sources by Filter Type

**IMPORTANT: The system currently only processes and stores data from 2 out of 7 available filter types.**

Based on the code analysis and the `filterTexts` from the JSON data, here's what's available vs. what's actually used:

#### **ACTUALLY PROCESSED AND STORED:**

#### 1. **filterByLocationBulk** - Wholesale Prices ✅
- **Source**: State Commission on Commodity Exchanges and Markets (ДКСБТ)
- **Data Type**: Wholesale prices with VAT
- **Frequency**: Weekly
- **Coverage**: National and regional wholesale markets
- **What it represents**: Bulk purchase prices for businesses and retailers
- **Status**: **ACTIVELY USED** → stored as `market_type = "wholesale"`

#### 2. **filterByLocation** - Retail Prices ✅
- **Source**: Direct market monitoring across regions
- **Data Type**: Retail prices with VAT
- **Frequency**: Weekly
- **Coverage**: All regions, including major retail chains and smaller stores
- **What it represents**: Consumer-facing prices in stores
- **Status**: **ACTIVELY USED** → stored as `market_type = "retail"`

---

#### **AVAILABLE IN RAW DATA BUT NOT PROCESSED:**

#### 3. **filterBySupplyMilk** - Milk Purchase Prices ❌
- **Source**: Registered agricultural producers
- **Data Type**: Milk purchase prices from farmers
- **Frequency**: Weekly
- **Coverage**: National milk supply chain
- **What it represents**: Farm gate prices for milk products
- **Status**: **NOT PROCESSED** → data exists in JSON but not stored in database

#### 4. **filterByCountry** - International Prices ❌
- **Source**: Economic advisors in Trade and Economic Affairs Services
- **Data Type**: International food prices
- **Frequency**: Weekly
- **Coverage**: Major trading partner countries
- **What it represents**: Import/export price benchmarks
- **Status**: **NOT PROCESSED** → data exists in JSON but not stored in database

#### 5. **filterCOOP** - Cooperative Prices ❌
- **Source**: Cooperative stores (e.g., Kalofer store in Plovdiv region)
- **Data Type**: Cooperative retail prices
- **Frequency**: Weekly
- **Coverage**: Specific cooperative locations
- **What it represents**: Alternative retail pricing models
- **Status**: **NOT PROCESSED** → data exists in JSON but not stored in database

#### 6. **filterLIDL** - LIDL Store Prices ❌
- **Source**: LIDL retail chain stores
- **Data Type**: LIDL-specific retail prices
- **Frequency**: Weekly
- **Coverage**: LIDL stores across Bulgaria
- **What it represents**: Major retailer pricing strategy
- **Status**: **NOT PROCESSED** → data exists in JSON but not stored in database

#### 7. **filterImport** - Import Prices ❌
- **Source**: NAP (National Revenue Agency) declarations and Intrastat
- **Data Type**: Wholesale import prices with VAT
- **Frequency**: Monthly (based on previous month declarations)
- **Coverage**: All imported food products
- **What it represents**: Cost of imported food entering Bulgaria
- **Status**: **NOT PROCESSED** → data exists in JSON but not stored in database

### Data Structure

The raw JSON data contains:
- **dates**: Array of weekly dates
- **unit**: Measurement unit (e.g., "брой", "килограм", "литър")
- **filters**: Object containing all filter types with location-specific price arrays
- **filterTexts**: Descriptive text explaining each filter's data source
- **filterContainers**: UI configuration for displaying filters

**Note**: While all 7 filter types exist in the raw JSON data, the `normalize.py` module only processes `filterByLocation` and `filterByLocationBulk`. The other 5 filter types contain valuable data but are not currently utilized by the system.

## Variable Calculations and Metrics

### Core Price Metrics

#### 1. **Price Returns (Weekly)**
```python
def _pct_change(s): 
    return s.pct_change(fill_method=None)
```
- **Calculation**: Week-over-week percentage change in prices
- **Purpose**: Identifies short-term price movements and volatility
- **Status**: ✅ **IMPLEMENTED CORRECTLY**

#### 2. **Year-over-Year (YoY) Changes**
```python
prev_year_idx = (g["date"] - (end - pd.DateOffset(years=1))).abs().values.argmin()
yoy = last / g["price"].iloc[prev_year_idx] - 1
```
- **Calculation**: Current price vs. price from approximately 1 year ago
- **Purpose**: Long-term inflation and trend analysis
- **Status**: ✅ **IMPLEMENTED CORRECTLY**

#### 3. **Price Trends (Slopes)**
```python
def slope(h):
    if len(h) < 2: return 0.0
    x = (h["date"] - h["date"].min()).dt.days.values
    return np.polyfit(x, h["price"].values, 1)[0]
```
- **Calculation**: Linear regression slope over specified time periods (8 weeks, 16 weeks)
- **Purpose**: Identifies directional price trends and momentum
- **Status**: ⚠️ **IMPLEMENTED WITH RISKS** - Returns 0.0 for insufficient data, could fail with NaN prices

### Market Signal Metrics

#### 1. **Product Risk Score**
```python
out["score"] = (
    2*out["HEAT_YOY"].astype(int) +           # High YoY inflation
    2*out["HEAT_MOM_PERSIST"].astype(int) +   # Persistent monthly growth
    1*out["VOL_REGIME"].astype(int) +          # Unusual volatility
    2*out["MARGIN_STRESS"].astype(int) +      # Margin pressure
    1*out["ELEVATED_LEVEL"].astype(int) +     # Above normal levels
    1*out["REVERSAL_UP"].astype(int)          # Trend reversal
)
```
- **Range**: 0-9 (higher = higher risk)
- **Purpose**: Identifies products requiring immediate attention
- **Status**: ⚠️ **IMPLEMENTED WITH ARBITRARY THRESHOLDS** - All thresholds are hardcoded constants without scientific justification

#### 2. **City Risk Score**
```python
out["score"] = (
    2*out["PREMIUM_RISING"].astype(int) +      # High and rising premiums
    2*out["BREADTH_UP"].astype(int) +          # Many products rising
    2*out["CITY_MARGIN_STRESS"].astype(int) +  # Margin stress
    1*out["AFFORDABILITY_ALERT"].astype(int)   # Affordability concerns
)
```
- **Range**: 0-7 (higher = higher risk)
- **Purpose**: Identifies cities with unusual price patterns
- **Status**: ⚠️ **IMPLEMENTED WITH LIMITATIONS** - Only considers retail prices, ignores wholesale data for cities

### Statistical Measures

#### 1. **Median Absolute Deviation (MAD)**
```python
def _mad(x):
    x = np.asarray(x); x = x[~np.isnan(x)]
    if x.size == 0: return 0.0
    med = np.median(x)
    return np.median(np.abs(x - med)) or 1e-9
```
- **Purpose**: Robust measure of price volatility
- **Used for**: Anomaly detection and volatility thresholds
- **Status**: ✅ **IMPLEMENTED CORRECTLY** - Robust handling of edge cases

#### 2. **Price Spreads**
```python
s["spread_abs"] = s["p90"] - s["p10"]
s["spread_rel"] = s["spread_abs"] / s["median"]
```
- **Absolute Spread**: Difference between 90th and 10th percentile prices
- **Relative Spread**: Spread as percentage of median price
- **Purpose**: Measures price dispersion across cities
- **Status**: ✅ **IMPLEMENTED CORRECTLY** - Clear percentile-based approach

#### 3. **Basket Index**
```python
x["base_price"] = x.groupby(["product_id","city","market_type"])["price"].transform(
    lambda s: s.ffill().bfill().iloc[0]
)
x["rel"] = 100.0 * x["price"] / x["base_price"]
```
- **Base**: First valid price per product-city-market combination
- **Index**: Current price as percentage of base price
- **Purpose**: Normalized price trends across different products and locations
- **Status**: ⚠️ **IMPLEMENTED WITH MISLEADING DESCRIPTION** - Documentation claims "equal-weight basket" but code shows equal weight per product, not equal weight across the market

### Margin Analysis

#### 1. **Store Profit Margins**
```python
piv["margin"] = piv.get("retail", pd.Series([np.nan] * len(piv))) - piv.get("wholesale", pd.Series([np.nan] * len(piv)))
```
- **Calculation**: Retail price - Wholesale price
- **Purpose**: Identifies profit margins and margin stress
- **Status**: ⚠️ **IMPLEMENTED WITH DATA DEPENDENCY** - Only works when both retail AND wholesale data exist for same product-city-date combination

#### 2. **Margin Stress Detection**
```python
z = (last - mu)/sd if sd and sd==sd else np.nan
return pd.Series({"MARGIN_STRESS": bool(z is not np.nan and z > MARGIN_Z)})
```
- **Threshold**: Z-score > 2.0 (MARGIN_Z = 2.0)
- **Purpose**: Identifies unusual margin compression
- **Status**: ⚠️ **IMPLEMENTED WITH ARBITRARY THRESHOLD** - Z-score threshold of 2.0 is hardcoded without statistical justification

## Technical Specifications

### Database Schema

#### **price_series** Table
- **product_id** (INTEGER): Product identifier
- **date** (TEXT): Date in YYYY-MM-DD format
- **city** (TEXT): City name
- **market_type** (TEXT): "retail" or "wholesale"
- **price** (REAL): Price in Bulgarian Lev (BGN)
- **unit** (TEXT): Measurement unit

#### **products** Table
- **product_id** (INTEGER): Product identifier
- **name** (TEXT): Product name in Bulgarian
- **unit** (TEXT): Measurement unit

#### **cities** Table
- **city** (TEXT): City name

#### **raw_cache** Table
- **product_id** (INTEGER): Product identifier
- **fetched_at** (TEXT): Timestamp of data fetch
- **payload** (TEXT): Raw JSON response

### Data Processing Pipeline

#### 1. **Data Harvesting** (`src/harvest_all.py`)
- Fetches data for all 36 products
- Implements rate limiting (1 request per second)
- Handles retries with exponential backoff
- Caches raw JSON responses

#### 2. **Data Normalization** (`src/normalize.py`)
- Converts JSON structure to tabular format
- Maps filter types to market types:
  - `filterByLocation` → "retail"
  - `filterByLocationBulk` → "wholesale"
- Extracts city names from location strings
- Handles missing/invalid data

#### 3. **Data Storage** (`src/ingest.py`)
- Upserts data into SQLite database
- Maintains data integrity and relationships
- Stores raw JSON for audit purposes

#### 4. **Analytics Engine** (`src/analytics/`)
- **signals.py**: Market signal detection
- **metrics.py**: Statistical calculations
- **app.py**: Streamlit web interface

### Performance Characteristics

- **Data Volume**: 103,858 price records
- **Coverage**: 36 products × 29 cities × 2 market types × ~52 weeks
- **Update Frequency**: Weekly data collection
- **Processing Time**: Real-time analytics on demand
- **Storage**: SQLite database with optimized queries

## Production Specifications

### System Requirements

- **Python**: 3.8+ with virtual environment
- **Dependencies**: pandas, numpy, streamlit, plotly, requests
- **Database**: SQLite (file-based, no server required)
- **Storage**: ~100MB for database + raw JSON cache
- **Network**: Internet access for data harvesting

### Deployment

- **Local Development**: `python -m src.harvest_all`
- **Web Interface**: `streamlit run src/app.py`
- **Batch Processing**: `python -m src.update_incremental`
- **Export Formats**: CSV, Parquet, SQLite

### Data Quality Controls

- **Validation**: JSON structure validation before processing
- **Error Handling**: Graceful failure with logging
- **Data Integrity**: Foreign key constraints and data type validation
- **Audit Trail**: Raw JSON cache for troubleshooting

### Monitoring and Alerting

- **Market Signals**: Automated detection of unusual price movements
- **Risk Scoring**: 0-9 scale for products, 0-7 scale for cities
- **Anomaly Detection**: Statistical outlier identification
- **Trend Analysis**: 8-week and 16-week momentum tracking

### Current Threshold Values (Hardcoded Constants)

**⚠️ WARNING: These thresholds are currently hardcoded without statistical justification**

#### **Product Risk Thresholds**
```python
YOY_HEAT = 0.08        # 8% year-over-year change triggers "HEAT_YOY"
MAD_MULT = 1.5         # 1.5x MAD triggers "HEAT_MOM_PERSIST"
VOL_MULT = 1.75        # 1.75x volatility triggers "VOL_REGIME"
MARGIN_Z = 2.0         # Z-score > 2.0 triggers "MARGIN_STRESS"
```

#### **City Risk Thresholds**
```python
PREMIUM = 0.05         # 5% price premium triggers "PREMIUM_RISING"
AFF_PREM = 0.07        # 7% premium triggers "AFFORDABILITY_ALERT"
AFF_LEVEL = 110.0      # 110% basket index triggers affordability concerns
BREADTH = 0.60         # 60% of products rising triggers "BREADTH_UP"
```

#### **Anomaly Detection**
```python
thresh = 3.5           # MAD-based threshold for price anomaly detection
```

**Recommendation**: These values should be calibrated based on historical data analysis and statistical significance testing rather than remaining as arbitrary constants.

## Filter Differences Summary

| Filter Type | Market Type | Data Source | Purpose | Frequency | Status |
|-------------|-------------|-------------|---------|-----------|---------|
| `filterByLocationBulk` | Wholesale | ДКСБТ | Bulk pricing for businesses | Weekly | ✅ **ACTIVE** |
| `filterByLocation` | Retail | Market monitoring | Consumer prices | Weekly | ✅ **ACTIVE** |
| `filterBySupplyMilk` | Farm gate | Agricultural producers | Supply chain pricing | Weekly | ❌ **INACTIVE** |
| `filterByCountry` | International | Economic advisors | Import/export benchmarks | Weekly | ❌ **INACTIVE** |
| `filterCOOP` | Cooperative | Cooperative stores | Alternative retail model | Weekly | ❌ **INACTIVE** |
| `filterLIDL` | Retail chain | LIDL stores | Major retailer pricing | Weekly | ❌ **INACTIVE** |
| `filterImport` | Import | NAP/Intrastat | Import cost tracking | Monthly | ❌ **INACTIVE** |

**Legend:**
- ✅ **ACTIVE**: Data is processed, stored in database, and used in analytics
- ❌ **INACTIVE**: Data exists in raw JSON but is not processed or stored

### Data Utilization Gap

**Current State**: The system only utilizes 28.6% (2/7) of available data sources, representing a significant opportunity for enhanced market intelligence.

**What's Missing from Analytics:**
- **Farm gate prices** (`filterBySupplyMilk`) - Could provide early warning of supply chain disruptions
- **International benchmarks** (`filterByCountry`) - Could help identify import/export price arbitrage opportunities
- **Retail chain specific data** (`filterLIDL`, `filterCOOP`) - Could reveal pricing strategies and market positioning
- **Import cost tracking** (`filterImport`) - Could help predict domestic price movements based on import costs

**Potential Impact of Full Utilization:**
- **Enhanced margin analysis** with farm-to-consumer price tracking
- **International price arbitrage** detection
- **Retail chain competitive analysis**
- **Import-driven inflation** early warning system
- **Supply chain stress** identification at multiple levels

**Implementation Effort**: Adding the missing 5 filter types would require:
1. Modifying `src/normalize.py` to process additional filter types
2. Extending database schema to handle new market types
3. Updating analytics to incorporate new data sources
4. Testing data quality and consistency across all filter types

## Current State vs. Desired State Analysis

### 🟢 **What's Working Well (Keep As-Is)**

#### **Data Infrastructure**
- ✅ **Robust data harvesting** with caching and error handling
- ✅ **Clean database schema** with proper relationships
- ✅ **Efficient data processing** pipeline
- ✅ **Comprehensive data coverage** for retail and wholesale markets

#### **Core Analytics**
- ✅ **Price return calculations** - mathematically sound
- ✅ **MAD-based volatility measures** - robust statistical approach
- ✅ **Price spread analysis** - clear percentile-based methodology
- ✅ **Data export capabilities** - multiple formats supported

### 🟡 **What Needs Improvement (Incremental Changes)**

#### **Risk Scoring System**
- **Current**: Hardcoded thresholds without scientific backing
- **Desired**: Configurable thresholds with statistical justification
- **Impact**: Low - can be improved without breaking existing functionality
- **Effort**: Medium - requires threshold research and configuration system

#### **Basket Index Calculation**
- **Current**: Equal weight per product, not per market
- **Desired**: True equal-weight market basket or configurable weighting
- **Impact**: Medium - affects market trend interpretation
- **Effort**: Low - modify weighting logic in `basket_index()` function

#### **City Analysis Scope**
- **Current**: Only retail prices considered
- **Desired**: Include wholesale data for comprehensive city analysis
- **Impact**: Medium - missing wholesale city insights
- **Effort**: Low - extend `city_watchlist()` to include wholesale

#### **Margin Analysis Robustness**
- **Current**: Only works with complete retail/wholesale data
- **Desired**: Handle missing data gracefully with interpolation
- **Impact**: Medium - many margin calculations return NaN
- **Effort**: Medium - add data imputation logic

### 🔴 **What Needs Major Rework (Strategic Decisions)**

#### **Threshold Calibration**
- **Current**: Arbitrary constants (8% YoY, 5% premium, etc.)
- **Desired**: Statistically calibrated thresholds based on historical data
- **Impact**: High - affects all risk assessments
- **Effort**: High - requires statistical analysis and backtesting

#### **Error Handling**
- **Current**: Minimal error handling in critical functions
- **Desired**: Comprehensive error handling with graceful degradation
- **Impact**: High - system could fail with edge cases
- **Effort**: Medium - add try-catch blocks and validation

#### **Data Quality Validation**
- **Current**: Basic validation in normalization
- **Desired**: Comprehensive data quality checks and alerts
- **Impact**: High - prevents bad data from affecting analytics
- **Effort**: Medium - implement validation framework

### 📊 **Implementation Priority Matrix**

| Component | Current Quality | Desired Quality | Impact | Effort | Priority |
|-----------|----------------|-----------------|---------|---------|----------|
| Thresholds | 🔴 Poor | 🟢 Excellent | High | High | **Phase 3** |
| Error Handling | 🟡 Fair | 🟢 Excellent | High | Medium | **Phase 2** |
| Basket Index | 🟡 Fair | 🟢 Excellent | Medium | Low | **Phase 1** |
| City Analysis | 🟡 Fair | 🟢 Excellent | Medium | Low | **Phase 1** |
| Margin Analysis | 🟡 Fair | 🟢 Excellent | Medium | Medium | **Phase 2** |
| Data Validation | 🟡 Fair | 🟢 Excellent | High | Medium | **Phase 2** |

### 🚀 **Recommended Implementation Phases**

#### **Phase 1: Quick Wins (Low Effort, Medium Impact)**
1. Fix basket index weighting
2. Extend city analysis to include wholesale
3. Update documentation to reflect actual limitations

#### **Phase 2: Quality Improvements (Medium Effort, High Impact)**
1. Add comprehensive error handling
2. Implement data quality validation
3. Improve margin analysis robustness

#### **Phase 3: Strategic Improvements (High Effort, High Impact)**
1. Calibrate risk thresholds statistically
2. Implement configurable parameters
3. Add backtesting and validation framework

## Key Insights

1. **Data Completeness**: The system provides comprehensive coverage of Bulgaria's food supply chain from farm to consumer
2. **Market Segmentation**: Clear distinction between retail, wholesale, and specialized market types
3. **Geographic Coverage**: 29 cities representing major population centers and agricultural regions
4. **Temporal Resolution**: Weekly data collection enables trend analysis and early warning systems
5. **Multi-source Integration**: Combines official statistics, market monitoring, and retail chain data
6. **Real-time Analytics**: Automated signal detection and risk scoring for immediate action
7. **Improvement Potential**: Significant opportunities for enhancement without disrupting core functionality

This documentation now accurately reflects both the current reality and the desired future state, providing a clear roadmap for incremental improvements while maintaining system stability.

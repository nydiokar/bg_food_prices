# 🍞 Bulgarian Food Price Intelligence Suite

A comprehensive food price monitoring and analysis system for Bulgaria, providing market signals, trend analysis, and price intelligence across retail and wholesale markets.

![Market Overview](assets/Screenshot%202025-08-31%20173500.png)
*Market Overview with national food price trends and data quality indicators*

## 🚀 Quick Start (New Machine Setup)

### Prerequisites
- Python 3.8+ 
- Git
- Windows PowerShell (or Command Prompt)

### 1. Clone & Setup Environment

#### Option A: Automated Setup (Recommended)
```powershell
# Clone the repository
git clone https://github.com/nydiokar/bg_food_prices.git
cd products-bg

# Run automated setup script
# For Windows Command Prompt:
setup.bat

# For PowerShell:
.\setup.ps1
```

#### Option B: Manual Setup
```powershell
# Clone the repository
git clone <your-repo-url>
cd products-bg

# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Setup Options

#### Option A: Use Pre-committed Database (Recommended for Development)
```powershell
# The SQLite database is already included in the repo
# No additional setup needed - you can start the app immediately
streamlit run src/app.py
```

#### Option B: Fetch Fresh Data (First Time Setup)
```powershell
# Harvest all historical data (one call per product, full history per payload)
python -m src.harvest_all

# For incremental updates later:
python -m src.update_incremental
```

### 3. Launch the Application
```powershell
# Start the Streamlit web app
streamlit run src/app.py

# Or run analytics directly
python -m src.analytics.run_all
```

## 📊 What This System Does

### Data Collection
- **Source**: Bulgarian Ministry of Agriculture food price monitoring system
- **Frequency**: Weekly updates
- **Products**: 46+ food items (bread, milk, meat, vegetables, etc.)
- **Markets**: Retail and wholesale prices across Bulgarian cities
- **Filters Available**: Location, bulk location, supply chain, market type

### Analytics & Intelligence
- **Market Signals**: Product and city watchlists for price stress
- **Basket Index**: National and city-level food price trends (base=100)
- **Margin Analysis**: Retail vs. wholesale profit margins
- **Anomaly Detection**: Unusual weekly price movements
- **Correlation Analysis**: City and product price relationships
- **Data Quality Assessment**: Coverage, completeness, and reliability metrics

## 📈 Understanding the Basket Index System

### What is the Basket Index?
The Basket Index is a **composite measure** that tracks how food prices change over time across Bulgaria. Think of it as a "food price thermometer" that shows whether prices are going up or down nationally.

### Key Terms Explained:

#### 🎯 **Baseline (100)**
- **What it is**: The starting point for all price comparisons
- **When it was set**: At the beginning of our data collection period
- **What it means**: When the index = 100, food prices are at their "normal" level
- **Why 100**: It's easier to understand "108.4" means "8.4% higher than normal" than dealing with absolute price values

#### 📊 **Current Index**
- **What it is**: The current value of the food price basket
- **Example**: 108.4 means food prices are currently 8.4% higher than the baseline
- **How it's calculated**: Weighted average of all product prices, normalized to the baseline period
- **What it tells us**: Whether food is getting more expensive or cheaper right now

#### 📈 **vs Baseline**
- **What it is**: The percentage change from baseline to current
- **Example**: +8.4% means prices have increased by 8.4% since the baseline period
- **Positive values**: Food is getting more expensive
- **Negative values**: Food is getting cheaper
- **Zero**: Prices are exactly at baseline levels

### Real-World Example:
- **Baseline (100)**: In January 2024, a typical food basket cost 100 BGN
- **Current Index (108.4)**: Today, the same basket costs 108.4 BGN
- **vs Baseline (+8.4%)**: Food prices have increased by 8.4% since January 2024

### Why This Matters:
- **Inflation tracking**: See if food prices are rising faster than general inflation
- **Regional comparison**: Compare how different cities perform against the national baseline
- **Trend analysis**: Identify if price increases are accelerating or stabilizing
- **Policy insights**: Help understand the impact of economic policies on food affordability

## 🖼️ App Screenshots

### Market Signals & Hot Products
![Market Signals](assets/Screenshot%202025-08-31%20173515.png)
*Product watchlist showing risk scores, trends, and human-readable risk explanations*

### City Alerts & Price Patterns
![City Alerts](assets/Screenshot%202025-08-31%20173609.png)
*City alerts with price premiums, unusual products percentage, and risk breakdowns*

### Price Anomalies Detection
![Price Anomalies](assets/Screenshot%202025-08-31%20173741.png)
*Weekly price anomaly detection with severity indicators and percentage changes*

## 🏗️ Project Structure

```
products-bg/
├── src/
│   ├── app.py                 # Streamlit web application
│   ├── harvest_all.py         # Initial data harvesting
│   ├── update_incremental.py  # Incremental updates
│   ├── fetch.py               # HTTP request handling
│   ├── normalize.py           # Data transformation
│   ├── ingest.py              # Database insertion
│   └── analytics/
│       ├── metrics.py         # Statistical calculations
│       ├── signals.py         # Market signal generation
│       └── run_all.py         # Analytics orchestration
├── data/
│   ├── foodprice.sqlite       # Main database (committed to repo)
│   ├── raw/                   # Raw JSON responses
│   └── exports/               # Processed CSV/Parquet files
├── assets/                    # App screenshots and visuals
├── requirements.txt            # Python dependencies
└── README.md                  # This file
```

## 🔄 Development Workflow

### Daily Development
```powershell
# Activate environment
.venv\Scripts\activate

# Make code changes
# Test the app
streamlit run src/app.py

# Run analytics tests
python test_phase2_completion.py
```

### Data Updates
```powershell
# When you need fresh data (optional)
python -m src.update_incremental

# Or full refresh (rarely needed)
python -m src.harvest_all
```

### Committing Changes
```powershell
# The SQLite database is included in commits
# This means new team members get the data immediately
git add .
git commit -m "Update analytics and data"
git push
```

## 📈 Key Features

### Market Overview
- National food price trend visualization
- Data quality indicators
- Coverage metrics (products, cities, market types)

### Market Signals
- **Hot Products**: Products with unusual price activity
- **City Alerts**: Cities with price stress patterns
- Risk scoring and human-readable explanations

### Deep Dive Analysis
- **Product Analysis**: Price trends, margins, percentiles
- **City Analysis**: Basket indices, product mix
- **Anomalies**: Weekly price change detection

## 🛠️ Technical Details

### Database Schema
- **facts**: Price records with product, city, date, market type
- **dims**: Product and city metadata
- **Indexes**: Optimized for time-series queries

### Data Processing Pipeline
1. **Fetch**: HTTP requests to ministry API
2. **Normalize**: Transform raw JSON to structured format
3. **Ingest**: Insert into SQLite with validation
4. **Analyze**: Generate metrics, signals, and insights

### Performance Considerations
- **Caching**: Streamlit data caching for app performance
- **Lazy Loading**: Data loaded only when needed
- **Optimized Queries**: Efficient SQL for large datasets

## 🚨 Troubleshooting

### Common Issues

#### App Won't Start
```powershell
# Check if database exists
ls data/foodprice.sqlite

# If missing, fetch data first
python -m src.harvest_all
```

#### Import Errors
```powershell
# Ensure virtual environment is activated
.venv\Scripts\activate

# Reinstall dependencies
pip install -r requirements.txt
```

#### Data Quality Issues
- Check the Data Quality Assessment in the app
- Run `python test_phase2_completion.py` to verify analytics
- Review logs in `data/exports/_failures.log`

### Getting Help
- Check the app's Data Quality Assessment section
- Review test logs for specific error messages
- Ensure all dependencies are installed correctly

## 🔮 Future Enhancements

### Phase 3 (Planned)
- Configurable parameters and backtesting
- Advanced correlation analysis
- Export functionality for reports
- API endpoints for external integration

### Data Expansion
- Additional types of filters
- More granular geographic coverage
- Increase coverage of metrics
- Real-time price feeds

## 📝 License & Attribution

Data source: Bulgarian Ministry of Economy and Industry price monitoring system 
Analysis and visualization: Custom analytics suite

---

**Note**: The SQLite database (`data/foodprice.sqlite`) is committed to the repository to provide immediate access to historical data for new team members and development environments. This eliminates the need for constant data fetching during development while maintaining the ability to update data when needed.

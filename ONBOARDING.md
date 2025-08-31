# 🚀 Quick Onboarding Guide

## ⚡ Get Running in 5 Minutes

### 1. Clone & Setup
```bash
git clone <your-repo-url>
cd products-bg
setup.bat  # or .\setup.ps1 for PowerShell
```

### 2. Start the App
```bash
streamlit run src/app.py
```

That's it! 🎉

## 🔍 What You'll See

- **Market Overview**: National food price trends and data quality
- **Market Signals**: Hot products and city alerts
- **Deep Dive**: Product analysis, city analysis, anomalies
- **Data Quality**: Coverage metrics and reliability indicators

## 🛠️ Development Commands

```bash
# Test analytics
python test_phase2_completion.py

# Run all analytics
python -m src.analytics.run_all

# Update data (optional)
python -m src.update_incremental
```

## 📊 Key Metrics Explained

- **Basket Index**: Food price trend (100 = baseline)
- **Unusual Products %**: % of products showing price stress in a city
- **Price Premium**: How much a city's prices differ from national average
- **Risk Score**: 0-5 scale for product risk assessment

## 🆘 Need Help?

- Check the app's Data Quality Assessment section
- Run test scripts to verify functionality
- Review `COMPREHENSIVE_DOCUMENTATION.md` for detailed explanations

## 💡 Pro Tips

- The SQL database is committed to the repo - no need to fetch data initially
- Use the search functionality in the app to find specific products
- Check the tooltips (hover over column headers) for explanations
- The app automatically caches data for performance

---


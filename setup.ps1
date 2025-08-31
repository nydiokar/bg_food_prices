# 🍞 Bulgarian Food Price Intelligence Suite Setup Script
Write-Host "🍞 Setting up Bulgarian Food Price Intelligence Suite..." -ForegroundColor Green
Write-Host ""

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python found" -ForegroundColor Green
    Write-Host $pythonVersion -ForegroundColor Cyan
} catch {
    Write-Host "❌ Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8+ from https://python.org" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if virtual environment exists
if (Test-Path ".venv") {
    Write-Host "✅ Virtual environment already exists" -ForegroundColor Green
} else {
    Write-Host "🔧 Creating virtual environment..." -ForegroundColor Yellow
    try {
        python -m venv .venv
        Write-Host "✅ Virtual environment created" -ForegroundColor Green
    } catch {
        Write-Host "❌ Failed to create virtual environment" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}

# Activate virtual environment
Write-Host "🔧 Activating virtual environment..." -ForegroundColor Yellow
& .venv\Scripts\Activate.ps1

# Install dependencies
Write-Host "🔧 Installing dependencies..." -ForegroundColor Yellow
try {
    pip install -r requirements.txt
    Write-Host "✅ Dependencies installed" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to install dependencies" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if database exists
if (Test-Path "data\foodprice.sqlite") {
    Write-Host "✅ Database found - ready to run!" -ForegroundColor Green
    Write-Host ""
    Write-Host "🚀 You can now start the app with:" -ForegroundColor Cyan
    Write-Host "   streamlit run src/app.py" -ForegroundColor White
    Write-Host ""
    Write-Host "Or run analytics with:" -ForegroundColor Cyan
    Write-Host "   python -m src.analytics.run_all" -ForegroundColor White
} else {
    Write-Host "⚠️  Database not found" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "🔧 Fetching initial data (this may take a while)..." -ForegroundColor Yellow
    try {
        python -m src.harvest_all
        Write-Host "✅ Data fetched successfully" -ForegroundColor Green
        Write-Host ""
        Write-Host "🚀 You can now start the app with:" -ForegroundColor Cyan
        Write-Host "   streamlit run src/app.py" -ForegroundColor White
    } catch {
        Write-Host "❌ Failed to fetch data" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}

Write-Host ""
Write-Host "🎉 Setup complete! Happy analyzing!" -ForegroundColor Green
Read-Host "Press Enter to exit"

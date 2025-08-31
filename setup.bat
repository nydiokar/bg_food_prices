@echo off
echo 🍞 Setting up Bulgarian Food Price Intelligence Suite...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo ✅ Python found
python --version

REM Check if virtual environment exists
if exist ".venv" (
    echo ✅ Virtual environment already exists
) else (
    echo 🔧 Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo ❌ Failed to create virtual environment
        pause
        exit /b 1
    )
    echo ✅ Virtual environment created
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call .venv\Scripts\activate.bat

REM Install dependencies
echo 🔧 Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)
echo ✅ Dependencies installed

REM Check if database exists
if exist "data\foodprice.sqlite" (
    echo ✅ Database found - ready to run!
    echo.
    echo 🚀 You can now start the app with:
    echo    streamlit run src/app.py
    echo.
    echo Or run analytics with:
    echo    python -m src.analytics.run_all
) else (
    echo ⚠️  Database not found
    echo.
    echo 🔧 Fetching initial data (this may take a while)...
    python -m src.harvest_all
    if errorlevel 1 (
        echo ❌ Failed to fetch data
        pause
        exit /b 1
    )
    echo ✅ Data fetched successfully
    echo.
    echo 🚀 You can now start the app with:
    echo    streamlit run src/app.py
)

echo.
echo 🎉 Setup complete! Happy analyzing!
pause

@echo off
REM Second-Generation Marriage Patterns Dashboard - Quick Setup
REM Run this script from your project directory

echo ======================================================================
echo SECOND-GENERATION MARRIAGE PATTERNS DASHBOARD - SETUP
echo ======================================================================
echo.

REM Check for Python
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found in PATH
    echo Please run with full Python path or add Python to PATH
    pause
    exit /b 1
)

REM Create directory structure
echo Creating directory structure...
if not exist "data\raw" mkdir data\raw
if not exist "data\processed" mkdir data\processed

echo.
echo Directory structure created:
echo   data\raw\        - Put your IPUMS extract (usa_00003.csv) here
echo   data\processed\  - Processed files will be saved here
echo.

REM Install dependencies
echo Installing Python dependencies...
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo.
    echo WARNING: Some packages may have failed to install
    echo Try running: pip install pandas numpy dash dash-bootstrap-components plotly
)

echo.
echo ======================================================================
echo SETUP COMPLETE
echo ======================================================================
echo.
echo NEXT STEPS:
echo   1. Place your IPUMS extract (usa_00003.csv) in data\raw\
echo   2. Run: python process_second_gen.py
echo   3. Run: python run_second_gen_dashboard.py
echo   4. Open: http://127.0.0.1:8050
echo.
pause

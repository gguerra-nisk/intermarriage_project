@echo off
echo ============================================
echo Setting up Intermarriage Dashboard Project
echo ============================================
echo.

:: Create directory structure
echo Creating folders...
mkdir data 2>nul
mkdir data\raw 2>nul
mkdir data\processed 2>nul
mkdir scripts 2>nul
mkdir dashboard 2>nul

echo.
echo Folder structure created:
echo.
echo   intermarriage-dashboard\
echo   ├── data\
echo   │   ├── raw\          (put IPUMS download here)
echo   │   └── processed\    (cleaned data goes here)
echo   ├── scripts\          (Python scripts)
echo   └── dashboard\        (output files)
echo.

:: Check if virtual environment exists
if exist venv (
    echo Virtual environment already exists.
) else (
    echo Creating virtual environment...
    python -m venv venv
)

:: Activate and install packages
echo.
echo Activating virtual environment and installing packages...
call venv\Scripts\activate.bat

pip install pandas numpy ipumspy pyarrow requests dash plotly dash-bootstrap-components --quiet

echo.
echo ============================================
echo Setup complete!
echo ============================================
echo.
echo NEXT STEPS:
echo 1. Download your IPUMS extract (you'll get an email)
echo 2. Put BOTH files (.csv.gz and .xml) in the data\raw folder
echo 3. Run: python scripts\process_ipums.py
echo 4. Run: python scripts\run_dashboard.py
echo.
pause

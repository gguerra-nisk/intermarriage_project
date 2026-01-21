# 💕 Immigrant Intermarriage Dashboard

An interactive visualization of marriage patterns among immigrants in the United States, spanning 1850-2023.

## Quick Start

### 1. Setup (one time)
Double-click `setup_project.bat` or run these commands:

```cmd
python -m venv venv
venv\Scripts\activate
pip install pandas numpy ipumspy pyarrow requests dash plotly dash-bootstrap-components
```

### 2. Add Your IPUMS Data
- Download your IPUMS extract when the email arrives
- Put **both files** (`.csv.gz` and `.xml`) in the `data\raw\` folder

### 3. Process the Data
```cmd
venv\Scripts\activate
python scripts\process_ipums.py
```

### 4. Launch the Dashboard
```cmd
python scripts\run_dashboard.py
```

Then open http://127.0.0.1:8050 in your browser!

---

## Project Structure

```
intermarriage-dashboard/
├── data/
│   ├── raw/           ← Put IPUMS downloads here
│   └── processed/     ← Cleaned data (auto-generated)
├── scripts/
│   ├── process_ipums.py    ← Data cleaning script
│   └── run_dashboard.py    ← Dashboard application
├── setup_project.bat  ← Windows setup script
└── README.md
```

## Data Source

This project uses data from [IPUMS USA](https://usa.ipums.org), University of Minnesota.

**Citation:**
> Steven Ruggles, Sarah Flood, Matthew Sobek, et al. IPUMS USA: Version 16.0 [dataset]. Minneapolis, MN: IPUMS, 2025. https://doi.org/10.18128/D010.V16.0

## Marriage Categories

- **Married US-Born**: Immigrant married to someone born in the United States
- **Same Country**: Both spouses from the same foreign country
- **Same Region**: Both foreign-born from the same world region (e.g., both from Europe, but different countries)
- **Different Region**: Both foreign-born from different world regions

---

Made with 💕 for Valentine's Day 2025

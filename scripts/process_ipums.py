"""
Immigrant Intermarriage Data Processing Script (Memory-Efficient Version)
==========================================================================
Processes IPUMS census data in chunks to handle large files.

Usage: python scripts/process_ipums.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("IMMIGRANT INTERMARRIAGE DATA PROCESSOR")
print("="*70)

# =============================================================================
# CONFIGURATION
# =============================================================================
RAW_DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Chunk size - adjust if still getting memory errors
CHUNK_SIZE = 500000  # Process 500k rows at a time

# =============================================================================
# STEP 1: LOCATE DATA FILE
# =============================================================================
print("\n[Step 1/7] Looking for IPUMS data file...")

data_files = list(RAW_DATA_DIR.glob("*.csv.gz")) + list(RAW_DATA_DIR.glob("*.csv"))

if not data_files:
    print("\n❌ ERROR: No data file found!")
    print(f"   Please put your IPUMS .csv.gz file in: {RAW_DATA_DIR.absolute()}")
    sys.exit(1)

DATA_FILE = data_files[0]
print(f"   ✓ Found: {DATA_FILE.name}")

# =============================================================================
# STEP 2: IDENTIFY COLUMNS
# =============================================================================
print("\n[Step 2/7] Identifying columns...")

sample_df = pd.read_csv(DATA_FILE, nrows=5)
all_columns = sample_df.columns.tolist()
print(f"   ✓ Found {len(all_columns)} columns in dataset")

# Define columns we want to use
desired_cols = [
    'YEAR', 'SERIAL', 'PERNUM', 'PERWT',
    'AGE', 'SEX', 'MARST',
    'SPLOC', 'SPRULE',
    'BPL', 'BPLD', 'NATIVITY', 'CITIZEN', 'YRIMMIG',
    'FBPL', 'MBPL',
    'RACE', 'HISPAN',
    'BPL_SP', 'BPLD_SP', 'NATIVITY_SP', 'RACE_SP', 'HISPAN_SP'
]

use_cols = [c for c in desired_cols if c in all_columns]
print(f"   Using {len(use_cols)} columns: {use_cols[:10]}...")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def map_to_region(bpl):
    """Map IPUMS BPL codes to readable region names."""
    if pd.isna(bpl):
        return "Unknown"
    bpl = int(bpl)
    
    if 1 <= bpl <= 56 or bpl == 99:
        return "United States"
    if 100 <= bpl <= 120:
        return "US Territory"
    if bpl == 150:
        return "Canada"
    if bpl == 200:
        return "Mexico"
    if 160 <= bpl <= 199 or 210 <= bpl <= 299:
        return "Central America/Caribbean"
    if 300 <= bpl <= 399:
        return "South America"
    if bpl in [400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410]:
        return "United Kingdom"
    if bpl == 414:
        return "Ireland"
    if bpl in [420, 421, 429]:
        return "Germany"
    if bpl in [430, 431, 432, 433, 434]:
        return "Scandinavia"
    if bpl == 460:
        return "Italy"
    if bpl in [450, 451, 452, 453, 455, 457]:
        return "Central Europe"
    if 500 <= bpl <= 510:
        return "Russia/USSR"
    if 400 <= bpl <= 499:
        return "Europe (Other)"
    if bpl == 500:
        return "China"
    if bpl == 501:
        return "Japan"
    if bpl == 502:
        return "Korea"
    if bpl == 515:
        return "Philippines"
    if bpl == 517:
        return "Vietnam"
    if bpl == 521:
        return "India"
    if 500 <= bpl <= 599:
        return "Asia (Other)"
    if 600 <= bpl <= 699:
        return "Africa"
    if 700 <= bpl <= 799:
        return "Oceania"
    return "Other/Unknown"


def map_to_country(bpl):
    """Map BPL codes to specific country names."""
    if pd.isna(bpl):
        return "Unknown"
    bpl = int(bpl)
    
    country_map = {
        150: "Canada", 200: "Mexico", 210: "Cuba", 250: "Puerto Rico",
        400: "England", 401: "Scotland", 402: "Wales", 414: "Ireland",
        420: "Germany", 426: "Switzerland",
        430: "Sweden", 431: "Norway", 432: "Denmark", 433: "Finland",
        436: "Netherlands", 437: "Belgium", 440: "France",
        450: "Austria", 451: "Hungary", 452: "Czechoslovakia", 453: "Poland",
        455: "Romania", 457: "Yugoslavia",
        460: "Italy", 465: "Spain", 467: "Portugal", 468: "Greece",
        480: "USSR/Russia",
        500: "China", 501: "Japan", 502: "Korea", 515: "Philippines",
        517: "Vietnam", 521: "India",
    }
    
    if 1 <= bpl <= 56 or bpl == 99:
        return "United States"
    
    return country_map.get(bpl, f"Code {bpl}")


def classify_marriage(row):
    """Classify marriage type based on birthplaces."""
    bpl = row['BPL']
    bpl_sp = row['BPL_SP']
    
    if pd.isna(bpl) or (1 <= bpl <= 56) or bpl == 99:
        return "Native-born"
    
    if pd.isna(bpl_sp):
        return "Unknown"
    
    if (1 <= bpl_sp <= 56) or bpl_sp == 99:
        return "Married US-born"
    
    if bpl == bpl_sp:
        return "Same Country"
    
    region = row['REGION']
    region_sp = row['REGION_SP']
    
    if region == region_sp:
        return "Same Region"
    else:
        return "Different Region"


def process_chunk(chunk):
    """Process a single chunk of data."""
    # Filter to married with spouse present
    df_married = chunk[(chunk['MARST'] == 1) & (chunk['SPLOC'] > 0)].copy()
    
    if len(df_married) == 0:
        return None
    
    # Map birthplaces
    df_married['REGION'] = df_married['BPL'].apply(map_to_region)
    df_married['REGION_SP'] = df_married['BPL_SP'].apply(map_to_region)
    df_married['COUNTRY'] = df_married['BPL'].apply(map_to_country)
    df_married['COUNTRY_SP'] = df_married['BPL_SP'].apply(map_to_country)
    
    # Classify marriages
    df_married['MARRIAGE_TYPE'] = df_married.apply(classify_marriage, axis=1)
    
    # Filter to immigrants only
    df_immigrants = df_married[~df_married['MARRIAGE_TYPE'].isin(["Native-born", "Unknown"])]
    
    return df_immigrants


# =============================================================================
# STEP 3: PROCESS DATA IN CHUNKS
# =============================================================================
print("\n[Step 3/7] Processing data in chunks (this may take 10-20 minutes)...")

all_chunks = []
total_rows = 0
immigrant_rows = 0
chunk_num = 0

start_time = time.time()

try:
    for chunk in pd.read_csv(DATA_FILE, usecols=use_cols, chunksize=CHUNK_SIZE, low_memory=False):
        chunk_num += 1
        total_rows += len(chunk)
        
        processed = process_chunk(chunk)
        
        if processed is not None and len(processed) > 0:
            immigrant_rows += len(processed)
            all_chunks.append(processed)
        
        elapsed = time.time() - start_time
        print(f"   Chunk {chunk_num}: {total_rows:,} rows processed, {immigrant_rows:,} immigrant marriages found ({elapsed:.0f}s)")

except Exception as e:
    print(f"\n❌ Error processing chunk {chunk_num}: {e}")
    if len(all_chunks) == 0:
        sys.exit(1)
    print("   Continuing with data processed so far...")

print(f"\n   ✓ Finished processing {total_rows:,} total rows")
print(f"   ✓ Found {immigrant_rows:,} immigrant marriage records")

# =============================================================================
# STEP 4: COMBINE CHUNKS
# =============================================================================
print("\n[Step 4/7] Combining processed data...")

if len(all_chunks) == 0:
    print("❌ ERROR: No immigrant marriages found in data!")
    sys.exit(1)

df_immigrants = pd.concat(all_chunks, ignore_index=True)
print(f"   ✓ Combined {len(df_immigrants):,} records")

# Free memory
del all_chunks

# =============================================================================
# STEP 5: COMPUTE STATISTICS
# =============================================================================
print("\n[Step 5/7] Computing statistics...")

def compute_stats(group):
    """Compute weighted intermarriage statistics."""
    total_weight = group['PERWT'].sum()
    n = len(group)
    
    if total_weight == 0:
        return pd.Series({
            'n_unweighted': n,
            'total_weighted': 0,
            'pct_married_us_born': 0,
            'pct_same_country': 0,
            'pct_same_region': 0,
            'pct_different_region': 0
        })
    
    result = {'n_unweighted': n, 'total_weighted': total_weight}
    
    for mtype, col_name in [
        ('Married US-born', 'pct_married_us_born'),
        ('Same Country', 'pct_same_country'),
        ('Same Region', 'pct_same_region'),
        ('Different Region', 'pct_different_region')
    ]:
        mask = group['MARRIAGE_TYPE'] == mtype
        weight = group.loc[mask, 'PERWT'].sum()
        result[col_name] = (weight / total_weight * 100)
    
    return pd.Series(result)


# By Year
print("   Computing by year...")
stats_by_year = df_immigrants.groupby('YEAR').apply(compute_stats).reset_index()

# By Year and Region
print("   Computing by year and region...")
stats_by_year_region = df_immigrants.groupby(['YEAR', 'REGION']).apply(compute_stats).reset_index()

# By Year and Country
print("   Computing by year and country...")
stats_by_year_country = df_immigrants.groupby(['YEAR', 'COUNTRY']).apply(compute_stats).reset_index()
stats_by_year_country = stats_by_year_country[stats_by_year_country['n_unweighted'] >= 50]

# =============================================================================
# STEP 6: SAVE FILES
# =============================================================================
print("\n[Step 6/7] Saving processed data...")

stats_by_year.to_csv(PROCESSED_DIR / "intermarriage_by_year.csv", index=False)
print(f"   ✓ Saved: intermarriage_by_year.csv")

stats_by_year_region.to_csv(PROCESSED_DIR / "intermarriage_by_year_region.csv", index=False)
print(f"   ✓ Saved: intermarriage_by_year_region.csv")

stats_by_year_country.to_csv(PROCESSED_DIR / "intermarriage_by_year_country.csv", index=False)
print(f"   ✓ Saved: intermarriage_by_year_country.csv")

# Save sample for inspection
sample_size = min(5000, len(df_immigrants))
df_sample = df_immigrants.sample(n=sample_size, random_state=42)
df_sample.to_csv(PROCESSED_DIR / "immigrant_marriages_sample.csv", index=False)
print(f"   ✓ Saved: immigrant_marriages_sample.csv ({sample_size:,} records)")

# =============================================================================
# STEP 7: SUMMARY
# =============================================================================
elapsed_total = time.time() - start_time

print("\n" + "="*70)
print("✅ PROCESSING COMPLETE!")
print("="*70)

print(f"""
Summary:
--------
  Total rows processed: {total_rows:,}
  Immigrant marriages: {len(df_immigrants):,}
  Processing time: {elapsed_total/60:.1f} minutes
  
  Census years: {sorted(df_immigrants['YEAR'].unique())}
  Origin regions: {df_immigrants['REGION'].nunique()}

Files saved to: {PROCESSED_DIR.absolute()}
""")

print("\nIntermarriage Rates Preview:")
print("-" * 50)
preview = stats_by_year[['YEAR', 'n_unweighted', 'pct_married_us_born', 'pct_same_country']].copy()
preview.columns = ['Year', 'N', '% US-born', '% Same Country']
preview['% US-born'] = preview['% US-born'].round(1)
preview['% Same Country'] = preview['% Same Country'].round(1)
print(preview.to_string(index=False))

print("\n" + "="*70)
print("NEXT STEP: Run the dashboard!")
print('  "C:\\Users\\Gilbert Guerra\\AppData\\Local\\Python\\pythoncore-3.14-64\\python.exe" scripts\\run_dashboard.py')
print("="*70)

"""
Second-Generation Immigrant Marriage Patterns Data Processor v7
================================================================
- Properly handles mixed-origin spouses
- Cleans country names (removes ", ns" suffixes etc.)

Usage: python process_second_gen.py
Author: Gil Guerra / Niskanen Center
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 300000
# Supports both .csv and .csv.gz files (pandas handles compression automatically)
INPUT_FILE = "usa_00004.csv.gz"

# Only process these census years - post-1930 censuses have sample-line limitations
# that prevent reliable spouse heritage comparison
VALID_YEARS = [1880, 1900, 1910, 1920, 1930]

# =============================================================================
# IPUMS BPL GENERAL CODES - FROM DDI CODEBOOK
# =============================================================================

US_STATE_CODES = set(range(1, 57))
US_STATE_CODES.add(90)   # Native American
US_STATE_CODES.add(99)   # United States, ns

# Clean country names (no ", ns" suffixes)
COUNTRY_CODES = {
    # US TERRITORIES
    100: "American Samoa", 105: "Guam", 110: "Puerto Rico",
    115: "US Virgin Islands", 120: "Other US Possessions",
    
    # NORTH AMERICA
    150: "Canada", 155: "St. Pierre and Miquelon", 
    160: "Atlantic Islands", 199: "North America (unspecified)",
    
    # MEXICO & CENTRAL AMERICA & CARIBBEAN
    200: "Mexico", 210: "Central America",
    250: "Cuba", 260: "West Indies", 299: "Americas (unspecified)",
    
    # SOUTH AMERICA
    300: "South America",
    
    # NORTHERN EUROPE
    400: "Denmark", 401: "Finland", 402: "Iceland", 403: "Lapland",
    404: "Norway", 405: "Sweden", 406: "Svalbard",
    410: "England", 411: "Scotland", 412: "Wales",
    413: "United Kingdom (unspecified)", 414: "Ireland", 
    419: "Northern Europe (unspecified)",
    
    # WESTERN EUROPE
    420: "Belgium", 421: "France", 422: "Liechtenstein", 423: "Luxembourg",
    424: "Monaco", 425: "Netherlands", 426: "Switzerland", 
    429: "Western Europe (unspecified)",
    
    # SOUTHERN EUROPE
    430: "Albania", 431: "Andorra", 432: "Gibraltar", 433: "Greece",
    434: "Italy", 435: "Malta", 436: "Portugal", 437: "San Marino",
    438: "Spain", 439: "Vatican City", 440: "Southern Europe (unspecified)",
    
    # CENTRAL & EASTERN EUROPE
    450: "Austria", 451: "Bulgaria", 452: "Czechoslovakia", 453: "Germany",
    454: "Hungary", 455: "Poland", 456: "Romania", 457: "Yugoslavia",
    458: "Central Europe (unspecified)", 459: "Eastern Europe (unspecified)",
    460: "Estonia", 461: "Latvia", 462: "Lithuania", 
    463: "Baltic States (unspecified)", 465: "Russia/USSR",
    499: "Europe (unspecified)",
    
    # ASIA
    500: "China", 501: "Japan", 502: "Korea", 509: "East Asia (unspecified)",
    510: "Brunei", 511: "Cambodia", 512: "Indonesia", 513: "Laos",
    514: "Malaysia", 515: "Philippines", 516: "Singapore", 517: "Thailand",
    518: "Vietnam", 519: "Southeast Asia (unspecified)",
    520: "Afghanistan", 521: "India", 522: "Iran", 523: "Maldives", 524: "Nepal",
    530: "Bahrain", 531: "Cyprus", 532: "Iraq", 533: "Iraq/Saudi Arabia",
    534: "Israel/Palestine", 535: "Jordan", 536: "Kuwait", 537: "Lebanon",
    538: "Oman", 539: "Qatar", 540: "Saudi Arabia", 541: "Syria", 542: "Turkey",
    543: "UAE", 544: "Yemen (North)", 545: "Yemen (South)",
    546: "Persian Gulf States (unspecified)", 547: "Middle East (unspecified)",
    548: "Southwest Asia (unspecified)", 549: "Asia Minor (unspecified)", 
    550: "South Asia (unspecified)", 599: "Asia (unspecified)",
    
    # AFRICA & OCEANIA
    600: "Africa", 700: "Australia/New Zealand", 710: "Pacific Islands",
    
    # OTHER
    800: "Antarctica", 900: "Abroad/At Sea", 950: "Other",
    997: "Unknown", 998: "Illegible", 999: "Missing",
    0: "N/A",
}

# Countries to treat as "not a real origin"
NON_ORIGIN = {"US-born", "Unknown", "N/A", "Abroad/At Sea", "Missing", "Illegible"}


def get_country(bpl_code):
    """Convert BPL code to country name."""
    if pd.isna(bpl_code):
        return "Unknown"
    bpl_code = int(bpl_code)
    if bpl_code in US_STATE_CODES:
        return "US-born"
    if bpl_code in COUNTRY_CODES:
        return COUNTRY_CODES[bpl_code]
    return f"Unknown (code {bpl_code})"


def is_foreign_origin(country):
    """Check if a country string represents foreign origin."""
    if country in NON_ORIGIN:
        return False
    if country.startswith("Unknown"):
        return False
    return True


def get_spouse_origins(bpl_sp, mbpl_sp, fbpl_sp):
    """Get ALL origins associated with the spouse."""
    spouse_bp = get_country(bpl_sp)
    spouse_mother = get_country(mbpl_sp)
    spouse_father = get_country(fbpl_sp)
    
    # Spouse is foreign-born = 1st generation immigrant
    if is_foreign_origin(spouse_bp):
        return {
            'generation': '1st gen immigrant',
            'origins': {spouse_bp},
            'primary_country': spouse_bp,
            'spouse_mother': spouse_mother,
            'spouse_father': spouse_father,
        }
    
    # Spouse is US-born - check parents
    sp_m_foreign = is_foreign_origin(spouse_mother)
    sp_f_foreign = is_foreign_origin(spouse_father)
    
    if sp_m_foreign or sp_f_foreign:
        origins = set()
        if sp_m_foreign:
            origins.add(spouse_mother)
        if sp_f_foreign:
            origins.add(spouse_father)
        
        if sp_f_foreign:
            primary = spouse_father
        else:
            primary = spouse_mother
            
        return {
            'generation': '2nd gen',
            'origins': origins,
            'primary_country': primary,
            'spouse_mother': spouse_mother,
            'spouse_father': spouse_father,
        }
    
    # 3rd+ generation
    return {
        'generation': '3rd+ gen American',
        'origins': set(),
        'primary_country': 'American',
        'spouse_mother': spouse_mother,
        'spouse_father': spouse_father,
    }


def classify_marriage(mother_origin, father_origin, spouse_info):
    """Classify marriage based on shared heritage."""
    spouse_gen = spouse_info['generation']
    spouse_origins = spouse_info['origins']
    
    if spouse_gen == '3rd+ gen American':
        return {
            'type': 'Married 3rd+ gen American',
            'shares_mother': False,
            'shares_father': False,
            'shared_countries': set(),
        }
    
    person_origins = set()
    if is_foreign_origin(mother_origin):
        person_origins.add(mother_origin)
    if is_foreign_origin(father_origin):
        person_origins.add(father_origin)
    
    shared = person_origins & spouse_origins
    shares_mother = is_foreign_origin(mother_origin) and mother_origin in spouse_origins
    shares_father = is_foreign_origin(father_origin) and father_origin in spouse_origins
    
    if shares_mother and shares_father:
        if mother_origin == father_origin:
            mtype = f'Married same origin ({spouse_gen})'
        else:
            mtype = f'Married someone sharing both heritages ({spouse_gen})'
    elif shares_mother:
        mtype = f"Married mother's origin ({spouse_gen})"
    elif shares_father:
        mtype = f"Married father's origin ({spouse_gen})"
    else:
        mtype = f'Married different origin ({spouse_gen})'
    
    return {
        'type': mtype,
        'shares_mother': shares_mother,
        'shares_father': shares_father,
        'shared_countries': shared,
    }


def process_chunk(chunk, chunk_num):
    """Process a chunk of census data."""

    required = ['BPL', 'MBPL', 'FBPL', 'MARST', 'SPLOC', 'PERWT', 'YEAR']
    spouse_cols = ['BPL_SP', 'MBPL_SP', 'FBPL_SP']

    if not all(c in chunk.columns for c in required):
        return pd.DataFrame()

    # Filter to valid years first
    valid_year = chunk['YEAR'].isin(VALID_YEARS)
    is_us_born = chunk['BPL'].isin(US_STATE_CODES)
    mother_foreign = chunk['MBPL'] > 99
    father_foreign = chunk['FBPL'] > 99
    has_immigrant_parent = mother_foreign | father_foreign
    is_married = chunk['MARST'] == 1
    has_spouse = chunk['SPLOC'] > 0

    df = chunk[valid_year & is_us_born & has_immigrant_parent & is_married & has_spouse].copy()
    
    if len(df) == 0 or not all(c in df.columns for c in spouse_cols):
        return pd.DataFrame()
    
    results = []
    for idx, row in df.iterrows():
        mother_origin = get_country(row['MBPL'])
        father_origin = get_country(row['FBPL'])
        
        spouse = get_spouse_origins(
            row['BPL_SP'], 
            row.get('MBPL_SP', 0), 
            row.get('FBPL_SP', 0)
        )
        
        marriage = classify_marriage(mother_origin, father_origin, spouse)
        
        # Convert year to int to avoid ".0" issue
        year = int(row['YEAR'])
        
        results.append({
            'YEAR': year,
            'PERWT': row['PERWT'],
            'SEX': row['SEX'],
            'AGE': row.get('AGE', None),
            
            'MOTHER_ORIGIN': mother_origin,
            'FATHER_ORIGIN': father_origin,
            
            'MBPL': row['MBPL'],
            'FBPL': row['FBPL'],
            
            'SPOUSE_GENERATION': spouse['generation'],
            'SPOUSE_COUNTRY': spouse['primary_country'],
            'SPOUSE_MOTHER_ORIGIN': spouse['spouse_mother'],
            'SPOUSE_FATHER_ORIGIN': spouse['spouse_father'],
            'SPOUSE_ALL_ORIGINS': '|'.join(sorted(spouse['origins'])) if spouse['origins'] else '',
            
            'BPL_SP': row['BPL_SP'],
            'MBPL_SP': row.get('MBPL_SP', 0),
            'FBPL_SP': row.get('FBPL_SP', 0),
            
            'MARRIAGE_TYPE': marriage['type'],
            'SHARES_MOTHER_HERITAGE': marriage['shares_mother'],
            'SHARES_FATHER_HERITAGE': marriage['shares_father'],
            'SHARED_COUNTRIES': '|'.join(sorted(marriage['shared_countries'])) if marriage['shared_countries'] else '',
        })
    
    return pd.DataFrame(results)


def main():
    print("\n" + "="*70)
    print("SECOND-GENERATION MARRIAGE PROCESSOR v8")
    print("="*70)

    input_path = RAW_DATA_DIR / INPUT_FILE
    if not input_path.exists():
        print(f"\nERROR: File not found: {input_path}")
        return

    print(f"\nInput: {input_path}")
    file_size_mb = input_path.stat().st_size / (1024 * 1024)
    print(f"Size: {file_size_mb:.1f} MB")
    print(f"Valid years: {VALID_YEARS}")
    
    start_time = time.time()
    all_results = []
    total_rows = 0
    
    for chunk_num, chunk in enumerate(pd.read_csv(input_path, chunksize=CHUNK_SIZE, low_memory=False), 1):
        total_rows += len(chunk)
        result = process_chunk(chunk, chunk_num)
        if len(result) > 0:
            all_results.append(result)
        print(f"   Chunk {chunk_num}: {len(chunk):,} -> {len(result):,} records")
    
    if not all_results:
        print("\nNo records found!")
        return
    
    df = pd.concat(all_results, ignore_index=True)
    elapsed = time.time() - start_time
    
    print(f"\nDone! {len(df):,} records in {elapsed/60:.1f} min")
    
    # Ensure YEAR is integer type
    df['YEAR'] = df['YEAR'].astype(int)
    
    # Save files
    df.to_csv(PROCESSED_DIR / "second_gen_marriages.csv", index=False)
    print(f"Saved: second_gen_marriages.csv")
    
    agg = df.groupby(['YEAR', 'MOTHER_ORIGIN', 'FATHER_ORIGIN', 'MARRIAGE_TYPE']).agg({
        'PERWT': 'sum'
    }).reset_index()
    agg['UNWEIGHTED_N'] = df.groupby(['YEAR', 'MOTHER_ORIGIN', 'FATHER_ORIGIN', 'MARRIAGE_TYPE']).size().values
    agg.columns = ['YEAR', 'MOTHER_ORIGIN', 'FATHER_ORIGIN', 'MARRIAGE_TYPE', 'WEIGHTED_COUNT', 'UNWEIGHTED_N']
    agg.to_csv(PROCESSED_DIR / "agg_by_parents.csv", index=False)
    print(f"Saved: agg_by_parents.csv")
    
    sankey = df.groupby([
        'YEAR', 'MOTHER_ORIGIN', 'FATHER_ORIGIN', 
        'SPOUSE_GENERATION', 'SPOUSE_COUNTRY', 
        'SPOUSE_MOTHER_ORIGIN', 'SPOUSE_FATHER_ORIGIN',
        'MARRIAGE_TYPE'
    ])['PERWT'].sum().reset_index()
    sankey.columns = ['YEAR', 'MOTHER_ORIGIN', 'FATHER_ORIGIN', 
                      'SPOUSE_GEN', 'SPOUSE_COUNTRY', 
                      'SPOUSE_MOTHER', 'SPOUSE_FATHER',
                      'MARRIAGE_TYPE', 'WEIGHTED_COUNT']
    sankey.to_csv(PROCESSED_DIR / "sankey_flows.csv", index=False)
    print(f"Saved: sankey_flows.csv")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nYears: {sorted(df['YEAR'].unique())}")
    
    print(f"\nMarriage types:")
    for mtype, count in df.groupby('MARRIAGE_TYPE')['PERWT'].sum().sort_values(ascending=False).items():
        pct = 100 * count / df['PERWT'].sum()
        print(f"   {mtype}: {count:,.0f} ({pct:.1f}%)")
    
    print("\n" + "="*70)
    print("NEXT: python run_second_gen_dashboard.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

"""
Preprocessing Script for Deployment
====================================
Creates optimized aggregated data files for the dashboard.
The raw data (usa_00004.csv.gz) is too large for free hosting,
so we pre-compute all aggregations needed by the dashboard.

Output files (in data/processed/):
- marriage_agg.csv: Main aggregation by year/origins/marriage type
- spouse_backgrounds.csv: Spouse background details for the table
- metadata.json: Valid origins, years, and presets info

Usage: python preprocess_for_deploy.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
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
INPUT_FILE = "usa_00004.csv.gz"

# Only process these census years
VALID_YEARS = [1880, 1900, 1910, 1920, 1930]

# Minimum sample size for origin to appear in dropdowns
MIN_SAMPLE_SIZE = 20000

# Origins to exclude from dropdowns
EXCLUDED_ORIGINS = {
    'Abroad/At Sea', 'Missing', 'Unknown', 'N/A',
    'Europe (unspecified)', 'Asia (unspecified)', 'Central Europe (unspecified)',
    'Asia Minor (unspecified)', 'Southwest Asia (unspecified)', 'United Kingdom (unspecified)',
    'Pacific Islands', 'Atlantic Islands', 'Other US Possessions',
    'Gibraltar', 'Liechtenstein', 'Malta', 'Guam', 'US Virgin Islands',
    'Indonesia', 'Thailand', 'Iran',
}

# =============================================================================
# IPUMS BPL CODES
# =============================================================================

US_STATE_CODES = set(range(1, 57))
US_STATE_CODES.add(90)
US_STATE_CODES.add(99)

COUNTRY_CODES = {
    100: "American Samoa", 105: "Guam", 110: "Puerto Rico",
    115: "US Virgin Islands", 120: "Other US Possessions",
    150: "Canada", 155: "St. Pierre and Miquelon",
    160: "Atlantic Islands", 199: "North America (unspecified)",
    200: "Mexico", 210: "Central America",
    250: "Cuba", 260: "West Indies", 299: "Americas (unspecified)",
    300: "South America",
    400: "Denmark", 401: "Finland", 402: "Iceland", 403: "Lapland",
    404: "Norway", 405: "Sweden", 406: "Svalbard",
    410: "England", 411: "Scotland", 412: "Wales",
    413: "United Kingdom (unspecified)", 414: "Ireland",
    419: "Northern Europe (unspecified)",
    420: "Belgium", 421: "France", 422: "Liechtenstein", 423: "Luxembourg",
    424: "Monaco", 425: "Netherlands", 426: "Switzerland",
    429: "Western Europe (unspecified)",
    430: "Albania", 431: "Andorra", 432: "Gibraltar", 433: "Greece",
    434: "Italy", 435: "Malta", 436: "Portugal", 437: "San Marino",
    438: "Spain", 439: "Vatican City", 440: "Southern Europe (unspecified)",
    450: "Austria", 451: "Bulgaria", 452: "Czechoslovakia", 453: "Germany",
    454: "Hungary", 455: "Poland", 456: "Romania", 457: "Yugoslavia",
    458: "Central Europe (unspecified)", 459: "Eastern Europe (unspecified)",
    460: "Estonia", 461: "Latvia", 462: "Lithuania",
    463: "Baltic States (unspecified)", 465: "Russia/USSR",
    499: "Europe (unspecified)",
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
    600: "Africa", 700: "Australia/New Zealand", 710: "Pacific Islands",
    800: "Antarctica", 900: "Abroad/At Sea", 950: "Other",
    997: "Unknown", 998: "Illegible", 999: "Missing",
    0: "N/A",
}

NON_ORIGIN = {"US-born", "Unknown", "N/A", "Abroad/At Sea", "Missing", "Illegible"}


def get_country(bpl_code):
    if pd.isna(bpl_code):
        return "Unknown"
    bpl_code = int(bpl_code)
    if bpl_code in US_STATE_CODES:
        return "US-born"
    if bpl_code in COUNTRY_CODES:
        return COUNTRY_CODES[bpl_code]
    return f"Unknown (code {bpl_code})"


def is_foreign_origin(country):
    if country in NON_ORIGIN:
        return False
    if country.startswith("Unknown"):
        return False
    return True


def get_spouse_info(bpl_sp, mbpl_sp, fbpl_sp):
    spouse_bp = get_country(bpl_sp)
    spouse_mother = get_country(mbpl_sp)
    spouse_father = get_country(fbpl_sp)

    if is_foreign_origin(spouse_bp):
        return '1st gen immigrant', spouse_bp, spouse_mother, spouse_father, {spouse_bp}

    sp_m_foreign = is_foreign_origin(spouse_mother)
    sp_f_foreign = is_foreign_origin(spouse_father)

    if sp_m_foreign or sp_f_foreign:
        origins = set()
        if sp_m_foreign:
            origins.add(spouse_mother)
        if sp_f_foreign:
            origins.add(spouse_father)
        primary = spouse_father if sp_f_foreign else spouse_mother
        return '2nd gen', primary, spouse_mother, spouse_father, origins

    return '3rd+ gen American', 'American', spouse_mother, spouse_father, set()


def classify_marriage(mother_origin, father_origin, spouse_gen, spouse_origins):
    if spouse_gen == '3rd+ gen American':
        return 'Married 3rd+ gen American'

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
            return f'Married same origin ({spouse_gen})'
        else:
            return f'Married someone sharing both heritages ({spouse_gen})'
    elif shares_mother:
        return f"Married mother's origin ({spouse_gen})"
    elif shares_father:
        return f"Married father's origin ({spouse_gen})"
    else:
        return f'Married different origin ({spouse_gen})'


def process_chunk(chunk):
    required = ['BPL', 'MBPL', 'FBPL', 'MARST', 'SPLOC', 'PERWT', 'YEAR']
    spouse_cols = ['BPL_SP', 'MBPL_SP', 'FBPL_SP']

    if not all(c in chunk.columns for c in required):
        return pd.DataFrame()

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

        spouse_gen, spouse_country, spouse_mom, spouse_dad, spouse_origins = get_spouse_info(
            row['BPL_SP'],
            row.get('MBPL_SP', 0),
            row.get('FBPL_SP', 0)
        )

        marriage_type = classify_marriage(mother_origin, father_origin, spouse_gen, spouse_origins)

        results.append({
            'YEAR': int(row['YEAR']),
            'PERWT': row['PERWT'],
            'MOTHER_ORIGIN': mother_origin,
            'FATHER_ORIGIN': father_origin,
            'SPOUSE_GENERATION': spouse_gen,
            'SPOUSE_COUNTRY': spouse_country,
            'SPOUSE_MOTHER_ORIGIN': spouse_mom,
            'SPOUSE_FATHER_ORIGIN': spouse_dad,
            'MARRIAGE_TYPE': marriage_type,
        })

    return pd.DataFrame(results)


def main():
    print("\n" + "="*70)
    print("PREPROCESSING FOR DEPLOYMENT")
    print("="*70)

    input_path = RAW_DATA_DIR / INPUT_FILE
    if not input_path.exists():
        print(f"\nERROR: File not found: {input_path}")
        print("Please ensure usa_00004.csv.gz is in data/raw/")
        return False

    print(f"\nInput: {input_path}")
    file_size_mb = input_path.stat().st_size / (1024 * 1024)
    print(f"Size: {file_size_mb:.1f} MB")
    print(f"Valid years: {VALID_YEARS}")

    # Process data
    start_time = time.time()
    all_results = []
    total_rows = 0

    print("\nProcessing chunks...")
    for chunk_num, chunk in enumerate(pd.read_csv(input_path, chunksize=CHUNK_SIZE, low_memory=False), 1):
        total_rows += len(chunk)
        result = process_chunk(chunk)
        if len(result) > 0:
            all_results.append(result)
        print(f"  Chunk {chunk_num}: {len(chunk):,} rows -> {len(result):,} records")

    if not all_results:
        print("\nNo records found!")
        return False

    df = pd.concat(all_results, ignore_index=True)
    elapsed = time.time() - start_time
    print(f"\nProcessed {len(df):,} records in {elapsed/60:.1f} min")

    # ==========================================================================
    # CREATE AGGREGATED FILES
    # ==========================================================================

    print("\nCreating aggregated files...")

    # 1. Main marriage aggregation
    print("  1. Marriage aggregation by year/origins/type...")
    marriage_agg = df.groupby([
        'YEAR', 'MOTHER_ORIGIN', 'FATHER_ORIGIN', 'MARRIAGE_TYPE'
    ]).agg({
        'PERWT': 'sum'
    }).reset_index()
    marriage_agg['UNWEIGHTED_N'] = df.groupby([
        'YEAR', 'MOTHER_ORIGIN', 'FATHER_ORIGIN', 'MARRIAGE_TYPE'
    ]).size().values
    marriage_agg.columns = ['YEAR', 'MOTHER_ORIGIN', 'FATHER_ORIGIN', 'MARRIAGE_TYPE',
                            'WEIGHTED_COUNT', 'UNWEIGHTED_N']
    marriage_agg.to_csv(PROCESSED_DIR / "marriage_agg.csv", index=False)
    print(f"     Saved: marriage_agg.csv ({len(marriage_agg):,} rows)")

    # 2. Spouse background details for the spouse table
    print("  2. Spouse backgrounds aggregation...")
    spouse_bg = df.groupby([
        'YEAR', 'MOTHER_ORIGIN', 'FATHER_ORIGIN',
        'SPOUSE_GENERATION', 'SPOUSE_COUNTRY',
        'SPOUSE_MOTHER_ORIGIN', 'SPOUSE_FATHER_ORIGIN'
    ])['PERWT'].sum().reset_index()
    spouse_bg.columns = ['YEAR', 'MOTHER_ORIGIN', 'FATHER_ORIGIN',
                         'SPOUSE_GEN', 'SPOUSE_COUNTRY',
                         'SPOUSE_MOTHER', 'SPOUSE_FATHER', 'WEIGHTED_COUNT']
    spouse_bg.to_csv(PROCESSED_DIR / "spouse_backgrounds.csv", index=False)
    print(f"     Saved: spouse_backgrounds.csv ({len(spouse_bg):,} rows)")

    # ==========================================================================
    # COMPUTE METADATA
    # ==========================================================================

    print("  3. Computing metadata...")

    # Get valid origins (those with sufficient sample size)
    mother_counts = df.groupby('MOTHER_ORIGIN')['PERWT'].sum()
    father_counts = df.groupby('FATHER_ORIGIN')['PERWT'].sum()

    valid_mother = set(mother_counts[mother_counts >= MIN_SAMPLE_SIZE].index.tolist())
    valid_father = set(father_counts[father_counts >= MIN_SAMPLE_SIZE].index.tolist())

    # Remove excluded origins
    valid_mother = valid_mother - EXCLUDED_ORIGINS
    valid_father = valid_father - EXCLUDED_ORIGINS

    # Use intersection for consistency
    valid_origins = sorted(valid_mother & valid_father)

    # Generate presets
    foreign_origins = [o for o in valid_origins if o != 'US-born']
    valid_df = df[df['MOTHER_ORIGIN'].isin(foreign_origins) & df['FATHER_ORIGIN'].isin(foreign_origins)]

    # Same-origin presets
    same_origin = valid_df[valid_df['MOTHER_ORIGIN'] == valid_df['FATHER_ORIGIN']]
    same_origin_sizes = same_origin.groupby('MOTHER_ORIGIN')['PERWT'].sum()
    same_origin_sizes = same_origin_sizes[same_origin_sizes >= MIN_SAMPLE_SIZE]
    top_same = same_origin_sizes.nlargest(8).index.tolist()

    # Mixed-origin presets
    mixed_origin = valid_df[valid_df['MOTHER_ORIGIN'] != valid_df['FATHER_ORIGIN']].copy()
    mixed_origin['combo'] = mixed_origin['MOTHER_ORIGIN'] + '|' + mixed_origin['FATHER_ORIGIN']
    top_mixed = mixed_origin.groupby('combo')['PERWT'].sum().nlargest(6).index.tolist()

    metadata = {
        'valid_origins': valid_origins,
        'years': VALID_YEARS,
        'presets_same_origin': top_same,
        'presets_mixed_origin': top_mixed,
        'min_sample_size': MIN_SAMPLE_SIZE,
    }

    with open(PROCESSED_DIR / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"     Saved: metadata.json")

    # ==========================================================================
    # SUMMARY
    # ==========================================================================

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    # File sizes
    marriage_size = (PROCESSED_DIR / "marriage_agg.csv").stat().st_size / 1024
    spouse_size = (PROCESSED_DIR / "spouse_backgrounds.csv").stat().st_size / 1024
    meta_size = (PROCESSED_DIR / "metadata.json").stat().st_size / 1024
    total_size = marriage_size + spouse_size + meta_size

    print(f"\nOutput files:")
    print(f"  marriage_agg.csv:       {marriage_size:,.0f} KB")
    print(f"  spouse_backgrounds.csv: {spouse_size:,.0f} KB")
    print(f"  metadata.json:          {meta_size:,.1f} KB")
    print(f"  TOTAL:                  {total_size:,.0f} KB ({total_size/1024:.1f} MB)")

    print(f"\nYears: {VALID_YEARS}")
    print(f"Valid origins: {len(valid_origins)} countries")
    print(f"Same-origin presets: {top_same}")
    print(f"Mixed-origin presets: {len(top_mixed)} combinations")

    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("  1. Update the dashboard to use aggregated files")
    print("  2. Test: python run_second_gen_dashboard.py")
    print("  3. Deploy to Render")
    print("="*70 + "\n")

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

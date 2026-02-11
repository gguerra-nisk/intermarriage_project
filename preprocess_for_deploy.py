"""
Preprocessing Script for Deployment
====================================
Creates optimized aggregated data files for the dashboard.
The raw data is too large for free hosting,
so we pre-compute all aggregations needed by the dashboard.

Output files (in data/processed/):
- marriage_agg.csv: Main aggregation by year/origins/marriage type
- spouse_backgrounds.csv: Spouse background details for the table
- geographic_agg.csv: State-level concentration vs outmarriage rates
- language_agg.csv: Language-level marriage patterns (1910-1930, MTONGUE)
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
INPUT_FILE = "usa_00008.csv.gz"

# Only process these census years
VALID_YEARS = [1880, 1900, 1910, 1920, 1930]

# Minimum sample size for origin to appear in dropdowns
MIN_SAMPLE_SIZE = 20000

# Minimum weighted count for a group-state cell in geographic analysis
MIN_GEO_SAMPLE = 5000

# FIPS state code to state name mapping
STATEFIP_NAMES = {
    1: 'Alabama', 2: 'Alaska', 4: 'Arizona', 5: 'Arkansas', 6: 'California',
    8: 'Colorado', 9: 'Connecticut', 10: 'Delaware', 11: 'District of Columbia',
    12: 'Florida', 13: 'Georgia', 15: 'Hawaii', 16: 'Idaho', 17: 'Illinois',
    18: 'Indiana', 19: 'Iowa', 20: 'Kansas', 21: 'Kentucky', 22: 'Louisiana',
    23: 'Maine', 24: 'Maryland', 25: 'Massachusetts', 26: 'Michigan',
    27: 'Minnesota', 28: 'Mississippi', 29: 'Missouri', 30: 'Montana',
    31: 'Nebraska', 32: 'Nevada', 33: 'New Hampshire', 34: 'New Jersey',
    35: 'New Mexico', 36: 'New York', 37: 'North Carolina', 38: 'North Dakota',
    39: 'Ohio', 40: 'Oklahoma', 41: 'Oregon', 42: 'Pennsylvania',
    44: 'Rhode Island', 45: 'South Carolina', 46: 'South Dakota',
    47: 'Tennessee', 48: 'Texas', 49: 'Utah', 50: 'Vermont', 51: 'Virginia',
    53: 'Washington', 54: 'West Virginia', 55: 'Wisconsin', 56: 'Wyoming',
}

# IPUMS MTONGUE codes (mother tongue)
MTONGUE_NAMES = {
    0: 'English/Not Reported',
    1: 'English',
    2: 'German',
    3: 'Yiddish/Hebrew',
    4: 'Dutch/Flemish',
    5: 'Swedish',
    6: 'Danish',
    7: 'Norwegian',
    10: 'Italian',
    11: 'French',
    12: 'Spanish',
    13: 'Portuguese',
    14: 'Romanian',
    15: 'Celtic/Gaelic',
    16: 'Greek',
    17: 'Albanian',
    18: 'Russian',
    19: 'Ruthenian',
    20: 'Czech',
    21: 'Polish',
    22: 'Slovak',
    23: 'Serbian/Croatian',
    24: 'Slovenian',
    25: 'Lithuanian',
    26: 'Latvian',
    28: 'Armenian',
    33: 'Finnish',
    34: 'Magyar',
    43: 'Chinese',
    48: 'Japanese',
}

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

        record = {
            'YEAR': int(row['YEAR']),
            'PERWT': row['PERWT'],
            'MOTHER_ORIGIN': mother_origin,
            'FATHER_ORIGIN': father_origin,
            'SPOUSE_GENERATION': spouse_gen,
            'SPOUSE_COUNTRY': spouse_country,
            'SPOUSE_MOTHER_ORIGIN': spouse_mom,
            'SPOUSE_FATHER_ORIGIN': spouse_dad,
            'MARRIAGE_TYPE': marriage_type,
        }
        if 'STATEFIP' in row.index:
            record['STATEFIP'] = int(row['STATEFIP'])
        if 'MTONGUE' in row.index:
            record['MTONGUE'] = int(row['MTONGUE']) if pd.notna(row.get('MTONGUE')) else 0
            record['MTONGUED'] = int(row['MTONGUED']) if pd.notna(row.get('MTONGUED')) else 0
        if 'MTONGUE_SP' in row.index:
            record['MTONGUE_SP'] = int(row['MTONGUE_SP']) if pd.notna(row.get('MTONGUE_SP')) else 0
            record['MTONGUED_SP'] = int(row['MTONGUED_SP']) if pd.notna(row.get('MTONGUED_SP')) else 0
        results.append(record)

    return pd.DataFrame(results)


def main():
    print("\n" + "="*70)
    print("PREPROCESSING FOR DEPLOYMENT")
    print("="*70)

    input_path = RAW_DATA_DIR / INPUT_FILE
    if not input_path.exists():
        print(f"\nERROR: File not found: {input_path}")
        print(f"Please ensure {INPUT_FILE} is in data/raw/")
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

    # 3. Geographic aggregation (state-level concentration vs outmarriage)
    has_statefip = 'STATEFIP' in df.columns
    if has_statefip:
        print("  3. Geographic aggregation by state/origin...")

        # Filter to same-origin parents only (cleaner analysis)
        geo_df = df[df['MOTHER_ORIGIN'] == df['FATHER_ORIGIN']].copy()
        geo_df = geo_df[~geo_df['MOTHER_ORIGIN'].isin(
            {'US-born', 'Unknown', 'N/A', 'Abroad/At Sea', 'Missing', 'Illegible'}
        )]
        geo_df = geo_df[geo_df['STATEFIP'].isin(STATEFIP_NAMES.keys())]
        geo_df['STATE_NAME'] = geo_df['STATEFIP'].map(STATEFIP_NAMES)

        # Pool all years for statistical power
        # Total 2nd-gen population per state
        state_totals = geo_df.groupby('STATEFIP')['PERWT'].sum().reset_index()
        state_totals.columns = ['STATEFIP', 'TOTAL_2NDGEN_IN_STATE']

        geo_results = []
        for (statefip, origin), grp in geo_df.groupby(['STATEFIP', 'MOTHER_ORIGIN']):
            weighted_n = grp['PERWT'].sum()
            unweighted_n = len(grp)

            if weighted_n < MIN_GEO_SAMPLE:
                continue

            state_total = state_totals.loc[
                state_totals['STATEFIP'] == statefip, 'TOTAL_2NDGEN_IN_STATE'
            ].values[0]
            group_share = weighted_n / state_total * 100 if state_total > 0 else 0

            # Marriage type breakdown
            type_counts = grp.groupby('MARRIAGE_TYPE')['PERWT'].sum()
            total = type_counts.sum()
            pcts = (type_counts / total * 100).to_dict() if total > 0 else {}

            heritage_rate = sum(v for k, v in pcts.items() if 'same origin' in k)
            third_gen_rate = sum(v for k, v in pcts.items() if '3rd+ gen' in k)
            outmarriage_rate = 100 - heritage_rate

            geo_results.append({
                'STATEFIP': int(statefip),
                'STATE_NAME': STATEFIP_NAMES.get(int(statefip), 'Unknown'),
                'ORIGIN_GROUP': origin,
                'TOTAL_2NDGEN_IN_STATE': state_total,
                'GROUP_COUNT_IN_STATE': weighted_n,
                'GROUP_SHARE_PCT': round(group_share, 2),
                'OUTMARRIAGE_RATE': round(outmarriage_rate, 2),
                'THIRD_GEN_RATE': round(third_gen_rate, 2),
                'HERITAGE_RATE': round(heritage_rate, 2),
                'WEIGHTED_N': weighted_n,
                'UNWEIGHTED_N': unweighted_n,
            })

        if geo_results:
            geo_agg = pd.DataFrame(geo_results)
            geo_agg.to_csv(PROCESSED_DIR / "geographic_agg.csv", index=False)
            print(f"     Saved: geographic_agg.csv ({len(geo_agg):,} rows)")
            print(f"     Origins with state data: {geo_agg['ORIGIN_GROUP'].nunique()}")
            print(f"     States with data: {geo_agg['STATE_NAME'].nunique()}")
        else:
            print("     WARNING: No geographic data met the minimum sample threshold")
            has_statefip = False
        # 3b. Geography-adjusted affinity matrix
        print("  3b. Computing geography-adjusted intermarriage affinities...")

        # Start from full df (has spouse details + STATEFIP)
        net_df = df[df['MOTHER_ORIGIN'] == df['FATHER_ORIGIN']].copy()
        net_df = net_df[~net_df['MOTHER_ORIGIN'].isin(
            {'US-born', 'Unknown', 'N/A', 'Abroad/At Sea', 'Missing', 'Illegible'}
        )]
        net_df = net_df[net_df['STATEFIP'].isin(STATEFIP_NAMES.keys())]

        EXCLUDED_FROM_NETWORK = {'Mexico', 'Cuba', 'West Indies', 'China', 'Japan'}
        net_df = net_df[~net_df['MOTHER_ORIGIN'].isin(EXCLUDED_FROM_NETWORK)]

        # Determine spouse heritage (same logic as app.py network)
        def get_spouse_heritage(row):
            if row['SPOUSE_GENERATION'] == '3rd+ gen American':
                return None
            elif row['SPOUSE_GENERATION'] == '1st gen immigrant':
                return row['SPOUSE_COUNTRY']
            else:
                dad = str(row['SPOUSE_FATHER_ORIGIN'])
                mom = str(row['SPOUSE_MOTHER_ORIGIN'])
                if dad not in ['US-born', 'Unknown', 'N/A', 'nan']:
                    return dad
                elif mom not in ['US-born', 'Unknown', 'N/A', 'nan']:
                    return mom
                return None

        net_df['SPOUSE_HERITAGE'] = net_df.apply(get_spouse_heritage, axis=1)
        net_df = net_df[net_df['SPOUSE_HERITAGE'].notna()]
        net_df = net_df[~net_df['SPOUSE_HERITAGE'].isin(
            ['Unknown', 'N/A', 'US-born'] + list(EXCLUDED_FROM_NETWORK)
        )]

        # Major groups (enough data nationally)
        MIN_NET_SAMPLE = 50000
        parent_totals_net = net_df.groupby('MOTHER_ORIGIN')['PERWT'].sum()
        major_groups = parent_totals_net[parent_totals_net >= MIN_NET_SAMPLE].index.tolist()

        net_df = net_df[net_df['MOTHER_ORIGIN'].isin(major_groups)]
        net_df = net_df[net_df['SPOUSE_HERITAGE'].isin(major_groups)]

        if len(net_df) > 0 and len(major_groups) > 1:
            # Pairwise counts per state
            state_pairs = net_df.groupby(
                ['STATEFIP', 'MOTHER_ORIGIN', 'SPOUSE_HERITAGE']
            )['PERWT'].sum().reset_index()
            state_pairs.columns = ['STATEFIP', 'PARENT', 'SPOUSE', 'COUNT']

            # Parent totals per state (denominator for observed rate)
            state_parent_tot = net_df.groupby(
                ['STATEFIP', 'MOTHER_ORIGIN']
            )['PERWT'].sum().reset_index()
            state_parent_tot.columns = ['STATEFIP', 'PARENT', 'PARENT_TOTAL']

            # Local spouse market shares per state
            state_spouse_tot = net_df.groupby(
                ['STATEFIP', 'SPOUSE_HERITAGE']
            )['PERWT'].sum().reset_index()
            state_spouse_tot.columns = ['STATEFIP', 'SPOUSE', 'SPOUSE_TOTAL']
            state_market = state_spouse_tot.groupby('STATEFIP')['SPOUSE_TOTAL'].sum().reset_index()
            state_market.columns = ['STATEFIP', 'MARKET_TOTAL']
            state_spouse_tot = state_spouse_tot.merge(state_market, on='STATEFIP')
            state_spouse_tot['MARKET_SHARE'] = (
                state_spouse_tot['SPOUSE_TOTAL'] / state_spouse_tot['MARKET_TOTAL']
            )

            # Merge and compute within-state affinities
            merged = state_pairs.merge(state_parent_tot, on=['STATEFIP', 'PARENT'])
            merged = merged.merge(
                state_spouse_tot[['STATEFIP', 'SPOUSE', 'MARKET_SHARE']],
                on=['STATEFIP', 'SPOUSE']
            )

            # Drop state-pair cells with tiny counts (prevents extreme ratios)
            MIN_PAIR_COUNT = 500
            merged = merged[merged['COUNT'] >= MIN_PAIR_COUNT]

            merged['OBSERVED_RATE'] = merged['COUNT'] / merged['PARENT_TOTAL']
            merged['AFFINITY'] = merged['OBSERVED_RATE'] / merged['MARKET_SHARE'].clip(lower=0.0001)

            # Weighted average affinity across states
            cross_group = merged[merged['PARENT'] != merged['SPOUSE']]
            adjusted = cross_group.groupby(['PARENT', 'SPOUSE']).apply(
                lambda g: np.average(g['AFFINITY'], weights=g['PARENT_TOTAL'])
            ).reset_index()
            adjusted.columns = ['SOURCE', 'TARGET', 'GEO_ADJUSTED_AFFINITY']

            # Symmetrize — require BOTH directions to exist (prevents one-sided artifacts)
            sym_results = []
            seen = set()
            for _, row in adjusted.iterrows():
                pair = tuple(sorted([row['SOURCE'], row['TARGET']]))
                if pair in seen:
                    continue
                seen.add(pair)
                ab = adjusted[
                    (adjusted['SOURCE'] == pair[0]) & (adjusted['TARGET'] == pair[1])
                ]['GEO_ADJUSTED_AFFINITY'].values
                ba = adjusted[
                    (adjusted['SOURCE'] == pair[1]) & (adjusted['TARGET'] == pair[0])
                ]['GEO_ADJUSTED_AFFINITY'].values
                if len(ab) == 0 or len(ba) == 0:
                    continue  # Skip one-directional pairs
                avg = (ab[0] + ba[0]) / 2
                sym_results.append({
                    'SOURCE': pair[0], 'TARGET': pair[1],
                    'GEO_ADJUSTED_AFFINITY': round(avg, 3)
                })

            if sym_results:
                adj_aff_df = pd.DataFrame(sym_results)
                adj_aff_df.to_csv(PROCESSED_DIR / "geo_adjusted_affinity.csv", index=False)
                print(f"     Saved: geo_adjusted_affinity.csv ({len(adj_aff_df)} pairs)")
            else:
                print("     WARNING: No adjusted affinity pairs computed")
        else:
            print("     WARNING: Insufficient data for adjusted affinity computation")

    else:
        print("  3. Skipping geographic aggregation (no STATEFIP column in data)")

    # ==========================================================================
    # LANGUAGE AGGREGATION (MTONGUE, 1910-1930 only)
    # ==========================================================================
    # MTONGUE is only populated for foreign-born individuals, so we process
    # first-generation immigrants separately from the raw data. We classify
    # their marriages by: whether the spouse shares their birthplace origin
    # (same heritage), is 3rd+ gen American, or is from a different origin.

    lang_results_exist = False
    print("  4. Language aggregation (MTONGUE, 1910-1930)...")
    print("     Scanning raw data for first-gen immigrants with MTONGUE...")

    lang_all = []
    for chunk_num, chunk in enumerate(pd.read_csv(input_path, chunksize=CHUNK_SIZE, low_memory=False), 1):
        if 'MTONGUE' not in chunk.columns or 'MTONGUE_SP' not in chunk.columns:
            continue

        # Filter: 1910-1930, foreign-born, married, has spouse
        valid_year = chunk['YEAR'].isin([1910, 1920, 1930])
        is_foreign = ~chunk['BPL'].isin(US_STATE_CODES) & (chunk['BPL'] > 99)
        is_married = chunk['MARST'] == 1
        has_spouse = chunk['SPLOC'] > 0
        has_tongue = chunk['MTONGUE'].notna() & (chunk['MTONGUE'] > 0)

        sub = chunk[valid_year & is_foreign & is_married & has_spouse & has_tongue].copy()
        if len(sub) == 0:
            continue

        # Ensure spouse columns exist
        if not all(c in sub.columns for c in ['BPL_SP', 'MTONGUE_SP']):
            continue

        # Map person's origin from BPL
        sub['ORIGIN'] = sub['BPL'].apply(lambda x: COUNTRY_CODES.get(int(x), 'Unknown') if pd.notna(x) else 'Unknown')
        sub = sub[~sub['ORIGIN'].isin({'Unknown', 'N/A', 'Abroad/At Sea', 'Missing', 'Illegible'})]

        # Map languages
        sub['LANGUAGE'] = sub['MTONGUE'].astype(int).map(MTONGUE_NAMES).fillna('Other')
        sub['SPOUSE_LANGUAGE'] = sub['MTONGUE_SP'].apply(
            lambda x: MTONGUE_NAMES.get(int(x), 'Other') if pd.notna(x) and int(x) > 0 else 'English/Not Reported'
        )

        # Classify marriage: does spouse share origin?
        sub['SPOUSE_ORIGIN'] = sub['BPL_SP'].apply(
            lambda x: COUNTRY_CODES.get(int(x), 'Unknown') if pd.notna(x) and int(x) > 99 and int(x) not in US_STATE_CODES else (
                'US-born' if pd.notna(x) and int(x) in US_STATE_CODES else 'Unknown'
            )
        )

        def classify_1stgen(row):
            if row['SPOUSE_ORIGIN'] == row['ORIGIN']:
                return 'Same heritage'
            elif row['SPOUSE_ORIGIN'] == 'US-born':
                return '3rd+ gen American'
            elif row['SPOUSE_ORIGIN'] in NON_ORIGIN or row['SPOUSE_ORIGIN'].startswith('Unknown'):
                return '3rd+ gen American'
            else:
                return 'Different origin'

        sub['MARRIAGE_CATEGORY'] = sub.apply(classify_1stgen, axis=1)

        lang_all.append(sub[['ORIGIN', 'LANGUAGE', 'MARRIAGE_CATEGORY', 'SPOUSE_LANGUAGE', 'PERWT']].copy())

    if lang_all:
        lang_df = pd.concat(lang_all, ignore_index=True)
        print(f"     Found {len(lang_df):,} first-gen immigrants with MTONGUE data")

        # Apply minimum threshold: origin-language groups with >= 500 weighted
        origin_lang_totals = lang_df.groupby(['ORIGIN', 'LANGUAGE'])['PERWT'].sum()
        valid_combos = origin_lang_totals[origin_lang_totals >= 500].index
        lang_df = lang_df[lang_df.set_index(['ORIGIN', 'LANGUAGE']).index.isin(valid_combos)]

        if len(lang_df) > 0:
            # Aggregate: ORIGIN x LANGUAGE x MARRIAGE_CATEGORY x SPOUSE_LANGUAGE
            lang_agg = lang_df.groupby([
                'ORIGIN', 'LANGUAGE', 'MARRIAGE_CATEGORY', 'SPOUSE_LANGUAGE'
            ]).agg(
                WEIGHTED_COUNT=('PERWT', 'sum'),
                UNWEIGHTED_N=('PERWT', 'count')
            ).reset_index()

            lang_agg.to_csv(PROCESSED_DIR / "language_agg.csv", index=False)
            print(f"     Saved: language_agg.csv ({len(lang_agg):,} rows)")
            print(f"     Origins with language data: {lang_agg['ORIGIN'].nunique()}")
            print(f"     Languages found: {lang_agg['LANGUAGE'].nunique()}")
            lang_results_exist = True

            # Identify multilingual origins (>= 2 languages with >= 1000 weighted)
            origin_lang_sums = lang_agg.groupby(['ORIGIN', 'LANGUAGE'])['WEIGHTED_COUNT'].sum()
            multilingual_origins = []
            for origin in lang_agg['ORIGIN'].unique():
                origin_langs = origin_lang_sums.loc[origin]
                big_langs = origin_langs[origin_langs >= 1000]
                if len(big_langs) >= 2:
                    multilingual_origins.append(origin)
            multilingual_origins.sort()
            print(f"     Multilingual origins: {multilingual_origins}")
        else:
            print("     WARNING: No origin-language groups met the minimum threshold")
    else:
        print("     WARNING: No MTONGUE data found in raw file")

    # ==========================================================================
    # COMPUTE METADATA
    # ==========================================================================

    print("  5. Computing metadata...")

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

    if has_statefip and geo_results:
        geographic_origins = sorted(geo_agg['ORIGIN_GROUP'].unique().tolist())
        metadata['geographic_origins'] = geographic_origins

    if lang_results_exist:
        metadata['language_years'] = [1910, 1920, 1930]
        metadata['multilingual_origins'] = multilingual_origins

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
    if has_statefip and (PROCESSED_DIR / "geographic_agg.csv").exists():
        geo_size = (PROCESSED_DIR / "geographic_agg.csv").stat().st_size / 1024
        total_size += geo_size
        print(f"  geographic_agg.csv:     {geo_size:,.0f} KB")
    if (PROCESSED_DIR / "language_agg.csv").exists():
        lang_size = (PROCESSED_DIR / "language_agg.csv").stat().st_size / 1024
        total_size += lang_size
        print(f"  language_agg.csv:       {lang_size:,.0f} KB")
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

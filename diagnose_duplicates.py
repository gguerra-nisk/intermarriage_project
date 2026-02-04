"""
Duplicate Diagnostic Script
============================
Analyzes potential duplicate individuals across census years to inform
deduplication methodology decisions.

This script answers:
1. How many individuals appear in multiple census years?
2. Are their characteristics (marriage type, spouse) consistent across years?
3. What is the magnitude of over-weighting for older cohorts?
4. What are the implications of different deduplication strategies?

Usage: python diagnose_duplicates.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

RAW_DATA_DIR = Path("data/raw")
INPUT_FILE = "usa_00004.csv.gz"
CHUNK_SIZE = 300000

VALID_YEARS = [1880, 1900, 1910, 1920, 1930]
US_STATE_CODES = set(range(1, 57)) | {90, 99}

# Output file for detailed results
OUTPUT_DIR = Path("data/diagnostics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# COUNTRY CODE MAPPING (simplified for diagnostics)
# =============================================================================

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
    997: "Unknown", 998: "Illegible", 999: "Missing", 0: "N/A",
}


def get_country(bpl_code):
    """Convert BPL code to country name."""
    if pd.isna(bpl_code) or bpl_code in US_STATE_CODES:
        return 'US-born'
    return COUNTRY_CODES.get(int(bpl_code), f'Code_{int(bpl_code)}')


def is_foreign_origin(origin):
    """Check if origin represents a foreign country."""
    return origin not in ('US-born', 'N/A', 'Unknown', 'Missing')


def get_spouse_generation(bpl_sp, mbpl_sp, fbpl_sp):
    """Classify spouse generation."""
    if pd.isna(bpl_sp):
        return 'Unknown'
    if bpl_sp not in US_STATE_CODES:
        return '1st gen immigrant'
    mother_foreign = not pd.isna(mbpl_sp) and mbpl_sp > 99
    father_foreign = not pd.isna(fbpl_sp) and fbpl_sp > 99
    if mother_foreign or father_foreign:
        return '2nd gen'
    return '3rd+ gen American'


def classify_marriage_simple(mother_origin, father_origin, spouse_gen, spouse_bpl, spouse_mbpl, spouse_fbpl):
    """Simplified marriage classification for diagnostics."""
    if spouse_gen == '3rd+ gen American':
        return '3rd+ gen American'

    person_origins = set()
    if is_foreign_origin(mother_origin):
        person_origins.add(mother_origin)
    if is_foreign_origin(father_origin):
        person_origins.add(father_origin)

    spouse_origins = set()
    if spouse_gen == '1st gen immigrant':
        spouse_country = get_country(spouse_bpl)
        if is_foreign_origin(spouse_country):
            spouse_origins.add(spouse_country)
    else:
        spouse_mom = get_country(spouse_mbpl)
        spouse_dad = get_country(spouse_fbpl)
        if is_foreign_origin(spouse_mom):
            spouse_origins.add(spouse_mom)
        if is_foreign_origin(spouse_dad):
            spouse_origins.add(spouse_dad)

    shared = person_origins & spouse_origins
    if shared == person_origins and len(person_origins) > 0:
        if len(person_origins) == 1 or (len(person_origins) == 2 and mother_origin == father_origin):
            return 'Same origin'
        else:
            return 'Both heritages'
    elif shared:
        return 'Partial heritage match'
    else:
        return 'Different origin'


def process_chunk_for_diagnostics(chunk):
    """Process a chunk and extract fields needed for duplicate analysis."""
    required = ['BPL', 'MBPL', 'FBPL', 'MARST', 'SPLOC', 'PERWT', 'YEAR', 'BIRTHYR', 'SEX', 'AGE']
    spouse_cols = ['BPL_SP', 'MBPL_SP', 'FBPL_SP', 'AGE_SP']

    if not all(c in chunk.columns for c in required):
        print(f"  Warning: Missing required columns")
        return pd.DataFrame()

    # Filter to second-generation married individuals
    valid_year = chunk['YEAR'].isin(VALID_YEARS)
    is_us_born = chunk['BPL'].isin(US_STATE_CODES)
    mother_foreign = chunk['MBPL'] > 99
    father_foreign = chunk['FBPL'] > 99
    has_immigrant_parent = mother_foreign | father_foreign
    is_married = chunk['MARST'] == 1
    has_spouse = chunk['SPLOC'] > 0

    df = chunk[valid_year & is_us_born & has_immigrant_parent & is_married & has_spouse].copy()

    if len(df) == 0:
        return pd.DataFrame()

    # Check for spouse columns
    for col in spouse_cols:
        if col not in df.columns:
            df[col] = np.nan

    results = []
    for idx, row in df.iterrows():
        mother_origin = get_country(row['MBPL'])
        father_origin = get_country(row['FBPL'])
        spouse_gen = get_spouse_generation(row['BPL_SP'], row.get('MBPL_SP'), row.get('FBPL_SP'))
        marriage_type = classify_marriage_simple(
            mother_origin, father_origin, spouse_gen,
            row['BPL_SP'], row.get('MBPL_SP'), row.get('FBPL_SP')
        )

        # Create unique person identifier
        # Using BIRTHYR + SEX + BPL (state) + MBPL + FBPL
        person_key = f"{int(row['BIRTHYR'])}_{int(row['SEX'])}_{int(row['BPL'])}_{int(row['MBPL'])}_{int(row['FBPL'])}"

        # Also create a spouse identifier to detect remarriage
        spouse_key = f"{int(row.get('BPL_SP', 0))}_{int(row.get('MBPL_SP', 0))}_{int(row.get('FBPL_SP', 0))}"

        results.append({
            'YEAR': int(row['YEAR']),
            'PERSON_KEY': person_key,
            'BIRTHYR': int(row['BIRTHYR']),
            'SEX': int(row['SEX']),
            'AGE': int(row['AGE']),
            'BPL': int(row['BPL']),
            'MOTHER_ORIGIN': mother_origin,
            'FATHER_ORIGIN': father_origin,
            'SPOUSE_KEY': spouse_key,
            'SPOUSE_GENERATION': spouse_gen,
            'MARRIAGE_TYPE': marriage_type,
            'PERWT': row['PERWT'],
        })

    return pd.DataFrame(results)


def main():
    print("\n" + "=" * 80)
    print("DUPLICATE DIAGNOSTIC ANALYSIS")
    print("=" * 80)
    print("\nThis analysis examines potential duplicate individuals across census years")
    print("to inform methodologically rigorous deduplication decisions.\n")

    input_path = RAW_DATA_DIR / INPUT_FILE
    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}")
        return

    # ==========================================================================
    # PHASE 1: Load and process data
    # ==========================================================================
    print("-" * 80)
    print("PHASE 1: DATA EXTRACTION")
    print("-" * 80)

    all_results = []
    total_raw_rows = 0

    for chunk_num, chunk in enumerate(pd.read_csv(input_path, chunksize=CHUNK_SIZE, low_memory=False), 1):
        total_raw_rows += len(chunk)
        result = process_chunk_for_diagnostics(chunk)
        if len(result) > 0:
            all_results.append(result)
        print(f"  Chunk {chunk_num}: {len(chunk):,} raw rows -> {len(result):,} eligible records")

    if not all_results:
        print("ERROR: No eligible records found!")
        return

    df = pd.concat(all_results, ignore_index=True)
    print(f"\nTotal raw rows processed: {total_raw_rows:,}")
    print(f"Total eligible records (married 2nd gen): {len(df):,}")

    # ==========================================================================
    # PHASE 2: Identify duplicates
    # ==========================================================================
    print("\n" + "-" * 80)
    print("PHASE 2: DUPLICATE IDENTIFICATION")
    print("-" * 80)

    # Count appearances per person
    person_counts = df.groupby('PERSON_KEY').size()
    unique_persons = len(person_counts)
    total_records = len(df)

    print(f"\nUnique person keys: {unique_persons:,}")
    print(f"Total records: {total_records:,}")
    print(f"Duplication rate: {(total_records - unique_persons) / total_records * 100:.2f}%")
    print(f"Average appearances per person: {total_records / unique_persons:.2f}")

    # Distribution of appearance counts
    appearance_dist = person_counts.value_counts().sort_index()
    print(f"\nDistribution of census appearances per person:")
    for count, num_persons in appearance_dist.items():
        pct = num_persons / unique_persons * 100
        print(f"  {count} census(es): {num_persons:,} persons ({pct:.1f}%)")

    # Identify multi-year individuals
    multi_year_keys = person_counts[person_counts > 1].index
    multi_year_df = df[df['PERSON_KEY'].isin(multi_year_keys)]
    single_year_df = df[~df['PERSON_KEY'].isin(multi_year_keys)]

    print(f"\nPersons appearing in multiple censuses: {len(multi_year_keys):,} ({len(multi_year_keys)/unique_persons*100:.1f}%)")
    print(f"Records from multi-year persons: {len(multi_year_df):,} ({len(multi_year_df)/total_records*100:.1f}%)")

    # ==========================================================================
    # PHASE 3: Birth year cohort analysis
    # ==========================================================================
    print("\n" + "-" * 80)
    print("PHASE 3: BIRTH COHORT ANALYSIS")
    print("-" * 80)

    # Which birth cohorts are over-represented?
    print("\nBirth decade distribution (records vs unique persons):")
    df['BIRTH_DECADE'] = (df['BIRTHYR'] // 10) * 10

    decade_records = df.groupby('BIRTH_DECADE').size()
    decade_persons = df.groupby('BIRTH_DECADE')['PERSON_KEY'].nunique()
    decade_ratio = decade_records / decade_persons

    print(f"\n{'Birth Decade':<15} {'Records':>12} {'Unique Persons':>15} {'Avg Appearances':>18}")
    print("-" * 60)
    for decade in sorted(decade_records.index):
        if decade >= 1820 and decade <= 1910:
            print(f"{int(decade)}-{int(decade)+9:<10} {decade_records[decade]:>12,} {decade_persons[decade]:>15,} {decade_ratio[decade]:>18.2f}")

    # Theoretical maximum appearances by birth year
    print("\n\nTheoretical census coverage by birth year:")
    print("(Assuming person is married and alive at each census)")
    print(f"\n{'Birth Year':<12} {'Possible Censuses':>20} {'Which Censuses':<30}")
    print("-" * 65)

    sample_years = [1840, 1850, 1860, 1870, 1880, 1890, 1900, 1910]
    for by in sample_years:
        possible = [y for y in VALID_YEARS if y - by >= 18 and y - by <= 80]  # Age 18-80
        print(f"{by:<12} {len(possible):>20} {str(possible):<30}")

    # ==========================================================================
    # PHASE 4: Consistency analysis for duplicates
    # ==========================================================================
    print("\n" + "-" * 80)
    print("PHASE 4: DUPLICATE CONSISTENCY ANALYSIS")
    print("-" * 80)

    print("\nAnalyzing whether the same person has consistent characteristics across censuses...")

    # For each multi-year person, check consistency
    consistency_results = {
        'same_spouse_same_type': 0,
        'same_spouse_diff_type': 0,
        'diff_spouse_same_type': 0,
        'diff_spouse_diff_type': 0,
    }

    spouse_change_details = []
    type_change_details = []

    for person_key in multi_year_keys:
        person_records = df[df['PERSON_KEY'] == person_key].sort_values('YEAR')

        if len(person_records) < 2:
            continue

        first_record = person_records.iloc[0]
        last_record = person_records.iloc[-1]

        same_spouse = first_record['SPOUSE_KEY'] == last_record['SPOUSE_KEY']
        same_type = first_record['MARRIAGE_TYPE'] == last_record['MARRIAGE_TYPE']

        if same_spouse and same_type:
            consistency_results['same_spouse_same_type'] += 1
        elif same_spouse and not same_type:
            consistency_results['same_spouse_diff_type'] += 1
            type_change_details.append({
                'person_key': person_key,
                'first_year': first_record['YEAR'],
                'last_year': last_record['YEAR'],
                'first_type': first_record['MARRIAGE_TYPE'],
                'last_type': last_record['MARRIAGE_TYPE'],
            })
        elif not same_spouse and same_type:
            consistency_results['diff_spouse_same_type'] += 1
            spouse_change_details.append({
                'person_key': person_key,
                'first_year': first_record['YEAR'],
                'last_year': last_record['YEAR'],
                'marriage_type': first_record['MARRIAGE_TYPE'],
            })
        else:
            consistency_results['diff_spouse_diff_type'] += 1
            spouse_change_details.append({
                'person_key': person_key,
                'first_year': first_record['YEAR'],
                'last_year': last_record['YEAR'],
                'first_type': first_record['MARRIAGE_TYPE'],
                'last_type': last_record['MARRIAGE_TYPE'],
            })

    total_multi = len(multi_year_keys)
    print(f"\nConsistency results for {total_multi:,} multi-census persons:")
    print(f"  Same spouse, same marriage type:      {consistency_results['same_spouse_same_type']:>8,} ({consistency_results['same_spouse_same_type']/total_multi*100:>5.1f}%)")
    print(f"  Same spouse, DIFFERENT marriage type: {consistency_results['same_spouse_diff_type']:>8,} ({consistency_results['same_spouse_diff_type']/total_multi*100:>5.1f}%)")
    print(f"  Different spouse, same marriage type: {consistency_results['diff_spouse_same_type']:>8,} ({consistency_results['diff_spouse_same_type']/total_multi*100:>5.1f}%)")
    print(f"  Different spouse, diff marriage type: {consistency_results['diff_spouse_diff_type']:>8,} ({consistency_results['diff_spouse_diff_type']/total_multi*100:>5.1f}%)")

    # Remarriage analysis
    remarriage_count = consistency_results['diff_spouse_same_type'] + consistency_results['diff_spouse_diff_type']
    print(f"\nPotential remarriages detected: {remarriage_count:,} ({remarriage_count/total_multi*100:.1f}% of multi-census persons)")

    # Data quality issues (same spouse, different type)
    if consistency_results['same_spouse_diff_type'] > 0:
        print(f"\n*** DATA QUALITY FLAG ***")
        print(f"Found {consistency_results['same_spouse_diff_type']:,} cases where same spouse has different marriage type classification.")
        print("This suggests classification edge cases or data inconsistencies.")

        # Sample some cases
        if type_change_details:
            print("\nSample cases of type changes with same spouse:")
            for case in type_change_details[:5]:
                print(f"  {case['first_year']}: {case['first_type']} -> {case['last_year']}: {case['last_type']}")

    # ==========================================================================
    # PHASE 5: Impact analysis by census year
    # ==========================================================================
    print("\n" + "-" * 80)
    print("PHASE 5: CENSUS YEAR IMPACT ANALYSIS")
    print("-" * 80)

    print("\nRecords and unique persons by census year:")
    print(f"\n{'Year':<8} {'Records':>12} {'Unique Persons':>15} {'Overlap Rate':>15} {'Weighted Pop':>15}")
    print("-" * 70)

    for year in VALID_YEARS:
        year_df = df[df['YEAR'] == year]
        year_records = len(year_df)
        year_persons = year_df['PERSON_KEY'].nunique()
        year_weighted = year_df['PERWT'].sum()

        # How many of these persons also appear in other years?
        year_keys = set(year_df['PERSON_KEY'])
        other_year_keys = set(df[df['YEAR'] != year]['PERSON_KEY'])
        overlap = len(year_keys & other_year_keys)
        overlap_rate = overlap / year_persons * 100 if year_persons > 0 else 0

        print(f"{year:<8} {year_records:>12,} {year_persons:>15,} {overlap_rate:>14.1f}% {year_weighted:>15,.0f}")

    # ==========================================================================
    # PHASE 6: Marriage type distribution comparison
    # ==========================================================================
    print("\n" + "-" * 80)
    print("PHASE 6: MARRIAGE TYPE DISTRIBUTION COMPARISON")
    print("-" * 80)

    print("\nComparing marriage type distributions:")
    print("  - 'All records': Current method (with duplicates)")
    print("  - 'First appearance': Keep only earliest census appearance")
    print("  - 'Last appearance': Keep only latest census appearance")

    # Current distribution (all records)
    current_dist = df.groupby('MARRIAGE_TYPE')['PERWT'].sum()
    current_pct = current_dist / current_dist.sum() * 100

    # First appearance distribution
    first_df = df.sort_values('YEAR').drop_duplicates('PERSON_KEY', keep='first')
    first_dist = first_df.groupby('MARRIAGE_TYPE')['PERWT'].sum()
    first_pct = first_dist / first_dist.sum() * 100

    # Last appearance distribution
    last_df = df.sort_values('YEAR').drop_duplicates('PERSON_KEY', keep='last')
    last_dist = last_df.groupby('MARRIAGE_TYPE')['PERWT'].sum()
    last_pct = last_dist / last_dist.sum() * 100

    print(f"\n{'Marriage Type':<25} {'All Records':>15} {'First Only':>15} {'Last Only':>15} {'Diff (Last-All)':>15}")
    print("-" * 90)

    all_types = sorted(set(current_pct.index) | set(first_pct.index) | set(last_pct.index))
    for mtype in all_types:
        curr = current_pct.get(mtype, 0)
        first = first_pct.get(mtype, 0)
        last = last_pct.get(mtype, 0)
        diff = last - curr
        print(f"{mtype:<25} {curr:>14.1f}% {first:>14.1f}% {last:>14.1f}% {diff:>+14.1f}pp")

    # ==========================================================================
    # PHASE 7: Sample size impact
    # ==========================================================================
    print("\n" + "-" * 80)
    print("PHASE 7: SAMPLE SIZE IMPACT")
    print("-" * 80)

    print("\nImpact of deduplication on sample sizes:")
    print(f"\n{'Metric':<40} {'All Records':>15} {'First Only':>15} {'Last Only':>15}")
    print("-" * 90)

    print(f"{'Total records':<40} {len(df):>15,} {len(first_df):>15,} {len(last_df):>15,}")
    print(f"{'Total weighted population':<40} {df['PERWT'].sum():>15,.0f} {first_df['PERWT'].sum():>15,.0f} {last_df['PERWT'].sum():>15,.0f}")
    print(f"{'Unique person-keys':<40} {df['PERSON_KEY'].nunique():>15,} {first_df['PERSON_KEY'].nunique():>15,} {last_df['PERSON_KEY'].nunique():>15,}")

    # Check sample sizes by major origin groups
    print("\n\nSample sizes for major origin groups (weighted, after deduplication):")

    # Get top origins
    origin_counts = df.groupby('MOTHER_ORIGIN')['PERWT'].sum().sort_values(ascending=False)
    top_origins = [o for o in origin_counts.head(10).index if o != 'US-born']

    print(f"\n{'Origin':<20} {'All Records':>15} {'First Only':>15} {'Last Only':>15}")
    print("-" * 70)

    for origin in top_origins:
        curr = df[df['MOTHER_ORIGIN'] == origin]['PERWT'].sum()
        first = first_df[first_df['MOTHER_ORIGIN'] == origin]['PERWT'].sum()
        last = last_df[last_df['MOTHER_ORIGIN'] == origin]['PERWT'].sum()
        print(f"{origin:<20} {curr:>15,.0f} {first:>15,.0f} {last:>15,.0f}")

    # ==========================================================================
    # PHASE 8: Age at marriage analysis
    # ==========================================================================
    print("\n" + "-" * 80)
    print("PHASE 8: AGE DISTRIBUTION ANALYSIS")
    print("-" * 80)

    print("\nAge distribution at census enumeration:")
    print("(This shows who is being captured under different strategies)")

    print(f"\n{'Statistic':<25} {'All Records':>15} {'First Only':>15} {'Last Only':>15}")
    print("-" * 75)

    print(f"{'Mean age':<25} {df['AGE'].mean():>15.1f} {first_df['AGE'].mean():>15.1f} {last_df['AGE'].mean():>15.1f}")
    print(f"{'Median age':<25} {df['AGE'].median():>15.1f} {first_df['AGE'].median():>15.1f} {last_df['AGE'].median():>15.1f}")
    print(f"{'Std dev':<25} {df['AGE'].std():>15.1f} {first_df['AGE'].std():>15.1f} {last_df['AGE'].std():>15.1f}")
    print(f"{'Min age':<25} {df['AGE'].min():>15.0f} {first_df['AGE'].min():>15.0f} {last_df['AGE'].min():>15.0f}")
    print(f"{'Max age':<25} {df['AGE'].max():>15.0f} {first_df['AGE'].max():>15.0f} {last_df['AGE'].max():>15.0f}")

    # ==========================================================================
    # PHASE 9: Recommendations
    # ==========================================================================
    print("\n" + "-" * 80)
    print("PHASE 9: METHODOLOGICAL RECOMMENDATIONS")
    print("-" * 80)

    dup_rate = (total_records - unique_persons) / total_records * 100
    remarriage_rate = remarriage_count / total_multi * 100 if total_multi > 0 else 0
    type_inconsistency_rate = consistency_results['same_spouse_diff_type'] / total_multi * 100 if total_multi > 0 else 0

    print(f"""
SUMMARY STATISTICS:
  - Duplication rate: {dup_rate:.1f}%
  - Multi-census persons with potential remarriage: {remarriage_rate:.1f}%
  - Classification inconsistencies (same spouse, diff type): {type_inconsistency_rate:.1f}%

METHODOLOGICAL OPTIONS:

1. KEEP FIRST APPEARANCE (earliest census)
   Pros: Captures marriage patterns at younger ages; less affected by mortality selection
   Cons: Smaller sample from later censuses; may miss remarriage patterns
   Best for: Understanding early marriage formation patterns

2. KEEP LAST APPEARANCE (latest census)
   Pros: Most complete information; captures stable marriages
   Cons: Survivor bias (only those who lived through period); older age profile
   Best for: Understanding who remained married; stable union analysis

3. KEEP ALL RECORDS (current approach)
   Pros: Maximizes sample size; captures full marriage history
   Cons: Over-weights older cohorts; same person counted multiple times
   Best for: Cross-sectional snapshots by year (but should note limitation)

4. RANDOM SELECTION (one observation per person)
   Pros: Unbiased selection; maintains representativeness
   Cons: Introduces randomness; less interpretable
   Best for: Sensitivity analysis

RECOMMENDATION:
""")

    if dup_rate > 20:
        print("  Given the substantial duplication rate ({:.1f}%), deduplication is STRONGLY recommended.".format(dup_rate))
    elif dup_rate > 10:
        print("  Given the moderate duplication rate ({:.1f}%), deduplication is RECOMMENDED.".format(dup_rate))
    else:
        print("  Given the low duplication rate ({:.1f}%), deduplication has modest impact.".format(dup_rate))

    if type_inconsistency_rate > 5:
        print(f"  WARNING: {type_inconsistency_rate:.1f}% of multi-census persons have inconsistent classifications.")
        print("  Consider investigating these cases before finalizing methodology.")

    if remarriage_rate > 10:
        print(f"  NOTE: {remarriage_rate:.1f}% of multi-census persons may have remarried.")
        print("  'First appearance' would capture initial marriage; 'last' would capture final marriage.")

    print("""
SUGGESTED APPROACH FOR RIGOROUS ANALYSIS:
  1. Use 'FIRST APPEARANCE' as primary methodology (captures marriage formation)
  2. Report 'LAST APPEARANCE' results as robustness check
  3. Document the deduplication in methodology section
  4. Note that remarriage cases ({:.1f}%) are captured at first observed marriage
""".format(remarriage_rate))

    # ==========================================================================
    # Save detailed results
    # ==========================================================================
    print("\n" + "-" * 80)
    print("SAVING DETAILED RESULTS")
    print("-" * 80)

    # Save the processed dataframe for further analysis
    df.to_csv(OUTPUT_DIR / "diagnostic_full_data.csv", index=False)
    print(f"  Saved: {OUTPUT_DIR / 'diagnostic_full_data.csv'}")

    # Save multi-year person details
    multi_year_summary = df[df['PERSON_KEY'].isin(multi_year_keys)].groupby('PERSON_KEY').agg({
        'YEAR': ['min', 'max', 'count'],
        'MARRIAGE_TYPE': lambda x: x.iloc[0],
        'SPOUSE_KEY': lambda x: len(x.unique()),
        'PERWT': 'mean',
    }).reset_index()
    multi_year_summary.columns = ['PERSON_KEY', 'FIRST_YEAR', 'LAST_YEAR', 'NUM_APPEARANCES',
                                   'FIRST_MARRIAGE_TYPE', 'NUM_UNIQUE_SPOUSES', 'AVG_WEIGHT']
    multi_year_summary.to_csv(OUTPUT_DIR / "multi_year_persons.csv", index=False)
    print(f"  Saved: {OUTPUT_DIR / 'multi_year_persons.csv'}")

    # Save comparison distributions
    comparison = pd.DataFrame({
        'MARRIAGE_TYPE': all_types,
        'ALL_RECORDS_PCT': [current_pct.get(t, 0) for t in all_types],
        'FIRST_ONLY_PCT': [first_pct.get(t, 0) for t in all_types],
        'LAST_ONLY_PCT': [last_pct.get(t, 0) for t in all_types],
    })
    comparison.to_csv(OUTPUT_DIR / "dedup_comparison.csv", index=False)
    print(f"  Saved: {OUTPUT_DIR / 'dedup_comparison.csv'}")

    print("\n" + "=" * 80)
    print("DIAGNOSTIC ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {OUTPUT_DIR.absolute()}")
    print("Review these results to inform deduplication methodology decision.\n")


if __name__ == "__main__":
    main()

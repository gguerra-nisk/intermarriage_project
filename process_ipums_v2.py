"""
Immigrant Intermarriage Data Processing Script v2.0
====================================================
- Comprehensive IPUMS birthplace code mappings
- Spouse-pair data for Sankey visualizations

Usage: python process_ipums_v2.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time

import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("IMMIGRANT INTERMARRIAGE DATA PROCESSOR v2.0")
print("="*70)

# =============================================================================
# CONFIGURATION
# =============================================================================
RAW_DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 500000

# =============================================================================
# COMPREHENSIVE IPUMS BPL CODE MAPPINGS
# =============================================================================

COUNTRY_CODES = {
    # US Territories (100-199)
    100: "Alaska (territory)", 105: "Hawaii (territory)", 110: "Puerto Rico",
    115: "US Virgin Islands", 120: "Guam", 150: "Canada",
    155: "St. Pierre and Miquelon", 160: "Atlantic Islands",
    
    # Mexico, Central America, Caribbean (200-299)
    200: "Mexico",
    210: "Central America (ns)", 211: "Belize", 212: "Costa Rica",
    213: "El Salvador", 214: "Guatemala", 215: "Honduras",
    216: "Nicaragua", 217: "Panama", 218: "Canal Zone",
    250: "Cuba", 260: "West Indies (ns)", 261: "Dominican Republic",
    262: "Haiti", 263: "Jamaica", 264: "Puerto Rico", 265: "Bahamas",
    266: "Barbados", 267: "Trinidad and Tobago", 268: "Bermuda",
    269: "Other Caribbean", 299: "Americas (ns)",
    
    # South America (300-399)
    300: "South America (ns)", 310: "Argentina", 315: "Bolivia",
    320: "Brazil", 325: "Chile", 330: "Colombia", 335: "Ecuador",
    340: "Guyana", 345: "Paraguay", 350: "Peru", 355: "Uruguay",
    360: "Venezuela", 365: "Suriname",
    
    # Europe - British Isles (400-419)
    400: "England", 401: "Scotland", 402: "Wales",
    403: "United Kingdom (ns)", 404: "Northern Ireland",
    405: "United Kingdom (ns)", 410: "Ireland",
    411: "Channel Islands", 412: "Isle of Man", 413: "United Kingdom (ns)",
    
    # Europe - Germany & neighbors (420-429)
    420: "Germany", 421: "Germany", 422: "Germany", 423: "Germany",
    424: "Germany", 425: "Austria", 426: "Switzerland", 429: "Germany",
    
    # Europe - Scandinavia & Low Countries (430-439)
    430: "Sweden", 431: "Norway", 432: "Denmark", 433: "Finland",
    434: "Iceland", 435: "Belgium", 436: "Netherlands",
    437: "Luxembourg", 438: "France", 439: "Monaco",
    
    # Europe - France (440-449)
    440: "France",
    
    # Europe - Central/Eastern (450-459)
    450: "Austria-Hungary (ns)", 451: "Hungary", 452: "Czechoslovakia",
    453: "Poland", 454: "Bulgaria", 455: "Romania", 456: "Albania",
    457: "Yugoslavia", 458: "North Macedonia", 459: "Serbia",
    
    # Europe - Italy & Mediterranean (460-469)
    460: "Italy", 461: "Italy", 462: "Italy", 463: "Italy",
    465: "Spain", 466: "Andorra", 467: "Portugal", 468: "Greece", 469: "Malta",
    
    # Europe - Eastern/Russia (480-499)
    480: "Russia/USSR", 481: "Russia", 482: "Ukraine", 483: "Belarus",
    484: "Latvia", 485: "Lithuania", 486: "Estonia", 487: "Moldova",
    489: "USSR (ns)", 490: "Europe (ns)", 499: "Europe (ns)",
    
    # Asia - East (500-519)
    500: "China", 501: "Japan", 502: "Korea", 503: "North Korea",
    504: "South Korea", 505: "Hong Kong", 506: "Taiwan", 507: "Macau",
    509: "Taiwan", 510: "Mongolia", 511: "Hong Kong", 512: "Macau",
    513: "Myanmar/Burma", 514: "Cambodia", 515: "Philippines",
    516: "Indonesia", 517: "Vietnam", 518: "Thailand", 519: "Laos",
    
    # Asia - South/Southeast (520-529)
    520: "Malaysia", 521: "India", 522: "Pakistan", 523: "Bangladesh",
    524: "Sri Lanka", 525: "Nepal", 526: "Bhutan", 527: "Singapore",
    528: "Brunei", 529: "Maldives",
    
    # Asia - Middle East (530-549)
    530: "Iran", 531: "Iraq", 532: "Israel", 533: "Palestine",
    534: "Syria", 535: "Turkey", 536: "Saudi Arabia", 537: "Iran",
    538: "Cyprus", 539: "Kuwait", 540: "Jordan", 541: "Lebanon",
    542: "Middle East (ns)", 543: "Afghanistan", 544: "Egypt",
    545: "Yemen", 546: "United Arab Emirates", 547: "Qatar",
    548: "Bahrain", 549: "Oman", 599: "Asia (ns)",
    
    # Africa (600-699)
    600: "Africa (ns)", 601: "North Africa (ns)", 610: "Egypt",
    611: "Sudan", 612: "Libya", 613: "Tunisia", 614: "Algeria",
    615: "Morocco", 620: "West Africa (ns)", 621: "Nigeria",
    622: "Ghana", 623: "Senegal", 624: "Liberia", 625: "Sierra Leone",
    626: "Cameroon", 627: "Ivory Coast", 630: "East Africa (ns)",
    631: "Ethiopia", 632: "Eritrea", 633: "Kenya", 634: "Somalia",
    635: "Tanzania", 636: "Uganda", 640: "Central Africa (ns)",
    641: "Congo", 642: "Zaire/DRC", 650: "South Africa",
    651: "Zimbabwe", 652: "Zambia", 653: "Botswana", 699: "Africa (ns)",
    
    # Oceania (700-799)
    700: "Australia", 710: "New Zealand", 720: "Oceania (ns)",
    730: "Fiji", 740: "Tonga", 750: "Samoa",
    
    # Special (900-999)
    900: "Born abroad (US parents)", 950: "At sea", 999: "Unknown",
}

REGION_CODES = {
    # US Territories
    100: "US Territory", 105: "US Territory", 110: "US Territory",
    115: "US Territory", 120: "US Territory",
    
    # North America
    150: "Canada", 155: "Canada", 160: "Atlantic Islands",
    
    # Mexico
    200: "Mexico",
    
    # Central America & Caribbean
    210: "Central America/Caribbean", 211: "Central America/Caribbean",
    212: "Central America/Caribbean", 213: "Central America/Caribbean",
    214: "Central America/Caribbean", 215: "Central America/Caribbean",
    216: "Central America/Caribbean", 217: "Central America/Caribbean",
    218: "Central America/Caribbean", 250: "Central America/Caribbean",
    260: "Central America/Caribbean", 261: "Central America/Caribbean",
    262: "Central America/Caribbean", 263: "Central America/Caribbean",
    264: "Central America/Caribbean", 265: "Central America/Caribbean",
    266: "Central America/Caribbean", 267: "Central America/Caribbean",
    268: "Central America/Caribbean", 269: "Central America/Caribbean",
    299: "Americas (ns)",
    
    # South America
    300: "South America", 310: "South America", 315: "South America",
    320: "South America", 325: "South America", 330: "South America",
    335: "South America", 340: "South America", 345: "South America",
    350: "South America", 355: "South America", 360: "South America",
    365: "South America",
    
    # UK & Ireland
    400: "United Kingdom", 401: "United Kingdom", 402: "United Kingdom",
    403: "United Kingdom", 404: "United Kingdom", 405: "United Kingdom",
    410: "Ireland", 411: "United Kingdom", 412: "United Kingdom",
    413: "United Kingdom",
    
    # Germany & Central Europe
    420: "Germany", 421: "Germany", 422: "Germany", 423: "Germany",
    424: "Germany", 425: "Austria", 426: "Switzerland", 429: "Germany",
    
    # Scandinavia
    430: "Scandinavia", 431: "Scandinavia", 432: "Scandinavia",
    433: "Scandinavia", 434: "Scandinavia",
    
    # Western Europe
    435: "Western Europe", 436: "Western Europe", 437: "Western Europe",
    438: "France", 439: "France", 440: "France",
    
    # Central/Eastern Europe
    450: "Central Europe", 451: "Central Europe", 452: "Central Europe",
    453: "Poland", 454: "Central Europe", 455: "Central Europe",
    456: "Central Europe", 457: "Central Europe", 458: "Central Europe",
    459: "Central Europe",
    
    # Southern Europe
    460: "Italy", 461: "Italy", 462: "Italy", 463: "Italy",
    465: "Spain/Portugal", 466: "Spain/Portugal", 467: "Spain/Portugal",
    468: "Greece", 469: "Southern Europe",
    
    # Eastern Europe / Russia
    480: "Russia/USSR", 481: "Russia/USSR", 482: "Russia/USSR",
    483: "Russia/USSR", 484: "Russia/USSR", 485: "Russia/USSR",
    486: "Russia/USSR", 487: "Russia/USSR", 489: "Russia/USSR",
    490: "Europe (Other)", 499: "Europe (Other)",
    
    # East Asia
    500: "China", 501: "Japan", 502: "Korea", 503: "Korea", 504: "Korea",
    505: "China", 506: "China", 507: "China", 509: "China",
    510: "East Asia", 511: "China", 512: "China",
    
    # Southeast Asia
    513: "Southeast Asia", 514: "Southeast Asia", 515: "Philippines",
    516: "Southeast Asia", 517: "Vietnam", 518: "Southeast Asia",
    519: "Southeast Asia", 520: "Southeast Asia",
    
    # South Asia
    521: "India", 522: "South Asia", 523: "South Asia", 524: "South Asia",
    525: "South Asia", 526: "South Asia", 527: "Southeast Asia",
    528: "Southeast Asia", 529: "South Asia",
    
    # Middle East
    530: "Middle East", 531: "Middle East", 532: "Middle East",
    533: "Middle East", 534: "Middle East", 535: "Middle East",
    536: "Middle East", 537: "Middle East", 538: "Middle East",
    539: "Middle East", 540: "Middle East", 541: "Middle East",
    542: "Middle East", 543: "Middle East", 544: "Middle East",
    545: "Middle East", 546: "Middle East", 547: "Middle East",
    548: "Middle East", 549: "Middle East", 599: "Asia (Other)",
    
    # Africa
    600: "Africa", 601: "Africa", 610: "Africa", 611: "Africa",
    612: "Africa", 613: "Africa", 614: "Africa", 615: "Africa",
    620: "Africa", 621: "Africa", 622: "Africa", 623: "Africa",
    624: "Africa", 625: "Africa", 626: "Africa", 627: "Africa",
    630: "Africa", 631: "Africa", 632: "Africa", 633: "Africa",
    634: "Africa", 635: "Africa", 636: "Africa", 640: "Africa",
    641: "Africa", 642: "Africa", 650: "Africa", 651: "Africa",
    652: "Africa", 653: "Africa", 699: "Africa",
    
    # Oceania
    700: "Oceania", 710: "Oceania", 720: "Oceania", 730: "Oceania",
    740: "Oceania", 750: "Oceania",
    
    # Special
    900: "Born Abroad (US parents)", 950: "At Sea", 999: "Unknown",
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def map_to_region(bpl):
    if pd.isna(bpl):
        return "Unknown"
    bpl = int(bpl)
    if 1 <= bpl <= 56 or bpl == 99:
        return "United States"
    if bpl in REGION_CODES:
        return REGION_CODES[bpl]
    # Fallback by range
    if 100 <= bpl <= 199: return "US Territory"
    if 200 <= bpl <= 299: return "Central America/Caribbean"
    if 300 <= bpl <= 399: return "South America"
    if 400 <= bpl <= 419: return "United Kingdom"
    if 420 <= bpl <= 429: return "Germany"
    if 430 <= bpl <= 439: return "Scandinavia"
    if 440 <= bpl <= 449: return "Western Europe"
    if 450 <= bpl <= 459: return "Central Europe"
    if 460 <= bpl <= 469: return "Southern Europe"
    if 470 <= bpl <= 499: return "Eastern Europe"
    if 500 <= bpl <= 529: return "Asia"
    if 530 <= bpl <= 599: return "Middle East/Asia"
    if 600 <= bpl <= 699: return "Africa"
    if 700 <= bpl <= 799: return "Oceania"
    return "Unknown"


def map_to_country(bpl):
    if pd.isna(bpl):
        return "Unknown"
    bpl = int(bpl)
    if 1 <= bpl <= 56 or bpl == 99:
        return "United States"
    if bpl in COUNTRY_CODES:
        return COUNTRY_CODES[bpl]
    return f"Code {bpl}"


def get_spouse_category(bpl_sp):
    """Categorize spouse for Sankey: US-born or specific country."""
    if pd.isna(bpl_sp):
        return "Unknown"
    bpl_sp = int(bpl_sp)
    if 1 <= bpl_sp <= 56 or bpl_sp == 99:
        return "US-Born"
    return map_to_country(bpl_sp)


def classify_marriage(row):
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
    return "Different Region"


def process_chunk(chunk):
    df_married = chunk[(chunk['MARST'] == 1) & (chunk['SPLOC'] > 0)].copy()
    if len(df_married) == 0:
        return None
    
    df_married['REGION'] = df_married['BPL'].apply(map_to_region)
    df_married['REGION_SP'] = df_married['BPL_SP'].apply(map_to_region)
    df_married['COUNTRY'] = df_married['BPL'].apply(map_to_country)
    df_married['COUNTRY_SP'] = df_married['BPL_SP'].apply(map_to_country)
    df_married['SPOUSE_CATEGORY'] = df_married['BPL_SP'].apply(get_spouse_category)
    df_married['MARRIAGE_TYPE'] = df_married.apply(classify_marriage, axis=1)
    
    df_immigrants = df_married[~df_married['MARRIAGE_TYPE'].isin(["Native-born", "Unknown"])]
    return df_immigrants


# =============================================================================
# MAIN PROCESSING
# =============================================================================

# Step 1: Find data
print("\n[Step 1/7] Looking for IPUMS data file...")
data_files = list(RAW_DATA_DIR.glob("*.csv.gz")) + list(RAW_DATA_DIR.glob("*.csv"))
if not data_files:
    print("❌ ERROR: No data file found!")
    sys.exit(1)
DATA_FILE = data_files[0]
print(f"   ✓ Found: {DATA_FILE.name}")

# Step 2: Get columns
print("\n[Step 2/7] Identifying columns...")
sample_df = pd.read_csv(DATA_FILE, nrows=5)
all_columns = sample_df.columns.tolist()
desired_cols = [
    'YEAR', 'SERIAL', 'PERNUM', 'PERWT', 'AGE', 'SEX', 'MARST', 'SPLOC',
    'BPL', 'BPLD', 'NATIVITY', 'RACE', 'HISPAN',
    'BPL_SP', 'BPLD_SP', 'NATIVITY_SP', 'RACE_SP', 'HISPAN_SP'
]
use_cols = [c for c in desired_cols if c in all_columns]
print(f"   ✓ Using {len(use_cols)} columns")

# Step 3: Process chunks
print("\n[Step 3/7] Processing data in chunks...")
all_chunks = []
total_rows = 0
immigrant_rows = 0
chunk_num = 0
start_time = time.time()

for chunk in pd.read_csv(DATA_FILE, usecols=use_cols, chunksize=CHUNK_SIZE, low_memory=False):
    chunk_num += 1
    total_rows += len(chunk)
    processed = process_chunk(chunk)
    if processed is not None and len(processed) > 0:
        immigrant_rows += len(processed)
        all_chunks.append(processed)
    elapsed = time.time() - start_time
    print(f"   Chunk {chunk_num}: {total_rows:,} rows, {immigrant_rows:,} immigrant marriages ({elapsed:.0f}s)")

print(f"\n   ✓ Processed {total_rows:,} total rows")
print(f"   ✓ Found {immigrant_rows:,} immigrant marriages")

# Step 4: Combine
print("\n[Step 4/7] Combining data...")
df_immigrants = pd.concat(all_chunks, ignore_index=True)
del all_chunks
print(f"   ✓ Combined {len(df_immigrants):,} records")

# Step 5: Compute standard statistics
print("\n[Step 5/7] Computing standard statistics...")

def compute_stats(group):
    total_weight = group['PERWT'].sum()
    n = len(group)
    if total_weight == 0:
        return pd.Series({
            'n_unweighted': n, 'total_weighted': 0,
            'pct_married_us_born': 0, 'pct_same_country': 0,
            'pct_same_region': 0, 'pct_different_region': 0
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

print("   By year...")
stats_by_year = df_immigrants.groupby('YEAR').apply(compute_stats).reset_index()

print("   By year and region...")
stats_by_year_region = df_immigrants.groupby(['YEAR', 'REGION']).apply(compute_stats).reset_index()

print("   By year and country...")
stats_by_year_country = df_immigrants.groupby(['YEAR', 'COUNTRY']).apply(compute_stats).reset_index()
stats_by_year_country = stats_by_year_country[stats_by_year_country['n_unweighted'] >= 50]

# Step 6: Compute spouse-pair data for Sankey
print("\n[Step 6/7] Computing spouse-pair data for Sankey...")

# Group by year, origin country, and spouse category
spouse_pairs = df_immigrants.groupby(['YEAR', 'COUNTRY', 'SPOUSE_CATEGORY']).agg({
    'PERWT': 'sum',
    'SERIAL': 'count'
}).reset_index()
spouse_pairs.columns = ['YEAR', 'ORIGIN', 'SPOUSE', 'WEIGHTED_COUNT', 'UNWEIGHTED_COUNT']

# Filter to pairs with enough observations
spouse_pairs = spouse_pairs[spouse_pairs['UNWEIGHTED_COUNT'] >= 20]

# Compute percentages within each origin-year
spouse_pairs['TOTAL_BY_ORIGIN'] = spouse_pairs.groupby(['YEAR', 'ORIGIN'])['WEIGHTED_COUNT'].transform('sum')
spouse_pairs['PERCENTAGE'] = (spouse_pairs['WEIGHTED_COUNT'] / spouse_pairs['TOTAL_BY_ORIGIN'] * 100).round(2)

print(f"   ✓ Generated {len(spouse_pairs):,} origin-spouse pair records")

# Step 7: Save all files
print("\n[Step 7/7] Saving files...")

stats_by_year.to_csv(PROCESSED_DIR / "intermarriage_by_year.csv", index=False)
print(f"   ✓ intermarriage_by_year.csv")

stats_by_year_region.to_csv(PROCESSED_DIR / "intermarriage_by_year_region.csv", index=False)
print(f"   ✓ intermarriage_by_year_region.csv")

stats_by_year_country.to_csv(PROCESSED_DIR / "intermarriage_by_year_country.csv", index=False)
print(f"   ✓ intermarriage_by_year_country.csv")

spouse_pairs.to_csv(PROCESSED_DIR / "spouse_pairs.csv", index=False)
print(f"   ✓ spouse_pairs.csv (for Sankey diagrams)")

# Save sample
sample_size = min(5000, len(df_immigrants))
df_sample = df_immigrants.sample(n=sample_size, random_state=42)
df_sample.to_csv(PROCESSED_DIR / "immigrant_marriages_sample.csv", index=False)
print(f"   ✓ immigrant_marriages_sample.csv")

# =============================================================================
# SUMMARY
# =============================================================================
elapsed_total = time.time() - start_time

print("\n" + "="*70)
print("✅ PROCESSING COMPLETE!")
print("="*70)

print(f"\nTime: {elapsed_total/60:.1f} minutes")
print(f"Total rows: {total_rows:,}")
print(f"Immigrant marriages: {len(df_immigrants):,}")

print(f"\nTop 20 origin countries:")
country_counts = df_immigrants['COUNTRY'].value_counts().head(20)
for country, count in country_counts.items():
    print(f"   {country}: {count:,}")

# Check unmapped
unmapped = df_immigrants[df_immigrants['COUNTRY'].str.startswith('Code')]['COUNTRY'].unique()
if len(unmapped) > 0:
    print(f"\n⚠️  Still unmapped codes: {list(unmapped)[:10]}")
else:
    print(f"\n✓ All country codes mapped!")

print("\n" + "="*70)
print("FILES CREATED:")
print("="*70)
print(f"   {PROCESSED_DIR / 'intermarriage_by_year.csv'}")
print(f"   {PROCESSED_DIR / 'intermarriage_by_year_region.csv'}")
print(f"   {PROCESSED_DIR / 'intermarriage_by_year_country.csv'}")
print(f"   {PROCESSED_DIR / 'spouse_pairs.csv'}")
print(f"   {PROCESSED_DIR / 'immigrant_marriages_sample.csv'}")

print("\n" + "="*70)
print("NEXT STEP:")
print("="*70)
print('   Run: python scripts/run_dashboard.py')
print("   Open: http://127.0.0.1:8050")
print("="*70)

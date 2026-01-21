"""
Data Validation Script - Check for anomalies in processed intermarriage data
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/processed")

print("="*70)
print("DATA VALIDATION REPORT")
print("="*70)

# Load all data files
df_year = pd.read_csv(DATA_DIR / "intermarriage_by_year.csv")
df_region = pd.read_csv(DATA_DIR / "intermarriage_by_year_region.csv")
df_country = pd.read_csv(DATA_DIR / "intermarriage_by_year_country.csv")

# =============================================================================
# CHECK 1: Percentages should sum to ~100%
# =============================================================================
print("\n" + "="*70)
print("CHECK 1: Do percentages sum to 100%?")
print("="*70)

df_year['total_pct'] = (df_year['pct_married_us_born'] + 
                        df_year['pct_same_country'] + 
                        df_year['pct_same_region'] + 
                        df_year['pct_different_region'])

print("\nBy Year - Percentage totals:")
print(df_year[['YEAR', 'total_pct']].to_string(index=False))

off_by = df_year[abs(df_year['total_pct'] - 100) > 1]
if len(off_by) > 0:
    print("\n⚠️  WARNING: These years don't sum to ~100%:")
    print(off_by[['YEAR', 'total_pct']])
else:
    print("\n✓ All years sum to 100% (within rounding)")

# =============================================================================
# CHECK 2: Sample sizes by year
# =============================================================================
print("\n" + "="*70)
print("CHECK 2: Sample sizes by year")
print("="*70)

print("\nSample sizes:")
print(df_year[['YEAR', 'n_unweighted', 'total_weighted']].to_string(index=False))

small_samples = df_year[df_year['n_unweighted'] < 1000]
if len(small_samples) > 0:
    print("\n⚠️  WARNING: These years have small samples (<1000):")
    print(small_samples[['YEAR', 'n_unweighted']])

# =============================================================================
# CHECK 3: Look for suspicious values
# =============================================================================
print("\n" + "="*70)
print("CHECK 3: Checking for suspicious values")
print("="*70)

# Check for negative percentages
for col in ['pct_married_us_born', 'pct_same_country', 'pct_same_region', 'pct_different_region']:
    neg = df_year[df_year[col] < 0]
    if len(neg) > 0:
        print(f"\n⚠️  WARNING: Negative values in {col}")
        print(neg)
    
    over100 = df_year[df_year[col] > 100]
    if len(over100) > 0:
        print(f"\n⚠️  WARNING: Values over 100% in {col}")
        print(over100)

print("\n✓ No negative or >100% values found")

# =============================================================================
# CHECK 4: Year coverage
# =============================================================================
print("\n" + "="*70)
print("CHECK 4: Year coverage")
print("="*70)

years = sorted(df_year['YEAR'].unique())
print(f"\nYears in dataset: {years}")
print(f"Total years: {len(years)}")

expected_years = [1850, 1860, 1870, 1880, 1900, 1910, 1920, 1930, 1940, 
                  1970, 1980, 1990, 2000, 2010, 2015, 2020, 2023]
missing = [y for y in expected_years if y not in years]
if missing:
    print(f"\n⚠️  Missing expected years: {missing}")
else:
    print("\n✓ All expected years present")

# =============================================================================
# CHECK 5: Trend sanity check
# =============================================================================
print("\n" + "="*70)
print("CHECK 5: Historical trends (sanity check)")
print("="*70)

print("\nIntermarriage with US-born over time:")
print("-" * 50)
for _, row in df_year.sort_values('YEAR').iterrows():
    bar = "█" * int(row['pct_married_us_born'] / 2)
    print(f"{int(row['YEAR'])}: {row['pct_married_us_born']:5.1f}% {bar}")

print("\nExpected pattern: Lower intermarriage in early years (ethnic enclaves),")
print("increasing over time as assimilation occurs.")

# =============================================================================
# CHECK 6: Region breakdown
# =============================================================================
print("\n" + "="*70)
print("CHECK 6: Regions in dataset")
print("="*70)

regions = df_region['REGION'].unique()
print(f"\nUnique regions ({len(regions)}):")
for r in sorted(regions):
    count = len(df_region[df_region['REGION'] == r])
    print(f"  - {r} ({count} year-observations)")

# =============================================================================
# CHECK 7: Country breakdown
# =============================================================================
print("\n" + "="*70)
print("CHECK 7: Countries in dataset")
print("="*70)

countries = df_country['COUNTRY'].unique()
print(f"\nUnique countries ({len(countries)}):")
for c in sorted(countries):
    total_n = df_country[df_country['COUNTRY'] == c]['n_unweighted'].sum()
    print(f"  - {c}: {total_n:,} total observations")

# Check for "Code XXX" entries (unmapped countries)
unmapped = [c for c in countries if 'Code' in str(c)]
if unmapped:
    print(f"\n⚠️  Unmapped country codes: {unmapped}")

# =============================================================================
# CHECK 8: Latest year detailed breakdown
# =============================================================================
print("\n" + "="*70)
print("CHECK 8: Latest year detailed breakdown")
print("="*70)

latest = df_year['YEAR'].max()
latest_data = df_year[df_year['YEAR'] == latest].iloc[0]

print(f"\nYear: {int(latest)}")
print(f"Sample size: {int(latest_data['n_unweighted']):,}")
print(f"\nMarriage patterns:")
print(f"  Married US-born:    {latest_data['pct_married_us_born']:.1f}%")
print(f"  Same country:       {latest_data['pct_same_country']:.1f}%")
print(f"  Same region:        {latest_data['pct_same_region']:.1f}%")
print(f"  Different region:   {latest_data['pct_different_region']:.1f}%")
print(f"  TOTAL:              {latest_data['pct_married_us_born'] + latest_data['pct_same_country'] + latest_data['pct_same_region'] + latest_data['pct_different_region']:.1f}%")

# =============================================================================
# CHECK 9: Compare early vs late periods
# =============================================================================
print("\n" + "="*70)
print("CHECK 9: Early vs Modern comparison")
print("="*70)

early = df_year[df_year['YEAR'] <= 1920]
modern = df_year[df_year['YEAR'] >= 2000]

if len(early) > 0 and len(modern) > 0:
    print("\n                        Early (≤1920)    Modern (≥2000)")
    print("-" * 55)
    print(f"Avg % Married US-born:  {early['pct_married_us_born'].mean():10.1f}%    {modern['pct_married_us_born'].mean():10.1f}%")
    print(f"Avg % Same Country:     {early['pct_same_country'].mean():10.1f}%    {modern['pct_same_country'].mean():10.1f}%")
    print(f"Total sample:           {int(early['n_unweighted'].sum()):>10,}    {int(modern['n_unweighted'].sum()):>10,}")

print("\n" + "="*70)
print("VALIDATION COMPLETE")
print("="*70)

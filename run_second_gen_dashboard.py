"""
Second-Generation Marriage Patterns Dashboard v10.3
====================================================
Bold visual design with Niskanen Center brand guidelines:
- Fonts: Neuton (headings), Hanken Grotesk (body)
- Full brand color palette with teals, golds, oranges, greens, purples
- Dramatic dark header hero banner
- Bold shadows and strong visual hierarchy

v10.3 Changes:
- Restricted to 1880-1930 censuses (reliable spouse heritage data)
- Excluded 1940-1960 censuses (missing spouse parent birthplace info)
- Filtered small sample same-origin combinations from presets
- Added methodology note explaining census data limitations

Usage: python run_second_gen_dashboard.py
Open: http://127.0.0.1:8050
"""

import pandas as pd
import numpy as np
from pathlib import Path
from urllib.parse import urlencode, parse_qs
import json

import dash
from dash import dcc, html, callback, Input, Output, State, ctx, no_update
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go

# =============================================================================
# NISKANEN CENTER BRAND COLORS - FULL PALETTE
# =============================================================================

COLORS = {
    # Primary Teals (main brand colors)
    'dark_teal': '#194852',
    'medium_teal': '#348397',
    'light_teal': '#7dceda',
    'very_light_teal': '#b7f6fc',
    'very_dark_teal': '#0c2a30',
    'muted_teal': '#78a0a3',

    # Neutrals
    'light_gray': '#d0dbdd',
    'very_light_gray': '#edf1f2',
    'white': '#ffffff',

    # Golds (for data viz)
    'dark_gold': '#52482a',
    'gold': '#bca45e',
    'light_gold': '#f4da91',
    'very_light_gold': '#fef0c7',

    # Oranges (for data viz)
    'dark_orange': '#411b08',
    'medium_orange': '#8d381c',
    'orange': '#da5831',
    'light_orange': '#f17d3a',

    # Greens (for data viz)
    'dark_green': '#2c3811',
    'green': '#709628',
    'light_green': '#b1d955',
    'very_light_green': '#d7f881',

    # Purples (for data viz)
    'dark_purple': '#503961',
    'purple': '#8655b2',
    'light_purple': '#ba88ef',
    'very_light_purple': '#e0c6fc',
}

# Chart colors for marriage categories
MARRIAGE_COLORS = {
    'Married same origin (1st gen immigrant)': COLORS['dark_teal'],
    'Married same origin (2nd gen)': COLORS['medium_teal'],
    "Married mother's origin (1st gen immigrant)": COLORS['purple'],
    "Married mother's origin (2nd gen)": COLORS['light_purple'],
    "Married father's origin (1st gen immigrant)": COLORS['orange'],
    "Married father's origin (2nd gen)": COLORS['light_orange'],
    'Married someone sharing both heritages (1st gen immigrant)': COLORS['green'],
    'Married someone sharing both heritages (2nd gen)': COLORS['light_green'],
    'Married different origin (1st gen immigrant)': COLORS['gold'],
    'Married different origin (2nd gen)': COLORS['light_gold'],
    'Married 3rd+ gen American': COLORS['light_teal'],
}

# Simplified category colors for legends
CATEGORY_COLORS = {
    '3rd+ gen American': COLORS['light_teal'],
    'Same origin': COLORS['dark_teal'],
    "Mother's origin": COLORS['purple'],
    "Father's origin": COLORS['orange'],
    'Both heritages': COLORS['green'],
    'Different origin': COLORS['gold'],
}

# =============================================================================
# DEMONYMS - Convert country names to proper adjective forms
# =============================================================================

DEMONYMS = {
    'Ireland': 'Irish',
    'Germany': 'German',
    'Italy': 'Italian',
    'Poland': 'Polish',
    'England': 'English',
    'Scotland': 'Scottish',
    'Wales': 'Welsh',
    'France': 'French',
    'Russia/USSR': 'Russian',
    'Austria': 'Austrian',
    'Hungary': 'Hungarian',
    'Sweden': 'Swedish',
    'Norway': 'Norwegian',
    'Denmark': 'Danish',
    'Netherlands': 'Dutch',
    'Belgium': 'Belgian',
    'Switzerland': 'Swiss',
    'Czechoslovakia': 'Czechoslovak',
    'Yugoslavia': 'Yugoslav',
    'Greece': 'Greek',
    'Portugal': 'Portuguese',
    'Spain': 'Spanish',
    'Canada': 'Canadian',
    'Mexico': 'Mexican',
    'Cuba': 'Cuban',
    'China': 'Chinese',
    'Japan': 'Japanese',
    'Philippines': 'Filipino',
    'India': 'Indian',
    'Finland': 'Finnish',
    'Lithuania': 'Lithuanian',
    'Romania': 'Romanian',
    'Luxembourg': 'Luxembourgish',
    'Puerto Rico': 'Puerto Rican',
    'West Indies': 'West Indian',
    'Syria': 'Syrian',
    'Turkey': 'Turkish',
    'Latvia': 'Latvian',
    'Iceland': 'Icelandic',
    'Albania': 'Albanian',
    'Lebanon': 'Lebanese',
    'Bulgaria': 'Bulgarian',
    'Korea': 'Korean',
    'Estonia': 'Estonian',
    'Australia/New Zealand': 'Australian/New Zealand',
    'South America': 'South American',
    'Central America': 'Central American',
    'Africa': 'African',
    'US-born': 'American',
}

def get_demonym(country):
    """Get the proper adjective form for a country name."""
    return DEMONYMS.get(country, country)

# =============================================================================
# EXCLUDED CATEGORIES
# =============================================================================

EXCLUDED_ORIGINS = {
    'Abroad/At Sea', 'Missing', 'Unknown', 'N/A',
    'Europe (unspecified)', 'Asia (unspecified)', 'Central Europe (unspecified)',
    'Asia Minor (unspecified)', 'Southwest Asia (unspecified)', 'United Kingdom (unspecified)',
    'Pacific Islands', 'Atlantic Islands', 'Other US Possessions',
    'Gibraltar', 'Liechtenstein', 'Malta', 'Guam', 'US Virgin Islands',
    'Indonesia', 'Thailand', 'Iran',
}

MIN_SAMPLE_SIZE = 20000

# =============================================================================
# DATA LOADING
# =============================================================================

PROCESSED_DIR = Path("data/processed")

def load_data():
    data = {}
    try:
        data['main'] = pd.read_csv(PROCESSED_DIR / "second_gen_marriages.csv", low_memory=False)
        data['main']['YEAR'] = data['main']['YEAR'].astype(int)
        # Filter to only include 1880-1930 census years
        # Later censuses (1940, 1950, 1960) only collected parent birthplace for sample-line
        # persons, making spouse heritage analysis unreliable
        VALID_YEARS = [1880, 1900, 1910, 1920, 1930]
        data['main'] = data['main'][data['main']['YEAR'].isin(VALID_YEARS)]
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    return data

print("Loading data...")
DATA = load_data()

if DATA is None:
    print("ERROR: Run process_second_gen.py first")
    exit(1)

def get_valid_origins(column):
    df = DATA['main']
    origin_counts = df.groupby(column)['PERWT'].sum()
    valid = origin_counts[origin_counts >= MIN_SAMPLE_SIZE].index.tolist()
    valid = [x for x in valid if x not in EXCLUDED_ORIGINS]
    return sorted(valid)

# Get origins valid in each column
_mother_origins = set(get_valid_origins('MOTHER_ORIGIN'))
_father_origins = set(get_valid_origins('FATHER_ORIGIN'))

# Use intersection to ensure consistency - an origin must have sufficient
# sample size in BOTH columns to appear in either dropdown
valid_origins = sorted(_mother_origins & _father_origins)

# Log excluded origins for transparency
_excluded_from_mother = _father_origins - _mother_origins
_excluded_from_father = _mother_origins - _father_origins
if _excluded_from_mother or _excluded_from_father:
    print(f"Origins excluded (insufficient sample in one column):")
    if _excluded_from_mother:
        print(f"  Not enough as mother's origin: {sorted(_excluded_from_mother)}")
    if _excluded_from_father:
        print(f"  Not enough as father's origin: {sorted(_excluded_from_father)}")

mother_origins = valid_origins
father_origins = valid_origins
years = sorted(DATA['main']['YEAR'].unique())

def get_origin_label(origin):
    """Get display label for an origin (US-born -> American)."""
    if origin == 'US-born':
        return 'American'
    return origin

def build_origin_dropdown_options(origins):
    """Build dropdown options with foreign countries and American in separate sections."""
    foreign = [o for o in origins if o != 'US-born']
    has_american = 'US-born' in origins

    options = [{'label': 'Any Foreign Country', 'value': 'Any'}]
    options.append({'label': '── Foreign Countries ──', 'value': '', 'disabled': True})
    options.extend([{'label': c, 'value': c} for c in foreign])

    if has_american:
        options.append({'label': '── Native-Born ──', 'value': '', 'disabled': True})
        options.append({'label': 'American', 'value': 'US-born'})

    return options

mother_dropdown_options = build_origin_dropdown_options(mother_origins)
father_dropdown_options = build_origin_dropdown_options(father_origins)

def generate_dynamic_presets():
    df = DATA['main']
    # Exclude US-born from presets (keep presets focused on immigrant origins)
    foreign_origins = [o for o in mother_origins if o != 'US-born']
    valid = df[df['MOTHER_ORIGIN'].isin(foreign_origins) & df['FATHER_ORIGIN'].isin(foreign_origins)]
    same_origin = valid[valid['MOTHER_ORIGIN'] == valid['FATHER_ORIGIN']]
    # Get sample sizes for same-origin combinations and filter out those below threshold
    same_origin_sizes = same_origin.groupby('MOTHER_ORIGIN')['PERWT'].sum()
    same_origin_sizes = same_origin_sizes[same_origin_sizes >= MIN_SAMPLE_SIZE]
    top_same = same_origin_sizes.nlargest(8)
    mixed_origin = valid[valid['MOTHER_ORIGIN'] != valid['FATHER_ORIGIN']].copy()
    mixed_origin['combo'] = mixed_origin['MOTHER_ORIGIN'] + '|' + mixed_origin['FATHER_ORIGIN']
    top_mixed = mixed_origin.groupby('combo')['PERWT'].sum().nlargest(6)

    presets = [{'label': 'Select a preset...', 'value': ''}]
    presets.append({'label': '── Same Origin Parents ──', 'value': '', 'disabled': True})
    for country in top_same.index:
        demonym = get_demonym(country)
        presets.append({'label': f'{demonym} + {demonym}', 'value': f'{country}|{country}'})
    presets.append({'label': '── Mixed Origin Parents ──', 'value': '', 'disabled': True})
    for combo in top_mixed.index:
        mother, father = combo.split('|')
        # Use shorter format: "Irish mom × German dad"
        presets.append({'label': f'{get_demonym(mother)} mom × {get_demonym(father)} dad', 'value': combo})
    return presets

DYNAMIC_PRESETS = generate_dynamic_presets()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_filtered_data(mother, father, year):
    df = DATA['main'].copy()
    if mother != 'Any':
        df = df[df['MOTHER_ORIGIN'] == mother]
    else:
        df = df[~df['MOTHER_ORIGIN'].isin(['US-born', 'Unknown', 'N/A', 'Abroad/At Sea', 'Missing'])]
    if father != 'Any':
        df = df[df['FATHER_ORIGIN'] == father]
    else:
        df = df[~df['FATHER_ORIGIN'].isin(['US-born', 'Unknown', 'N/A', 'Abroad/At Sea', 'Missing'])]
    if year != 'All':
        df = df[df['YEAR'] == int(year)]
    return df


def get_comparison_stats(mother_origin, father_origin, year='All'):
    """Get marriage statistics for a specific origin combination for comparison."""
    df = DATA['main'].copy()

    # Always filter out non-country values to ensure valid comparisons
    non_countries = ['US-born', 'Unknown', 'N/A', 'Abroad/At Sea', 'Missing']

    if mother_origin != 'Any':
        df = df[df['MOTHER_ORIGIN'] == mother_origin]
    else:
        df = df[~df['MOTHER_ORIGIN'].isin(non_countries)]

    if father_origin != 'Any':
        df = df[df['FATHER_ORIGIN'] == father_origin]
    else:
        df = df[~df['FATHER_ORIGIN'].isin(non_countries)]

    if year != 'All':
        df = df[df['YEAR'] == int(year)]

    if len(df) < 30:
        return None

    marriage_stats = df.groupby('MARRIAGE_TYPE')['PERWT'].sum()
    total = marriage_stats.sum()
    if total == 0:
        return None

    pcts = (marriage_stats / total * 100).to_dict()
    return {
        'third_gen': sum(v for k, v in pcts.items() if '3rd+ gen' in k),
        'same': sum(v for k, v in pcts.items() if 'same origin' in k),
        'mother': sum(v for k, v in pcts.items() if "mother's origin" in k),
        'father': sum(v for k, v in pcts.items() if "father's origin" in k),
        'both': sum(v for k, v in pcts.items() if 'both heritages' in k),
        'diff': sum(v for k, v in pcts.items() if 'different origin' in k),
        'n': total
    }


def get_trend_data(mother, father):
    """Get trend data across years for the given selection."""
    df = DATA['main'].copy()
    if mother != 'Any':
        df = df[df['MOTHER_ORIGIN'] == mother]
    else:
        df = df[~df['MOTHER_ORIGIN'].isin(['US-born', 'Unknown', 'N/A', 'Abroad/At Sea', 'Missing'])]
    if father != 'Any':
        df = df[df['FATHER_ORIGIN'] == father]
    else:
        df = df[~df['FATHER_ORIGIN'].isin(['US-born', 'Unknown', 'N/A', 'Abroad/At Sea', 'Missing'])]

    # Minimum weighted population per year to include in trends
    # Use 10,000 as threshold - smaller samples produce unreliable percentages
    MIN_TREND_SAMPLE = 10000

    trends = {}
    for yr in sorted(df['YEAR'].unique()):
        yr_df = df[df['YEAR'] == yr]
        marriage_stats = yr_df.groupby('MARRIAGE_TYPE')['PERWT'].sum()
        total = marriage_stats.sum()
        # Skip years with insufficient weighted sample size
        if total < MIN_TREND_SAMPLE:
            continue
        pcts = (marriage_stats / total * 100).to_dict()
        trends[yr] = {
            'third_gen': sum(v for k, v in pcts.items() if '3rd+ gen' in k),
            'same': sum(v for k, v in pcts.items() if 'same origin' in k),
            'ethnic_total': sum(v for k, v in pcts.items() if 'same origin' in k or "mother's origin" in k or "father's origin" in k or 'both heritages' in k),
            'sample_size': total,  # Track sample size for transparency
        }
    return trends


def get_top_spouse_backgrounds(df, exclude_heritage=None, top_n=5):
    """Get the most common spouse backgrounds from the data using efficient groupby.

    Args:
        df: Filtered dataframe
        exclude_heritage: Set of origins to exclude (e.g., the subject's heritage)
        top_n: Number of top backgrounds to return

    Returns:
        List of tuples: (background_description, percentage)
    """
    if len(df) == 0:
        return []

    if exclude_heritage is None:
        exclude_heritage = set()

    total = df['PERWT'].sum()
    results = {}

    # For 1st gen immigrant spouses - use groupby
    immigrants = df[df['SPOUSE_GENERATION'] == '1st gen immigrant']
    if len(immigrants) > 0:
        imm_by_country = immigrants.groupby('SPOUSE_COUNTRY')['PERWT'].sum()
        for country, weight in imm_by_country.items():
            if country not in exclude_heritage and country not in ['Unknown', 'N/A', 'American']:
                key = f"{get_demonym(country)} immigrant"
                results[key] = results.get(key, 0) + weight

    # For 2nd gen spouses - use groupby on parent origins
    second_gen = df[df['SPOUSE_GENERATION'] == '2nd gen']
    if len(second_gen) > 0:
        # Group by both parent origins
        grouped = second_gen.groupby(['SPOUSE_MOTHER_ORIGIN', 'SPOUSE_FATHER_ORIGIN'])['PERWT'].sum()
        for (sp_mom, sp_dad), weight in grouped.items():
            sp_mom = str(sp_mom) if pd.notna(sp_mom) else 'Unknown'
            sp_dad = str(sp_dad) if pd.notna(sp_dad) else 'Unknown'

            invalid = {'Unknown', 'N/A', 'US-born'}

            if sp_mom == sp_dad and sp_mom not in exclude_heritage and sp_mom not in invalid:
                key = f"2nd-gen {get_demonym(sp_mom)}"
            elif sp_mom not in invalid and sp_dad not in invalid:
                # Mixed heritage spouse
                mom_excluded = sp_mom in exclude_heritage
                dad_excluded = sp_dad in exclude_heritage
                if mom_excluded and dad_excluded:
                    continue
                elif mom_excluded:
                    key = f"2nd-gen {get_demonym(sp_dad)}"
                elif dad_excluded:
                    key = f"2nd-gen {get_demonym(sp_mom)}"
                else:
                    # Sort for consistency
                    origins = sorted([get_demonym(sp_mom), get_demonym(sp_dad)])
                    key = f"2nd-gen {origins[0]}×{origins[1]}"
            elif sp_mom not in invalid and sp_mom not in exclude_heritage:
                key = f"2nd-gen {get_demonym(sp_mom)}"
            elif sp_dad not in invalid and sp_dad not in exclude_heritage:
                key = f"2nd-gen {get_demonym(sp_dad)}"
            else:
                continue

            results[key] = results.get(key, 0) + weight

    # Convert to list, sort, and return top N with percentages
    result_list = [(key, weight / total * 100) for key, weight in results.items()]
    result_list.sort(key=lambda x: x[1], reverse=True)
    return result_list[:top_n]


def generate_summary(df, mother, father, year):
    """Generate detailed summary with trends, comparisons, and rich analysis."""
    if len(df) == 0:
        return "No data available for this selection."

    weighted_n = df['PERWT'].sum()
    unweighted_n = len(df)
    mother_dem = get_demonym(mother) if mother != 'Any' else None
    father_dem = get_demonym(father) if father != 'Any' else None

    # Build subject descriptions
    if mother != 'Any' and father != 'Any':
        if mother == father:
            subject = f"children of two **{mother_dem}** immigrant parents"
            subject_short = f"{mother_dem}-Americans"
        elif mother == 'US-born':
            # American mother, immigrant father
            subject = f"children of **{father_dem}** immigrant fathers and **American** mothers"
            subject_short = f"{father_dem}-Americans (with American mothers)"
        elif father == 'US-born':
            # Immigrant mother, American father
            subject = f"children of **{mother_dem}** immigrant mothers and **American** fathers"
            subject_short = f"{mother_dem}-Americans (with American fathers)"
        else:
            subject = f"children of **{mother_dem}** mothers and **{father_dem}** fathers"
            subject_short = f"{mother_dem}-{father_dem} Americans"
    elif mother != 'Any':
        subject = f"children of **{mother_dem}** mothers (with any foreign-born father)"
        subject_short = f"children of {mother_dem} mothers"
    elif father != 'Any':
        subject = f"children of **{father_dem}** fathers (with any foreign-born mother)"
        subject_short = f"children of {father_dem} fathers"
    else:
        subject = "all second-generation Americans (US-born children of immigrants)"
        subject_short = "second-generation Americans"

    if year != 'All':
        time_phrase = f"in **{year}**"
    else:
        min_year, max_year = int(df['YEAR'].min()), int(df['YEAR'].max())
        time_phrase = f"from **{min_year} to {max_year}**"

    lines = []
    lines.append(f"### Marriage Patterns of {subject_short.title()}")
    lines.append("")
    lines.append(f"This analysis examines {subject} {time_phrase}.")
    lines.append("")

    # Sample size warnings
    VERY_SMALL_SAMPLE = 10000
    SMALL_SAMPLE = 50000

    if unweighted_n < 30 or weighted_n < VERY_SMALL_SAMPLE:
        lines.append(f"**⚠️ CAUTION: Very small sample size** — Only {unweighted_n:,} records ({weighted_n:,.0f} weighted individuals). Results may not be statistically reliable and should be interpreted with extreme caution.")
        lines.append("")
        lines.append("*Consider selecting 'Any' for broader coverage or a different year range.*")
        return "\n".join(lines)

    if weighted_n < SMALL_SAMPLE:
        lines.append(f"*Based on {weighted_n:,.0f} individuals ({unweighted_n:,} census records)*")
        lines.append("")
        lines.append(f"**⚠️ Note: Small sample size** — Results based on fewer than 50,000 weighted individuals. Percentages may have higher margins of error.")
        lines.append("")
    else:
        lines.append(f"*Based on {weighted_n:,.0f} individuals ({unweighted_n:,} census records)*")
        lines.append("")

    # Calculate current statistics from the FILTERED dataframe
    # These values are used throughout all sections for consistency
    marriage_stats = df.groupby('MARRIAGE_TYPE')['PERWT'].sum()
    total = marriage_stats.sum()
    marriage_pcts = (marriage_stats / total * 100).to_dict()

    third_gen_pct = sum(v for k, v in marriage_pcts.items() if '3rd+ gen' in k)
    same_pct = sum(v for k, v in marriage_pcts.items() if 'same origin' in k)
    mother_pct = sum(v for k, v in marriage_pcts.items() if "mother's origin" in k)
    father_pct = sum(v for k, v in marriage_pcts.items() if "father's origin" in k)
    both_pct = sum(v for k, v in marriage_pcts.items() if 'both heritages' in k)
    diff_pct = sum(v for k, v in marriage_pcts.items() if 'different origin' in k)

    same_1st = marriage_pcts.get("Married same origin (1st gen immigrant)", 0)
    same_2nd = marriage_pcts.get("Married same origin (2nd gen)", 0)

    # Historical context for groups with disrupted patterns
    HISTORICAL_CONTEXT = {
        'Japan': {
            'warning': '⚠️ **Historical Context:** Japanese-American communities were severely disrupted by WWII internment (1942-1945), which forcibly relocated ~120,000 people from West Coast ethnic enclaves. Post-war marriage patterns reflect this displacement rather than typical assimilation trends.',
            'disruption_year': 1940
        },
        'Germany': {
            'warning': '**Historical Note:** Anti-German sentiment during WWI (1917-1918) led some German-Americans to downplay their heritage, potentially affecting marriage pattern reporting.',
            'disruption_year': 1920
        }
    }

    # ==================== FINDINGS ====================
    lines.append("---")
    lines.append("")
    lines.append("#### Findings")
    lines.append("")

    if mother != 'Any' and father != 'Any' and mother == father:
        # Check for historical disruption warnings
        if mother in HISTORICAL_CONTEXT:
            lines.append(HISTORICAL_CONTEXT[mother]['warning'])
            lines.append("")

        lines.append(f"**Ethnic retention:** {same_pct:.1f}% of {subject_short} married someone of {mother_dem} heritage:")
        if same_1st >= 0.5:
            lines.append(f"  - {same_1st:.1f}% married a {mother_dem} immigrant (born in {mother})")
        if same_2nd >= 0.5:
            lines.append(f"  - {same_2nd:.1f}% married a second-generation {mother_dem}-American")
        lines.append("")
        lines.append(f"**Mainstream assimilation:** {third_gen_pct:.1f}% married a 3rd+ generation American — someone whose parents were also US-born.")
        lines.append("")
        if diff_pct >= 0.5:
            lines.append(f"**Cross-ethnic marriage:** {diff_pct:.1f}% married someone from a different immigrant background.")
            lines.append("")

        # Add year-specific breakdown for same-origin selections
        if year == 'All':
            year_data = []
            for yr in sorted(df['YEAR'].unique()):
                yr_df = df[df['YEAR'] == yr]
                yr_total = yr_df['PERWT'].sum()
                if yr_total < 5000:  # Skip years with very small samples
                    continue
                yr_stats = yr_df.groupby('MARRIAGE_TYPE')['PERWT'].sum()
                yr_pcts = (yr_stats / yr_total * 100).to_dict()
                yr_third = sum(v for k, v in yr_pcts.items() if '3rd+ gen' in k)
                yr_same = sum(v for k, v in yr_pcts.items() if 'same origin' in k)
                year_data.append((yr, yr_third, yr_same, yr_total))

            if len(year_data) >= 2:
                # Check if there's significant variation across years
                third_vals = [d[1] for d in year_data]
                same_vals = [d[2] for d in year_data]
                third_range = max(third_vals) - min(third_vals)
                same_range = max(same_vals) - min(same_vals)

                if third_range > 15 or same_range > 15:  # Show breakdown if significant variation
                    lines.append("**Trends by census year:**")
                    lines.append("")
                    lines.append("| Year | Ethnic Retention | 3rd+ Gen American |")
                    lines.append("|------|------------------|-------------------|")
                    for yr, yr_third, yr_same, yr_total in year_data:
                        sample_note = " *(small sample)*" if yr_total < 20000 else ""
                        lines.append(f"| {yr} | {yr_same:.0f}% | {yr_third:.0f}%{sample_note} |")
                    lines.append("")

                    # Add interpretation of the trend
                    first_same = year_data[0][2]
                    last_same = year_data[-1][2]
                    if last_same < first_same - 20:
                        lines.append(f"*Ethnic retention declined substantially from {first_same:.0f}% to {last_same:.0f}% over this period, suggesting increasing integration into mainstream American society over generations.*")
                        lines.append("")
                    elif last_same > first_same + 20:
                        lines.append(f"*Ethnic retention increased from {first_same:.0f}% to {last_same:.0f}%, possibly reflecting growth of ethnic community institutions or chain migration patterns.*")
                        lines.append("")
    elif mother != 'Any' and father != 'Any' and mother != father:
        # Check if one parent is American (US-born)
        mother_is_american = mother == 'US-born'
        father_is_american = father == 'US-born'

        if mother_is_american or father_is_american:
            # One parent is American - different narrative structure
            immigrant_parent = father if mother_is_american else mother
            immigrant_dem = get_demonym(immigrant_parent)
            immigrant_pct = father_pct if mother_is_american else mother_pct
            parent_type = "father" if mother_is_american else "mother"

            lines.append(f"**{immigrant_dem} heritage:** {immigrant_pct:.1f}% married someone of {immigrant_dem} origin (connecting to their immigrant {parent_type}'s background)")
            lines.append("")
            lines.append(f"**Mainstream assimilation:** {third_gen_pct:.1f}% married a 3rd+ generation American (aligning with their American {('mother' if mother_is_american else 'father')}'s background)")
            lines.append("")
            if diff_pct >= 0.5:
                lines.append(f"**Other immigrant backgrounds:** {diff_pct:.1f}% married someone from a different immigrant community (not {immigrant_dem})")
                lines.append("")
        else:
            # Both parents are foreign-born - standard mixed heritage
            lines.append(f"**Mother's heritage ({mother_dem}):** {mother_pct:.1f}% married someone of {mother_dem} origin")
            lines.append("")
            lines.append(f"**Father's heritage ({father_dem}):** {father_pct:.1f}% married someone of {father_dem} origin")
            lines.append("")
            if both_pct >= 0.5:
                lines.append(f"**Both heritages:** {both_pct:.1f}% married someone sharing both {mother_dem} and {father_dem} ancestry")
                lines.append("")
            lines.append(f"**Mainstream assimilation:** {third_gen_pct:.1f}% married a 3rd+ generation American")
            lines.append("")
            if diff_pct >= 0.5:
                lines.append(f"**Other immigrant backgrounds:** {diff_pct:.1f}% married someone from a different immigrant community (neither {mother_dem} nor {father_dem})")
                lines.append("")
    elif mother != 'Any' or father != 'Any':
        # Partial selection: one specific origin + one "Any"
        specific_origin = mother if mother != 'Any' else father
        specific_dem = get_demonym(specific_origin)
        parent_type = "mother" if mother != 'Any' else "father"

        # Calculate heritage-based total (connection to the specific parent's heritage)
        # For partial selection, relevant_pct is the match to the specified parent
        relevant_pct = mother_pct if mother != 'Any' else father_pct
        heritage_total = same_pct + relevant_pct + both_pct

        # Determine the largest category for accurate language
        if heritage_total >= third_gen_pct and heritage_total >= diff_pct:
            lines.append(f"**{specific_dem} heritage predominated:** {heritage_total:.1f}% married someone with {specific_dem} ancestry.")
        elif third_gen_pct >= heritage_total and third_gen_pct >= diff_pct:
            lines.append(f"**Mainstream assimilation:** {third_gen_pct:.1f}% married a 3rd+ generation American.")
        else:
            lines.append(f"**Cross-ethnic patterns:** {diff_pct:.1f}% married someone from a different immigrant background.")
        lines.append("")

        # Full breakdown of all categories
        lines.append("**Full breakdown:**")
        breakdown_items = []
        if heritage_total >= 0.5:
            breakdown_items.append(f"- {specific_dem} spouse: {heritage_total:.1f}%")
        if third_gen_pct >= 0.5:
            breakdown_items.append(f"- 3rd+ gen American: {third_gen_pct:.1f}%")
        if diff_pct >= 0.5:
            breakdown_items.append(f"- Other immigrant backgrounds: {diff_pct:.1f}%")

        for item in breakdown_items:
            lines.append(item)
        lines.append("")
    else:
        # General selection (Any x Any)
        heritage_total = same_pct + mother_pct + father_pct + both_pct
        non_heritage = third_gen_pct + diff_pct

        # Determine which is actually largest and use accurate language
        if heritage_total > third_gen_pct and heritage_total > diff_pct:
            lines.append(f"**Heritage-based marriages led:** {heritage_total:.1f}% married someone connected to their immigrant heritage — the largest category.")
        elif third_gen_pct > heritage_total and third_gen_pct > diff_pct:
            lines.append(f"**Mainstream assimilation led:** {third_gen_pct:.1f}% married a 3rd+ generation American — the largest category.")
        else:
            lines.append(f"**Balanced patterns:** Marriage to 3rd+ gen Americans ({third_gen_pct:.1f}%) and heritage-based marriages ({heritage_total:.1f}%) occurred at similar rates.")
        lines.append("")

        # Show breakdown without assuming which is "primary"
        lines.append(f"**3rd+ generation American:** {third_gen_pct:.1f}%")
        lines.append("")
        lines.append(f"**Heritage-based marriages:** {heritage_total:.1f}%")
        if same_pct >= 0.5:
            lines.append(f"  - {same_pct:.1f}% married someone matching both parents' origins")
        if mother_pct + father_pct + both_pct >= 0.5:
            lines.append(f"  - {mother_pct + father_pct + both_pct:.1f}% married someone matching one parent's origin (among mixed-heritage individuals)")
        lines.append("")
        if diff_pct >= 0.5:
            lines.append(f"**Other immigrant backgrounds:** {diff_pct:.1f}%")
            lines.append("")

    # ==================== TOP SPOUSE BACKGROUNDS ====================
    # Determine what heritage to exclude (the subject's own heritage)
    if mother != 'Any' and father != 'Any' and mother == father:
        exclude_set = {mother}
    elif mother != 'Any' and father != 'Any':
        exclude_set = {mother, father}
    elif mother != 'Any':
        exclude_set = {mother}
    elif father != 'Any':
        exclude_set = {father}
    else:
        exclude_set = set()

    # Get top cross-ethnic backgrounds (excluding the subject's heritage)
    top_backgrounds = get_top_spouse_backgrounds(df, exclude_heritage=exclude_set, top_n=5)

    # Filter to only show backgrounds with >= 0.5% before deciding to display section
    displayable_backgrounds = [(bg, pct) for bg, pct in top_backgrounds if pct >= 0.5]

    if displayable_backgrounds and diff_pct >= 1:
        lines.append("---")
        lines.append("")
        lines.append("#### Most Common Cross-Ethnic Marriages")
        lines.append("")

        if mother != 'Any' and father != 'Any' and mother == father:
            lines.append(f"Among {subject_short} who married outside the {mother_dem} community, the most common spouse backgrounds were:")
        elif mother != 'Any' and father != 'Any':
            lines.append(f"Among those who married outside both {mother_dem} and {father_dem} communities, the most common spouse backgrounds were:")
        elif mother != 'Any':
            lines.append(f"Among children of {get_demonym(mother)} mothers who married outside {get_demonym(mother)} communities, the most common spouse backgrounds were:")
        elif father != 'Any':
            lines.append(f"Among children of {get_demonym(father)} fathers who married outside {get_demonym(father)} communities, the most common spouse backgrounds were:")
        else:
            lines.append("The most common cross-ethnic spouse backgrounds were:")
        lines.append("")

        for bg, pct in displayable_backgrounds:
            lines.append(f"- **{bg}:** {pct:.1f}%")

        lines.append("")

    # ==================== TRENDS OVER TIME ====================
    if year == 'All':
        trends = get_trend_data(mother, father)
        if len(trends) >= 2:
            years_list = sorted(trends.keys())

            lines.append("---")
            lines.append("")
            lines.append("#### Trends Over Time")
            lines.append("")

            # Always show first and last data points
            first_year = years_list[0]
            last_year = years_list[-1]
            first_third = trends[first_year]['third_gen']
            last_third = trends[last_year]['third_gen']
            first_ethnic = trends[first_year]['ethnic_total']
            last_ethnic = trends[last_year]['ethnic_total']

            change_third = last_third - first_third
            change_ethnic = last_ethnic - first_ethnic

            lines.append(f"**From {first_year} to {last_year}:**")
            lines.append("")

            # Marriage to 3rd+ gen
            if abs(change_third) >= 3:
                direction = "rose" if change_third > 0 else "fell"
                lines.append(f"- Marriage to 3rd+ generation Americans {direction} from {first_third:.0f}% to {last_third:.0f}% ({'+' if change_third > 0 else ''}{change_third:.0f} points)")
            else:
                lines.append(f"- Marriage to 3rd+ generation Americans remained relatively stable (~{(first_third + last_third)/2:.0f}%)")

            # Ethnic retention
            if abs(change_ethnic) >= 3:
                direction = "increased" if change_ethnic > 0 else "decreased"
                lines.append(f"- Heritage-based marriage {direction} from {first_ethnic:.0f}% to {last_ethnic:.0f}% ({'+' if change_ethnic > 0 else ''}{change_ethnic:.0f} points)")
            else:
                lines.append(f"- Heritage-based marriage remained relatively stable (~{(first_ethnic + last_ethnic)/2:.0f}%)")

            lines.append("")

            # Find peak/trough years if enough data
            if len(years_list) >= 4:
                max_third_year = max(years_list, key=lambda y: trends[y]['third_gen'])
                min_third_year = min(years_list, key=lambda y: trends[y]['third_gen'])

                if max_third_year != first_year and max_third_year != last_year:
                    lines.append(f"**Peak mainstream marriage:** {trends[max_third_year]['third_gen']:.0f}% in {max_third_year}")
                    lines.append("")
                elif min_third_year != first_year and min_third_year != last_year:
                    lines.append(f"**Lowest mainstream marriage:** {trends[min_third_year]['third_gen']:.0f}% in {min_third_year}")
                    lines.append("")

    # ==================== COMPARATIVE CONTEXT ====================
    lines.append("---")
    lines.append("")
    lines.append("#### Comparative Context")
    lines.append("")

    # Calculate out-group marriage rate (everything except same heritage)
    outgroup_pct = 100 - same_pct if mother != 'Any' and father != 'Any' and mother == father else 100 - (same_pct + mother_pct + father_pct + both_pct)

    # For mixed-heritage selections, compare to each parent group
    if mother != 'Any' and father != 'Any' and mother != father:
        # Check if one parent is American (US-born)
        mother_is_american = mother == 'US-born'
        father_is_american = father == 'US-born'

        if mother_is_american or father_is_american:
            # One parent is American - simplified comparison
            immigrant_parent = father if mother_is_american else mother
            immigrant_dem = get_demonym(immigrant_parent)
            immigrant_pct = father_pct if mother_is_american else mother_pct

            # Compare to same-origin immigrant group
            immigrant_stats = get_comparison_stats(immigrant_parent, immigrant_parent, year)
            overall_stats = get_comparison_stats('Any', 'Any', year)

            if immigrant_stats or overall_stats:
                lines.append(f"**How do children of {immigrant_dem}-American marriages compare?**")
                lines.append("")
                lines.append("| Group | 3rd+ Gen | Ethnic Retention |")
                lines.append("|-------|----------|------------------|")
                lines.append(f"| **{immigrant_dem} × American (selected)** | {third_gen_pct:.0f}% | {immigrant_pct:.0f}% |")

                if immigrant_stats:
                    lines.append(f"| {immigrant_dem} × {immigrant_dem} | {immigrant_stats['third_gen']:.0f}% | {immigrant_stats['same']:.0f}% |")
                if overall_stats:
                    lines.append(f"| All 2nd-gen Americans | {overall_stats['third_gen']:.0f}% | {overall_stats['same']:.0f}% |")
                lines.append("")

                if immigrant_stats:
                    diff = third_gen_pct - immigrant_stats['third_gen']
                    if diff > 10:
                        lines.append(f"**Higher mainstream integration:** Children with one American parent married 3rd+ generation Americans at much higher rates ({third_gen_pct:.0f}%) than children of two {immigrant_dem} parents ({immigrant_stats['third_gen']:.0f}%). Having one native-born parent likely provided greater exposure to mainstream American social networks.")
                    elif diff < -10:
                        lines.append(f"**Strong ethnic ties despite mixed parentage:** Even with one American parent, these children maintained strong connections to {immigrant_dem} communities, marrying within the ethnic group at {immigrant_pct:.0f}%.")
                    else:
                        lines.append(f"**Similar patterns:** Children of {immigrant_dem}-American marriages showed similar mainstream integration rates to children of two {immigrant_dem} parents.")
                    lines.append("")
        else:
            # Both parents are foreign-born - standard comparison
            comparisons = []

            # Compare to mother x mother
            mom_stats = get_comparison_stats(mother, mother, year)
            if mom_stats:
                comparisons.append((f"{mother_dem} × {mother_dem}", mom_stats, mother_dem))

            # Compare to father x father
            dad_stats = get_comparison_stats(father, father, year)
            if dad_stats:
                comparisons.append((f"{father_dem} × {father_dem}", dad_stats, father_dem))

            # Compare to reverse (father x mother)
            reverse_stats = get_comparison_stats(father, mother, year)
            if reverse_stats:
                comparisons.append((f"{father_dem} × {mother_dem}", reverse_stats, None))

            if comparisons:
                lines.append("**How does this mixed-heritage group compare?**")
                lines.append("")
                lines.append("| Group | 3rd+ Gen | Ethnic Retention |")
                lines.append("|-------|----------|------------------|")

                current_ethnic = mother_pct + father_pct + both_pct
                lines.append(f"| **{mother_dem} × {father_dem} (selected)** | {third_gen_pct:.0f}% | {current_ethnic:.0f}% |")

                for comp_name, stats, origin_dem in comparisons:
                    comp_ethnic = stats['same'] if origin_dem else stats['mother'] + stats['father'] + stats['both']
                    lines.append(f"| {comp_name} | {stats['third_gen']:.0f}% | {comp_ethnic:.0f}% |")

                lines.append("")

                # Detailed analysis
                all_third = [(third_gen_pct, f"{mother_dem} × {father_dem}")] + [(s['third_gen'], n) for n, s, _ in comparisons]
                all_third.sort(reverse=True)
                rank = [i for i, (v, n) in enumerate(all_third) if f"{mother_dem} × {father_dem}" in n][0] + 1

                if rank == 1:
                    lines.append(f"**Highest assimilation rate:** Children of {mother_dem}-{father_dem} mixed marriages showed the highest rate of marriage to 3rd+ generation Americans among all comparison groups. This suggests that growing up between two ethnic communities may have facilitated integration into the American mainstream, possibly because these children were less embedded in any single ethnic enclave.")
                elif rank == len(all_third):
                    lines.append(f"**Strongest ethnic ties:** Despite their mixed heritage, children of {mother_dem}-{father_dem} marriages maintained stronger ties to ethnic communities than either single-heritage group. This may indicate that having connections to two immigrant communities reinforced ethnic identity rather than diluting it.")
                else:
                    lines.append(f"**Intermediate position:** Children of {mother_dem}-{father_dem} marriages fell between the single-heritage comparison groups, suggesting their dual background created a balance between ethnic retention and mainstream assimilation.")
                lines.append("")

                # Heritage preference
                if mother_pct > 0 or father_pct > 0:
                    if abs(mother_pct - father_pct) > 5:
                        stronger = mother_dem if mother_pct > father_pct else father_dem
                        weaker = father_dem if mother_pct > father_pct else mother_dem
                        lines.append(f"**Maternal vs. paternal heritage:** These individuals married into {stronger} communities at notably higher rates ({max(mother_pct, father_pct):.1f}%) than {weaker} communities ({min(mother_pct, father_pct):.1f}%). This asymmetry may reflect which parent's community was larger, more geographically concentrated, or more culturally influential in the household.")
                    else:
                        lines.append(f"**Balanced dual identity:** Marriage into both parents' communities occurred at similar rates ({mother_pct:.1f}% {mother_dem}, {father_pct:.1f}% {father_dem}), indicating these children maintained meaningful connections to both sides of their heritage.")
                    lines.append("")

    elif mother != 'Any' and father != 'Any' and mother == father:
        # Same-origin: compare to overall and other groups
        overall_stats = get_comparison_stats('Any', 'Any', year)

        comparison_origins = ['Ireland', 'Germany', 'Italy', 'Poland', 'England', 'Russia/USSR', 'Sweden', 'Norway']
        comparison_origins = [o for o in comparison_origins if o != mother and o in mother_origins][:4]

        comparisons = []
        for origin in comparison_origins:
            stats = get_comparison_stats(origin, origin, year)
            if stats:
                comparisons.append((get_demonym(origin), stats))

        lines.append("**How do {}-Americans compare to other groups?**".format(mother_dem))
        lines.append("")
        lines.append("| Group | 3rd+ Gen | Co-ethnic |")
        lines.append("|-------|----------|-----------|")
        lines.append(f"| **{mother_dem}-Americans (selected)** | {third_gen_pct:.0f}% | {same_pct:.0f}% |")

        if overall_stats:
            lines.append(f"| All 2nd-gen Americans | {overall_stats['third_gen']:.0f}% | {overall_stats['same']:.0f}% |")

        for comp_name, stats in comparisons:
            lines.append(f"| {comp_name}-Americans | {stats['third_gen']:.0f}% | {stats['same']:.0f}% |")

        lines.append("")

        # Analysis based on ethnic retention rate - thresholds based on outmarriage rate
        outmarriage = 100 - same_pct
        if same_pct < 30:
            lines.append(f"**Low ethnic retention:** With {outmarriage:.0f}% marrying outside the {mother_dem} community, this group showed weak co-ethnic marriage patterns. Only {same_pct:.1f}% married within their heritage. This high outmarriage rate suggests {mother_dem}-Americans were geographically dispersed, faced a limited co-ethnic marriage market, or actively integrated into broader American society.")
        elif same_pct < 50:
            lines.append(f"**Moderate ethnic retention:** About {same_pct:.0f}% of {mother_dem}-Americans married within their ethnic community, while {outmarriage:.0f}% married outside it. This represents a rough balance between maintaining ethnic ties and integrating into the broader population.")
        else:
            lines.append(f"**Strong ethnic retention:** {same_pct:.1f}% of {mother_dem}-Americans married within their ethnic community—a majority staying in-group. Only {outmarriage:.0f}% married outside, suggesting strong ethnic institutions, concentrated residential patterns, or cultural preferences for in-group marriage.")
        lines.append("")

        if overall_stats:
            diff = third_gen_pct - overall_stats['third_gen']
            ethnic_diff = same_pct - overall_stats['same']
            if abs(diff) > 5 or abs(ethnic_diff) > 5:
                if diff > 5:
                    lines.append(f"**Above-average mainstream integration:** {mother_dem}-Americans married 3rd+ generation Americans at rates {diff:.0f} percentage points above the overall average, suggesting this group experienced relatively smooth incorporation into American society.")
                elif diff < -5:
                    lines.append(f"**Below-average mainstream integration:** {mother_dem}-Americans married 3rd+ generation Americans at rates {abs(diff):.0f} percentage points below average, instead showing stronger connections to ethnic and immigrant communities.")
                lines.append("")

    elif mother != 'Any' or father != 'Any':
        # Partial selection: one specific origin + one "Any"
        specific_origin = mother if mother != 'Any' else father
        specific_dem = get_demonym(specific_origin)
        parent_type = "mother" if mother != 'Any' else "father"
        other_parent = "father" if mother != 'Any' else "mother"

        # Calculate heritage-based for this partial selection
        heritage_based_pct = same_pct + mother_pct + father_pct + both_pct
        relevant_pct = mother_pct if mother != 'Any' else father_pct

        # Compare to same-origin group (the main comparison)
        same_origin_stats = get_comparison_stats(specific_origin, specific_origin, year)

        lines.append(f"**How do children of {specific_dem} {parent_type}s compare?**")
        lines.append("")

        if same_origin_stats:
            lines.append(f"| Group | 3rd+ Gen | {specific_dem} Spouse |")
            lines.append("|-------|----------|----------------|")
            lines.append(f"| **{specific_dem} {parent_type} × Any {other_parent} (selected)** | {third_gen_pct:.0f}% | {relevant_pct + same_pct:.0f}% |")
            lines.append(f"| {specific_dem} × {specific_dem} (both parents) | {same_origin_stats['third_gen']:.0f}% | {same_origin_stats['same']:.0f}% |")
            lines.append("")

            diff_third = third_gen_pct - same_origin_stats['third_gen']
            diff_ethnic = (relevant_pct + same_pct) - same_origin_stats['same']

            # Analyze the differences
            if abs(diff_third) > 5 or abs(diff_ethnic) > 5:
                if diff_third > 5:
                    lines.append(f"**Higher mainstream integration:** Children with only one {specific_dem} parent married 3rd+ generation Americans at {third_gen_pct:.0f}% vs {same_origin_stats['third_gen']:.0f}% for those with two {specific_dem} parents. Having a non-{specific_dem} {other_parent} appears to have facilitated integration into the American mainstream.")
                elif diff_third < -5:
                    lines.append(f"**Lower mainstream integration:** Children with only one {specific_dem} parent showed {third_gen_pct:.0f}% marriage to 3rd+ gen Americans, compared to {same_origin_stats['third_gen']:.0f}% for those with two {specific_dem} parents.")

                if diff_ethnic < -5:
                    lines.append(f"")
                    lines.append(f"**Weaker ethnic retention:** Only {relevant_pct + same_pct:.0f}% married a {specific_dem} spouse, compared to {same_origin_stats['same']:.0f}% for {specific_dem}×{specific_dem} households. Having only one {specific_dem} parent significantly reduced the likelihood of marrying within that ethnic community.")
                elif diff_ethnic > 5:
                    lines.append(f"")
                    lines.append(f"**Stronger ethnic connection:** Despite having only one {specific_dem} parent, {relevant_pct + same_pct:.0f}% still married a {specific_dem} spouse—higher than the {same_origin_stats['same']:.0f}% among {specific_dem}×{specific_dem} households.")
            else:
                lines.append(f"Marriage patterns were similar whether both parents or only the {parent_type} was {specific_dem}, suggesting the {specific_dem} community maintained strong influence even in mixed-origin households.")
        else:
            lines.append(f"Children of {specific_dem} {parent_type}s showed {third_gen_pct:.0f}% marriage to 3rd+ generation Americans and {relevant_pct + same_pct:.0f}% marriage to {specific_dem} spouses.")

    else:
        # General selection (Any x Any)
        lines.append("**Overall second-generation marriage patterns:**")
        lines.append("")

        # Calculate total heritage-based marriages (same + any parent's heritage)
        heritage_based_pct = same_pct + mother_pct + father_pct + both_pct
        non_heritage_pct = third_gen_pct + diff_pct

        # Use accurate, dynamic language based on actual comparison
        if heritage_based_pct > third_gen_pct:
            lines.append(f"**Heritage connections remained strong:** {heritage_based_pct:.1f}% of second-generation Americans married someone connected to their immigrant heritage, making this the most common pattern. Meanwhile, {third_gen_pct:.1f}% married 3rd+ generation Americans (those whose parents were also US-born).")
        elif third_gen_pct > heritage_based_pct + 5:
            lines.append(f"**Assimilation predominated:** {third_gen_pct:.1f}% of second-generation Americans married 3rd+ generation Americans (those whose parents were also US-born), making mainstream integration the most common outcome. {heritage_based_pct:.1f}% married someone connected to their immigrant heritage.")
        else:
            lines.append(f"**Split between assimilation and heritage:** Second-generation Americans were nearly evenly divided, with {third_gen_pct:.1f}% marrying 3rd+ generation Americans and {heritage_based_pct:.1f}% marrying someone connected to their immigrant heritage.")
        lines.append("")

        # Breakdown of heritage-based
        lines.append(f"**Heritage-based breakdown:**")
        lines.append(f"  - {same_pct:.1f}% married someone matching both parents' origins")
        if mother_pct > 0.5 or father_pct > 0.5 or both_pct > 0.5:
            lines.append(f"  - {mother_pct + father_pct + both_pct:.1f}% married someone matching one parent's origin (among mixed-heritage individuals)")
        lines.append("")

        if diff_pct > 1:
            lines.append(f"**Cross-ethnic immigrant marriage:** {diff_pct:.1f}% married immigrants or second-generation Americans from different ethnic backgrounds — marriage *between* immigrant communities rather than into them or out to the mainstream.")
            lines.append("")

        # Accurate summary
        if heritage_based_pct > non_heritage_pct:
            lines.append(f"**Summary:** {heritage_based_pct:.0f}% maintained ethnic ties through marriage, while {non_heritage_pct:.0f}% married outside their parents' ethnic communities ({third_gen_pct:.0f}% into the mainstream American population, {diff_pct:.0f}% into other immigrant communities).")
        else:
            lines.append(f"**Summary:** {non_heritage_pct:.0f}% married outside their parents' ethnic communities ({third_gen_pct:.0f}% into the mainstream American population, {diff_pct:.0f}% into other immigrant communities), while {heritage_based_pct:.0f}% maintained ethnic ties through marriage.")

    return "\n".join(lines)


# =============================================================================
# COMPREHENSIVE CUSTOM CSS
# =============================================================================

CUSTOM_CSS = """
/* ============================================
   NISKANEN CENTER BRAND STYLES v10.1
   Bold, professional, visually striking
   ============================================ */

/* Import fonts */
@import url('https://fonts.googleapis.com/css2?family=Neuton:wght@400;700&family=Hanken+Grotesk:wght@300;400;600&display=swap');

/* Base Reset */
* {
    box-sizing: border-box;
}

body {
    font-family: 'Hanken Grotesk', -apple-system, BlinkMacSystemFont, sans-serif;
    font-weight: 400;
    color: #194852;
    background: #edf1f2;
    min-height: 100vh;
    margin: 0;
    padding: 0;
}

/* Typography */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Neuton', Georgia, serif;
    font-weight: 700;
    color: #194852;
}

strong, b {
    font-weight: 600;
}

/* ============================================
   DRAMATIC HERO HEADER
   ============================================ */

.dashboard-wrapper {
    position: relative;
    background: #edf1f2;
}

.dashboard-wrapper::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 320px;
    background: linear-gradient(135deg, #0c2a30 0%, #194852 50%, #348397 100%);
    z-index: 0;
}

.dashboard-wrapper::after {
    content: '';
    position: absolute;
    top: 280px;
    left: 0;
    right: 0;
    height: 80px;
    background: linear-gradient(to bottom, rgba(25, 72, 82, 0.1) 0%, transparent 100%);
    z-index: 0;
}

/* ============================================
   HEADER STYLES
   ============================================ */

.header-section {
    position: relative;
    padding: 3rem 0 2.5rem 0;
    margin-bottom: 2rem;
    z-index: 1;
}

.main-title {
    font-family: 'Neuton', Georgia, serif;
    font-size: 2.75rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 0.75rem;
    letter-spacing: -0.01em;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
}

.subtitle {
    font-family: 'Hanken Grotesk', sans-serif;
    font-weight: 300;
    font-size: 1.2rem;
    color: #7dceda;
    margin-bottom: 0;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.15);
}

/* ============================================
   CARD STYLES - Bold with strong shadows
   ============================================ */

.brand-card {
    background: #ffffff;
    border: none;
    border-radius: 12px;
    box-shadow: 0 10px 40px rgba(12, 42, 48, 0.15),
                0 2px 8px rgba(12, 42, 48, 0.08);
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
    z-index: 1;
}

.brand-card:hover {
    box-shadow: 0 15px 50px rgba(12, 42, 48, 0.2),
                0 4px 12px rgba(12, 42, 48, 0.1);
    transform: translateY(-3px);
}

.brand-card-header {
    background: linear-gradient(135deg, #194852 0%, #0c2a30 100%);
    color: #ffffff;
    padding: 1.25rem 1.5rem;
    border-radius: 12px 12px 0 0;
    font-family: 'Hanken Grotesk', sans-serif;
    font-weight: 600;
    font-size: 1rem;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}

.brand-card-header-light {
    background: linear-gradient(135deg, #348397 0%, #194852 100%);
    color: #ffffff;
    padding: 1.25rem 1.5rem;
    border-radius: 12px 12px 0 0;
    font-family: 'Hanken Grotesk', sans-serif;
    font-weight: 600;
    font-size: 1rem;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}

.brand-card-header-gold {
    background: linear-gradient(135deg, #bca45e 0%, #52482a 100%);
    color: #ffffff;
    padding: 1.25rem 1.5rem;
    border-radius: 12px 12px 0 0;
    font-family: 'Hanken Grotesk', sans-serif;
    font-weight: 600;
    font-size: 1rem;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}

.brand-card-body {
    padding: 1.75rem;
}

/* Summary card special styling */
.summary-card {
    border-left: 5px solid #bca45e;
}

.summary-card .brand-card-body {
    background: linear-gradient(135deg, #fffef9 0%, #fef8e8 100%);
}

/* ============================================
   FILTER CONTROLS
   ============================================ */

.filter-section {
    background: #ffffff;
    border-radius: 12px;
    padding: 1.75rem;
    margin-bottom: 2rem;
    box-shadow: 0 10px 40px rgba(12, 42, 48, 0.12),
                0 2px 8px rgba(12, 42, 48, 0.06);
    border: none;
    position: relative;
    z-index: 2;
}

.filter-label {
    font-family: 'Hanken Grotesk', sans-serif;
    font-weight: 600;
    font-size: 0.85rem;
    color: #194852;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
    display: block;
}

/* Dropdown styling */
.Select-control {
    border-radius: 10px !important;
    border: 2px solid #d0dbdd !important;
    background: #ffffff !important;
    transition: all 0.2s ease !important;
}

.Select-control:hover {
    border-color: #348397 !important;
}

.is-focused .Select-control {
    border-color: #348397 !important;
    box-shadow: 0 0 0 3px rgba(52, 131, 151, 0.15) !important;
}

.Select-menu-outer {
    border-radius: 10px !important;
    box-shadow: 0 8px 24px rgba(25, 72, 82, 0.15) !important;
    border: 1px solid #d0dbdd !important;
    margin-top: 4px !important;
    min-width: 200px !important;
}

.Select-option {
    font-family: 'Hanken Grotesk', sans-serif !important;
    font-weight: 300 !important;
    padding: 14px 14px !important;
    line-height: 1.4 !important;
    white-space: nowrap !important;
    overflow: visible !important;
}

.Select-value-label {
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}

/* Dash dropdown specific styling */
.VirtualizedSelectOption {
    padding: 16px 14px !important;
    line-height: 1.5 !important;
    min-height: 44px !important;
}

.VirtualizedSelectFocusedOption {
    background: #edf1f2 !important;
}

/* Alternative dropdown selectors */
[class*="option"] {
    padding-top: 12px !important;
    padding-bottom: 12px !important;
}

.Select-option.is-focused {
    background: #edf1f2 !important;
}

.Select-option.is-selected {
    background: #348397 !important;
    color: #ffffff !important;
    font-weight: 600 !important;
}

/* ============================================
   BUTTONS
   ============================================ */

.brand-btn {
    font-family: 'Hanken Grotesk', sans-serif;
    font-weight: 600;
    font-size: 0.85rem;
    padding: 0.75rem 1.5rem;
    border-radius: 8px;
    transition: all 0.25s ease;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    cursor: pointer;
}

.brand-btn-primary {
    background: linear-gradient(135deg, #348397 0%, #194852 100%);
    border: none;
    color: #ffffff;
    box-shadow: 0 4px 15px rgba(25, 72, 82, 0.3);
}

.brand-btn-primary:hover {
    background: linear-gradient(135deg, #7dceda 0%, #348397 100%);
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(25, 72, 82, 0.4);
    color: #ffffff;
}

.brand-btn-secondary {
    background: #ffffff;
    border: 2px solid #348397;
    color: #348397;
}

.brand-btn-secondary:hover {
    background: #348397;
    color: #ffffff;
    transform: translateY(-2px);
    box-shadow: 0 4px 15px rgba(52, 131, 151, 0.3);
}

/* ============================================
   STATS DISPLAY
   ============================================ */

.stat-display {
    text-align: center;
    padding: 1.25rem;
    background: linear-gradient(135deg, #194852 0%, #0c2a30 100%);
    border-radius: 10px;
    box-shadow: 0 4px 15px rgba(12, 42, 48, 0.2);
}

.stat-value {
    font-family: 'Neuton', Georgia, serif;
    font-size: 2.25rem;
    font-weight: 700;
    color: #ffffff;
    line-height: 1.2;
}

.stat-label {
    font-family: 'Hanken Grotesk', sans-serif;
    font-weight: 400;
    font-size: 0.8rem;
    color: #7dceda;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 0.25rem;
}

/* ============================================
   TABS
   ============================================ */

.nav-tabs {
    border-bottom: none;
    margin-bottom: 0;
    background: linear-gradient(135deg, #194852 0%, #0c2a30 100%);
    border-radius: 12px 12px 0 0;
    padding: 0.5rem 0.5rem 0 0.5rem;
}

.nav-tabs .nav-link {
    font-family: 'Hanken Grotesk', sans-serif;
    font-weight: 600;
    font-size: 0.9rem;
    color: rgba(255, 255, 255, 0.7);
    border: none;
    border-radius: 8px 8px 0 0;
    padding: 1rem 1.75rem;
    margin: 0;
    transition: all 0.25s ease;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    background: transparent;
}

.nav-tabs .nav-link:hover {
    color: #ffffff;
    background: rgba(255, 255, 255, 0.1);
}

.nav-tabs .nav-link.active {
    color: #194852;
    background: #ffffff;
    font-weight: 600;
}

/* ============================================
   TABLE STYLES
   ============================================ */

.brand-table {
    font-family: 'Hanken Grotesk', sans-serif;
    font-weight: 400;
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
}

.brand-table thead th {
    background: linear-gradient(135deg, #194852 0%, #0c2a30 100%);
    color: #ffffff;
    font-weight: 600;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    padding: 1rem 1.25rem;
    border: none;
    position: sticky;
    top: 0;
}

.brand-table thead th:first-child {
    border-radius: 8px 0 0 0;
}

.brand-table thead th:last-child {
    border-radius: 0 8px 0 0;
}

.brand-table tbody td {
    padding: 1rem 1.25rem;
    border-bottom: 1px solid #edf1f2;
    vertical-align: middle;
}

.brand-table tbody tr:hover {
    background: linear-gradient(135deg, rgba(125, 206, 218, 0.1) 0%, rgba(52, 131, 151, 0.08) 100%);
}

.brand-table tbody tr:last-child td:first-child {
    border-radius: 0 0 0 8px;
}

.brand-table tbody tr:last-child td:last-child {
    border-radius: 0 0 8px 0;
}

/* ============================================
   MARKDOWN STYLES (for summary)
   ============================================ */

.summary-markdown {
    font-family: 'Hanken Grotesk', sans-serif;
    font-weight: 400;
    font-size: 1rem;
    line-height: 1.85;
    color: #194852;
}

.summary-markdown h3 {
    font-family: 'Neuton', Georgia, serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #194852;
    margin-bottom: 1.25rem;
    padding-bottom: 0.75rem;
    border-bottom: 3px solid #bca45e;
}

.summary-markdown h4 {
    font-family: 'Hanken Grotesk', sans-serif;
    font-weight: 600;
    font-size: 1rem;
    color: #194852;
    background: linear-gradient(135deg, #348397 0%, #194852 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-top: 1.75rem;
    margin-bottom: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.summary-markdown strong {
    font-weight: 600;
    color: #194852;
}

.summary-markdown hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, #d0dbdd, transparent);
    margin: 1.5rem 0;
}

.summary-markdown ul {
    padding-left: 1.5rem;
}

.summary-markdown li {
    margin-bottom: 0.5rem;
}

/* Comparison tables in summary */
.summary-markdown table {
    width: 100%;
    border-collapse: collapse;
    margin: 1.25rem 0;
    font-size: 0.95rem;
    background: #ffffff;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 4px 15px rgba(12, 42, 48, 0.1);
}

.summary-markdown table thead tr {
    background: linear-gradient(135deg, #194852 0%, #0c2a30 100%);
    color: #ffffff;
}

.summary-markdown table th {
    padding: 1rem 1.25rem;
    text-align: left;
    font-weight: 600;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.summary-markdown table td {
    padding: 0.875rem 1.25rem;
    border-bottom: 1px solid #edf1f2;
}

.summary-markdown table tbody tr:hover {
    background: rgba(125, 206, 218, 0.08);
}

.summary-markdown table tbody tr:last-child td {
    border-bottom: none;
}

/* Highlight selected row */
.summary-markdown table tbody tr:first-child {
    background: linear-gradient(135deg, rgba(188, 164, 94, 0.15) 0%, rgba(188, 164, 94, 0.08) 100%);
    font-weight: 600;
}

.summary-markdown table tbody tr:first-child td {
    border-bottom: 2px solid #bca45e;
}

/* ============================================
   RESPONSIVE ADJUSTMENTS
   ============================================ */

@media (max-width: 992px) {
    .main-title {
        font-size: 2rem;
    }
    .filter-section {
        padding: 1rem;
    }
}

@media (max-width: 768px) {
    .main-title {
        font-size: 1.5rem;
    }
    .subtitle {
        font-size: 1rem;
    }
    .header-section {
        padding: 1.5rem 0 1rem 0;
    }
    .brand-card-body {
        padding: 1rem;
    }
    .nav-tabs .nav-link {
        padding: 0.75rem 1rem;
        font-size: 0.8rem;
    }
}

/* ============================================
   LOADING STATES
   ============================================ */

._dash-loading {
    background: rgba(255, 255, 255, 0.9) !important;
}

._dash-loading-callback {
    background: rgba(255, 255, 255, 0.9) !important;
}

/* ============================================
   CHART TRANSITIONS
   ============================================ */

.js-plotly-plot .plotly .main-svg {
    transition: all 0.4s ease-out;
}

/* ============================================
   FOOTER
   ============================================ */

.footer-section {
    margin-top: 3rem;
    padding: 2.5rem 0;
    background: linear-gradient(135deg, #194852 0%, #0c2a30 100%);
    border-radius: 12px;
    text-align: center;
}

.footer-text {
    font-family: 'Hanken Grotesk', sans-serif;
    font-weight: 400;
    font-size: 0.95rem;
    color: rgba(255, 255, 255, 0.8);
}

.footer-text a {
    color: #7dceda;
    text-decoration: none;
    font-weight: 600;
    transition: color 0.2s ease;
}

.footer-text a:hover {
    color: #ffffff;
}
"""

# =============================================================================
# DASH APP
# =============================================================================

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    suppress_callback_exceptions=True
)
app.title = "Marriage and the Melting Pot, 1880–1930 | Niskanen Center"

app.index_string = f'''
<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>{{%title%}}</title>
        {{%favicon%}}
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Neuton:wght@400;700&family=Hanken+Grotesk:wght@300;400;600&display=swap" rel="stylesheet">
        {{%css%}}
        <style>
{CUSTOM_CSS}
        </style>
    </head>
    <body>
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
    </body>
</html>
'''

# =============================================================================
# LAYOUT
# =============================================================================

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dcc.Clipboard(id='clipboard', style={'display': 'none'}),

    html.Div([
        dbc.Container([
            # Header Section
            html.Div([
                html.H1("Marriage and the Melting Pot, 1880–1930",
                        className='main-title text-center'),
                html.P("Whom did the US-born children of immigrants marry?",
                       className='subtitle text-center'),
            ], className='header-section'),

            # Filter Section
            html.Div([
                dbc.Row([
                    dbc.Col([
                        html.Label("Mother's Birthplace", className='filter-label'),
                        dcc.Dropdown(
                            id='mother-dropdown',
                            options=mother_dropdown_options,
                            value='Any',
                            clearable=False,
                        )
                    ], lg=3, md=6, xs=12, className='mb-3'),
                    dbc.Col([
                        html.Label("Father's Birthplace", className='filter-label'),
                        dcc.Dropdown(
                            id='father-dropdown',
                            options=father_dropdown_options,
                            value='Any',
                            clearable=False,
                        )
                    ], lg=3, md=6, xs=12, className='mb-3'),
                    dbc.Col([
                        html.Label("Census Year", className='filter-label'),
                        dcc.Dropdown(
                            id='year-dropdown',
                            options=[{'label': 'All Years', 'value': 'All'}] +
                                    [{'label': str(y), 'value': y} for y in years],
                            value='All',
                            clearable=False,
                        )
                    ], lg=2, md=4, xs=6, className='mb-3'),
                    dbc.Col([
                        html.Label("Quick Presets", className='filter-label'),
                        dcc.Dropdown(
                            id='preset-dropdown',
                            options=DYNAMIC_PRESETS,
                            value='',
                            clearable=False,
                            optionHeight=45,
                            maxHeight=400,
                        )
                    ], lg=2, md=4, xs=6, className='mb-3'),
                    dbc.Col([
                        html.Label("Actions", className='filter-label'),
                        html.Div([
                            html.Button("Reset", id='reset-btn',
                                       className='brand-btn brand-btn-secondary'),
                        ], className='d-flex gap-3')
                    ], lg=2, md=4, xs=12, className='mb-3'),
                ]),
            ], className='filter-section'),

            # Sample Size & Export Row
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Div(id='sample-size', className='stat-value'),
                        html.Div("Weighted Sample Size", className='stat-label'),
                    ], className='stat-display')
                ], md=3, xs=6),
                dbc.Col([
                    html.Div([
                        html.Button("Download CSV", id='download-csv-btn',
                                   className='brand-btn brand-btn-secondary me-2'),
                        dcc.Download(id='download-csv'),
                        html.Button("Share Link", id='copy-link-btn',
                                   className='brand-btn brand-btn-secondary'),
                        html.Span(id='link-copied-msg',
                                  style={'marginLeft': '10px', 'color': COLORS['green'],
                                         'fontWeight': '600', 'fontSize': '0.85rem'}),
                    ], className='text-end pt-3')
                ], md=9, xs=6),
            ], className='mb-4 align-items-center'),

            # Summary Card
            html.Div([
                html.Div("Analysis Summary", className='brand-card-header-gold'),
                html.Div([
                    dcc.Loading(
                        type='circle',
                        color=COLORS['medium_teal'],
                        children=[
                            dcc.Markdown(id='auto-summary', className='summary-markdown')
                        ]
                    )
                ], className='brand-card-body')
            ], className='brand-card summary-card mb-4'),

            # Visualization Tabs
            dbc.Tabs([
                dbc.Tab(label="Main Chart", tab_id="tab-main"),
                dbc.Tab(label="Trends Over Time", tab_id="tab-trends"),
            ], id='viz-tabs', active_tab='tab-main', className='mb-0'),

            # Tab Content
            html.Div([
                html.Div(id='tab-content')
            ], className='brand-card mb-4', style={'borderRadius': '0 0 16px 16px'}),

            # Spouse Backgrounds Table
            html.Div([
                html.Div("Spouse Backgrounds (Top 15)", className='brand-card-header-light'),
                html.Div([
                    dcc.Loading(
                        type='circle',
                        color=COLORS['medium_teal'],
                        children=[
                            html.Div(id='spouse-table', style={'maxHeight': '400px', 'overflowY': 'auto'})
                        ]
                    )
                ], className='brand-card-body')
            ], className='brand-card mb-4'),

            # Methodology
            html.Div([
                html.Div("Methodology & Data Sources", className='brand-card-header'),
                html.Div([
                    # Data Source Section
                    html.H4("Data Source", style={'color': COLORS['dark_teal'], 'marginTop': '0', 'marginBottom': '0.5rem', 'fontFamily': 'Neuton, serif'}),
                    html.P([
                        "This dashboard uses microdata from ",
                        html.A("IPUMS USA", href="https://usa.ipums.org", target="_blank", style={'color': COLORS['medium_teal']}),
                        " (Integrated Public Use Microdata Series), a harmonized collection of U.S. Census samples maintained by the Minnesota Population Center. We selected the largest available samples for each census year to maximize statistical reliability:"
                    ], style={'marginBottom': '0.75rem'}),

                    # Sample size table
                    html.Table([
                        html.Thead(html.Tr([
                            html.Th("Census Year", style={'padding': '0.5rem', 'borderBottom': f'2px solid {COLORS["dark_teal"]}', 'textAlign': 'left'}),
                            html.Th("Sample Size", style={'padding': '0.5rem', 'borderBottom': f'2px solid {COLORS["dark_teal"]}', 'textAlign': 'left'}),
                            html.Th("Approx. Records", style={'padding': '0.5rem', 'borderBottom': f'2px solid {COLORS["dark_teal"]}', 'textAlign': 'left'}),
                        ])),
                        html.Tbody([
                            html.Tr([html.Td("1880", style={'padding': '0.4rem'}), html.Td("10%", style={'padding': '0.4rem'}), html.Td("~5 million", style={'padding': '0.4rem'})]),
                            html.Tr([html.Td("1900", style={'padding': '0.4rem'}), html.Td("5%", style={'padding': '0.4rem'}), html.Td("~3.8 million", style={'padding': '0.4rem'})], style={'backgroundColor': COLORS['very_light_gray']}),
                            html.Tr([html.Td("1910", style={'padding': '0.4rem'}), html.Td("1%", style={'padding': '0.4rem'}), html.Td("~0.9 million", style={'padding': '0.4rem'})]),
                            html.Tr([html.Td("1920", style={'padding': '0.4rem'}), html.Td("1%", style={'padding': '0.4rem'}), html.Td("~1.0 million", style={'padding': '0.4rem'})], style={'backgroundColor': COLORS['very_light_gray']}),
                            html.Tr([html.Td("1930", style={'padding': '0.4rem'}), html.Td("5%", style={'padding': '0.4rem'}), html.Td("~6.1 million", style={'padding': '0.4rem'})]),
                        ])
                    ], style={'width': '100%', 'marginBottom': '1.25rem', 'fontSize': '0.9rem', 'borderCollapse': 'collapse'}),

                    # Historical Context: 1880-1930
                    html.H4("Historical Context: 1880-1930", style={'color': COLORS['dark_teal'], 'marginBottom': '0.5rem', 'fontFamily': 'Neuton, serif'}),
                    html.P([
                        "Between 1880 and 1930, over 27 million immigrants arrived in the United States, transforming the nation's demographic composition. The early decades saw continued flows from Ireland, Germany, and Scandinavia, while the 1890s through 1920s brought large-scale immigration from Southern and Eastern Europe—Italians, Poles, Jews from the Russian Empire, and others. The Immigration Act of 1924 dramatically curtailed these flows, making this period a natural unit of analysis for studying immigrant integration through intermarriage."
                    ], style={'marginBottom': '0.75rem'}),
                    html.P([
                        "The children of these immigrants—the second generation—came of age and married during this era, making 1880-1930 ideal for examining whether ethnic boundaries persisted or dissolved across generations. Censuses after 1940 are excluded due to data limitations that prevent reliable spouse heritage comparison."
                    ], style={'marginBottom': '1.25rem'}),

                    # Population Studied
                    html.H4("Population Studied", style={'color': COLORS['dark_teal'], 'marginBottom': '0.5rem', 'fontFamily': 'Neuton, serif'}),
                    html.P([
                        html.Strong("Second-generation immigrants"), ": US-born individuals with at least one foreign-born parent, who were married with spouse present at census enumeration. We identify these individuals using:"
                    ], style={'marginBottom': '0.25rem'}),
                    html.Ul([
                        html.Li([html.Code("BPL"), " (birthplace) — must be a US state"]),
                        html.Li([html.Code("MBPL"), " (mother's birthplace) and/or ", html.Code("FBPL"), " (father's birthplace) — at least one must be foreign"]),
                    ], style={'marginBottom': '0.5rem', 'paddingLeft': '1.5rem'}),
                    html.P([
                        "Spouse information is obtained through IPUMS's \"Attach Characteristics\" feature, which links each person to their spouse (via ", html.Code("SPLOC"), ") and provides the spouse's birthplace and parental birthplace data (", html.Code("BPL_SP"), ", ", html.Code("MBPL_SP"), ", ", html.Code("FBPL_SP"), ")."
                    ], style={'marginBottom': '1.25rem'}),

                    # Heritage Classification
                    html.H4("Heritage Classification", style={'color': COLORS['dark_teal'], 'marginBottom': '0.5rem', 'fontFamily': 'Neuton, serif'}),
                    html.P([
                        "Users can filter by ", html.Strong("mother's origin country"), " and ", html.Strong("father's origin country"),
                        " independently, enabling analysis of mixed-heritage individuals (e.g., children of Irish mothers and German fathers)."
                    ], style={'marginBottom': '0.5rem'}),
                    html.P(html.Strong("Spouse categories:"), style={'marginBottom': '0.25rem'}),
                    html.Ul([
                        html.Li([html.Strong("1st generation immigrant: "), "Spouse born outside the US (", html.Code("BPL_SP"), " indicates foreign country)"]),
                        html.Li([html.Strong("2nd generation: "), "US-born spouse with at least one foreign-born parent (checked via ", html.Code("MBPL_SP"), " and ", html.Code("FBPL_SP"), ")"]),
                        html.Li([html.Strong("3rd+ generation American: "), "US-born spouse with both parents US-born"]),
                    ], style={'marginBottom': '0.5rem', 'paddingLeft': '1.5rem'}),
                    html.P([
                        "For 2nd-generation spouses, heritage matching checks whether either of the spouse's parents shares an origin country with the reference person's parent(s). The dashboard identifies cases where spouses share maternal heritage, paternal heritage, both, or neither."
                    ], style={'marginBottom': '0.5rem'}),
                    html.P([
                        html.Strong("\"American\" parent option: "),
                        "The dropdown includes \"American\" to select individuals with one US-born parent. This identifies children of mixed native/immigrant marriages (e.g., a German immigrant father who married an American woman). ",
                        html.Em("Note: "),
                        "The census recorded parent birthplace but not grandparent birthplace, so we cannot verify whether an \"American\" parent was themselves 3rd+ generation or 2nd generation with immigrant grandparents."
                    ], style={'marginBottom': '1.25rem'}),

                    # Statistical Notes
                    html.H4("Statistical Notes", style={'color': COLORS['dark_teal'], 'marginBottom': '0.5rem', 'fontFamily': 'Neuton, serif'}),
                    html.Ul([
                        html.Li(["Results are weighted using ", html.Code("PERWT"), " (person weight) to produce population-representative estimates."]),
                        html.Li(["Sample size warnings appear when cell counts fall below thresholds that may produce unreliable estimates."]),
                        html.Li(["Country codes follow IPUMS BPL classifications, which consolidate historical territories (e.g., Austria-Hungary, Russian Poland) into broader categories where necessary."]),
                    ], style={'marginBottom': '1.25rem', 'paddingLeft': '1.5rem'}),

                    # Caveats & Limitations
                    html.H4("Caveats & Limitations", style={'color': COLORS['dark_teal'], 'marginBottom': '0.5rem', 'fontFamily': 'Neuton, serif'}),
                    html.Ol([
                        html.Li([html.Strong("\"3rd+ generation\" is a residual category: "), "It includes anyone whose grandparents' origins cannot be traced through parent birthplace—whether their family arrived in 1630 or 1870."]),
                        html.Li([html.Strong("Boundary changes: "), "European borders shifted dramatically during this period. \"Germany,\" \"Poland,\" \"Austria,\" and \"Russia\" in census records may reflect different territories across census years. IPUMS harmonizes these where possible, but some ambiguity remains."]),
                        html.Li([html.Strong("Enumerator variation: "), "Census responses depended on enumerator transcription and respondent knowledge. Parent birthplace was sometimes recorded as \"Unknown\" or a broad region."]),
                        html.Li([html.Strong("Selection into marriage: "), "This analysis captures who ", html.Em("was"), " married at census time, not marriage formation rates. It cannot distinguish recent marriages from decades-long unions."]),
                        html.Li([html.Strong("Missing spouse data: "), "Records where spouse information could not be linked (e.g., spouse absent, incomplete enumeration) are excluded."]),
                    ], style={'marginBottom': '1.25rem', 'paddingLeft': '1.5rem'}),

                    # Development Process
                    html.H4("Development Process", style={'color': COLORS['dark_teal'], 'marginBottom': '0.5rem', 'fontFamily': 'Neuton, serif'}),
                    html.P([
                        "This dashboard was developed iteratively using ", html.Strong("Claude Code"), " (Anthropic), an AI-assisted coding tool. Claude assisted with data processing pipeline development, debugging, visualization design, and methodology refinement. The underlying analysis logic and research design were directed by the researcher."
                    ], style={'marginBottom': '1.25rem'}),

                    # Citation
                    html.H4("Citation", style={'color': COLORS['dark_teal'], 'marginBottom': '0.5rem', 'fontFamily': 'Neuton, serif'}),
                    html.P("IPUMS data should be cited as:", style={'marginBottom': '0.25rem'}),
                    html.Blockquote([
                        "Steven Ruggles, Sarah Flood, Matthew Sobek, Daniel Backman, Annie Chen, Grace Cooper, Stephanie Richards, Renae Rogers, and Megan Schouweiler. ",
                        html.Em("IPUMS USA: Version 15.0"), " [dataset]. Minneapolis, MN: IPUMS, 2024. ",
                        html.A("https://doi.org/10.18128/D010.V15.0", href="https://doi.org/10.18128/D010.V15.0", target="_blank", style={'color': COLORS['medium_teal']})
                    ], style={
                        'borderLeft': f'3px solid {COLORS["gold"]}',
                        'paddingLeft': '1rem',
                        'marginLeft': '0',
                        'fontStyle': 'normal',
                        'fontSize': '0.9rem',
                        'color': COLORS['dark_teal'],
                        'backgroundColor': COLORS['very_light_gold'],
                        'padding': '0.75rem 1rem',
                        'borderRadius': '0 4px 4px 0',
                    }),

                ], className='brand-card-body', style={'fontSize': '0.95rem', 'lineHeight': '1.6'})
            ], className='brand-card mb-4'),

            # Feedback
            html.Div([
                html.Div("Feedback", className='brand-card-header'),
                html.Div([
                    html.P([
                        "Comments, inquiries, and corrections are welcome. Please contact ",
                        html.A("Gil Guerra", href="mailto:gguerra@niskanencenter.org", style={'color': COLORS['medium_teal']}),
                        " at ",
                        html.A("gguerra@niskanencenter.org", href="mailto:gguerra@niskanencenter.org", style={'color': COLORS['medium_teal']}),
                        "."
                    ])
                ], className='brand-card-body')
            ], className='brand-card mb-4'),

            # Footer
            html.Div([
                html.P([
                    "Data: ",
                    html.A("IPUMS USA", href="https://usa.ipums.org", target="_blank"),
                    " | Analysis: ",
                    html.A("Niskanen Center, Gil Guerra", href="https://www.niskanencenter.org/author/gguerra/", target="_blank")
                ], className='footer-text'),
            ], className='footer-section'),

        ], fluid=True, style={'maxWidth': '1400px', 'position': 'relative', 'zIndex': 1})
    ], className='dashboard-wrapper', style={'minHeight': '100vh', 'padding': '1rem'})
])


# =============================================================================
# CALLBACKS
# =============================================================================

@callback(
    [Output('mother-dropdown', 'value', allow_duplicate=True),
     Output('father-dropdown', 'value', allow_duplicate=True),
     Output('year-dropdown', 'value', allow_duplicate=True)],
    [Input('url', 'search')],
    prevent_initial_call='initial_duplicate'
)
def load_url_state(search):
    if not search:
        return no_update, no_update, no_update
    params = parse_qs(search.lstrip('?'))
    mother = params.get('mother', ['Any'])[0]
    father = params.get('father', ['Any'])[0]
    year = params.get('year', ['All'])[0]
    if mother not in mother_origins and mother != 'Any':
        mother = 'Any'
    if father not in father_origins and father != 'Any':
        father = 'Any'
    if year != 'All':
        try:
            year = int(year)
            if year not in years:
                year = 'All'
        except:
            year = 'All'
    return mother, father, year


@callback(
    Output('url', 'search'),
    [Input('mother-dropdown', 'value'),
     Input('father-dropdown', 'value'),
     Input('year-dropdown', 'value')]
)
def update_url(mother, father, year):
    params = {}
    if mother != 'Any':
        params['mother'] = mother
    if father != 'Any':
        params['father'] = father
    if year != 'All':
        params['year'] = str(year)
    if params:
        return '?' + urlencode(params)
    return ''


@callback(
    [Output('mother-dropdown', 'value'),
     Output('father-dropdown', 'value')],
    [Input('preset-dropdown', 'value')],
    prevent_initial_call=True
)
def apply_preset(preset):
    if not preset or '|' not in preset:
        return no_update, no_update
    parts = preset.split('|')
    if len(parts) == 2:
        return parts[0], parts[1]
    return no_update, no_update


@callback(
    [Output('mother-dropdown', 'value', allow_duplicate=True),
     Output('father-dropdown', 'value', allow_duplicate=True),
     Output('year-dropdown', 'value', allow_duplicate=True),
     Output('preset-dropdown', 'value')],
    [Input('reset-btn', 'n_clicks')],
    prevent_initial_call=True
)
def reset_filters(n_clicks):
    if n_clicks:
        return 'Any', 'Any', 'All', ''
    return no_update, no_update, no_update, no_update


@callback(
    [Output('clipboard', 'content'),
     Output('link-copied-msg', 'children')],
    [Input('copy-link-btn', 'n_clicks')],
    [State('url', 'href')],
    prevent_initial_call=True
)
def copy_link(n_clicks, href):
    if n_clicks:
        return href, "Copied!"
    return no_update, ""


@callback(
    Output('download-csv', 'data'),
    [Input('download-csv-btn', 'n_clicks')],
    [State('mother-dropdown', 'value'),
     State('father-dropdown', 'value'),
     State('year-dropdown', 'value')],
    prevent_initial_call=True
)
def download_csv(n_clicks, mother, father, year):
    if n_clicks:
        df = get_filtered_data(mother, father, year)
        agg = df.groupby('MARRIAGE_TYPE')['PERWT'].sum().reset_index()
        agg.columns = ['Marriage Type', 'Weighted Count']
        total = agg['Weighted Count'].sum()
        agg['Percentage'] = (agg['Weighted Count'] / total * 100).round(2)
        agg['Unweighted Count'] = df.groupby('MARRIAGE_TYPE').size().values
        filename = f"marriage_patterns_{mother}_{father}_{year}.csv".replace(' ', '_')
        return dcc.send_data_frame(agg.to_csv, filename, index=False)
    return no_update


@callback(
    [Output('sample-size', 'children'),
     Output('auto-summary', 'children')],
    [Input('mother-dropdown', 'value'),
     Input('father-dropdown', 'value'),
     Input('year-dropdown', 'value')]
)
def update_summary(mother, father, year):
    df = get_filtered_data(mother, father, year)
    weighted = df['PERWT'].sum()
    unweighted = len(df)
    if unweighted < 30:
        size_text = html.Span([
            f"{weighted:,.0f}",
            html.Br(),
            html.Span(f"Small sample ({unweighted})",
                      style={'fontSize': '0.7rem', 'color': COLORS['orange']})
        ])
    else:
        size_text = f"{weighted:,.0f}"
    summary = generate_summary(df, mother, father, year)
    return size_text, summary


@callback(
    Output('tab-content', 'children'),
    [Input('viz-tabs', 'active_tab'),
     Input('mother-dropdown', 'value'),
     Input('father-dropdown', 'value'),
     Input('year-dropdown', 'value')]
)
def render_tab_content(active_tab, mother, father, year):
    df = get_filtered_data(mother, father, year)
    if active_tab == 'tab-main':
        return html.Div([
            dcc.Loading(
                type='circle',
                color=COLORS['medium_teal'],
                children=[dcc.Graph(id='main-chart', figure=create_main_chart(df, mother, father, year),
                                   config={'displayModeBar': True, 'toImageButtonOptions': {
                                       'format': 'png', 'filename': 'marriage_patterns'}})]
            )
        ], style={'padding': '1rem'})
    elif active_tab == 'tab-trends':
        return html.Div([
            dcc.Loading(
                type='circle',
                color=COLORS['medium_teal'],
                children=[dcc.Graph(id='time-chart', figure=create_time_chart(mother, father),
                                   config={'displayModeBar': True})]
            )
        ], style={'padding': '1rem'})
    return html.Div()


def create_main_chart(df, mother, father, year):
    """Create the main horizontal bar chart with brand styling and sample size warnings."""
    if len(df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data for this selection", x=0.5, y=0.5, showarrow=False,
                          font=dict(size=16, color=COLORS['muted_teal'], family='Hanken Grotesk'))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        return fig

    agg = df.groupby('MARRIAGE_TYPE')['PERWT'].sum().reset_index()
    agg.columns = ['Marriage Type', 'Count']
    agg = agg.sort_values('Count', ascending=True)
    total = agg['Count'].sum()
    agg['Percent'] = (agg['Count'] / total * 100).round(1)
    agg['Label'] = agg.apply(lambda r: f"{r['Percent']:.1f}%", axis=1)
    agg['Color'] = agg['Marriage Type'].map(MARRIAGE_COLORS).fillna(COLORS['muted_teal'])

    fig = go.Figure(go.Bar(
        x=agg['Count'],
        y=agg['Marriage Type'],
        orientation='h',
        marker_color=agg['Color'],
        marker_line_width=0,
        text=agg['Label'],
        textposition='outside',
        textfont=dict(family='Hanken Grotesk', size=12, color=COLORS['dark_teal']),
        hovertemplate='<b>%{y}</b><br>Count: %{x:,.0f}<br>Percent: %{text}<extra></extra>'
    ))

    mother_dem = get_demonym(mother) if mother != 'Any' else None
    father_dem = get_demonym(father) if father != 'Any' else None

    if mother != 'Any' and father != 'Any':
        if mother == father:
            title = f"Whom Did Children of {mother_dem} Parents Marry?"
        elif mother == 'US-born':
            title = f"Whom Did {father_dem}-Americans (with American Mothers) Marry?"
        elif father == 'US-born':
            title = f"Whom Did {mother_dem}-Americans (with American Fathers) Marry?"
        else:
            title = f"Whom Did {mother_dem} × {father_dem} Children Marry?"
    elif mother != 'Any':
        title = f"Whom Did Children of {mother_dem} Mothers Marry?"
    elif father != 'Any':
        title = f"Whom Did Children of {father_dem} Fathers Marry?"
    else:
        title = "Marriage Patterns: All Second-Generation Americans"

    unweighted_n = len(df)
    weighted_n = total

    # Determine sample size warning level
    SMALL_SAMPLE = 50000
    VERY_SMALL_SAMPLE = 10000

    if weighted_n < VERY_SMALL_SAMPLE:
        sample_note = f"n = {unweighted_n:,} records ({weighted_n:,.0f} weighted) — CAUTION: Very small sample, results may be unreliable"
        sample_color = COLORS['medium_orange']
    elif weighted_n < SMALL_SAMPLE:
        sample_note = f"n = {unweighted_n:,} records ({weighted_n:,.0f} weighted) — Note: Small sample size"
        sample_color = COLORS['gold']
    else:
        sample_note = f"n = {unweighted_n:,} records ({weighted_n:,.0f} weighted)"
        sample_color = COLORS['muted_teal']

    fig.update_layout(
        title=dict(text=title, font=dict(family='Neuton', size=22, color=COLORS['dark_teal']),
                  x=0, xanchor='left'),
        xaxis_title="Weighted Count",
        xaxis=dict(
            title_font=dict(family='Hanken Grotesk', size=12, color=COLORS['muted_teal']),
            tickfont=dict(family='Hanken Grotesk', size=11, color=COLORS['muted_teal']),
            gridcolor=COLORS['light_gray'],
            gridwidth=1,
            zeroline=False,
        ),
        yaxis=dict(
            tickfont=dict(family='Hanken Grotesk', size=11, color=COLORS['dark_teal']),
            gridcolor='rgba(0,0,0,0)',
        ),
        height=max(400, len(agg) * 45 + 120),
        margin=dict(l=320, r=80, t=80, b=60),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        annotations=[dict(
            text=sample_note,
            xref='paper', yref='paper',
            x=1, y=-0.08,
            showarrow=False,
            font=dict(family='Hanken Grotesk', size=11, color=sample_color)
        )],
        transition=dict(duration=400, easing='cubic-in-out')
    )
    return fig


def create_time_chart(mother, father):
    """Create trends over time chart with brand styling and sample size warnings."""
    df = get_filtered_data(mother, father, 'All')
    if len(df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data", x=0.5, y=0.5, showarrow=False)
        return fig

    # Calculate yearly totals and filter by sample size
    MIN_TREND_SAMPLE = 10000  # Minimum weighted population per year
    WARN_SAMPLE = 50000  # Show warning marker below this threshold

    yearly = df.groupby(['YEAR', 'MARRIAGE_TYPE'])['PERWT'].sum().reset_index()
    yearly_total = df.groupby('YEAR')['PERWT'].sum().reset_index()
    yearly_total.columns = ['YEAR', 'TOTAL']
    yearly = yearly.merge(yearly_total, on='YEAR')
    yearly['Percent'] = yearly['PERWT'] / yearly['TOTAL'] * 100

    # Track which years have small samples
    small_sample_years = yearly_total[yearly_total['TOTAL'] < MIN_TREND_SAMPLE]['YEAR'].tolist()
    warn_sample_years = yearly_total[(yearly_total['TOTAL'] >= MIN_TREND_SAMPLE) &
                                      (yearly_total['TOTAL'] < WARN_SAMPLE)]['YEAR'].tolist()

    # Filter out years with insufficient samples
    yearly = yearly[~yearly['YEAR'].isin(small_sample_years)]

    if len(yearly) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient sample size for trend analysis.<br>Try selecting 'Any' for broader coverage.",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color=COLORS['muted_teal'], family='Hanken Grotesk')
        )
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300)
        return fig

    def simplify(mtype):
        if '3rd+ gen' in mtype:
            return '3rd+ gen American'
        elif 'same origin' in mtype:
            return 'Same origin'
        elif 'both heritages' in mtype:
            return 'Both heritages'
        elif "mother's origin" in mtype:
            return "Mother's origin"
        elif "father's origin" in mtype:
            return "Father's origin"
        else:
            return 'Different origin'

    yearly['Category'] = yearly['MARRIAGE_TYPE'].apply(simplify)
    yearly_agg = yearly.groupby(['YEAR', 'Category'])['Percent'].sum().reset_index()

    fig = go.Figure()
    for category in CATEGORY_COLORS.keys():
        cat_data = yearly_agg[yearly_agg['Category'] == category]
        if len(cat_data) > 0:
            # Create hover text with sample size info
            hover_texts = []
            for _, row in cat_data.iterrows():
                yr = row['YEAR']
                sample = yearly_total[yearly_total['YEAR'] == yr]['TOTAL'].values[0]
                warning = " (small sample)" if yr in warn_sample_years else ""
                hover_texts.append(f"<b>{row['Percent']:.1f}%</b><br>n={sample:,.0f}{warning}")

            fig.add_trace(go.Scatter(
                x=cat_data['YEAR'],
                y=cat_data['Percent'],
                name=category,
                mode='lines+markers',
                line=dict(color=CATEGORY_COLORS[category], width=3),
                marker=dict(size=8, line=dict(width=2, color='white')),
                hovertemplate='%{customdata}<extra>' + category + '</extra>',
                customdata=hover_texts
            ))

    # Add warning markers for years with small samples
    annotations = []
    if warn_sample_years:
        for yr in warn_sample_years:
            # Find max percent for this year to place annotation
            yr_max = yearly_agg[yearly_agg['YEAR'] == yr]['Percent'].max()
            annotations.append(dict(
                x=yr, y=yr_max + 5,
                text="*",
                showarrow=False,
                font=dict(size=16, color=COLORS['medium_orange'], family='Hanken Grotesk')
            ))

    # Build disclaimer text
    disclaimer_parts = []
    if small_sample_years:
        excluded = ", ".join(str(y) for y in sorted(small_sample_years))
        disclaimer_parts.append(f"Years excluded (n<10,000): {excluded}")
    if warn_sample_years:
        warned = ", ".join(str(y) for y in sorted(warn_sample_years))
        disclaimer_parts.append(f"* Small sample (n<50,000): {warned}")

    disclaimer_text = " | ".join(disclaimer_parts) if disclaimer_parts else ""

    fig.update_layout(
        title=dict(text="Trends Over Time", font=dict(family='Neuton', size=22, color=COLORS['dark_teal']),
                  x=0, xanchor='left'),
        xaxis_title="Census Year",
        yaxis_title="% of Marriages",
        xaxis=dict(
            title_font=dict(family='Hanken Grotesk', size=12, color=COLORS['muted_teal']),
            tickfont=dict(family='Hanken Grotesk', size=11, color=COLORS['muted_teal']),
            gridcolor=COLORS['light_gray'],
            dtick=10,
            zeroline=False,
        ),
        yaxis=dict(
            title_font=dict(family='Hanken Grotesk', size=12, color=COLORS['muted_teal']),
            tickfont=dict(family='Hanken Grotesk', size=11, color=COLORS['muted_teal']),
            gridcolor=COLORS['light_gray'],
            range=[0, max(yearly_agg['Percent'].max() * 1.15, 50)],
            zeroline=False,
        ),
        height=480,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5,
            font=dict(family='Hanken Grotesk', size=11, color=COLORS['dark_teal']),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor=COLORS['light_gray'],
            borderwidth=1
        ),
        margin=dict(l=60, r=30, t=100, b=100 if disclaimer_text else 60),
        annotations=annotations + ([dict(
            text=disclaimer_text,
            xref='paper', yref='paper',
            x=0.5, y=-0.22,
            showarrow=False,
            font=dict(family='Hanken Grotesk', size=10, color=COLORS['medium_orange'])
        )] if disclaimer_text else []),
        transition=dict(duration=400, easing='cubic-in-out')
    )
    return fig


@callback(
    Output('spouse-table', 'children'),
    [Input('mother-dropdown', 'value'),
     Input('father-dropdown', 'value'),
     Input('year-dropdown', 'value')]
)
def update_table(mother, father, year):
    df = get_filtered_data(mother, father, year)
    if len(df) == 0:
        return html.P("No data available", style={'color': COLORS['muted_teal']})

    df = df.copy()
    df['spouse_country_dem'] = df['SPOUSE_COUNTRY'].map(lambda x: get_demonym(x))
    df['spouse_mom_dem'] = df['SPOUSE_MOTHER_ORIGIN'].fillna('Unknown').map(lambda x: get_demonym(x))
    df['spouse_dad_dem'] = df['SPOUSE_FATHER_ORIGIN'].fillna('Unknown').map(lambda x: get_demonym(x))

    conditions = [
        df['SPOUSE_GENERATION'] == '1st gen immigrant',
        df['SPOUSE_GENERATION'] == '3rd+ gen',
    ]
    choices = [
        df['spouse_country_dem'] + ' immigrant',
        '3rd+ gen American',
    ]

    second_gen_mask = df['SPOUSE_GENERATION'] == '2nd gen'
    same_origin_mask = df['spouse_mom_dem'] == df['spouse_dad_dem']

    df['Spouse Background'] = np.select(conditions, choices, default='2nd-gen')
    second_gen_same = second_gen_mask & same_origin_mask
    second_gen_mixed = second_gen_mask & ~same_origin_mask

    df.loc[second_gen_same, 'Spouse Background'] = '2nd-gen ' + df.loc[second_gen_same, 'spouse_mom_dem']
    df.loc[second_gen_mixed, 'Spouse Background'] = '2nd-gen ' + df.loc[second_gen_mixed, 'spouse_mom_dem'] + ' × ' + df.loc[second_gen_mixed, 'spouse_dad_dem']

    agg = df.groupby('Spouse Background')['PERWT'].sum().reset_index()
    agg.columns = ['Spouse Background', 'Count']
    agg = agg.sort_values('Count', ascending=False).head(15)
    total = df['PERWT'].sum()
    agg['Share'] = (agg['Count'] / total * 100).round(1).astype(str) + '%'
    agg['Count'] = agg['Count'].apply(lambda x: f"{x:,.0f}")

    return html.Table([
        html.Thead([
            html.Tr([
                html.Th("Spouse Background", style={'textAlign': 'left'}),
                html.Th("Count", style={'textAlign': 'right'}),
                html.Th("Share", style={'textAlign': 'right'}),
            ])
        ]),
        html.Tbody([
            html.Tr([
                html.Td(row['Spouse Background']),
                html.Td(row['Count'], style={'textAlign': 'right'}),
                html.Td(row['Share'], style={'textAlign': 'right', 'fontWeight': '600'}),
            ]) for _, row in agg.iterrows()
        ])
    ], className='brand-table', style={'width': '100%'})


# =============================================================================
# RUN
# =============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("SECOND-GENERATION MARRIAGE DASHBOARD v10.3")
    print("Niskanen Center - Bold Design Edition")
    print("="*60)
    print("\nChanges in v10.3:")
    print("  - Restricted to 1880-1930 censuses (reliable spouse heritage data)")
    print("  - Excluded 1940-1960 (missing spouse parent birthplace info)")
    print("  - Filtered small sample same-origin combinations from presets")
    print("  - Added methodology note explaining census data limitations")
    print("="*60)
    print("\nOpen: http://127.0.0.1:8050")
    print("Press Ctrl+C to stop\n")

    app.run(debug=True, host='127.0.0.1', port=8050)

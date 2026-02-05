"""
Second-Generation Marriage Patterns Dashboard - Production Version
===================================================================
Uses pre-aggregated data files for efficient deployment on Render.
Run preprocessing: python preprocess_for_deploy.py
Local dev: python app.py
Production: gunicorn app:server --workers 2 --bind 0.0.0.0:$PORT
"""

import pandas as pd
import numpy as np
from pathlib import Path
from urllib.parse import urlencode, parse_qs
import json
import os
from scipy.spatial.distance import pdist, squareform

import dash
from dash import dcc, html, callback, Input, Output, State, ctx, no_update
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go

# =============================================================================
# NISKANEN CENTER BRAND COLORS
# =============================================================================

COLORS = {
    'dark_teal': '#194852',
    'medium_teal': '#348397',
    'light_teal': '#7dceda',
    'very_light_teal': '#b7f6fc',
    'very_dark_teal': '#0c2a30',
    'muted_teal': '#78a0a3',
    'light_gray': '#d0dbdd',
    'very_light_gray': '#edf1f2',
    'white': '#ffffff',
    'dark_gold': '#52482a',
    'gold': '#bca45e',
    'light_gold': '#f4da91',
    'very_light_gold': '#fef0c7',
    'dark_orange': '#411b08',
    'medium_orange': '#8d381c',
    'orange': '#da5831',
    'light_orange': '#f17d3a',
    'dark_green': '#2c3811',
    'green': '#709628',
    'light_green': '#b1d955',
    'very_light_green': '#d7f881',
    'dark_purple': '#503961',
    'purple': '#8655b2',
    'light_purple': '#ba88ef',
    'very_light_purple': '#e0c6fc',
}

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

CATEGORY_COLORS = {
    '3rd+ gen American': COLORS['light_teal'],
    'Same origin': COLORS['dark_teal'],
    "Mother's origin": COLORS['purple'],
    "Father's origin": COLORS['orange'],
    'Both heritages': COLORS['green'],
    'Different origin': COLORS['gold'],
}

# =============================================================================
# DEMONYMS
# =============================================================================

DEMONYMS = {
    'Ireland': 'Irish', 'Germany': 'German', 'Italy': 'Italian', 'Poland': 'Polish',
    'England': 'English', 'Scotland': 'Scottish', 'Wales': 'Welsh', 'France': 'French',
    'Russia/USSR': 'Russian', 'Austria': 'Austrian', 'Hungary': 'Hungarian',
    'Sweden': 'Swedish', 'Norway': 'Norwegian', 'Denmark': 'Danish',
    'Netherlands': 'Dutch', 'Belgium': 'Belgian', 'Switzerland': 'Swiss',
    'Czechoslovakia': 'Czechoslovak', 'Yugoslavia': 'Yugoslav', 'Greece': 'Greek',
    'Portugal': 'Portuguese', 'Spain': 'Spanish', 'Canada': 'Canadian',
    'Mexico': 'Mexican', 'Cuba': 'Cuban', 'China': 'Chinese', 'Japan': 'Japanese',
    'Philippines': 'Filipino', 'India': 'Indian', 'Finland': 'Finnish',
    'Lithuania': 'Lithuanian', 'Romania': 'Romanian', 'Luxembourg': 'Luxembourgish',
    'Puerto Rico': 'Puerto Rican', 'West Indies': 'West Indian', 'Syria': 'Syrian',
    'Turkey': 'Turkish', 'Latvia': 'Latvian', 'Iceland': 'Icelandic',
    'Albania': 'Albanian', 'Lebanon': 'Lebanese', 'Bulgaria': 'Bulgarian',
    'Korea': 'Korean', 'Estonia': 'Estonian', 'Australia/New Zealand': 'Australian/New Zealand',
    'South America': 'South American', 'Central America': 'Central American',
    'Africa': 'African', 'US-born': 'American',
}

def get_demonym(country):
    return DEMONYMS.get(country, country)

# =============================================================================
# DATA LOADING (Aggregated Files)
# =============================================================================

PROCESSED_DIR = Path("data/processed")

def load_data():
    """Load pre-aggregated data files."""
    data = {}
    try:
        # Main marriage aggregation
        data['marriage_agg'] = pd.read_csv(PROCESSED_DIR / "marriage_agg.csv", low_memory=False)
        data['marriage_agg']['YEAR'] = data['marriage_agg']['YEAR'].astype(int)

        # Spouse backgrounds
        data['spouse_bg'] = pd.read_csv(PROCESSED_DIR / "spouse_backgrounds.csv", low_memory=False)
        data['spouse_bg']['YEAR'] = data['spouse_bg']['YEAR'].astype(int)

        # Metadata
        with open(PROCESSED_DIR / "metadata.json", 'r') as f:
            data['metadata'] = json.load(f)

        return data
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Run 'python preprocess_for_deploy.py' first to generate aggregated files.")
        return None

print("Loading aggregated data...")
DATA = load_data()

if DATA is None:
    print("ERROR: Run preprocess_for_deploy.py first")
    exit(1)

# Extract from metadata
valid_origins = DATA['metadata']['valid_origins']
mother_origins = valid_origins
father_origins = valid_origins
years = DATA['metadata']['years']

def build_origin_dropdown_options(origins):
    foreign = [o for o in origins if o != 'US-born']
    has_american = 'US-born' in origins
    options = [{'label': 'Any Foreign Country', 'value': 'Any'}]
    options.append({'label': '-- Foreign Countries --', 'value': '', 'disabled': True})
    options.extend([{'label': c, 'value': c} for c in foreign])
    if has_american:
        options.append({'label': '-- Native-Born --', 'value': '', 'disabled': True})
        options.append({'label': 'American', 'value': 'US-born'})
    return options

mother_dropdown_options = build_origin_dropdown_options(mother_origins)
father_dropdown_options = build_origin_dropdown_options(father_origins)

def generate_dynamic_presets():
    presets = [{'label': 'Select a preset...', 'value': ''}]
    presets.append({'label': '-- Same Origin Parents --', 'value': '', 'disabled': True})
    for country in DATA['metadata']['presets_same_origin']:
        demonym = get_demonym(country)
        presets.append({'label': f'{demonym} + {demonym}', 'value': f'{country}|{country}'})
    presets.append({'label': '-- Mixed Origin Parents --', 'value': '', 'disabled': True})
    for combo in DATA['metadata']['presets_mixed_origin']:
        mother, father = combo.split('|')
        presets.append({'label': f'{get_demonym(mother)} mom x {get_demonym(father)} dad', 'value': combo})
    return presets

DYNAMIC_PRESETS = generate_dynamic_presets()

# =============================================================================
# DATA ACCESS FUNCTIONS (Work with aggregated data)
# =============================================================================

NON_COUNTRIES = {'US-born', 'Unknown', 'N/A', 'Abroad/At Sea', 'Missing'}

def get_filtered_marriage_data(mother, father, year):
    """Filter the marriage aggregation data."""
    df = DATA['marriage_agg'].copy()
    if mother != 'Any':
        df = df[df['MOTHER_ORIGIN'] == mother]
    else:
        df = df[~df['MOTHER_ORIGIN'].isin(NON_COUNTRIES)]
    if father != 'Any':
        df = df[df['FATHER_ORIGIN'] == father]
    else:
        df = df[~df['FATHER_ORIGIN'].isin(NON_COUNTRIES)]
    if year != 'All':
        df = df[df['YEAR'] == int(year)]
    return df


def get_filtered_spouse_data(mother, father, year):
    """Filter the spouse background data."""
    df = DATA['spouse_bg'].copy()
    if mother != 'Any':
        df = df[df['MOTHER_ORIGIN'] == mother]
    else:
        df = df[~df['MOTHER_ORIGIN'].isin(NON_COUNTRIES)]
    if father != 'Any':
        df = df[df['FATHER_ORIGIN'] == father]
    else:
        df = df[~df['FATHER_ORIGIN'].isin(NON_COUNTRIES)]
    if year != 'All':
        df = df[df['YEAR'] == int(year)]
    return df


def get_marriage_stats(mother, father, year):
    """Get marriage type statistics."""
    df = get_filtered_marriage_data(mother, father, year)
    if len(df) == 0:
        return None, 0, 0
    stats = df.groupby('MARRIAGE_TYPE').agg({
        'WEIGHTED_COUNT': 'sum',
        'UNWEIGHTED_N': 'sum'
    }).reset_index()
    weighted_total = stats['WEIGHTED_COUNT'].sum()
    unweighted_total = stats['UNWEIGHTED_N'].sum()
    return stats, weighted_total, unweighted_total


def get_comparison_stats(mother_origin, father_origin, year='All'):
    """Get marriage statistics for comparison."""
    df = get_filtered_marriage_data(mother_origin, father_origin, year)
    if len(df) == 0:
        return None

    stats = df.groupby('MARRIAGE_TYPE')['WEIGHTED_COUNT'].sum()
    total = stats.sum()
    if total == 0:
        return None

    pcts = (stats / total * 100).to_dict()
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
    """Get trend data across years."""
    df = get_filtered_marriage_data(mother, father, 'All')
    if len(df) == 0:
        return {}

    MIN_TREND_SAMPLE = 10000
    trends = {}

    for yr in sorted(df['YEAR'].unique()):
        yr_df = df[df['YEAR'] == yr]
        stats = yr_df.groupby('MARRIAGE_TYPE')['WEIGHTED_COUNT'].sum()
        total = stats.sum()
        if total < MIN_TREND_SAMPLE:
            continue
        pcts = (stats / total * 100).to_dict()
        trends[yr] = {
            'third_gen': sum(v for k, v in pcts.items() if '3rd+ gen' in k),
            'same': sum(v for k, v in pcts.items() if 'same origin' in k),
            'ethnic_total': sum(v for k, v in pcts.items() if any(x in k for x in ['same origin', "mother's origin", "father's origin", 'both heritages'])),
            'sample_size': total,
        }
    return trends


def get_year_composition(mother, father):
    """Get the composition of data by census year for the current selection."""
    df = get_filtered_marriage_data(mother, father, 'All')
    if len(df) == 0:
        return {}

    year_totals = df.groupby('YEAR')['WEIGHTED_COUNT'].sum()
    total = year_totals.sum()

    composition = {}
    for yr, weight in year_totals.items():
        composition[int(yr)] = {
            'weighted': weight,
            'pct': weight / total * 100 if total > 0 else 0
        }
    return composition


def get_integration_ranking(year='All'):
    """Get all same-origin groups ranked by outmarriage rates."""
    df = DATA['marriage_agg'].copy()
    if year != 'All':
        df = df[df['YEAR'] == int(year)]

    # Filter to same-origin parents only
    df = df[df['MOTHER_ORIGIN'] == df['FATHER_ORIGIN']]
    df = df[~df['MOTHER_ORIGIN'].isin(NON_COUNTRIES)]

    results = []
    MIN_SAMPLE = 20000  # Lowered to include more countries

    for origin in df['MOTHER_ORIGIN'].unique():
        origin_df = df[df['MOTHER_ORIGIN'] == origin]
        total = origin_df['WEIGHTED_COUNT'].sum()

        if total < MIN_SAMPLE:
            continue

        stats = origin_df.groupby('MARRIAGE_TYPE')['WEIGHTED_COUNT'].sum()
        pcts = (stats / total * 100).to_dict()

        same_origin_rate = sum(v for k, v in pcts.items() if 'same origin' in k)
        third_gen_rate = sum(v for k, v in pcts.items() if '3rd+ gen' in k)
        diff_origin_rate = sum(v for k, v in pcts.items() if 'different origin' in k)

        results.append({
            'origin': origin,
            'demonym': get_demonym(origin),
            'same_origin_rate': same_origin_rate,
            'integration_rate': 100 - same_origin_rate,  # Openness to cross-ethnic
            'third_gen_rate': third_gen_rate,
            'diff_origin_rate': diff_origin_rate,
            'population': total
        })

    # Sort by integration rate (most integrated first)
    results.sort(key=lambda x: x['integration_rate'], reverse=True)
    return results


def get_spouse_generation_data(mother, father, year):
    """Get spouse generation composition for the current selection."""
    df = get_filtered_spouse_data(mother, father, year)
    if len(df) == 0:
        return {}

    gen_totals = df.groupby('SPOUSE_GEN')['WEIGHTED_COUNT'].sum()
    total = gen_totals.sum()

    result = {}
    for gen in ['1st gen immigrant', '2nd gen', '3rd+ gen American']:
        if gen in gen_totals.index:
            result[gen] = {
                'count': gen_totals[gen],
                'pct': gen_totals[gen] / total * 100 if total > 0 else 0
            }
        else:
            result[gen] = {'count': 0, 'pct': 0}

    return result


def get_clustering_data(year='All'):
    """Get marriage clustering data: who did children of each group marry?

    Returns a matrix showing what % of children from same-origin Group X
    married someone from Group Y (including 3rd+ gen Americans).
    """
    df = DATA['spouse_bg'].copy()
    if year != 'All':
        df = df[df['YEAR'] == int(year)]

    # Filter to same-origin parents for cleaner clustering view
    df = df[df['MOTHER_ORIGIN'] == df['FATHER_ORIGIN']]
    df = df[~df['MOTHER_ORIGIN'].isin(NON_COUNTRIES)]

    # Determine spouse heritage
    def get_spouse_heritage(row):
        if row['SPOUSE_GEN'] == '3rd+ gen American':
            return 'American'
        elif row['SPOUSE_GEN'] == '1st gen immigrant':
            return row['SPOUSE_COUNTRY']
        else:  # 2nd gen - use primary parent origin
            if str(row['SPOUSE_FATHER']) not in ['US-born', 'Unknown', 'N/A', 'nan']:
                return row['SPOUSE_FATHER']
            elif str(row['SPOUSE_MOTHER']) not in ['US-born', 'Unknown', 'N/A', 'nan']:
                return row['SPOUSE_MOTHER']
            return 'Unknown'

    df['SPOUSE_HERITAGE'] = df.apply(get_spouse_heritage, axis=1)

    # Filter out unknown spouse heritage
    df = df[~df['SPOUSE_HERITAGE'].isin(['Unknown', 'N/A', 'US-born'])]

    # Get major parent origins (those with substantial data)
    MIN_SAMPLE = 100000
    parent_totals = df.groupby('MOTHER_ORIGIN')['WEIGHTED_COUNT'].sum()
    major_parents = parent_totals[parent_totals >= MIN_SAMPLE].index.tolist()

    # Get major spouse heritages
    spouse_totals = df.groupby('SPOUSE_HERITAGE')['WEIGHTED_COUNT'].sum()
    major_spouses = spouse_totals[spouse_totals >= 50000].index.tolist()

    # Ensure 'American' is included if present
    if 'American' in spouse_totals.index and 'American' not in major_spouses:
        major_spouses.append('American')

    # Build clustering matrix
    clustering = df[df['MOTHER_ORIGIN'].isin(major_parents) & df['SPOUSE_HERITAGE'].isin(major_spouses)]
    clustering = clustering.groupby(['MOTHER_ORIGIN', 'SPOUSE_HERITAGE'])['WEIGHTED_COUNT'].sum().reset_index()

    # Pivot and convert to percentages
    pivot = clustering.pivot_table(index='MOTHER_ORIGIN', columns='SPOUSE_HERITAGE',
                                   values='WEIGHTED_COUNT', fill_value=0)
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    return pivot_pct, major_parents, major_spouses


def get_network_data(year='All'):
    """Get network graph data for ethnic clustering visualization.

    Returns nodes (ethnic groups) and edges (intermarriage connections).
    Edge weights are POPULATION-ADJUSTED AFFINITY: observed rate / expected rate.
    Affinity > 1 means groups married MORE than random chance would predict.
    This corrects for large groups (like Germans) dominating raw percentages.
    """
    df = DATA['spouse_bg'].copy()
    if year != 'All':
        df = df[df['YEAR'] == int(year)]

    # Filter to same-origin parents
    df = df[df['MOTHER_ORIGIN'] == df['FATHER_ORIGIN']]
    df = df[~df['MOTHER_ORIGIN'].isin(NON_COUNTRIES)]

    # Exclude groups that don't cluster well with European immigrants
    EXCLUDED_FROM_NETWORK = {'Mexico', 'Cuba', 'West Indies', 'China', 'Japan'}
    df = df[~df['MOTHER_ORIGIN'].isin(EXCLUDED_FROM_NETWORK)]

    # Get spouse heritage (exclude 3rd+ gen American for cleaner network)
    def get_spouse_heritage(row):
        if row['SPOUSE_GEN'] == '3rd+ gen American':
            return None  # Exclude from network
        elif row['SPOUSE_GEN'] == '1st gen immigrant':
            return row['SPOUSE_COUNTRY']
        else:
            if str(row['SPOUSE_FATHER']) not in ['US-born', 'Unknown', 'N/A', 'nan']:
                return row['SPOUSE_FATHER']
            elif str(row['SPOUSE_MOTHER']) not in ['US-born', 'Unknown', 'N/A', 'nan']:
                return row['SPOUSE_MOTHER']
            return None

    df['SPOUSE_HERITAGE'] = df.apply(get_spouse_heritage, axis=1)
    df = df[df['SPOUSE_HERITAGE'].notna()]
    df = df[~df['SPOUSE_HERITAGE'].isin(['Unknown', 'N/A', 'US-born'] + list(EXCLUDED_FROM_NETWORK))]

    # Get groups with substantial data
    MIN_SAMPLE = 50000
    parent_totals = df.groupby('MOTHER_ORIGIN')['WEIGHTED_COUNT'].sum()
    major_groups = parent_totals[parent_totals >= MIN_SAMPLE].index.tolist()

    # Calculate market share for each group (expected rate if random matching)
    spouse_totals = df.groupby('SPOUSE_HERITAGE')['WEIGHTED_COUNT'].sum()
    total_market = spouse_totals.sum()
    market_share = spouse_totals / total_market

    # Calculate population-adjusted affinity for all pairs
    directed_affinity = {}
    for parent in major_groups:
        parent_df = df[df['MOTHER_ORIGIN'] == parent]
        total = parent_df['WEIGHTED_COUNT'].sum()
        if total == 0:
            continue

        spouse_counts = parent_df.groupby('SPOUSE_HERITAGE')['WEIGHTED_COUNT'].sum()

        for spouse, count in spouse_counts.items():
            if spouse in major_groups and spouse != parent:
                observed_rate = count / total
                expected_rate = market_share.get(spouse, 0)
                if expected_rate > 0:
                    affinity = observed_rate / expected_rate
                    directed_affinity[(parent, spouse)] = affinity

    # Build symmetric edges (average affinity of both directions)
    edges = []
    seen_pairs = set()
    for (source, target), affinity in directed_affinity.items():
        pair = tuple(sorted([source, target]))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)

        # Get affinity in both directions and average
        aff_ab = directed_affinity.get((source, target), 0)
        aff_ba = directed_affinity.get((target, source), 0)
        avg_affinity = (aff_ab + aff_ba) / 2

        # Only show edges where groups married MORE than random chance (affinity > 1)
        if avg_affinity >= 1.0:
            edges.append({
                'source': pair[0],
                'target': pair[1],
                'weight': avg_affinity
            })

    # Only include nodes that have at least one edge
    connected_groups = set()
    for edge in edges:
        connected_groups.add(edge['source'])
        connected_groups.add(edge['target'])

    nodes = []
    for group in connected_groups:
        pop = parent_totals.get(group, 0)
        nodes.append({
            'id': group,
            'label': get_demonym(group),
            'population': pop
        })

    # Compute positions using force-directed layout
    positions = _compute_force_layout(nodes, edges)

    return nodes, edges, positions


def _compute_force_layout(nodes, edges, iterations=100):
    """Simple force-directed layout algorithm."""
    import random
    random.seed(42)  # For reproducibility

    n = len(nodes)
    if n == 0:
        return {}

    # Initialize random positions
    pos = {node['id']: [random.uniform(-1, 1), random.uniform(-1, 1)] for node in nodes}

    # Build adjacency with weights
    adj = {node['id']: {} for node in nodes}
    for edge in edges:
        s, t, w = edge['source'], edge['target'], edge['weight']
        if s in adj and t in adj:
            adj[s][t] = adj[s].get(t, 0) + w
            adj[t][s] = adj[t].get(s, 0) + w

    # Force-directed iterations
    k = 1.0 / np.sqrt(n)  # Optimal distance
    temp = 0.5

    for _ in range(iterations):
        displacement = {node['id']: [0.0, 0.0] for node in nodes}

        # Repulsive forces (all pairs)
        for i, n1 in enumerate(nodes):
            for n2 in nodes[i+1:]:
                id1, id2 = n1['id'], n2['id']
                dx = pos[id1][0] - pos[id2][0]
                dy = pos[id1][1] - pos[id2][1]
                dist = max(np.sqrt(dx*dx + dy*dy), 0.01)
                force = k * k / dist
                displacement[id1][0] += dx / dist * force
                displacement[id1][1] += dy / dist * force
                displacement[id2][0] -= dx / dist * force
                displacement[id2][1] -= dy / dist * force

        # Attractive forces (connected pairs)
        for edge in edges:
            s, t, w = edge['source'], edge['target'], edge['weight']
            if s not in pos or t not in pos:
                continue
            dx = pos[s][0] - pos[t][0]
            dy = pos[s][1] - pos[t][1]
            dist = max(np.sqrt(dx*dx + dy*dy), 0.01)
            force = dist * dist / k * (w / 10)  # Weight influences attraction
            displacement[s][0] -= dx / dist * force
            displacement[s][1] -= dy / dist * force
            displacement[t][0] += dx / dist * force
            displacement[t][1] += dy / dist * force

        # Apply displacements with temperature
        for node in nodes:
            nid = node['id']
            disp_len = max(np.sqrt(displacement[nid][0]**2 + displacement[nid][1]**2), 0.01)
            pos[nid][0] += displacement[nid][0] / disp_len * min(disp_len, temp)
            pos[nid][1] += displacement[nid][1] / disp_len * min(disp_len, temp)
            # Keep within bounds
            pos[nid][0] = max(-2, min(2, pos[nid][0]))
            pos[nid][1] = max(-2, min(2, pos[nid][1]))

        temp *= 0.95  # Cool down

    return pos


def get_scatter_data(year='All'):
    """Get population size vs ethnic retention for scatter plot."""
    df = DATA['marriage_agg'].copy()
    if year != 'All':
        df = df[df['YEAR'] == int(year)]

    # Filter to same-origin parents
    df = df[df['MOTHER_ORIGIN'] == df['FATHER_ORIGIN']]
    df = df[~df['MOTHER_ORIGIN'].isin(NON_COUNTRIES)]

    results = []
    MIN_SAMPLE = 20000

    for origin in df['MOTHER_ORIGIN'].unique():
        origin_df = df[df['MOTHER_ORIGIN'] == origin]
        total = origin_df['WEIGHTED_COUNT'].sum()

        if total < MIN_SAMPLE:
            continue

        stats = origin_df.groupby('MARRIAGE_TYPE')['WEIGHTED_COUNT'].sum()
        pcts = (stats / total * 100).to_dict()

        same_origin_rate = sum(v for k, v in pcts.items() if 'same origin' in k)

        results.append({
            'origin': origin,
            'demonym': get_demonym(origin),
            'population': total,
            'same_origin_rate': same_origin_rate
        })

    return results


def detect_anomalies(year='All'):
    """Detect groups with counter-intuitive trends (like Italy's high retention)."""
    anomalies = []

    # Get trend data for major same-origin groups
    df = DATA['marriage_agg'].copy()
    df = df[df['MOTHER_ORIGIN'] == df['FATHER_ORIGIN']]
    df = df[~df['MOTHER_ORIGIN'].isin(NON_COUNTRIES)]

    major_origins = df.groupby('MOTHER_ORIGIN')['WEIGHTED_COUNT'].sum()
    major_origins = major_origins[major_origins >= 100000].index.tolist()

    # First pass: collect all trend data to find the average
    all_trends = {}
    for origin in major_origins:
        trends = get_trend_data(origin, origin)
        if len(trends) >= 2:
            years_list = sorted(trends.keys())
            first_ethnic = trends[years_list[0]]['ethnic_total']
            last_ethnic = trends[years_list[-1]]['ethnic_total']
            all_trends[origin] = {
                'trends': trends,
                'first_year': years_list[0],
                'last_year': years_list[-1],
                'first_rate': first_ethnic,
                'last_rate': last_ethnic,
                'change': last_ethnic - first_ethnic
            }

    if not all_trends:
        return anomalies

    # Calculate average change across all groups
    avg_change = np.mean([t['change'] for t in all_trends.values()])

    # Detect anomalies
    for origin, data in all_trends.items():
        change = data['change']
        first_ethnic = data['first_rate']
        last_ethnic = data['last_rate']

        # Counter-trend: Most groups decline, but this one increased or stayed stable
        if avg_change < -5 and change > 0:
            anomalies.append({
                'origin': origin,
                'demonym': get_demonym(origin),
                'first_year': data['first_year'],
                'last_year': data['last_year'],
                'first_rate': first_ethnic,
                'last_rate': last_ethnic,
                'change': change,
                'avg_change': avg_change,
                'type': 'counter_trend'
            })
        # Consistently high retention (>70%) - notable for late-arriving groups
        elif first_ethnic > 70 and last_ethnic > 70:
            anomalies.append({
                'origin': origin,
                'demonym': get_demonym(origin),
                'first_year': data['first_year'],
                'last_year': data['last_year'],
                'first_rate': first_ethnic,
                'last_rate': last_ethnic,
                'change': change,
                'avg_change': avg_change,
                'type': 'consistently_high'
            })

    return anomalies


def get_single_origin_overview(origin, year='All'):
    """Get marriage outcomes for all parent combinations involving a single origin.

    Returns categories that sum to 100%:
    - heritage_rate: Married within immigrant heritage (same, mother's, father's, or both heritages)
    - diff_origin_rate: Married different immigrant origin
    - third_gen_rate: Married 3rd+ gen American
    """
    df = DATA['marriage_agg'].copy()
    if year != 'All':
        df = df[df['YEAR'] == int(year)]

    # Filter to combinations where either mother OR father is the selected origin
    df = df[(df['MOTHER_ORIGIN'] == origin) | (df['FATHER_ORIGIN'] == origin)]
    df = df[~df['MOTHER_ORIGIN'].isin(NON_COUNTRIES)]
    df = df[~df['FATHER_ORIGIN'].isin(NON_COUNTRIES)]

    results = []
    MIN_SAMPLE = 20000

    # Get unique combinations
    combinations = df.groupby(['MOTHER_ORIGIN', 'FATHER_ORIGIN']).agg({
        'WEIGHTED_COUNT': 'sum',
        'UNWEIGHTED_N': 'sum'
    }).reset_index()

    for _, row in combinations.iterrows():
        mother = row['MOTHER_ORIGIN']
        father = row['FATHER_ORIGIN']
        total = row['WEIGHTED_COUNT']

        if total < MIN_SAMPLE:
            continue

        # Get marriage type breakdown
        combo_df = df[(df['MOTHER_ORIGIN'] == mother) & (df['FATHER_ORIGIN'] == father)]
        stats = combo_df.groupby('MARRIAGE_TYPE')['WEIGHTED_COUNT'].sum()
        pcts = (stats / total * 100).to_dict()

        # Calculate categories that sum to 100%
        # Heritage-based: same origin, mother's origin, father's origin, both heritages
        heritage_rate = sum(v for k, v in pcts.items()
                          if any(x in k for x in ['same origin', "mother's origin", "father's origin", 'both heritages']))
        # Different immigrant origin
        diff_origin_rate = sum(v for k, v in pcts.items() if 'different origin' in k)
        # 3rd+ gen American
        third_gen_rate = sum(v for k, v in pcts.items() if '3rd+ gen' in k)

        # Create label
        if mother == father:
            label = f"{get_demonym(mother)} x {get_demonym(father)}"
            combo_type = "same"
        elif mother == origin:
            label = f"{get_demonym(mother)} mom x {get_demonym(father)} dad"
            combo_type = "mother"
        else:
            label = f"{get_demonym(mother)} mom x {get_demonym(father)} dad"
            combo_type = "father"

        results.append({
            'mother': mother,
            'father': father,
            'label': label,
            'combo_type': combo_type,
            'population': total,
            'heritage_rate': heritage_rate,
            'diff_origin_rate': diff_origin_rate,
            'third_gen_rate': third_gen_rate,
        })

    # Sort by population
    results.sort(key=lambda x: x['population'], reverse=True)
    return results


def get_available_origins_for_overview():
    """Get list of origins with enough data for single-origin overview."""
    df = DATA['marriage_agg'].copy()

    # Get origins that appear as either mother or father with substantial data
    mother_totals = df.groupby('MOTHER_ORIGIN')['WEIGHTED_COUNT'].sum()
    father_totals = df.groupby('FATHER_ORIGIN')['WEIGHTED_COUNT'].sum()

    MIN_SAMPLE = 100000

    valid_origins = set()
    for origin in mother_totals[mother_totals >= MIN_SAMPLE].index:
        if origin not in NON_COUNTRIES:
            valid_origins.add(origin)
    for origin in father_totals[father_totals >= MIN_SAMPLE].index:
        if origin not in NON_COUNTRIES:
            valid_origins.add(origin)

    # Sort by total population
    origin_list = []
    for origin in valid_origins:
        total = mother_totals.get(origin, 0) + father_totals.get(origin, 0)
        origin_list.append((origin, total))

    origin_list.sort(key=lambda x: x[1], reverse=True)
    return [o[0] for o in origin_list]


def get_top_spouse_backgrounds(mother, father, year, exclude_heritage=None, top_n=5):
    """Get top spouse backgrounds from aggregated data."""
    df = get_filtered_spouse_data(mother, father, year)
    if len(df) == 0:
        return []

    if exclude_heritage is None:
        exclude_heritage = set()

    total = df['WEIGHTED_COUNT'].sum()
    results = {}

    # 1st gen immigrants
    imm = df[df['SPOUSE_GEN'] == '1st gen immigrant']
    if len(imm) > 0:
        for _, row in imm.groupby('SPOUSE_COUNTRY')['WEIGHTED_COUNT'].sum().items():
            country, weight = _, row
            if country not in exclude_heritage and country not in ['Unknown', 'N/A', 'American']:
                key = f"{get_demonym(country)} immigrant"
                results[key] = results.get(key, 0) + weight

    # 2nd gen
    second = df[df['SPOUSE_GEN'] == '2nd gen']
    if len(second) > 0:
        grouped = second.groupby(['SPOUSE_MOTHER', 'SPOUSE_FATHER'])['WEIGHTED_COUNT'].sum()
        for (sp_mom, sp_dad), weight in grouped.items():
            sp_mom = str(sp_mom) if pd.notna(sp_mom) else 'Unknown'
            sp_dad = str(sp_dad) if pd.notna(sp_dad) else 'Unknown'
            invalid = {'Unknown', 'N/A', 'US-born'}

            if sp_mom == sp_dad and sp_mom not in exclude_heritage and sp_mom not in invalid:
                key = f"2nd-gen {get_demonym(sp_mom)}"
            elif sp_mom not in invalid and sp_dad not in invalid:
                mom_excluded = sp_mom in exclude_heritage
                dad_excluded = sp_dad in exclude_heritage
                if mom_excluded and dad_excluded:
                    continue
                elif mom_excluded:
                    key = f"2nd-gen {get_demonym(sp_dad)}"
                elif dad_excluded:
                    key = f"2nd-gen {get_demonym(sp_mom)}"
                else:
                    origins = sorted([get_demonym(sp_mom), get_demonym(sp_dad)])
                    key = f"2nd-gen {origins[0]}x{origins[1]}"
            elif sp_mom not in invalid and sp_mom not in exclude_heritage:
                key = f"2nd-gen {get_demonym(sp_mom)}"
            elif sp_dad not in invalid and sp_dad not in exclude_heritage:
                key = f"2nd-gen {get_demonym(sp_dad)}"
            else:
                continue

            results[key] = results.get(key, 0) + weight

    result_list = [(key, weight / total * 100) for key, weight in results.items()]
    result_list.sort(key=lambda x: x[1], reverse=True)
    return result_list[:top_n]


def generate_summary(mother, father, year):
    """Generate detailed summary with trends, comparisons, and rich analysis."""
    stats, weighted_n, unweighted_n = get_marriage_stats(mother, father, year)
    if stats is None or weighted_n == 0:
        return "No data available for this selection."

    mother_dem = get_demonym(mother) if mother != 'Any' else None
    father_dem = get_demonym(father) if father != 'Any' else None

    # Build subject descriptions
    if mother != 'Any' and father != 'Any':
        if mother == father:
            subject = f"children of two **{mother_dem}** immigrant parents"
            subject_short = f"{mother_dem}-Americans"
        elif mother == 'US-born':
            subject = f"children of **{father_dem}** immigrant fathers and **American** mothers"
            subject_short = f"{father_dem}-Americans (with American mothers)"
        elif father == 'US-born':
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
        subject = "all married second-generation Americans (US-born children of immigrants)"
        subject_short = "married second-generation Americans"

    # Get year range from data
    df = get_filtered_marriage_data(mother, father, year)
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
        lines.append(f"**Warning: Very small sample size** - Only {unweighted_n:,} records ({weighted_n:,.0f} weighted individuals). Results may not be statistically reliable.")
        lines.append("")
        lines.append("*Consider selecting 'Any' for broader coverage or a different year range.*")
        return "\n".join(lines)

    if weighted_n < SMALL_SAMPLE:
        lines.append(f"*Based on {weighted_n:,.0f} individuals ({unweighted_n:,} census records)*")
        lines.append("")
        lines.append(f"**Note: Small sample size** - Results based on fewer than 50,000 weighted individuals.")
        lines.append("")
    else:
        lines.append(f"*Based on {weighted_n:,.0f} individuals ({unweighted_n:,} census records)*")
        lines.append("")

    # Add year-specific methodology notes with composition
    if year == 'All':
        year_comp = get_year_composition(mother, father)
        if year_comp:
            # Find the dominant year
            max_year = max(year_comp.keys(), key=lambda y: year_comp[y]['pct'])
            max_pct = year_comp[max_year]['pct']
            year_breakdown = ", ".join([f"{yr}: {data['pct']:.0f}%" for yr, data in sorted(year_comp.items())])
            lines.append(f"*Year composition: {year_breakdown}*")
            if max_pct > 40:
                lines.append(f"*Note: {max_year} contributes {max_pct:.0f}% of the pooled sample. Select individual years for temporal specificity.*")
            lines.append("")
    else:
        lines.append(f"*This {year} cross-section provides a snapshot free from repeated observations across census years.*")
        lines.append("")

    # Calculate percentages
    total = stats['WEIGHTED_COUNT'].sum()
    pcts = dict(zip(stats['MARRIAGE_TYPE'], stats['WEIGHTED_COUNT'] / total * 100))

    third_gen_pct = sum(v for k, v in pcts.items() if '3rd+ gen' in k)
    same_pct = sum(v for k, v in pcts.items() if 'same origin' in k)
    mother_pct = sum(v for k, v in pcts.items() if "mother's origin" in k)
    father_pct = sum(v for k, v in pcts.items() if "father's origin" in k)
    both_pct = sum(v for k, v in pcts.items() if 'both heritages' in k)
    diff_pct = sum(v for k, v in pcts.items() if 'different origin' in k)

    same_1st = pcts.get("Married same origin (1st gen immigrant)", 0)
    same_2nd = pcts.get("Married same origin (2nd gen)", 0)

    # Historical context
    HISTORICAL_CONTEXT = {
        'Germany': {
            'warning': '**Historical Note:** Anti-German sentiment during WWI (1917-1918) led some German-Americans to downplay their heritage, potentially affecting marriage pattern reporting.',
        }
    }

    # ==================== FINDINGS ====================
    lines.append("---")
    lines.append("")
    lines.append("#### Findings")
    lines.append("")

    if mother != 'Any' and father != 'Any' and mother == father:
        if mother in HISTORICAL_CONTEXT:
            lines.append(HISTORICAL_CONTEXT[mother]['warning'])
            lines.append("")

        lines.append(f"**Ethnic retention:** {same_pct:.1f}% of {subject_short} married someone of {mother_dem} heritage:")
        if same_1st >= 0.5:
            lines.append(f"  - {same_1st:.1f}% married a {mother_dem} immigrant (born in {mother})")
        if same_2nd >= 0.5:
            lines.append(f"  - {same_2nd:.1f}% married a second-generation {mother_dem}-American")
        lines.append("")
        lines.append(f"**Mainstream assimilation:** {third_gen_pct:.1f}% married a 3rd+ generation American - someone whose parents were also US-born.")
        lines.append("")
        if diff_pct >= 0.5:
            lines.append(f"**Cross-ethnic marriage:** {diff_pct:.1f}% married someone from a different immigrant background.")
            lines.append("")

        # Year-specific breakdown for same-origin selections
        if year == 'All':
            year_data = []
            for yr in sorted(df['YEAR'].unique()):
                yr_df = df[df['YEAR'] == yr]
                yr_total = yr_df['WEIGHTED_COUNT'].sum()
                if yr_total < 5000:
                    continue
                yr_stats = yr_df.groupby('MARRIAGE_TYPE')['WEIGHTED_COUNT'].sum()
                yr_pcts = (yr_stats / yr_total * 100).to_dict()
                yr_third = sum(v for k, v in yr_pcts.items() if '3rd+ gen' in k)
                yr_same = sum(v for k, v in yr_pcts.items() if 'same origin' in k)
                year_data.append((yr, yr_third, yr_same, yr_total))

            if len(year_data) >= 2:
                third_vals = [d[1] for d in year_data]
                same_vals = [d[2] for d in year_data]
                third_range = max(third_vals) - min(third_vals)
                same_range = max(same_vals) - min(same_vals)

                if third_range > 15 or same_range > 15:
                    lines.append("**Trends by census year:**")
                    lines.append("")
                    lines.append("| Year | Ethnic Retention | 3rd+ Gen American |")
                    lines.append("|------|------------------|-------------------|")
                    for yr, yr_third, yr_same, yr_total in year_data:
                        sample_note = " *(small sample)*" if yr_total < 20000 else ""
                        lines.append(f"| {yr} | {yr_same:.0f}% | {yr_third:.0f}%{sample_note} |")
                    lines.append("")

                    first_same = year_data[0][2]
                    last_same = year_data[-1][2]
                    if last_same < first_same - 20:
                        lines.append(f"*Ethnic retention declined substantially from {first_same:.0f}% to {last_same:.0f}% over this period, suggesting increasing integration into mainstream American society.*")
                        lines.append("")
                    elif last_same > first_same + 20:
                        lines.append(f"*Ethnic retention increased from {first_same:.0f}% to {last_same:.0f}%, possibly reflecting growth of ethnic community institutions or chain migration patterns.*")
                        lines.append("")

    elif mother != 'Any' and father != 'Any' and mother != father:
        mother_is_american = mother == 'US-born'
        father_is_american = father == 'US-born'

        if mother_is_american or father_is_american:
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
        specific_origin = mother if mother != 'Any' else father
        specific_dem = get_demonym(specific_origin)
        parent_type = "mother" if mother != 'Any' else "father"
        relevant_pct = mother_pct if mother != 'Any' else father_pct
        heritage_total = same_pct + relevant_pct + both_pct

        if heritage_total >= third_gen_pct and heritage_total >= diff_pct:
            lines.append(f"**{specific_dem} heritage predominated:** {heritage_total:.1f}% married someone with {specific_dem} ancestry.")
        elif third_gen_pct >= heritage_total and third_gen_pct >= diff_pct:
            lines.append(f"**Mainstream assimilation:** {third_gen_pct:.1f}% married a 3rd+ generation American.")
        else:
            lines.append(f"**Cross-ethnic patterns:** {diff_pct:.1f}% married someone from a different immigrant background.")
        lines.append("")

        lines.append("**Full breakdown:**")
        if heritage_total >= 0.5:
            lines.append(f"- {specific_dem} spouse: {heritage_total:.1f}%")
        if third_gen_pct >= 0.5:
            lines.append(f"- 3rd+ gen American: {third_gen_pct:.1f}%")
        if diff_pct >= 0.5:
            lines.append(f"- Other immigrant backgrounds: {diff_pct:.1f}%")
        lines.append("")

    else:
        heritage_total = same_pct + mother_pct + father_pct + both_pct

        # Calculate mixed-origin parent statistics
        mixed_origin_df = df[df['MOTHER_ORIGIN'] != df['FATHER_ORIGIN']]
        mixed_origin_weighted = mixed_origin_df['WEIGHTED_COUNT'].sum()
        mixed_origin_pct = (mixed_origin_weighted / total * 100) if total > 0 else 0

        # Calculate the "outside community" total
        outside_community_pct = third_gen_pct + diff_pct

        if heritage_total > third_gen_pct and heritage_total > diff_pct:
            # Heritage-based is largest single category, but check if majority married outside
            if outside_community_pct > 50:
                lines.append(f"**Heritage-based marriages were the most common category** at {heritage_total:.1f}%, but a majority ({outside_community_pct:.1f}%) married outside their parents' immigrant communities - either a 3rd+ generation American ({third_gen_pct:.1f}%) or someone from a different immigrant background ({diff_pct:.1f}%).")
            else:
                lines.append(f"**Heritage-based marriages led:** {heritage_total:.1f}% married someone connected to their immigrant heritage - the largest category.")
        elif third_gen_pct > heritage_total and third_gen_pct > diff_pct:
            lines.append(f"**Mainstream assimilation led:** {third_gen_pct:.1f}% married a 3rd+ generation American - the largest category.")
        else:
            lines.append(f"**Balanced patterns:** Marriage to 3rd+ gen Americans ({third_gen_pct:.1f}%) and heritage-based marriages ({heritage_total:.1f}%) occurred at similar rates.")
        lines.append("")

        lines.append(f"**Heritage-based marriages:** {heritage_total:.1f}%")
        if same_pct >= 0.5:
            lines.append(f"  - {same_pct:.1f}% married someone matching both parents' origins")
        if mother_pct + father_pct + both_pct >= 0.5:
            lines.append(f"  - {mother_pct + father_pct + both_pct:.1f}% married someone matching one parent's origin (among those with mixed-origin parents)")
        lines.append("")
        lines.append(f"**Married outside parents' communities:** {outside_community_pct:.1f}%")
        lines.append(f"  - {third_gen_pct:.1f}% married a 3rd+ generation American")
        if diff_pct >= 0.5:
            lines.append(f"  - {diff_pct:.1f}% married someone from a different immigrant background")
        lines.append("")

        # Add mixed-origin parent context
        if mixed_origin_pct >= 0.5:
            lines.append("---")
            lines.append("")
            lines.append("#### Mixed-Origin Families")
            lines.append("")
            lines.append(f"**{mixed_origin_pct:.1f}%** of married second-generation Americans had parents from different countries (mixed-origin parents).")
            if both_pct >= 0.1:
                # Calculate what % of mixed-origin individuals married someone sharing both heritages
                both_of_mixed_pct = (both_pct / mixed_origin_pct * 100) if mixed_origin_pct > 0 else 0
                lines.append(f"Among this group, {both_pct:.1f}% of the total population ({both_of_mixed_pct:.1f}% of those with mixed-origin parents) married someone who shared both of their parents' heritages.")
            lines.append("")

    # ==================== TOP SPOUSE BACKGROUNDS ====================
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

    top_backgrounds = get_top_spouse_backgrounds(mother, father, year, exclude_heritage=exclude_set, top_n=5)
    displayable_backgrounds = [(bg, pct) for bg, pct in top_backgrounds if pct >= 0.5]

    if displayable_backgrounds and diff_pct >= 1:
        lines.append("---")
        lines.append("")
        lines.append("#### Most Common Cross-Ethnic Marriages")
        lines.append("")

        if mother != 'Any' and father != 'Any' and mother == father:
            lines.append(f"Among {subject_short} who married outside the {mother_dem} community, the most common spouse backgrounds were:")
        elif mother != 'Any' and father != 'Any':
            lines.append(f"Among those who married outside both {mother_dem} and {father_dem} communities:")
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

            if abs(change_third) >= 3:
                direction = "rose" if change_third > 0 else "fell"
                lines.append(f"- Marriage to 3rd+ generation Americans {direction} from {first_third:.0f}% to {last_third:.0f}% ({'+' if change_third > 0 else ''}{change_third:.0f} points)")
            else:
                lines.append(f"- Marriage to 3rd+ generation Americans remained relatively stable (~{(first_third + last_third)/2:.0f}%)")

            if abs(change_ethnic) >= 3:
                direction = "increased" if change_ethnic > 0 else "decreased"
                lines.append(f"- Heritage-based marriage {direction} from {first_ethnic:.0f}% to {last_ethnic:.0f}% ({'+' if change_ethnic > 0 else ''}{change_ethnic:.0f} points)")
            else:
                lines.append(f"- Heritage-based marriage remained relatively stable (~{(first_ethnic + last_ethnic)/2:.0f}%)")
            lines.append("")

            # Show marriage patterns by census year
            lines.append("**Marriage Patterns by Census Year:**")
            lines.append("")
            lines.append("| Year | 3rd+ Gen | Heritage-Based | Other Immigrant |")
            lines.append("|------|----------|----------------|-----------------|")
            for yr in years_list:
                yr_third = trends[yr]['third_gen']
                yr_ethnic = trends[yr]['ethnic_total']
                yr_other = 100 - yr_third - yr_ethnic
                lines.append(f"| {yr} | {yr_third:.0f}% | {yr_ethnic:.0f}% | {yr_other:.0f}% |")
            lines.append("")

    # ==================== COMPARATIVE CONTEXT ====================
    # Skip this section entirely for Any x Any (info already in Findings)
    if mother == 'Any' and father == 'Any':
        pass  # No comparative context needed
    elif mother != 'Any' and father != 'Any' and mother != father:
        lines.append("---")
        lines.append("")
        lines.append("#### Comparative Context")
        lines.append("")
        mother_is_american = mother == 'US-born'
        father_is_american = father == 'US-born'

        if mother_is_american or father_is_american:
            immigrant_parent = father if mother_is_american else mother
            immigrant_dem = get_demonym(immigrant_parent)
            immigrant_pct = father_pct if mother_is_american else mother_pct

            immigrant_stats = get_comparison_stats(immigrant_parent, immigrant_parent, year)
            overall_stats = get_comparison_stats('Any', 'Any', year)

            if immigrant_stats or overall_stats:
                lines.append(f"**How do children of {immigrant_dem}-American marriages compare?**")
                lines.append("")
                lines.append("| Group | 3rd+ Gen | Ethnic Retention |")
                lines.append("|-------|----------|------------------|")
                lines.append(f"| **{immigrant_dem} x American (selected)** | {third_gen_pct:.0f}% | {immigrant_pct:.0f}% |")
                if immigrant_stats:
                    lines.append(f"| {immigrant_dem} x {immigrant_dem} | {immigrant_stats['third_gen']:.0f}% | {immigrant_stats['same']:.0f}% |")
                if overall_stats:
                    lines.append(f"| All 2nd-gen Americans | {overall_stats['third_gen']:.0f}% | {overall_stats['same']:.0f}% |")
                lines.append("")
        else:
            comparisons = []
            mom_stats = get_comparison_stats(mother, mother, year)
            if mom_stats:
                comparisons.append((f"{mother_dem} x {mother_dem}", mom_stats, mother_dem))
            dad_stats = get_comparison_stats(father, father, year)
            if dad_stats:
                comparisons.append((f"{father_dem} x {father_dem}", dad_stats, father_dem))

            if comparisons:
                lines.append("**How does this mixed-heritage group compare?**")
                lines.append("")
                lines.append("| Group | 3rd+ Gen | Ethnic Retention |")
                lines.append("|-------|----------|------------------|")
                current_ethnic = mother_pct + father_pct + both_pct
                lines.append(f"| **{mother_dem} x {father_dem} (selected)** | {third_gen_pct:.0f}% | {current_ethnic:.0f}% |")
                for comp_name, stats, origin_dem in comparisons:
                    comp_ethnic = stats['same'] if origin_dem else stats['mother'] + stats['father'] + stats['both']
                    lines.append(f"| {comp_name} | {stats['third_gen']:.0f}% | {comp_ethnic:.0f}% |")
                lines.append("")

    elif mother != 'Any' and father != 'Any' and mother == father:
        lines.append("---")
        lines.append("")
        lines.append("#### Comparative Context")
        lines.append("")

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

        outmarriage = 100 - same_pct
        if same_pct < 30:
            lines.append(f"**Low ethnic retention:** With {outmarriage:.0f}% marrying outside the {mother_dem} community, this group showed weak co-ethnic marriage patterns.")
        elif same_pct < 50:
            lines.append(f"**Moderate ethnic retention:** About {same_pct:.0f}% of {mother_dem}-Americans married within their ethnic community, while {outmarriage:.0f}% married outside it.")
        else:
            lines.append(f"**Strong ethnic retention:** {same_pct:.1f}% of {mother_dem}-Americans married within their ethnic community - a majority staying in-group.")
        lines.append("")

        if overall_stats:
            diff = third_gen_pct - overall_stats['third_gen']
            if diff > 5:
                lines.append(f"**Above-average mainstream integration:** {mother_dem}-Americans married 3rd+ generation Americans at rates {diff:.0f} percentage points above the overall average.")
            elif diff < -5:
                lines.append(f"**Below-average mainstream integration:** {mother_dem}-Americans married 3rd+ generation Americans at rates {abs(diff):.0f} percentage points below average.")

    elif mother != 'Any' or father != 'Any':
        lines.append("---")
        lines.append("")
        lines.append("#### Comparative Context")
        lines.append("")

        specific_origin = mother if mother != 'Any' else father
        specific_dem = get_demonym(specific_origin)
        parent_type = "mother" if mother != 'Any' else "father"
        other_parent = "father" if mother != 'Any' else "mother"
        relevant_pct = mother_pct if mother != 'Any' else father_pct

        same_origin_stats = get_comparison_stats(specific_origin, specific_origin, year)
        lines.append(f"**How do children of {specific_dem} {parent_type}s compare?**")
        lines.append("")

        if same_origin_stats:
            lines.append(f"| Group | 3rd+ Gen | {specific_dem} Spouse |")
            lines.append("|-------|----------|----------------|")
            lines.append(f"| **{specific_dem} {parent_type} x Any {other_parent}** | {third_gen_pct:.0f}% | {relevant_pct + same_pct:.0f}% |")
            lines.append(f"| {specific_dem} x {specific_dem} (both parents) | {same_origin_stats['third_gen']:.0f}% | {same_origin_stats['same']:.0f}% |")
            lines.append("")

    return "\n".join(lines)

# =============================================================================
# CUSTOM CSS
# =============================================================================

CUSTOM_CSS = """
/* Base */
body {
    background: linear-gradient(180deg, #edf1f2 0%, #d0dbdd 100%);
    font-family: 'Hanken Grotesk', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #194852;
    min-height: 100vh;
}

/* Header */
.header-section {
    padding: 2rem 0 1.5rem 0;
    text-align: center;
}

.main-title {
    font-family: 'Neuton', Georgia, serif;
    font-size: 2.5rem;
    font-weight: 700;
    color: #194852;
    margin-bottom: 0.5rem;
    letter-spacing: -0.02em;
}

.subtitle {
    font-family: 'Hanken Grotesk', sans-serif;
    font-size: 1.2rem;
    font-weight: 400;
    color: #348397;
    margin: 0;
}

/* Filter Section */
.filter-section {
    background: linear-gradient(135deg, #194852 0%, #0c2a30 100%);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(12, 42, 48, 0.25);
}

.filter-label {
    color: #7dceda;
    font-size: 0.85rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Cards */
.brand-card {
    background: #ffffff;
    border-radius: 16px;
    box-shadow: 0 4px 20px rgba(12, 42, 48, 0.1);
    overflow: hidden;
}

.brand-card-header {
    background: linear-gradient(135deg, #194852 0%, #0c2a30 100%);
    color: #ffffff;
    padding: 1rem 1.5rem;
    font-family: 'Neuton', serif;
    font-size: 1.2rem;
    font-weight: 700;
}

.brand-card-header-gold {
    background: linear-gradient(135deg, #bca45e 0%, #52482a 100%);
    color: #ffffff;
    padding: 1rem 1.5rem;
    font-family: 'Neuton', serif;
    font-size: 1.2rem;
    font-weight: 700;
}

.brand-card-header-light {
    background: linear-gradient(135deg, #348397 0%, #194852 100%);
    color: #ffffff;
    padding: 1rem 1.5rem;
    font-family: 'Neuton', serif;
    font-size: 1.2rem;
    font-weight: 700;
}

.brand-card-body {
    padding: 1.5rem;
}

/* Buttons */
.brand-btn {
    font-family: 'Hanken Grotesk', sans-serif;
    font-weight: 600;
    padding: 0.6rem 1.2rem;
    border-radius: 8px;
    border: none;
    cursor: pointer;
    transition: all 0.2s ease;
}

.brand-btn-secondary {
    background: rgba(255,255,255,0.15);
    color: #ffffff;
    border: 1px solid rgba(255,255,255,0.3);
}

.brand-btn-secondary:hover {
    background: rgba(255,255,255,0.25);
}

/* Stats */
.stat-display {
    text-align: left;
    padding: 0.5rem 0;
}

.stat-value {
    font-family: 'Neuton', serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #194852;
}

.stat-label {
    font-size: 0.8rem;
    color: #78a0a3;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Tabs */
.nav-tabs {
    border: none;
    background: #194852;
    border-radius: 16px 16px 0 0;
    padding: 0.5rem 0.5rem 0 0.5rem;
}

.nav-tabs .nav-link {
    color: rgba(255,255,255,0.7);
    border: none;
    border-radius: 12px 12px 0 0;
    padding: 0.75rem 1.5rem;
    font-weight: 600;
    margin-right: 0.25rem;
}

.nav-tabs .nav-link.active {
    background: #ffffff;
    color: #194852;
}

/* Summary */
.summary-markdown {
    line-height: 1.7;
}

.summary-markdown h3, .summary-markdown h4 {
    font-family: 'Neuton', serif;
    color: #194852;
}

.summary-markdown table {
    width: 100%;
    border-collapse: collapse;
    margin: 1rem 0;
    font-size: 0.9rem;
}

.summary-markdown table th {
    background: #194852;
    color: white;
    padding: 0.75rem;
    text-align: left;
}

.summary-markdown table td {
    padding: 0.75rem;
    border-bottom: 1px solid #edf1f2;
}

/* Tables */
.brand-table {
    width: 100%;
    border-collapse: collapse;
}

.brand-table th {
    background: linear-gradient(135deg, #194852 0%, #0c2a30 100%);
    color: white;
    padding: 0.75rem 1rem;
    font-weight: 600;
}

.brand-table td {
    padding: 0.75rem 1rem;
    border-bottom: 1px solid #edf1f2;
}

.brand-table tr:hover {
    background: rgba(125, 206, 218, 0.08);
}

/* Footer */
.footer-section {
    margin-top: 2rem;
    padding: 2rem 0;
    background: linear-gradient(135deg, #194852 0%, #0c2a30 100%);
    border-radius: 12px;
    text-align: center;
}

.footer-text {
    color: rgba(255,255,255,0.8);
    font-size: 0.9rem;
}

.footer-text a {
    color: #7dceda;
    text-decoration: none;
}

@media (max-width: 768px) {
    .main-title { font-size: 1.5rem; }
    .subtitle { font-size: 1rem; }
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
app.title = "Marriage and the Melting Pot, 1880-1930 | Niskanen Center"

# Expose server for Gunicorn
server = app.server

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
        <style>{CUSTOM_CSS}</style>
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
            # Header
            html.Div([
                html.H1("Marriage and the Melting Pot, 1880-1930", className='main-title text-center'),
                html.P("Whom did the US-born children of immigrants marry?", className='subtitle text-center'),
            ], className='header-section'),

            # Filters
            html.Div([
                dbc.Row([
                    dbc.Col([
                        html.Label("Mother's Birthplace", className='filter-label'),
                        dcc.Dropdown(id='mother-dropdown', options=mother_dropdown_options,
                                    value='Any', clearable=False)
                    ], lg=3, md=6, xs=12, className='mb-3'),
                    dbc.Col([
                        html.Label("Father's Birthplace", className='filter-label'),
                        dcc.Dropdown(id='father-dropdown', options=father_dropdown_options,
                                    value='Any', clearable=False)
                    ], lg=3, md=6, xs=12, className='mb-3'),
                    dbc.Col([
                        html.Label("Census Year", className='filter-label'),
                        dcc.Dropdown(id='year-dropdown',
                                    options=[{'label': 'All Years', 'value': 'All'}] +
                                            [{'label': str(y), 'value': y} for y in years],
                                    value='All', clearable=False)
                    ], lg=2, md=4, xs=6, className='mb-3'),
                    dbc.Col([
                        html.Label("Quick Presets", className='filter-label'),
                        dcc.Dropdown(id='preset-dropdown', options=DYNAMIC_PRESETS,
                                    value='', clearable=False, optionHeight=45, maxHeight=400)
                    ], lg=2, md=4, xs=6, className='mb-3'),
                    dbc.Col([
                        html.Label("Actions", className='filter-label'),
                        html.Div([
                            html.Button("Reset", id='reset-btn', className='brand-btn brand-btn-secondary'),
                        ], className='d-flex gap-3')
                    ], lg=2, md=4, xs=12, className='mb-3'),
                ]),
            ], className='filter-section'),

            # Sample Size & Actions
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Div(id='sample-size', className='stat-value'),
                        html.Div("Weighted Sample Size", className='stat-label'),
                    ], className='stat-display')
                ], md=3, xs=6),
                dbc.Col([
                    html.Div([
                        html.Button("Download CSV", id='download-csv-btn', className='brand-btn brand-btn-secondary me-2'),
                        dcc.Download(id='download-csv'),
                        html.Button("Share Link", id='copy-link-btn', className='brand-btn brand-btn-secondary'),
                        html.Span(id='link-copied-msg', style={'marginLeft': '10px', 'color': COLORS['green'], 'fontWeight': '600'}),
                    ], className='text-end pt-3')
                ], md=9, xs=6),
            ], className='mb-4 align-items-center'),

            # Summary Card
            html.Div([
                html.Div("Analysis Summary", className='brand-card-header-gold'),
                html.Div([
                    dcc.Loading(type='circle', color=COLORS['medium_teal'],
                               children=[dcc.Markdown(id='auto-summary', className='summary-markdown')])
                ], className='brand-card-body')
            ], className='brand-card summary-card mb-4'),

            # Selection-Based Visualization Tabs (respond to heritage dropdowns)
            html.Div([
                html.Div([
                    html.Span("Your Selection", style={
                        'fontFamily': 'Neuton, serif',
                        'fontSize': '1.1rem',
                        'fontWeight': '600',
                        'color': COLORS['dark_teal'],
                        'textTransform': 'uppercase',
                        'letterSpacing': '0.05em'
                    }),
                    html.Span(" — responds to heritage selections above", style={
                        'fontFamily': 'Hanken Grotesk, sans-serif',
                        'fontSize': '0.85rem',
                        'color': COLORS['muted_teal'],
                        'marginLeft': '0.5rem'
                    })
                ], style={'marginBottom': '0.75rem'}),
                dbc.Tabs([
                    dbc.Tab(label="Main Chart", tab_id="tab-main"),
                    dbc.Tab(label="Trends Over Time", tab_id="tab-trends"),
                    dbc.Tab(label="Spouse Generation", tab_id="tab-spouse-gen"),
                ], id='viz-tabs', active_tab='tab-main', className='mb-0'),
                html.Div([html.Div(id='tab-content')], className='brand-card',
                        style={'borderRadius': '0 0 16px 16px'})
            ], className='mb-4'),

            # Spouse Backgrounds Table
            html.Div([
                html.Div("Spouse Backgrounds (Top 15)", className='brand-card-header-light'),
                html.Div([
                    dcc.Loading(type='circle', color=COLORS['medium_teal'],
                               children=[html.Div(id='spouse-table', style={'maxHeight': '400px', 'overflowY': 'auto'})])
                ], className='brand-card-body')
            ], className='brand-card mb-4'),

            # Compare All Groups Section
            html.Div([
                html.Div([
                    html.Span("Compare All Groups", style={
                        'fontFamily': 'Neuton, serif',
                        'fontSize': '1.1rem',
                        'fontWeight': '600',
                        'color': COLORS['dark_teal'],
                        'textTransform': 'uppercase',
                        'letterSpacing': '0.05em'
                    }),
                    html.Span(" — cross-group comparisons", style={
                        'fontFamily': 'Hanken Grotesk, sans-serif',
                        'fontSize': '0.85rem',
                        'color': COLORS['muted_teal'],
                        'marginLeft': '0.5rem'
                    })
                ], style={'marginBottom': '0.75rem'}),
                dbc.Tabs([
                    dbc.Tab(label="Outmarriage Rates", tab_id="tab-outmarriage"),
                    dbc.Tab(label="Clustering Network", tab_id="tab-heatmap"),
                    dbc.Tab(label="Single Origin Overview", tab_id="tab-single-origin"),
                ], id='overview-tabs', active_tab='tab-outmarriage', className='mb-0'),
                html.Div([html.Div(id='overview-tab-content')], className='brand-card',
                        style={'borderRadius': '0 0 16px 16px'})
            ], className='mb-4'),

            # Methodology
            html.Div([
                html.Div("Methodology & Data Sources", className='brand-card-header'),
                html.Div([
                    html.H4("Data Source", style={'color': COLORS['dark_teal'], 'marginTop': '0', 'marginBottom': '0.5rem', 'fontFamily': 'Neuton, serif'}),
                    html.P([
                        "This dashboard uses microdata from ",
                        html.A("IPUMS USA", href="https://usa.ipums.org", target="_blank", style={'color': COLORS['medium_teal']}),
                        " (Integrated Public Use Microdata Series), a harmonized collection of U.S. Census samples maintained by the Minnesota Population Center."
                    ], style={'marginBottom': '0.75rem'}),

                    html.Table([
                        html.Thead(html.Tr([
                            html.Th("Census Year", style={'padding': '0.5rem', 'borderBottom': f'2px solid {COLORS["dark_teal"]}', 'textAlign': 'left'}),
                            html.Th("Sample Size", style={'padding': '0.5rem', 'borderBottom': f'2px solid {COLORS["dark_teal"]}', 'textAlign': 'left'}),
                            html.Th("Approx. Records", style={'padding': '0.5rem', 'borderBottom': f'2px solid {COLORS["dark_teal"]}', 'textAlign': 'left'}),
                        ])),
                        html.Tbody([
                            html.Tr([html.Td("1880"), html.Td("10%"), html.Td("~5 million")]),
                            html.Tr([html.Td("1900"), html.Td("5%"), html.Td("~3.8 million")], style={'backgroundColor': COLORS['very_light_gray']}),
                            html.Tr([html.Td("1910"), html.Td("1%"), html.Td("~0.9 million")]),
                            html.Tr([html.Td("1920"), html.Td("1%"), html.Td("~1.0 million")], style={'backgroundColor': COLORS['very_light_gray']}),
                            html.Tr([html.Td("1930"), html.Td("5%"), html.Td("~6.1 million")]),
                        ])
                    ], style={'width': '100%', 'marginBottom': '1.25rem', 'fontSize': '0.9rem', 'borderCollapse': 'collapse'}),

                    html.H4("Historical Context: 1880-1930", style={'color': COLORS['dark_teal'], 'marginBottom': '0.5rem', 'fontFamily': 'Neuton, serif'}),
                    html.P([
                        "Between 1880 and 1930, over 27 million immigrants arrived in the United States. The early decades saw continued flows from Ireland, Germany, and Scandinavia, while the 1890s through 1920s brought large-scale immigration from Southern and Eastern Europe. The Immigration Act of 1924 dramatically curtailed these flows, making this period a natural unit of analysis for studying immigrant integration through intermarriage."
                    ], style={'marginBottom': '1.25rem'}),

                    html.H4("Population Studied", style={'color': COLORS['dark_teal'], 'marginBottom': '0.5rem', 'fontFamily': 'Neuton, serif'}),
                    html.P([
                        html.Strong("Children of immigrants"), " (second generation): US-born individuals with at least one foreign-born parent, who were married with spouse present at census enumeration."
                    ], style={'marginBottom': '0.5rem'}),
                    html.P([
                        "Spouse information is obtained through IPUMS's \"Attach Characteristics\" feature, which links each person to their spouse and provides the spouse's birthplace and parental birthplace data."
                    ], style={'marginBottom': '1.25rem'}),

                    html.H4("Heritage Classification", style={'color': COLORS['dark_teal'], 'marginBottom': '0.5rem', 'fontFamily': 'Neuton, serif'}),
                    html.P(html.Strong("Spouse categories:"), style={'marginBottom': '0.25rem'}),
                    html.Ul([
                        html.Li([html.Strong("1st generation immigrant: "), "Spouse born outside the US"]),
                        html.Li([html.Strong("2nd generation: "), "US-born spouse with at least one foreign-born parent"]),
                        html.Li([html.Strong("3rd+ generation American: "), "US-born spouse with both parents US-born"]),
                    ], style={'marginBottom': '1.25rem', 'paddingLeft': '1.5rem'}),

                    html.H4("Caveats & Limitations", style={'color': COLORS['dark_teal'], 'marginBottom': '0.5rem', 'fontFamily': 'Neuton, serif'}),
                    html.Ol([
                        html.Li([html.Strong("\"3rd+ generation\" is a residual category: "), "It includes anyone whose grandparents' origins cannot be traced through parent birthplace."]),
                        html.Li([html.Strong("Boundary changes: "), "European borders shifted dramatically during this period. \"Germany,\" \"Poland,\" \"Austria,\" and \"Russia\" may reflect different territories across census years."]),
                        html.Li([html.Strong("Selection into marriage: "), "This analysis captures who ", html.Em("was"), " married at census time, not marriage formation rates."]),
                        html.Li([html.Strong("Potential repeated observations: "), "When viewing \"All Years,\" individuals who remained married across multiple census years may be counted more than once. Historical census microdata lacks longitudinal person identifiers, making individual-level deduplication infeasible. As a result, older birth cohorts (who could appear in more censuses) may be over-represented in pooled analyses. For robustness, users can select individual census years to examine cross-sectional snapshots free from this issue."]),
                    ], style={'marginBottom': '1.25rem', 'paddingLeft': '1.5rem'}),

                    html.H4("Citation", style={'color': COLORS['dark_teal'], 'marginBottom': '0.5rem', 'fontFamily': 'Neuton, serif'}),
                    html.P("IPUMS data should be cited as:", style={'marginBottom': '0.25rem'}),
                    html.Blockquote([
                        "Steven Ruggles, Sarah Flood, Matthew Sobek, et al. ",
                        html.Em("IPUMS USA: Version 15.0"), " [dataset]. Minneapolis, MN: IPUMS, 2024. ",
                        html.A("https://doi.org/10.18128/D010.V15.0", href="https://doi.org/10.18128/D010.V15.0", target="_blank", style={'color': COLORS['medium_teal']})
                    ], style={
                        'borderLeft': f'3px solid {COLORS["gold"]}',
                        'paddingLeft': '1rem',
                        'marginLeft': '0',
                        'fontSize': '0.9rem',
                        'backgroundColor': COLORS['very_light_gold'],
                        'padding': '0.75rem 1rem',
                        'borderRadius': '0 4px 4px 0',
                    }),

                    html.P("To cite this dashboard:", style={'marginTop': '1rem', 'marginBottom': '0.25rem'}),
                    html.Blockquote([
                        "Guerra, Gil. ",
                        html.Em("Marriage and the Melting Pot, 1880-1930"),
                        " [interactive dashboard]. Washington, DC: Niskanen Center, 2025. ",
                        html.A("https://www.niskanencenter.org/intermarriage-dashboard/", href="https://www.niskanencenter.org/intermarriage-dashboard/", target="_blank", style={'color': COLORS['medium_teal']})
                    ], style={
                        'borderLeft': f'3px solid {COLORS["gold"]}',
                        'paddingLeft': '1rem',
                        'marginLeft': '0',
                        'fontSize': '0.9rem',
                        'backgroundColor': COLORS['very_light_gold'],
                        'padding': '0.75rem 1rem',
                        'borderRadius': '0 4px 4px 0',
                    }),

                    html.H4("Development Process", style={'color': COLORS['dark_teal'], 'marginTop': '1.25rem', 'marginBottom': '0.5rem', 'fontFamily': 'Neuton, serif'}),
                    html.P([
                        "This dashboard was developed iteratively using ",
                        html.Strong("Claude Code"),
                        " (Anthropic), an AI-assisted coding tool. Claude assisted with data processing pipeline development, debugging, visualization design, and methodology refinement. The underlying analysis logic and research design were directed by the researcher."
                    ], style={'marginBottom': '0'}),
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
                    "Data: ", html.A("IPUMS USA", href="https://usa.ipums.org", target="_blank"),
                    " | Analysis: ", html.A("Niskanen Center, Gil Guerra", href="https://www.niskanencenter.org/author/gguerra/", target="_blank")
                ], className='footer-text'),
            ], className='footer-section'),

        ], fluid=True, style={'maxWidth': '1400px'})
    ], style={'minHeight': '100vh', 'padding': '1rem'})
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


@callback(Output('url', 'search'),
          [Input('mother-dropdown', 'value'), Input('father-dropdown', 'value'), Input('year-dropdown', 'value')])
def update_url(mother, father, year):
    params = {}
    if mother != 'Any':
        params['mother'] = mother
    if father != 'Any':
        params['father'] = father
    if year != 'All':
        params['year'] = str(year)
    return '?' + urlencode(params) if params else ''


@callback([Output('mother-dropdown', 'value'), Output('father-dropdown', 'value')],
          [Input('preset-dropdown', 'value')], prevent_initial_call=True)
def apply_preset(preset):
    if not preset or '|' not in preset:
        return no_update, no_update
    parts = preset.split('|')
    return (parts[0], parts[1]) if len(parts) == 2 else (no_update, no_update)


@callback(
    [Output('mother-dropdown', 'value', allow_duplicate=True),
     Output('father-dropdown', 'value', allow_duplicate=True),
     Output('year-dropdown', 'value', allow_duplicate=True),
     Output('preset-dropdown', 'value')],
    [Input('reset-btn', 'n_clicks')], prevent_initial_call=True)
def reset_filters(n_clicks):
    return ('Any', 'Any', 'All', '') if n_clicks else (no_update, no_update, no_update, no_update)


@callback([Output('clipboard', 'content'), Output('link-copied-msg', 'children')],
          [Input('copy-link-btn', 'n_clicks')], [State('url', 'href')], prevent_initial_call=True)
def copy_link(n_clicks, href):
    return (href, "Copied!") if n_clicks else (no_update, "")


@callback(Output('download-csv', 'data'),
          [Input('download-csv-btn', 'n_clicks')],
          [State('mother-dropdown', 'value'), State('father-dropdown', 'value'), State('year-dropdown', 'value')],
          prevent_initial_call=True)
def download_csv(n_clicks, mother, father, year):
    if n_clicks:
        stats, total, _ = get_marriage_stats(mother, father, year)
        if stats is not None:
            stats['Percentage'] = (stats['WEIGHTED_COUNT'] / total * 100).round(2)
            filename = f"marriage_patterns_{mother}_{father}_{year}.csv".replace(' ', '_')
            return dcc.send_data_frame(stats.to_csv, filename, index=False)
    return no_update


@callback([Output('sample-size', 'children'), Output('auto-summary', 'children')],
          [Input('mother-dropdown', 'value'), Input('father-dropdown', 'value'), Input('year-dropdown', 'value')])
def update_summary(mother, father, year):
    stats, weighted, unweighted = get_marriage_stats(mother, father, year)
    if stats is None:
        return "0", "No data available"
    if unweighted < 30:
        size_text = html.Span([f"{weighted:,.0f}", html.Br(),
                               html.Span(f"Small sample ({unweighted})", style={'fontSize': '0.7rem', 'color': COLORS['orange']})])
    else:
        size_text = f"{weighted:,.0f}"
    summary = generate_summary(mother, father, year)
    return size_text, summary


@callback(Output('tab-content', 'children'),
          [Input('viz-tabs', 'active_tab'), Input('mother-dropdown', 'value'),
           Input('father-dropdown', 'value'), Input('year-dropdown', 'value')])
def render_tab_content(active_tab, mother, father, year):
    """Render selection-based tabs that respond to heritage dropdowns."""
    if active_tab == 'tab-main':
        return html.Div([
            dcc.Loading(type='circle', color=COLORS['medium_teal'],
                       children=[dcc.Graph(id='main-chart', figure=create_main_chart(mother, father, year),
                                          config={'displayModeBar': True})])
        ], style={'padding': '1rem'})
    elif active_tab == 'tab-trends':
        return html.Div([
            dcc.Loading(type='circle', color=COLORS['medium_teal'],
                       children=[dcc.Graph(id='time-chart', figure=create_time_chart(mother, father),
                                          config={'displayModeBar': True})])
        ], style={'padding': '1rem'})
    elif active_tab == 'tab-spouse-gen':
        return html.Div([
            html.P("Distribution of spouse generations for the current selection. Shows whether children of immigrants "
                   "married recent immigrants (1st gen), fellow second-generation Americans (2nd gen), or established Americans (3rd+ gen).",
                   style={'color': COLORS['muted_teal'], 'fontSize': '0.9rem', 'marginBottom': '1rem'}),
            dcc.Loading(type='circle', color=COLORS['medium_teal'],
                       children=[dcc.Graph(id='spouse-gen-chart', figure=create_spouse_gen_chart(mother, father, year),
                                          config={'displayModeBar': True})])
        ], style={'padding': '1rem'})
    return html.Div()


@callback(Output('overview-tab-content', 'children'),
          [Input('overview-tabs', 'active_tab'), Input('year-dropdown', 'value')])
def render_overview_tab_content(active_tab, year):
    """Render overview tabs that show cross-group comparisons."""
    if active_tab == 'tab-outmarriage':
        return html.Div([
            html.P("Outmarriage rates by ethnic group. Higher rates indicate more marriages outside one's own ethnic community.",
                   style={'color': COLORS['muted_teal'], 'fontSize': '0.9rem', 'marginBottom': '1rem'}),
            html.Div([
                html.Label("Sort by:", style={'fontWeight': '500', 'marginRight': '10px', 'color': COLORS['dark_teal']}),
                dcc.Dropdown(
                    id='outmarriage-sort-dropdown',
                    options=[
                        {'label': 'Total Outmarriage Rate', 'value': 'total'},
                        {'label': 'Outmarriage to 3rd+ Gen Americans', 'value': 'american'},
                        {'label': 'Outmarriage to Different Immigrant Groups', 'value': 'other_immigrant'},
                    ],
                    value='total',
                    style={'width': '300px', 'display': 'inline-block', 'verticalAlign': 'middle'},
                    clearable=False
                ),
            ], style={'marginBottom': '1rem'}),
            dcc.Loading(type='circle', color=COLORS['medium_teal'],
                       children=[html.Div(id='outmarriage-chart-container')])
        ], style={'padding': '1rem'})
    elif active_tab == 'tab-heatmap':
        return html.Div([
            html.P("Network showing true ethnic affinity, adjusted for population size. Connections only appear when groups married "
                   "MORE than random chance would predict. This corrects for large groups like Germans appearing in many marriages simply due to their size.",
                   style={'color': COLORS['muted_teal'], 'fontSize': '0.9rem', 'marginBottom': '1rem'}),
            dcc.Loading(type='circle', color=COLORS['medium_teal'],
                       children=[dcc.Graph(id='heatmap-chart', figure=create_heatmap_chart(year),
                                          config={'displayModeBar': True})])
        ], style={'padding': '1rem'})
    elif active_tab == 'tab-single-origin':
        available_origins = get_available_origins_for_overview()
        default_origin = available_origins[0] if available_origins else 'Germany'
        return html.Div([
            html.P("Select an origin to see marriage patterns for the most common immigrant parent combinations involving that heritage "
                   "(excludes US-born parents). Compare how children with same-origin vs mixed-origin parents differ in their marriage choices.",
                   style={'color': COLORS['muted_teal'], 'fontSize': '0.9rem', 'marginBottom': '1rem'}),
            html.Div([
                html.Label("Select Origin:", style={'fontWeight': '500', 'marginRight': '10px', 'color': COLORS['dark_teal']}),
                dcc.Dropdown(
                    id='single-origin-dropdown',
                    options=[{'label': o, 'value': o} for o in available_origins],
                    value=default_origin,
                    style={'width': '200px', 'display': 'inline-block', 'verticalAlign': 'middle'}
                ),
            ], style={'marginBottom': '1rem'}),
            dcc.Loading(type='circle', color=COLORS['medium_teal'],
                       children=[html.Div(id='single-origin-chart-container')])
        ], style={'padding': '1rem'})
    return html.Div()


@callback(Output('outmarriage-chart-container', 'children'),
          [Input('outmarriage-sort-dropdown', 'value'), Input('year-dropdown', 'value')])
def update_outmarriage_chart(sort_by, year):
    """Update the outmarriage rates chart based on sorting selection."""
    return dcc.Graph(id='outmarriage-chart', figure=create_outmarriage_chart(year, sort_by),
                     config={'displayModeBar': True})


@callback(Output('single-origin-chart-container', 'children'),
          [Input('single-origin-dropdown', 'value'), Input('year-dropdown', 'value')])
def update_single_origin_chart(origin, year):
    """Update the single origin overview chart."""
    if not origin:
        return html.P("Please select an origin", style={'color': COLORS['muted_teal']})
    return dcc.Graph(id='single-origin-chart', figure=create_single_origin_chart(origin, year),
                     config={'displayModeBar': True})


def create_main_chart(mother, father, year):
    """Create the main horizontal bar chart."""
    stats, total, unweighted_n = get_marriage_stats(mother, father, year)
    if stats is None or total == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data for this selection", x=0.5, y=0.5, showarrow=False,
                          font=dict(size=16, color=COLORS['muted_teal']))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        return fig

    agg = stats.copy()
    agg = agg.sort_values('WEIGHTED_COUNT', ascending=True)
    agg['Percent'] = (agg['WEIGHTED_COUNT'] / total * 100).round(1)
    agg['Label'] = agg['Percent'].apply(lambda x: f"{x:.1f}%")
    agg['Color'] = agg['MARRIAGE_TYPE'].map(MARRIAGE_COLORS).fillna(COLORS['muted_teal'])

    fig = go.Figure(go.Bar(
        x=agg['WEIGHTED_COUNT'], y=agg['MARRIAGE_TYPE'], orientation='h',
        marker_color=agg['Color'], text=agg['Label'], textposition='outside',
        textfont=dict(family='Hanken Grotesk', size=12, color=COLORS['dark_teal']),
        hovertemplate='<b>%{y}</b><br>Count: %{x:,.0f}<br>Percent: %{text}<extra></extra>'
    ))

    mother_dem = get_demonym(mother) if mother != 'Any' else None
    father_dem = get_demonym(father) if father != 'Any' else None

    if mother != 'Any' and father != 'Any':
        if mother == father:
            title = f"Whom Did Children of {mother_dem} Parents Marry?"
        else:
            title = f"Whom Did {mother_dem} x {father_dem} Children Marry?"
    elif mother != 'Any':
        title = f"Whom Did Children of {mother_dem} Mothers Marry?"
    elif father != 'Any':
        title = f"Whom Did Children of {father_dem} Fathers Marry?"
    else:
        title = "Marriage Patterns: All Second-Generation Americans"

    sample_note = f"n = {unweighted_n:,} records ({total:,.0f} weighted)"

    fig.update_layout(
        title=dict(text=title, font=dict(family='Neuton', size=22, color=COLORS['dark_teal']), x=0, xanchor='left'),
        xaxis_title="Weighted Count",
        xaxis=dict(title_font=dict(family='Hanken Grotesk', size=12), gridcolor=COLORS['light_gray']),
        yaxis=dict(tickfont=dict(family='Hanken Grotesk', size=11)),
        height=max(400, len(agg) * 45 + 120),
        margin=dict(l=320, r=80, t=80, b=60),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        annotations=[dict(text=sample_note, xref='paper', yref='paper', x=1, y=-0.08, showarrow=False,
                         font=dict(family='Hanken Grotesk', size=11, color=COLORS['muted_teal']))]
    )
    return fig


def create_time_chart(mother, father):
    """Create trends over time chart."""
    df = get_filtered_marriage_data(mother, father, 'All')
    if len(df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data", x=0.5, y=0.5, showarrow=False)
        return fig

    MIN_TREND_SAMPLE = 10000
    yearly = df.groupby(['YEAR', 'MARRIAGE_TYPE'])['WEIGHTED_COUNT'].sum().reset_index()
    yearly_total = df.groupby('YEAR')['WEIGHTED_COUNT'].sum().reset_index()
    yearly_total.columns = ['YEAR', 'TOTAL']
    yearly = yearly.merge(yearly_total, on='YEAR')
    yearly['Percent'] = yearly['WEIGHTED_COUNT'] / yearly['TOTAL'] * 100

    small_sample_years = yearly_total[yearly_total['TOTAL'] < MIN_TREND_SAMPLE]['YEAR'].tolist()
    yearly = yearly[~yearly['YEAR'].isin(small_sample_years)]

    if len(yearly) == 0:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient sample size for trend analysis", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300)
        return fig

    def simplify(mtype):
        if '3rd+ gen' in mtype: return '3rd+ gen American'
        elif 'same origin' in mtype: return 'Same origin'
        elif 'both heritages' in mtype: return 'Both heritages'
        elif "mother's origin" in mtype: return "Mother's origin"
        elif "father's origin" in mtype: return "Father's origin"
        else: return 'Different origin'

    yearly['Category'] = yearly['MARRIAGE_TYPE'].apply(simplify)
    yearly_agg = yearly.groupby(['YEAR', 'Category'])['Percent'].sum().reset_index()

    fig = go.Figure()
    for category in CATEGORY_COLORS.keys():
        cat_data = yearly_agg[yearly_agg['Category'] == category]
        if len(cat_data) > 0:
            fig.add_trace(go.Scatter(
                x=cat_data['YEAR'], y=cat_data['Percent'], name=category, mode='lines+markers',
                line=dict(color=CATEGORY_COLORS[category], width=3),
                marker=dict(size=8, line=dict(width=2, color='white'))
            ))

    fig.update_layout(
        title=dict(text="Trends Over Time", font=dict(family='Neuton', size=22, color=COLORS['dark_teal']), x=0),
        xaxis_title="Census Year", yaxis_title="% of Marriages",
        xaxis=dict(gridcolor=COLORS['light_gray'], dtick=10),
        yaxis=dict(gridcolor=COLORS['light_gray'], range=[0, max(yearly_agg['Percent'].max() * 1.15, 50)]),
        height=480, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
    )
    return fig


def create_outmarriage_chart(year, sort_by='total'):
    """Create horizontal bar chart showing outmarriage rates with different sorting options."""
    ranking = get_integration_ranking(year)

    if not ranking:
        fig = go.Figure()
        fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400)
        return fig

    # Sort by selected metric
    if sort_by == 'american':
        ranking.sort(key=lambda x: x['third_gen_rate'], reverse=True)
        title = "Outmarriage Rates: Into American Mainstream"
        xaxis_title = "Outmarriage to 3rd+ Gen Americans (%)"
        value_key = 'third_gen_rate'
    elif sort_by == 'other_immigrant':
        ranking.sort(key=lambda x: x['diff_origin_rate'], reverse=True)
        title = "Outmarriage Rates: Into Other Immigrant Groups"
        xaxis_title = "Outmarriage to Different Immigrant Groups (%)"
        value_key = 'diff_origin_rate'
    else:  # total
        ranking.sort(key=lambda x: x['integration_rate'], reverse=True)
        title = "Outmarriage Rates: Total (Outside Own Ethnic Group)"
        xaxis_title = "Total Outmarriage Rate (%)"
        value_key = 'integration_rate'

    # Reverse for horizontal bar chart (so highest is at top)
    ranking = ranking[::-1]

    demonyms = [r['demonym'] + '-Americans' for r in ranking]
    values = [r[value_key] for r in ranking]
    populations = [r['population'] for r in ranking]

    # Create hover text with breakdown
    hover_texts = []
    for r in ranking:
        hover_texts.append(
            f"<b>{r['demonym']}-Americans</b><br>"
            f"Total outmarriage: {r['integration_rate']:.1f}%<br>"
            f"  To 3rd+ gen Americans: {r['third_gen_rate']:.1f}%<br>"
            f"  To other immigrant groups: {r['diff_origin_rate']:.1f}%<br>"
            f"Population: {r['population']:,.0f}"
        )

    fig = go.Figure(go.Bar(
        x=values,
        y=demonyms,
        orientation='h',
        marker_color=COLORS['medium_teal'],
        text=[f"{v:.0f}%" for v in values],
        textposition='outside',
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover_texts
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(family='Neuton', size=22, color=COLORS['dark_teal']), x=0),
        xaxis_title=xaxis_title,
        xaxis=dict(gridcolor=COLORS['light_gray'], range=[0, max(values) * 1.15 if values else 100]),
        yaxis=dict(gridcolor=COLORS['light_gray']),
        height=max(400, len(ranking) * 28 + 100),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=150, r=60)
    )
    return fig


def create_spouse_gen_chart(mother, father, year):
    """Create stacked bar chart showing spouse generation composition."""
    spouse_gen = get_spouse_generation_data(mother, father, year)

    if not spouse_gen or all(v['count'] == 0 for v in spouse_gen.values()):
        fig = go.Figure()
        fig.add_annotation(text="No spouse generation data available", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300)
        return fig

    generations = ['1st gen immigrant', '2nd gen', '3rd+ gen American']
    percentages = [spouse_gen[g]['pct'] for g in generations]
    counts = [spouse_gen[g]['count'] for g in generations]

    gen_colors = {
        '1st gen immigrant': COLORS['dark_teal'],
        '2nd gen': COLORS['medium_teal'],
        '3rd+ gen American': COLORS['light_teal']
    }

    fig = go.Figure(go.Bar(
        x=generations,
        y=percentages,
        marker_color=[gen_colors[g] for g in generations],
        text=[f"{p:.1f}%" for p in percentages],
        textposition='outside',
        hovertemplate="<b>%{x}</b><br>%{y:.1f}%<br>Count: %{customdata:,.0f}<extra></extra>",
        customdata=counts
    ))

    fig.update_layout(
        title=dict(text="Spouse Generation Distribution", font=dict(family='Neuton', size=22, color=COLORS['dark_teal']), x=0),
        yaxis_title="% of Spouses",
        xaxis=dict(gridcolor=COLORS['light_gray']),
        yaxis=dict(gridcolor=COLORS['light_gray'], range=[0, max(percentages) * 1.2 if percentages else 100]),
        height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig


def create_heatmap_chart(year):
    """Create network graph showing ethnic clustering patterns."""
    try:
        nodes, edges, positions = get_network_data(year)
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient data for clustering analysis", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400)
        return fig

    if not nodes or not edges:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient data for network visualization", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400)
        return fig

    fig = go.Figure()

    # Draw edges (connections between groups)
    max_weight = max(e['weight'] for e in edges) if edges else 1
    for edge in edges:
        x0, y0 = positions[edge['source']]
        x1, y1 = positions[edge['target']]
        # Line width proportional to marriage rate
        width = max(0.5, edge['weight'] / max_weight * 6)
        # Opacity based on weight
        opacity = min(0.8, 0.2 + edge['weight'] / max_weight * 0.6)

        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode='lines',
            line=dict(width=width, color=COLORS['medium_teal']),
            opacity=opacity,
            hoverinfo='skip',
            showlegend=False
        ))

    # Draw nodes
    max_pop = max(n['population'] for n in nodes) if nodes else 1
    node_x = [positions[n['id']][0] for n in nodes]
    node_y = [positions[n['id']][1] for n in nodes]
    node_sizes = [15 + 35 * (n['population'] / max_pop) for n in nodes]
    node_labels = [n['label'] for n in nodes]
    node_pops = [n['population'] for n in nodes]

    # Node colors - use a gradient based on population
    node_colors = [COLORS['dark_teal'] for _ in nodes]

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=2, color='white')
        ),
        text=node_labels,
        textposition='top center',
        textfont=dict(family='Hanken Grotesk', size=11, color=COLORS['dark_teal']),
        hovertemplate="<b>%{text}</b><br>Population: %{customdata:,.0f}<extra></extra>",
        customdata=node_pops,
        showlegend=False
    ))

    fig.update_layout(
        title=dict(
            text="Ethnic Clustering Network (Population-Adjusted)",
            font=dict(family='Neuton', size=22, color=COLORS['dark_teal']),
            x=0
        ),
        showlegend=False,
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
        height=550,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=40, t=80, b=60),
        annotations=[
            dict(
                text="Connected groups married more than random chance would predict · Line thickness = affinity strength",
                xref='paper', yref='paper', x=0.5, y=-0.05,
                showarrow=False, font=dict(size=10, color=COLORS['muted_teal']),
                xanchor='center'
            )
        ]
    )
    return fig


def create_scatter_chart(year):
    """Create scatter plot showing population size vs ethnic retention."""
    scatter_data = get_scatter_data(year)

    if not scatter_data:
        fig = go.Figure()
        fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400)
        return fig

    df = pd.DataFrame(scatter_data)

    # Size bubbles by population (log scale for better visibility)
    df['bubble_size'] = np.log10(df['population']) * 8

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['population'],
        y=df['same_origin_rate'],
        mode='markers+text',
        marker=dict(
            size=df['bubble_size'],
            color=df['same_origin_rate'],
            colorscale=[[0, COLORS['light_teal']], [0.5, COLORS['medium_teal']], [1, COLORS['dark_teal']]],
            showscale=True,
            colorbar=dict(title="Same-Origin<br>Rate (%)", ticksuffix="%"),
            line=dict(width=1, color='white')
        ),
        text=df['demonym'],
        textposition='top center',
        textfont=dict(size=10, color=COLORS['dark_teal']),
        hovertemplate="<b>%{text}</b><br>Population: %{x:,.0f}<br>Same-origin rate: %{y:.1f}%<extra></extra>"
    ))

    fig.update_layout(
        title=dict(text="Population Size vs Ethnic Retention", font=dict(family='Neuton', size=22, color=COLORS['dark_teal']), x=0),
        xaxis_title="Population (Same-Origin Parents)",
        yaxis_title="Same-Origin Marriage Rate (%)",
        xaxis=dict(type='log', gridcolor=COLORS['light_gray']),
        yaxis=dict(gridcolor=COLORS['light_gray'], range=[0, 100]),
        height=500, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    return fig


def create_single_origin_chart(origin, year):
    """Create stacked bar chart showing marriage patterns for all combinations involving an origin.

    Categories sum to 100%:
    - Within Heritage: Married within immigrant heritage (same, mother's, father's, or both)
    - Different Immigrant: Married different immigrant origin
    - 3rd+ Gen American: Married established American
    """
    data = get_single_origin_overview(origin, year)

    if not data:
        fig = go.Figure()
        fig.add_annotation(text=f"No data available for {origin}", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400)
        return fig

    df = pd.DataFrame(data)

    # Sort by heritage rate for clearer presentation
    df = df.sort_values('heritage_rate', ascending=True)

    fig = go.Figure()

    # Add bars for different marriage outcome categories (these sum to 100%)
    fig.add_trace(go.Bar(
        y=df['label'], x=df['heritage_rate'], name='Within Heritage',
        orientation='h', marker_color=COLORS['dark_teal'],
        text=[f"{v:.0f}%" if v >= 5 else "" for v in df['heritage_rate']], textposition='inside',
        hovertemplate="<b>%{y}</b><br>Within Heritage: %{x:.1f}%<extra></extra>"
    ))

    fig.add_trace(go.Bar(
        y=df['label'], x=df['diff_origin_rate'], name='Different Immigrant',
        orientation='h', marker_color=COLORS['medium_teal'],
        text=[f"{v:.0f}%" if v >= 5 else "" for v in df['diff_origin_rate']], textposition='inside',
        hovertemplate="<b>%{y}</b><br>Different Immigrant: %{x:.1f}%<extra></extra>"
    ))

    fig.add_trace(go.Bar(
        y=df['label'], x=df['third_gen_rate'], name='3rd+ Gen American',
        orientation='h', marker_color=COLORS['light_teal'],
        text=[f"{v:.0f}%" if v >= 5 else "" for v in df['third_gen_rate']], textposition='inside',
        hovertemplate="<b>%{y}</b><br>3rd+ Gen American: %{x:.1f}%<extra></extra>"
    ))

    dem = get_demonym(origin)
    fig.update_layout(
        title=dict(text=f"Marriage Patterns: Most Common {dem} Parent Combinations",
                   font=dict(family='Neuton', size=22, color=COLORS['dark_teal']), x=0),
        xaxis_title="Percentage (categories sum to 100%)",
        xaxis=dict(gridcolor=COLORS['light_gray'], range=[0, 105], ticksuffix='%'),
        yaxis=dict(tickfont=dict(family='Hanken Grotesk', size=11)),
        barmode='stack',
        height=max(400, len(df) * 45 + 120),
        margin=dict(l=220, r=60, t=80, b=60),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
    )
    return fig


@callback(Output('spouse-table', 'children'),
          [Input('mother-dropdown', 'value'), Input('father-dropdown', 'value'), Input('year-dropdown', 'value')])
def update_table(mother, father, year):
    df = get_filtered_spouse_data(mother, father, year)
    if len(df) == 0:
        return html.P("No data available", style={'color': COLORS['muted_teal']})

    total = df['WEIGHTED_COUNT'].sum()

    # Build spouse background labels
    def get_bg_label(row):
        gen = row['SPOUSE_GEN']
        if gen == '1st gen immigrant':
            return f"{get_demonym(row['SPOUSE_COUNTRY'])} immigrant"
        elif gen == '3rd+ gen American':
            return '3rd+ gen American'
        else:
            mom = str(row['SPOUSE_MOTHER']) if pd.notna(row['SPOUSE_MOTHER']) else 'Unknown'
            dad = str(row['SPOUSE_FATHER']) if pd.notna(row['SPOUSE_FATHER']) else 'Unknown'
            if mom == dad and mom not in ['Unknown', 'N/A', 'US-born']:
                return f"2nd-gen {get_demonym(mom)}"
            elif mom not in ['Unknown', 'N/A', 'US-born'] and dad not in ['Unknown', 'N/A', 'US-born']:
                return f"2nd-gen {get_demonym(mom)} x {get_demonym(dad)}"
            elif mom not in ['Unknown', 'N/A', 'US-born']:
                return f"2nd-gen {get_demonym(mom)}"
            elif dad not in ['Unknown', 'N/A', 'US-born']:
                return f"2nd-gen {get_demonym(dad)}"
            return "2nd-gen (mixed)"

    df = df.copy()
    df['Background'] = df.apply(get_bg_label, axis=1)
    agg = df.groupby('Background')['WEIGHTED_COUNT'].sum().reset_index()
    agg = agg.sort_values('WEIGHTED_COUNT', ascending=False).head(15)
    agg['Share'] = (agg['WEIGHTED_COUNT'] / total * 100).round(1).astype(str) + '%'
    agg['Count'] = agg['WEIGHTED_COUNT'].apply(lambda x: f"{x:,.0f}")

    return html.Table([
        html.Thead([html.Tr([
            html.Th("Spouse Background", style={'textAlign': 'left'}),
            html.Th("Count", style={'textAlign': 'right'}),
            html.Th("Share", style={'textAlign': 'right'}),
        ])]),
        html.Tbody([
            html.Tr([
                html.Td(row['Background']),
                html.Td(row['Count'], style={'textAlign': 'right'}),
                html.Td(row['Share'], style={'textAlign': 'right', 'fontWeight': '600'}),
            ]) for _, row in agg.iterrows()
        ])
    ], className='brand-table', style={'width': '100%'})


# =============================================================================
# RUN
# =============================================================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    print(f"\nStarting dashboard on port {port}...")
    print(f"Debug mode: {debug}")
    print(f"Open: http://127.0.0.1:{port}\n")
    app.run(debug=debug, host='0.0.0.0', port=port)

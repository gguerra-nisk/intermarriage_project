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
import dash
from dash import dcc, html, callback, Input, Output, State, ctx, no_update, ALL
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

# Shorter display labels for chart y-axes
MARRIAGE_DISPLAY_LABELS = {
    'Married same origin (1st gen immigrant)': 'Same origin (1st gen)',
    'Married same origin (2nd gen)': 'Same origin (2nd gen)',
    "Married mother's origin (1st gen immigrant)": "Mother's origin (1st gen)",
    "Married mother's origin (2nd gen)": "Mother's origin (2nd gen)",
    "Married father's origin (1st gen immigrant)": "Father's origin (1st gen)",
    "Married father's origin (2nd gen)": "Father's origin (2nd gen)",
    'Married someone sharing both heritages (1st gen immigrant)': 'Both heritages (1st gen)',
    'Married someone sharing both heritages (2nd gen)': 'Both heritages (2nd gen)',
    'Married different origin (1st gen immigrant)': 'Different origin (1st gen)',
    'Married different origin (2nd gen)': 'Different origin (2nd gen)',
    'Married 3rd+ gen American': '3rd+ gen American',
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

        # Geographic aggregation (optional - only if preprocessing included STATEFIP)
        geo_path = PROCESSED_DIR / "geographic_agg.csv"
        if geo_path.exists():
            data['geographic'] = pd.read_csv(geo_path, low_memory=False)
            print(f"  Loaded geographic data: {len(data['geographic'])} rows")
        else:
            data['geographic'] = None

        # Geography-adjusted affinity matrix (optional)
        adj_path = PROCESSED_DIR / "geo_adjusted_affinity.csv"
        if adj_path.exists():
            data['geo_adjusted_affinity'] = pd.read_csv(adj_path)
            print(f"  Loaded adjusted affinities: {len(data['geo_adjusted_affinity'])} pairs")
        else:
            data['geo_adjusted_affinity'] = None

        # Language aggregation (optional - MTONGUE data for 1910-1930)
        lang_path = PROCESSED_DIR / "language_agg.csv"
        if lang_path.exists():
            data['language'] = pd.read_csv(lang_path, low_memory=False)
            print(f"  Loaded language data: {len(data['language'])} rows")
        else:
            data['language'] = None

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

    # Build connection info for each node (for tooltips)
    node_connections = {g: [] for g in connected_groups}
    for edge in edges:
        src, tgt, wt = edge['source'], edge['target'], edge['weight']
        node_connections[src].append((tgt, wt))
        node_connections[tgt].append((src, wt))

    nodes = []
    for group in connected_groups:
        pop = parent_totals.get(group, 0)
        # Sort connections by affinity strength
        conns = sorted(node_connections[group], key=lambda x: x[1], reverse=True)
        conn_text = [f"{get_demonym(c)}: {w:.1f}x" for c, w in conns[:5]]
        nodes.append({
            'id': group,
            'label': get_demonym(group),
            'population': pop,
            'connections': conn_text
        })

    # Compute positions using force-directed layout (more iterations for better spacing)
    positions = _compute_force_layout(nodes, edges, iterations=150)

    return nodes, edges, positions


def _compute_force_layout(nodes, edges, iterations=500):
    """Spring-based layout with target distances based on connection strength."""
    import random
    random.seed(42)

    n = len(nodes)
    if n == 0:
        return {}

    # Build adjacency
    adj = {node['id']: {} for node in nodes}
    for edge in edges:
        s, t, w = edge['source'], edge['target'], edge['weight']
        if s in adj and t in adj:
            adj[s][t] = w
            adj[t][s] = w

    # Initialize in a small random cluster
    pos = {}
    for node in nodes:
        pos[node['id']] = [random.gauss(0, 0.3), random.gauss(0, 0.3)]

    # Spring layout parameters
    ideal_edge_length = 0.4
    repulsion_radius = 1.5

    for iteration in range(iterations):
        # Decreasing step size
        step = 0.05 * (1 - iteration / iterations) + 0.005

        force = {node['id']: [0.0, 0.0] for node in nodes}

        # Process all pairs
        for i, n1 in enumerate(nodes):
            for j, n2 in enumerate(nodes):
                if i >= j:
                    continue

                id1, id2 = n1['id'], n2['id']
                dx = pos[id2][0] - pos[id1][0]
                dy = pos[id2][1] - pos[id1][1]
                dist = np.sqrt(dx*dx + dy*dy)

                if dist < 0.001:
                    dx, dy = random.gauss(0, 0.01), random.gauss(0, 0.01)
                    dist = 0.01

                ux, uy = dx / dist, dy / dist

                if id2 in adj[id1]:
                    # Connected: spring to target distance (closer for higher weight)
                    weight = adj[id1][id2]
                    target = ideal_edge_length / (0.5 + weight * 0.2)
                    # Spring force proportional to displacement from target
                    f = (dist - target) * 0.3
                else:
                    # Not connected: only repel if too close
                    if dist < repulsion_radius:
                        f = -0.1 * (repulsion_radius - dist)
                    else:
                        f = 0

                force[id1][0] += f * ux
                force[id1][1] += f * uy
                force[id2][0] -= f * ux
                force[id2][1] -= f * uy

        # Apply forces
        for node in nodes:
            nid = node['id']
            pos[nid][0] += force[nid][0] * step
            pos[nid][1] += force[nid][1] * step

    # Center and normalize
    xs = [pos[n['id']][0] for n in nodes]
    ys = [pos[n['id']][1] for n in nodes]
    cx, cy = np.mean(xs), np.mean(ys)
    max_r = max(np.sqrt((x-cx)**2 + (y-cy)**2) for x, y in zip(xs, ys))
    scale = 1.5 / max_r if max_r > 0 else 1

    for node in nodes:
        pos[node['id']][0] = (pos[node['id']][0] - cx) * scale
        pos[node['id']][1] = (pos[node['id']][1] - cy) * scale

    return pos


def get_adjusted_network_data():
    """Build network data using geography-adjusted affinities.

    Uses pre-computed within-state affinities so that co-location in the
    same states doesn't masquerade as cultural affinity.
    """
    adj_df = DATA.get('geo_adjusted_affinity')
    if adj_df is None or len(adj_df) == 0:
        return [], [], {}

    # Get population data for node sizing (reuse spouse_bg like raw network)
    df = DATA['spouse_bg'].copy()
    df = df[df['MOTHER_ORIGIN'] == df['FATHER_ORIGIN']]
    df = df[~df['MOTHER_ORIGIN'].isin(NON_COUNTRIES)]
    parent_totals = df.groupby('MOTHER_ORIGIN')['WEIGHTED_COUNT'].sum()

    # Build edges from adjusted affinities (only keep affinity >= 1.0)
    edges = []
    for _, row in adj_df.iterrows():
        if row['GEO_ADJUSTED_AFFINITY'] >= 1.0:
            edges.append({
                'source': row['SOURCE'],
                'target': row['TARGET'],
                'weight': row['GEO_ADJUSTED_AFFINITY']
            })

    # Connected groups only
    connected_groups = set()
    for edge in edges:
        connected_groups.add(edge['source'])
        connected_groups.add(edge['target'])

    # Build connection info for tooltips
    node_connections = {g: [] for g in connected_groups}
    for edge in edges:
        src, tgt, wt = edge['source'], edge['target'], edge['weight']
        node_connections[src].append((tgt, wt))
        node_connections[tgt].append((src, wt))

    nodes = []
    for group in connected_groups:
        pop = parent_totals.get(group, 0)
        conns = sorted(node_connections[group], key=lambda x: x[1], reverse=True)
        conn_text = [f"{get_demonym(c)}: {w:.1f}x" for c, w in conns[:5]]
        nodes.append({
            'id': group,
            'label': get_demonym(group),
            'population': pop,
            'connections': conn_text
        })

    positions = _compute_force_layout(nodes, edges, iterations=150)
    return nodes, edges, positions


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
    MIN_SAMPLE = 10000  # Lowered to show more combinations for comparison

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
    """Get list of origins with enough data for single-origin overview.

    Only includes origins that have at least 2 parent combinations with
    sufficient sample size, so users can make meaningful comparisons.
    """
    df = DATA['marriage_agg'].copy()

    # Get origins that appear as either mother or father with substantial data
    mother_totals = df.groupby('MOTHER_ORIGIN')['WEIGHTED_COUNT'].sum()
    father_totals = df.groupby('FATHER_ORIGIN')['WEIGHTED_COUNT'].sum()

    MIN_TOTAL_SAMPLE = 100000  # Minimum total population for origin
    MIN_COMBO_SAMPLE = 10000   # Must match threshold in get_single_origin_overview

    candidate_origins = set()
    for origin in mother_totals[mother_totals >= MIN_TOTAL_SAMPLE].index:
        if origin not in NON_COUNTRIES:
            candidate_origins.add(origin)
    for origin in father_totals[father_totals >= MIN_TOTAL_SAMPLE].index:
        if origin not in NON_COUNTRIES:
            candidate_origins.add(origin)

    # Filter to origins that have at least 2 combinations above threshold
    valid_origins = []
    for origin in candidate_origins:
        origin_df = df[(df['MOTHER_ORIGIN'] == origin) | (df['FATHER_ORIGIN'] == origin)]
        origin_df = origin_df[~origin_df['MOTHER_ORIGIN'].isin(NON_COUNTRIES)]
        origin_df = origin_df[~origin_df['FATHER_ORIGIN'].isin(NON_COUNTRIES)]
        combos = origin_df.groupby(['MOTHER_ORIGIN', 'FATHER_ORIGIN'])['WEIGHTED_COUNT'].sum()
        num_combos = len(combos[combos >= MIN_COMBO_SAMPLE])

        if num_combos >= 2:  # Must have at least 2 combinations for comparison
            total = mother_totals.get(origin, 0) + father_totals.get(origin, 0)
            valid_origins.append((origin, total))

    # Sort by total population
    valid_origins.sort(key=lambda x: x[1], reverse=True)
    return [o[0] for o in valid_origins]


def get_geographic_data():
    """Return the full geographic aggregation dataframe, or None if unavailable."""
    return DATA.get('geographic')


def get_group_geographic_data(origin):
    """Get state-level data for a specific ethnic group."""
    df = get_geographic_data()
    if df is None:
        return None
    return df[df['ORIGIN_GROUP'] == origin]


def get_language_data():
    """Return the language aggregation dataframe, or None if unavailable."""
    return DATA.get('language')


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
            year_breakdown = ", ".join([f"{data['pct']:.0f}% from {yr}" for yr, data in sorted(year_comp.items())])
            lines.append(f"*This pooled sample draws from multiple censuses: {year_breakdown}.*")
            if max_pct > 40:
                lines.append(f"*The {max_year} census contributes the largest share ({max_pct:.0f}%). Select individual years above for temporal specificity.*")
            lines.append("")
    else:
        lines.append(f"*Showing {year} census data only, providing a single point-in-time snapshot.*")
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
    overscroll-behavior: contain;
}

/* Header */
.header-section {
    background: linear-gradient(135deg, #194852 0%, #0c2a30 100%);
    border-radius: 16px;
    padding: 0;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 20px rgba(12, 42, 48, 0.25);
    overflow: hidden;
    position: relative;
}

.header-bar {
    display: flex;
    align-items: center;
    padding: 1.5rem 2rem;
    gap: 2rem;
}

.header-logo {
    flex-shrink: 0;
}

.header-logo img {
    height: 60px;
    filter: brightness(0) invert(1);
}

.header-divider {
    width: 1px;
    height: 50px;
    background: rgba(255,255,255,0.2);
    flex-shrink: 0;
}

.poe-logo-wrap img {
    filter: none;
    height: 75px;
    opacity: 0.95;
}

.header-text {
    flex: 1;
    text-align: center !important;
}
.header-text h1,
.header-text p {
    text-align: center !important;
}

.header-section::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #7dceda 0%, #bca45e 50%, #7dceda 100%);
}

.header-intro {
    padding: 1.25rem 2rem;
    background: rgba(255,255,255,0.05);
}

.header-intro p {
    color: rgba(255,255,255,0.85);
    font-size: 0.9rem;
    line-height: 1.6;
    margin-bottom: 0.4rem;
}

.header-intro p:last-child {
    margin-bottom: 0;
}

.main-title {
    font-family: 'Neuton', Georgia, serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 0.25rem;
    letter-spacing: -0.02em;
    line-height: 1.2;
}

.subtitle {
    font-family: 'Hanken Grotesk', sans-serif;
    font-size: 1.1rem;
    font-weight: 400;
    color: #7dceda;
    margin: 0;
}

/* Filter Section */
.filter-section {
    background: linear-gradient(135deg, #194852 0%, #0c2a30 100%);
    border-radius: 16px;
    padding: 1.25rem 1.5rem 1rem 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(12, 42, 48, 0.25);
    overflow: visible;
}

/* Accent stripe removed to avoid rectangular edge on rounded container */

/* Anchor link navigation - full width bar */
.anchor-nav {
    display: flex;
    justify-content: center;
    gap: 0;
    padding: 0;
    background: linear-gradient(135deg, #194852 0%, #0c2a30 100%);
    border-radius: 12px 12px 0 0;
    margin-bottom: 0;
    box-shadow: 0 -2px 16px rgba(12, 42, 48, 0.15);
    overflow: hidden;
}

/* Integrated summary panel below anchor nav */
.summary-panel {
    background: #ffffff;
    padding: 1.25rem 1.5rem;
    border-left: 1px solid rgba(12, 42, 48, 0.08);
    border-right: 1px solid rgba(12, 42, 48, 0.08);
    border-bottom: 1px solid rgba(12, 42, 48, 0.08);
    border-radius: 0 0 12px 12px;
    margin-bottom: 1.25rem;
    box-shadow: 0 4px 16px rgba(12, 42, 48, 0.08);
}

.narrative-snapshot {
    font-family: 'Hanken Grotesk', sans-serif;
    font-size: 0.95rem;
    line-height: 1.6;
    color: #194852;
    margin-bottom: 1rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid rgba(12, 42, 48, 0.08);
}

.summary-panel-stats {
    display: flex;
    align-items: center;
    gap: 1rem;
    flex-wrap: wrap;
}

.summary-panel-actions {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-left: auto;
    flex-shrink: 0;
}

.anchor-link {
    color: rgba(255,255,255,0.85);
    text-decoration: none;
    font-size: 0.85rem;
    font-weight: 600;
    padding: 0.75rem 1.25rem;
    transition: all 0.2s ease;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    flex: 1;
    text-align: center;
    border-right: 1px solid rgba(255,255,255,0.1);
}

.anchor-link:last-child {
    border-right: none;
}

.anchor-link:hover {
    background: rgba(125, 206, 218, 0.2);
    color: #ffffff;
}

/* Action buttons - visible on light background */
.action-btn {
    background: linear-gradient(135deg, #194852 0%, #0c2a30 100%);
    color: #ffffff !important;
    border: none;
    padding: 0.5rem 1rem;
    border-radius: 6px;
    font-weight: 600;
    font-size: 0.8rem;
    cursor: pointer;
    transition: all 0.2s ease;
    font-family: 'Hanken Grotesk', sans-serif;
}

.action-btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(12, 42, 48, 0.25);
}

/* Social share buttons */
.social-share {
    display: flex;
    gap: 0.5rem;
    align-items: center;
}

.social-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    border-radius: 6px;
    color: #ffffff !important;
    text-decoration: none;
    font-size: 0.9rem;
    font-weight: bold;
    transition: all 0.2s ease;
}

.social-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    color: #ffffff !important;
    text-decoration: none;
}

.social-btn-twitter { background: #1DA1F2; }
.social-btn-linkedin { background: #0A66C2; }
.social-btn-facebook { background: #1877F2; }
.social-btn-email { background: #6c757d; }

/* Smooth scroll behavior */
html {
    scroll-behavior: smooth;
}

/* Scroll margin for anchored sections */
[id] {
    scroll-margin-top: 20px;
}

.filter-label {
    color: #7dceda;
    font-size: 0.8rem;
    font-weight: 600;
    margin-bottom: 0.25rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Dropdown Styling */
.Select-control {
    transition: all 0.2s ease !important;
}

.Select-control:hover {
    border-color: #7dceda !important;
}

/* Dropdown option highlights - light teal */
.VirtualizedSelectOption {
    transition: background 0.1s ease;
}

.VirtualizedSelectFocusedOption {
    background-color: rgba(125, 206, 218, 0.2) !important;
}

.VirtualizedSelectSelectedOption {
    background-color: rgba(125, 206, 218, 0.35) !important;
}

/* Dash dropdown menu option hover */
.Select-option.is-focused {
    background-color: rgba(125, 206, 218, 0.2) !important;
}

.Select-option.is-selected {
    background-color: rgba(125, 206, 218, 0.35) !important;
    color: #194852 !important;
}

/* Text selection highlight */
::selection {
    background: #7dceda;
    color: #194852;
}

::-moz-selection {
    background: #7dceda;
    color: #194852;
}

/* Cards */
.brand-card {
    background: #ffffff;
    border-radius: 16px;
    box-shadow: 0 4px 20px rgba(12, 42, 48, 0.1);
    overflow: hidden;
    transition: box-shadow 0.3s ease, transform 0.3s ease;
}

.brand-card:hover {
    box-shadow: 0 8px 30px rgba(12, 42, 48, 0.15);
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
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}

.brand-btn-secondary {
    background: rgba(255,255,255,0.15);
    color: #ffffff;
    border: 1px solid rgba(255,255,255,0.3);
}

.brand-btn-secondary:hover {
    background: rgba(255,255,255,0.3);
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

.brand-btn-secondary:active {
    transform: translateY(0);
}

/* Current Selection Display */
.current-selection {
    background: rgba(125, 206, 218, 0.15);
    border-radius: 8px;
    padding: 0.5rem 1rem;
    display: inline-block;
    border-left: 3px solid #7dceda;
}

.current-selection-label {
    color: rgba(255,255,255,0.7);
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-right: 0.5rem;
}

.current-selection-value {
    color: #ffffff;
    font-weight: 600;
    font-size: 0.95rem;
}

/* Stats */
.stat-display {
    text-align: left;
    padding: 0.75rem 1rem;
    background: linear-gradient(135deg, rgba(25, 72, 82, 0.03) 0%, rgba(52, 131, 151, 0.06) 100%);
    border-radius: 12px;
    border-left: 3px solid #348397;
    transition: all 0.2s ease;
}

.stat-value {
    font-family: 'Neuton', serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #194852;
    line-height: 1.2;
}

.stat-label {
    font-size: 0.7rem;
    color: #78a0a3;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.2rem;
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
    transition: all 0.2s ease;
}

.nav-tabs .nav-link:hover:not(.active) {
    color: rgba(255,255,255,0.95);
    background: rgba(255,255,255,0.1);
}

.nav-tabs .nav-link.active {
    background: #ffffff;
    color: #194852;
}

/* Collapsible Summary */
.summary-collapsed {
    max-height: 0;
    overflow: hidden;
    padding: 0 1.5rem !important;
    transition: max-height 0.3s ease, padding 0.3s ease;
}

.summary-expanded {
    max-height: 3000px;
    transition: max-height 0.5s ease;
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
    text-transform: uppercase;
    font-size: 0.8rem;
    letter-spacing: 0.03em;
}

.brand-table td {
    padding: 0.75rem 1rem;
    border-bottom: 1px solid #edf1f2;
    transition: background 0.15s ease;
}

.brand-table tr:hover td {
    background: rgba(125, 206, 218, 0.1);
}

.brand-table tr:nth-child(even) td {
    background: rgba(237, 241, 242, 0.5);
}

.brand-table tr:nth-child(even):hover td {
    background: rgba(125, 206, 218, 0.12);
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

/* Section Headers */
.section-header {
    display: flex;
    align-items: center;
    margin-bottom: 0.75rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(25, 72, 82, 0.1);
}

.section-title {
    font-family: 'Neuton', serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #194852;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.section-subtitle {
    font-family: 'Hanken Grotesk', sans-serif;
    font-size: 0.85rem;
    color: #78a0a3;
    margin-left: 0.5rem;
}

.section-header-prominent {
    display: flex;
    align-items: center;
    margin-bottom: 0.75rem;
    padding: 0.75rem 1.25rem;
    background: linear-gradient(135deg, #194852 0%, #0c2a30 100%);
    border-radius: 12px;
}

.section-header-prominent .section-title {
    color: #ffffff;
}

.section-header-prominent .section-subtitle {
    color: rgba(255,255,255,0.6);
}

/* Chart containers: responsive by default, scroll only when necessary */
.chart-scroll {
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
}

/* Desktop: enforce min-widths for dense charts */
@media (min-width: 769px) {
    .chart-scroll-wide .js-plotly-plot {
        min-width: 540px;
    }
    .chart-scroll-medium .js-plotly-plot {
        min-width: 480px;
    }
}

/* Mobile: let charts fit screen width */
@media (max-width: 768px) {
    .chart-scroll-wide .js-plotly-plot,
    .chart-scroll-medium .js-plotly-plot {
        min-width: 0 !important;
        width: 100% !important;
    }
    /* Tighter chart padding */
    .js-plotly-plot .plotly .main-svg {
        overflow: visible;
    }
}

/* Loading Animation Enhancement */
.dash-spinner {
    margin: 2rem auto;
}

/* Smooth content transitions */
.brand-card-body {
    transition: all 0.3s ease;
}

@media (max-width: 768px) {
    .header-bar {
        flex-direction: column;
        gap: 0.75rem;
        padding: 1.25rem 1.25rem;
        text-align: center;
    }
    .header-divider { display: none; }
    .header-logo img { height: 45px; }
    .poe-logo-wrap img { height: 50px; }
    .header-intro { padding: 1rem 1.25rem; }
    .main-title { font-size: 1.5rem; }
    .subtitle { font-size: 1rem; }
    .stat-display { padding: 0.5rem 0.75rem; }
    .stat-value { font-size: 1.3rem; }
    .current-selection { font-size: 0.85rem; }
    .narrative-snapshot { font-size: 0.85rem; }
    .summary-panel { padding: 1rem !important; }

    /* Anchor nav: allow horizontal scroll on small screens */
    .anchor-nav {
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
    }
    .anchor-link {
        font-size: 0.7rem;
        padding: 0.6rem 0.75rem;
        flex: 0 0 auto;
        white-space: nowrap;
    }

    /* Summary panel: stack actions below stats */
    .summary-panel-stats {
        flex-direction: column;
        align-items: stretch;
        gap: 0.75rem;
    }
    .summary-panel-actions {
        margin-left: 0;
        flex-wrap: wrap;
        justify-content: flex-start;
    }

    /* Tabs: scrollable on mobile to prevent text clipping */
    .nav-tabs {
        flex-wrap: nowrap;
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
        padding: 0.4rem 0.4rem 0 0.4rem;
    }
    .nav-tabs .nav-link {
        font-size: 0.75rem;
        padding: 0.6rem 0.75rem;
        white-space: nowrap;
        flex: 0 0 auto;
    }

    /* Brand cards: reduce padding on mobile */
    .brand-card-body {
        padding: 0.75rem !important;
    }
    .brand-card-header, .brand-card-header-gold, .brand-card-header-light {
        padding: 0.6rem 0.75rem !important;
        font-size: 0.9rem;
    }

    /* Methodology blocks: tighter on mobile */
    .methodology-content {
        padding: 0.75rem 1rem;
        font-size: 0.8rem;
    }

    /* Filter panel: tighter */
    .filter-label {
        font-size: 0.8rem;
    }
}

/* Welcome / Landing Section */
.welcome-section {
    background: #ffffff;
    border-radius: 16px;
    padding: 2rem 2rem 1.5rem 2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 20px rgba(12, 42, 48, 0.1);
}

.welcome-intro {
    font-size: 1.05rem;
    line-height: 1.6;
    color: #194852;
    text-align: center;
    max-width: 800px;
    margin: 0 auto 0.25rem auto;
}

.welcome-stats {
    display: flex;
    justify-content: space-around;
    padding: 1.25rem 0;
    margin: 1rem 0 1.5rem 0;
    border-top: 1px solid rgba(12, 42, 48, 0.08);
    border-bottom: 1px solid rgba(12, 42, 48, 0.08);
}

.welcome-stat-item {
    text-align: center;
    flex: 1;
    padding: 0 0.75rem;
}

.welcome-stat-value {
    font-family: 'Neuton', serif;
    font-size: 2rem;
    font-weight: 700;
    color: #194852;
    line-height: 1;
    margin-bottom: 0.3rem;
}

.welcome-stat-value-alt {
    font-family: 'Neuton', serif;
    font-size: 2rem;
    font-weight: 700;
    color: #bca45e;
    line-height: 1;
    margin-bottom: 0.3rem;
}

.welcome-stat-value-neutral {
    font-family: 'Neuton', serif;
    font-size: 2rem;
    font-weight: 700;
    color: #7dceda;
    line-height: 1;
    margin-bottom: 0.3rem;
}

.welcome-stat-label {
    font-size: 0.78rem;
    color: #78a0a3;
    line-height: 1.3;
}

.nav-cards-label {
    font-size: 0.8rem;
    font-weight: 600;
    color: #78a0a3;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.75rem;
}

.nav-cards {
    display: flex;
    gap: 1rem;
}

.nav-card {
    flex: 1;
    padding: 1.25rem;
    border-radius: 12px;
    border: 1px solid rgba(12, 42, 48, 0.08);
    cursor: pointer;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    background: #ffffff;
}

.nav-card:hover {
    box-shadow: 0 6px 20px rgba(12, 42, 48, 0.12);
    transform: translateY(-2px);
    border-color: rgba(12, 42, 48, 0.15);
}

.nav-card-compare { border-left: 4px solid #348397; }
.nav-card-geo { border-left: 4px solid #bca45e; }
.nav-card-explore { border-left: 4px solid #da5831; }

.nav-card-title {
    font-family: 'Neuton', serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #194852;
    margin-bottom: 0.4rem;
}

.nav-card-desc {
    font-size: 0.85rem;
    color: #78a0a3;
    margin: 0;
    line-height: 1.4;
}

@media (max-width: 768px) {
    .welcome-section {
        padding: 1.5rem 1.25rem 1.25rem 1.25rem;
    }
    .welcome-intro {
        font-size: 0.95rem;
    }
    .welcome-stats {
        flex-wrap: wrap;
        gap: 0.75rem;
    }
    .welcome-stat-item {
        flex: 0 0 45%;
    }
    .welcome-stat-value,
    .welcome-stat-value-alt,
    .welcome-stat-value-neutral {
        font-size: 1.5rem;
    }
    .nav-cards {
        flex-direction: column;
    }
}

/* Methodology collapsible blocks */
.methodology-toggle {
    background: none;
    border: 1px solid rgba(12, 42, 48, 0.15);
    color: #78a0a3;
    font-family: 'Hanken Grotesk', sans-serif;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 0.3rem 0.75rem;
    border-radius: 6px;
    cursor: pointer;
    transition: all 0.2s ease;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    margin-top: 0.75rem;
}

.methodology-toggle:hover {
    background: rgba(125, 206, 218, 0.1);
    border-color: #7dceda;
    color: #348397;
}

.methodology-content {
    background: linear-gradient(135deg, rgba(25, 72, 82, 0.03) 0%, rgba(52, 131, 151, 0.05) 100%);
    border-left: 3px solid #7dceda;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.25rem;
    margin-top: 0.75rem;
    font-size: 0.85rem;
    line-height: 1.6;
    color: #194852;
}

.methodology-content p {
    margin-bottom: 0.5rem;
}

.methodology-content p:last-child {
    margin-bottom: 0;
}

.methodology-content strong {
    color: #194852;
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
        <meta property="og:title" content="Marriage and the Melting Pot, 1880-1930">
        <meta property="og:description" content="Explore marriage patterns of second-generation Americans using census data from 1880-1930.">
        <meta property="og:type" content="website">
        <meta name="twitter:card" content="summary">
        <meta name="twitter:title" content="Marriage and the Melting Pot, 1880-1930 | Niskanen Center">
        <meta name="twitter:description" content="Explore marriage patterns of second-generation Americans using census data from 1880-1930.">
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
        <script>
            // Broadcast content height to parent window for iframe embedding.
            // Uses actual rendered height (not scrollHeight which doesn't shrink
            // reliably after CSS max-height collapse transitions).
            (function() {{
                var lastHeight = 0;
                var pending = null;
                var isIframe = window !== window.parent;

                // When embedded in an iframe, remove min-height: 100vh from body.
                // Otherwise 100vh = iframe height, creating a circular dependency
                // that prevents the iframe from shrinking on content collapse.
                if (isIframe) {{
                    document.documentElement.style.minHeight = '0';
                    document.body.style.minHeight = '0';
                }}

                function measureHeight() {{
                    // Measure the actual Dash app content container
                    var app = document.getElementById('_dash-app-content');
                    if (!app) app = document.getElementById('react-entry-point');
                    if (app) {{
                        var rect = app.getBoundingClientRect();
                        return Math.ceil(rect.top + window.scrollY + rect.height);
                    }}
                    // Fallback: walk top-level body children to find actual bottom
                    var maxBottom = 0;
                    var children = document.body.children;
                    for (var i = 0; i < children.length; i++) {{
                        var r = children[i].getBoundingClientRect();
                        var bottom = r.top + window.scrollY + r.height;
                        if (bottom > maxBottom) maxBottom = bottom;
                    }}
                    return Math.ceil(maxBottom) || document.documentElement.scrollHeight;
                }}

                function postHeight() {{
                    var h = measureHeight();
                    if (h !== lastHeight) {{
                        lastHeight = h;
                        window.parent.postMessage({{type: 'dashboardResize', height: h}}, '*');
                    }}
                }}

                // Schedule a height check after CSS transitions complete
                function scheduleCheck() {{
                    if (pending) clearTimeout(pending);
                    // Immediate check for expansions
                    requestAnimationFrame(postHeight);
                    // Delayed checks for collapse transitions (0.3s-0.5s CSS durations)
                    setTimeout(postHeight, 150);
                    pending = setTimeout(postHeight, 500);
                }}

                window.addEventListener('load', scheduleCheck);
                window.addEventListener('resize', postHeight);

                // Watch for DOM changes (tab switches, dropdown selections, collapse/expand)
                var observer = new MutationObserver(scheduleCheck);
                observer.observe(document.body, {{childList: true, subtree: true, attributes: true}});

                // Listen for CSS transition ends (collapse/expand animations)
                document.addEventListener('transitionend', function(e) {{
                    if (e.propertyName === 'max-height' || e.propertyName === 'height'
                        || e.propertyName === 'padding') {{
                        postHeight();
                    }}
                }});

                // Initial polls to catch async chart renders
                var polls = 0;
                var poller = setInterval(function() {{
                    postHeight();
                    if (++polls > 20) clearInterval(poller);
                }}, 500);
            }})();
        </script>
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
            # Header with Logo
            html.Div([
                html.Div([
                    html.Div([
                        html.A(
                            html.Img(src='/assets/niskanen-logo.png', alt='Niskanen Center',
                                    id='niskanen-logo'),
                            href='https://www.niskanencenter.org/', target='_blank',
                        ),
                    ], className='header-logo'),
                    html.Div(className='header-divider'),
                    html.Div([
                        html.H1("Marriage and the Melting Pot, 1880-1930", className='main-title'),
                        html.P("Whom did the US-born children of immigrants marry?", className='subtitle'),
                        html.P("Gil Guerra | Niskanen Center",
                               style={'color': 'rgba(255,255,255,0.5)', 'fontSize': '0.8rem',
                                      'margin': '0.3rem 0 0 0', 'fontWeight': '400'}),
                    ], className='header-text', style={'textAlign': 'center'}),
                    html.Div(className='header-divider'),
                    html.Div([
                        html.A(
                            html.Img(src='/assets/POE Logo Plain.png', alt='POE',
                                    id='poe-logo'),
                            href='https://www.points-of-entry.com/', target='_blank',
                        ),
                    ], className='header-logo poe-logo-wrap'),
                ], className='header-bar'),
            ], className='header-section'),

            # Welcome / Landing Section
            html.Div([
                html.P([
                    "This dashboard explores the marriage patterns of second-generation Americans"
                    "\u2014U.S.-born individuals with at least one immigrant parent\u2014using "
                    "census data from 1880 to 1930. Read the author\u2019s analysis of the findings ",
                    html.A("here", href="https://www.points-of-entry.com/p/marriage-and-the-melting-pot",
                           target="_blank", style={'color': COLORS['medium_teal'], 'textDecoration': 'underline'}),
                    "."
                ], className='welcome-intro'),
                # Topline stats
                html.Div([
                    html.Div([
                        html.Div("51.0%", className='welcome-stat-value'),
                        html.Div("married outside their heritage", className='welcome-stat-label'),
                    ], className='welcome-stat-item'),
                    html.Div([
                        html.Div("28.7%", className='welcome-stat-value'),
                        html.Div("married a 3rd+ gen American", className='welcome-stat-label'),
                    ], className='welcome-stat-item'),
                    html.Div([
                        html.Div("22.3%", className='welcome-stat-value'),
                        html.Div("married into a different recent-immigrant community", className='welcome-stat-label'),
                    ], className='welcome-stat-item'),
                    html.Div([
                        html.Div("48.9%", className='welcome-stat-value-alt'),
                        html.Div("married within their ethnic heritage", className='welcome-stat-label'),
                    ], className='welcome-stat-item'),
                    html.Div([
                        html.Div("17.8M", className='welcome-stat-value-neutral'),
                        html.Div("census records drawn from", className='welcome-stat-label'),
                    ], className='welcome-stat-item'),
                ], className='welcome-stats'),
                # Navigation cards
                html.Div("Explore the data:", className='nav-cards-label'),
                html.Div([
                    html.Div([
                        html.Div("How did groups compare?", className='nav-card-title'),
                        html.P("Compare outmarriage rates across 26 ethnic origins, "
                               "with and without geographic adjustment.",
                               className='nav-card-desc'),
                    ], id='nav-card-compare', n_clicks=0,
                       className='nav-card nav-card-compare'),
                    html.Div([
                        html.Div("How did geography affect immigrant assimilation?", className='nav-card-title'),
                        html.P("See how local ethnic composition affected "
                               "intermarriage patterns.",

                               className='nav-card-desc'),
                    ], id='nav-card-geo', n_clicks=0,
                       className='nav-card nav-card-geo'),
                    html.Div([
                        html.Div("Explore a specific group", className='nav-card-title'),
                        html.P("Select parental origins to see detailed marriage patterns, "
                               "trends, and spouse backgrounds.",
                               className='nav-card-desc'),
                    ], id='nav-card-explore', n_clicks=0,
                       className='nav-card nav-card-explore'),
                ], className='nav-cards'),
            ], className='welcome-section'),

            # Filters
            html.Div([
                html.Div("Select parental origins below to explore specific groups, "
                         "or use \"Quick Presets\" for common combinations.",
                         style={'color': 'rgba(255,255,255,0.7)', 'fontSize': '0.85rem',
                                'marginBottom': '0.75rem'}),
                # Current Selection Display
                html.Div(id='current-selection-display', style={'marginBottom': '1rem'}),
                dbc.Row([
                    dbc.Col([
                        html.Label("Mother's Birth Country", className='filter-label'),
                        html.Div("Where was the subject's mother born?",
                                style={'fontSize': '0.75rem', 'color': 'rgba(255,255,255,0.6)', 'marginBottom': '0.25rem'}),
                        dcc.Dropdown(id='mother-dropdown', options=mother_dropdown_options,
                                    value='Any', clearable=False)
                    ], lg=3, md=6, xs=12, className='mb-3'),
                    dbc.Col([
                        html.Label("Father's Birth Country", className='filter-label'),
                        html.Div("Where was the subject's father born?",
                                style={'fontSize': '0.75rem', 'color': 'rgba(255,255,255,0.6)', 'marginBottom': '0.25rem'}),
                        dcc.Dropdown(id='father-dropdown', options=father_dropdown_options,
                                    value='Any', clearable=False)
                    ], lg=3, md=6, xs=12, className='mb-3'),
                    dbc.Col([
                        html.Label("Census Year", className='filter-label'),
                        html.Div("Filter by census",
                                style={'fontSize': '0.75rem', 'color': 'rgba(255,255,255,0.6)', 'marginBottom': '0.25rem'}),
                        dcc.Dropdown(id='year-dropdown',
                                    options=[{'label': 'All Years (1880-1930)', 'value': 'All'}] +
                                            [{'label': str(y), 'value': y} for y in years],
                                    value='All', clearable=False)
                    ], lg=2, md=4, xs=6, className='mb-3'),
                    dbc.Col([
                        html.Label("Quick Presets", className='filter-label'),
                        html.Div("Common combinations",
                                style={'fontSize': '0.75rem', 'color': 'rgba(255,255,255,0.6)', 'marginBottom': '0.25rem'}),
                        dcc.Dropdown(id='preset-dropdown', options=DYNAMIC_PRESETS,
                                    value='', clearable=False, optionHeight=45, maxHeight=400,
                                    placeholder="Try a preset...")
                    ], lg=2, md=4, xs=6, className='mb-3'),
                    dbc.Col([
                        html.Label("Actions", className='filter-label'),
                        html.Div(" ", style={'fontSize': '0.75rem', 'marginBottom': '0.25rem'}),
                        html.Div([
                            html.Button("Reset All", id='reset-btn', className='brand-btn brand-btn-secondary'),
                        ], className='d-flex gap-3')
                    ], lg=2, md=4, xs=12, className='mb-3'),
                ]),
            ], id='filters', className='filter-section'),

            # Anchor Navigation + Integrated Summary Panel
            html.Div([
                html.A("Charts", href='#charts', className='anchor-link'),
                html.A("Analysis", href='#analysis', className='anchor-link'),
                html.A("Spouse Backgrounds", href='#spouse-table', className='anchor-link'),
                html.A("Compare Groups", href='#compare', className='anchor-link'),
                html.A("Deeper Analysis", href='#deeper-analysis', className='anchor-link'),
                html.A("Methodology", href='#methodology', className='anchor-link'),
            ], className='anchor-nav'),
            html.Div([
                # Narrative snapshot
                html.Div(id='narrative-snapshot', className='narrative-snapshot'),
                # Stats + actions row
                html.Div([
                    html.Div([
                        html.Div(id='sample-size', className='stat-value'),
                        html.Div("Sample Size", className='stat-label'),
                    ], className='stat-display', title='Weighted population estimate based on census sampling'),
                    html.Div([
                        html.Div(id='key-stat-heritage', className='stat-value'),
                        html.Div("Within Heritage", className='stat-label'),
                    ], className='stat-display', title='Married someone connected to their immigrant heritage (same, mother\'s, father\'s, or both parents\' origins)'),
                    html.Div([
                        html.Div(id='key-stat-american', className='stat-value'),
                        html.Div("3rd+ Gen American", className='stat-label'),
                    ], className='stat-display', title='Married someone whose parents were both US-born (established American family)'),
                    html.Div([
                        html.Button("Download CSV", id='download-csv-btn', className='action-btn me-2'),
                        dcc.Download(id='download-csv'),
                        html.Button("Copy Link", id='copy-link-btn', className='action-btn me-2'),
                        html.Span(id='link-copied-msg', style={'marginRight': '0.5rem', 'color': COLORS['green'], 'fontWeight': '600', 'fontSize': '0.85rem'}),
                        html.Span([
                            html.A("X", href='#', id='share-twitter', className='social-btn social-btn-twitter', title='Share on X/Twitter', target='_blank'),
                            html.A("in", href='#', id='share-linkedin', className='social-btn social-btn-linkedin', title='Share on LinkedIn', target='_blank'),
                            html.A("f", href='#', id='share-facebook', className='social-btn social-btn-facebook', title='Share on Facebook', target='_blank'),
                            html.A("✉", href='#', id='share-email', className='social-btn social-btn-email', title='Share via Email'),
                        ], className='social-share'),
                    ], className='summary-panel-actions'),
                ], className='summary-panel-stats'),
            ], className='summary-panel'),

            # Selection-Based Visualization Tabs (respond to heritage dropdowns)
            html.Div([
                dbc.Tabs([
                    dbc.Tab(label="Main Chart", tab_id="tab-main"),
                    dbc.Tab(label="Trends Over Time", tab_id="tab-trends"),
                    dbc.Tab(label="Spouse Generation", tab_id="tab-spouse-gen"),
                ], id='viz-tabs', active_tab='tab-main', className='mb-0'),
                html.Div([html.Div(id='tab-content')], className='brand-card',
                        style={'borderRadius': '0 0 16px 16px'})
            ], id='charts', className='mb-4'),

            # Summary Card - Now collapsible and after the chart
            html.Div([
                html.Div([
                    html.Span("Detailed Analysis", style={'flex': '1'}),
                    html.Button("Show", id='toggle-summary-btn', className='brand-btn',
                               style={'background': 'transparent', 'color': '#fff', 'border': '1px solid rgba(255,255,255,0.3)',
                                      'padding': '0.25rem 0.75rem', 'fontSize': '0.8rem'})
                ], className='brand-card-header-gold', style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'space-between'}),
                html.Div([
                    dcc.Loading(type='circle', color=COLORS['medium_teal'],
                               children=[dcc.Markdown(id='auto-summary', className='summary-markdown')])
                ], id='summary-body', className='brand-card-body summary-collapsed')
            ], id='analysis', className='brand-card summary-card mb-4'),

            # Spouse Backgrounds Table
            html.Div([
                html.Div("Spouse Backgrounds (Top 15)", className='brand-card-header-light'),
                html.Div([
                    dcc.Loading(type='circle', color=COLORS['medium_teal'],
                               children=[html.Div(id='spouse-table-content', style={'maxHeight': '400px', 'overflowY': 'auto'})])
                ], className='brand-card-body')
            ], id='spouse-table', className='brand-card mb-4'),

            # Compare All Groups Section
            html.Div([
                html.Div([
                    html.Span("Compare All Groups", className='section-title'),
                    html.Span("— cross-group patterns (independent of filter selections)", className='section-subtitle')
                ], className='section-header-prominent'),
                dbc.Tabs([
                    dbc.Tab(label="Outmarriage Rates", tab_id="tab-outmarriage"),
                    dbc.Tab(label="Clustering Network", tab_id="tab-heatmap"),
                    dbc.Tab(label="Single Origin Overview", tab_id="tab-single-origin"),
                ], id='overview-tabs', active_tab='tab-outmarriage', className='mb-0'),
                html.Div([html.Div(id='overview-tab-content')], className='brand-card',
                        style={'borderRadius': '0 0 16px 16px'})
            ], id='compare', className='mb-4'),

            # What Explains These Patterns?
            html.Div([
                html.Div([
                    html.Span("What Explains These Patterns?", className='section-title'),
                    html.Span("— geography and culture beyond national origin", className='section-subtitle')
                ], className='section-header-prominent'),
                dbc.Tabs([
                    dbc.Tab(label="Concentration vs. Outmarriage", tab_id="tab-geo-scatter",
                            disabled=DATA.get('geographic') is None),
                    dbc.Tab(label="By State", tab_id="tab-geo-state",
                            disabled=DATA.get('geographic') is None),
                    dbc.Tab(label="Beyond Geography", tab_id="tab-geo-residuals",
                            disabled=DATA.get('geographic') is None),
                ], id='explain-tabs', active_tab='tab-geo-scatter', className='mb-0'),
                html.Div([html.Div(id='explain-tab-content')], className='brand-card',
                        style={'borderRadius': '0 0 16px 16px'})
            ], id='deeper-analysis', className='mb-4'),

            # Methodology
            html.Div([
                html.Div("Methodology & Data Sources", className='brand-card-header'),
                html.Div([
                    html.H4("Data Source", style={'color': COLORS['dark_teal'], 'marginTop': '0', 'marginBottom': '0.5rem', 'fontFamily': 'Neuton, serif'}),
                    html.P([
                        "This dashboard draws on 17.8 million individual census records from ",
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
                            html.Tr([html.Td("1880"), html.Td("10%"), html.Td("5.9 million")]),
                            html.Tr([html.Td("1900"), html.Td("5%"), html.Td("3.9 million")], style={'backgroundColor': COLORS['very_light_gray']}),
                            html.Tr([html.Td("1910"), html.Td("1%"), html.Td("0.9 million")]),
                            html.Tr([html.Td("1920"), html.Td("1%"), html.Td("1.1 million")], style={'backgroundColor': COLORS['very_light_gray']}),
                            html.Tr([html.Td("1930"), html.Td("5%"), html.Td("6.1 million")]),
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
                        "Guerra, Gil. \u201cMarriage and the Melting Pot, 1880\u20131930.\u201d Interactive dashboard. Washington, DC: Niskanen Center, 2026. ",
                        html.A("https://www.niskanencenter.org/marriage-meltingpot/", href="https://www.niskanencenter.org/marriage-meltingpot/", target="_blank", style={'color': COLORS['medium_teal']}),
                        "."
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
            ], id='methodology', className='brand-card mb-4'),

            # Stay Connected
            html.Div([
                html.Div("Stay Connected", className='brand-card-header'),
                html.Div([
                    html.P([
                        "For more analysis of these findings and other immigration research, follow ",
                        html.A("Points of Entry", href="https://www.points-of-entry.com/", target="_blank",
                               style={'color': COLORS['medium_teal'], 'fontWeight': '600'}),
                        " on Substack."
                    ], style={'marginBottom': '0.75rem'}),
                    html.P([
                        "Comments, inquiries, and corrections are welcome. Please contact ",
                        html.A("Gil Guerra", href="mailto:gguerra@niskanencenter.org", style={'color': COLORS['medium_teal']}),
                        " at ",
                        html.A("gguerra@niskanencenter.org", href="mailto:gguerra@niskanencenter.org", style={'color': COLORS['medium_teal']}),
                        "."
                    ], style={'marginBottom': '0'}),
                ], className='brand-card-body')
            ], className='brand-card mb-4'),

            # Footer
            html.Div([
                html.P([
                    "Data: ", html.A("IPUMS USA", href="https://usa.ipums.org", target="_blank"),
                    " | Analysis: ", html.A("Niskanen Center, Gil Guerra", href="https://www.niskanencenter.org/author/gguerra/", target="_blank"),
                    " | Follow: ", html.A("Points of Entry", href="https://www.points-of-entry.com/", target="_blank"),
                ], className='footer-text'),
            ], className='footer-section'),

        ], fluid=True, style={'maxWidth': '1400px'})
    ], style={'minHeight': '100vh', 'padding': '1rem'})
])

# =============================================================================
# WELCOME CARD NAVIGATION (clientside)
# =============================================================================

app.clientside_callback(
    """
    function(n1, n2, n3) {
        var triggered = dash_clientside.callback_context.triggered;
        if (!triggered || triggered.length === 0) return [dash_clientside.no_update, dash_clientside.no_update];
        var id = triggered[0].prop_id.split('.')[0];
        if (id === 'nav-card-compare') {
            var el = document.getElementById('compare');
            if (el) el.scrollIntoView({behavior: 'smooth'});
            return ['tab-outmarriage', dash_clientside.no_update];
        } else if (id === 'nav-card-geo') {
            var el = document.getElementById('deeper-analysis');
            if (el) el.scrollIntoView({behavior: 'smooth'});
            return [dash_clientside.no_update, 'tab-geo-scatter'];
        } else if (id === 'nav-card-explore') {
            var el = document.getElementById('filters');
            if (el) el.scrollIntoView({behavior: 'smooth'});
            return [dash_clientside.no_update, dash_clientside.no_update];
        }
        return [dash_clientside.no_update, dash_clientside.no_update];
    }
    """,
    [Output('overview-tabs', 'active_tab', allow_duplicate=True),
     Output('explain-tabs', 'active_tab', allow_duplicate=True)],
    [Input('nav-card-compare', 'n_clicks'),
     Input('nav-card-geo', 'n_clicks'),
     Input('nav-card-explore', 'n_clicks')],
    prevent_initial_call=True
)

# =============================================================================
# METHODOLOGY BLOCK HELPER
# =============================================================================

def _methodology_block(chart_id, paragraphs):
    """Return a collapsible 'Detailed Methodology' block for a chart.

    Parameters
    ----------
    chart_id : str
        Unique identifier for this chart's methodology toggle.
    paragraphs : list[str]
        List of paragraph strings to display when expanded.
    """
    return html.Div([
        html.Button(
            "Detailed Methodology",
            id={'type': 'methodology-btn', 'index': chart_id},
            n_clicks=0,
            className='methodology-toggle',
        ),
        dbc.Collapse(
            html.Div(
                [html.P(p) for p in paragraphs],
                className='methodology-content',
            ),
            id={'type': 'methodology-collapse', 'index': chart_id},
            is_open=False,
        ),
    ])


@callback(
    Output({'type': 'methodology-collapse', 'index': ALL}, 'is_open'),
    Input({'type': 'methodology-btn', 'index': ALL}, 'n_clicks'),
    State({'type': 'methodology-collapse', 'index': ALL}, 'is_open'),
    prevent_initial_call=True,
)
def toggle_methodology(n_clicks_list, is_open_list):
    """Toggle whichever methodology block was clicked."""
    triggered = ctx.triggered_id
    if triggered is None:
        return [no_update] * len(is_open_list)
    results = []
    for i, btn_id in enumerate(ctx.inputs_list[0]):
        if btn_id['id']['index'] == triggered['index']:
            results.append(not is_open_list[i])
        else:
            results.append(no_update)
    return results


# =============================================================================
# NARRATIVE SNAPSHOT HELPER
# =============================================================================

def _build_narrative_snapshot(mother, father, year, mother_dem, father_dem,
                              heritage_pct, american_pct, diff_pct, weighted, unweighted):
    """Build a 2-3 sentence narrative summary for the control panel."""
    # Build subject phrase
    if mother != 'Any' and father != 'Any':
        if mother == father:
            subject = f"second-generation {mother_dem}-Americans"
        elif mother == 'US-born':
            subject = f"children of {father_dem} fathers and American mothers"
        elif father == 'US-born':
            subject = f"children of {mother_dem} mothers and American fathers"
        else:
            subject = f"children of {mother_dem} mothers and {father_dem} fathers"
    elif mother != 'Any':
        subject = f"children of {mother_dem} mothers"
    elif father != 'Any':
        subject = f"children of {father_dem} fathers"
    else:
        subject = "second-generation Americans"

    time_phrase = f"in {year}" if year != 'All' else "across 1880\u20131930"

    if unweighted < 30:
        return f"Very small sample for {subject} {time_phrase} ({unweighted} records). Results may not be reliable."

    # Sentence 1: Main finding
    if heritage_pct >= american_pct:
        sentence1 = (f"Among {subject} {time_phrase}, {heritage_pct:.1f}% married within "
                     f"their ethnic heritage community, while {american_pct:.1f}% married a "
                     f"3rd+ generation American.")
    else:
        sentence1 = (f"Among {subject} {time_phrase}, {american_pct:.1f}% married a "
                     f"3rd+ generation American\u2014the most common outcome\u2014while "
                     f"{heritage_pct:.1f}% married within their ethnic heritage.")

    # Sentence 2: Cross-ethnic or interpretive color
    if diff_pct >= 5:
        sentence2 = (f"Another {diff_pct:.1f}% married someone from a different immigrant "
                     f"background entirely.")
    elif heritage_pct >= 60:
        sentence2 = "This group showed strong ethnic retention in marriage patterns."
    elif american_pct >= 50:
        sentence2 = "This group was notably assimilated into the broader American population."
    else:
        sentence2 = "Marriage patterns were relatively mixed across categories."

    # Sentence 3: Sample context
    if weighted >= 100000:
        sentence3 = f"Based on a robust sample of {weighted:,.0f} individuals."
    else:
        sentence3 = f"Based on {weighted:,.0f} individuals ({unweighted:,} census records)."

    return f"{sentence1} {sentence2} {sentence3}"


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
        except (ValueError, TypeError):
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


app.clientside_callback(
    """
    function(n_clicks, href) {
        if (!n_clicks) return [dash_clientside.no_update, ''];
        // Build shareable URL pointing to the Niskanen embed page
        var shareUrl = 'https://www.niskanencenter.org/marriage-meltingpot/';
        // Copy to clipboard (works in both standalone and iframe contexts)
        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(shareUrl).catch(function() {});
        }
        // Clear "Copied!" after 3 seconds
        setTimeout(function() {
            var el = document.getElementById('link-copied-msg');
            if (el) el.textContent = '';
        }, 3000);
        return [shareUrl, 'Copied!'];
    }
    """,
    [Output('clipboard', 'content'), Output('link-copied-msg', 'children')],
    [Input('copy-link-btn', 'n_clicks')],
    [State('url', 'href')],
    prevent_initial_call=True
)


# Social share button URLs
@callback(
    [Output('share-twitter', 'href'),
     Output('share-linkedin', 'href'),
     Output('share-facebook', 'href'),
     Output('share-email', 'href')],
    [Input('url', 'href')]
)
def update_social_links(href):
    # Always share the Niskanen embed page
    share_url = "https://www.niskanencenter.org/marriage-meltingpot/"

    title = "Marriage and the Melting Pot, 1880-1930 - Niskanen Center"
    description = "Explore marriage patterns of second-generation Americans using census data from 1880-1930."

    from urllib.parse import quote

    twitter_url = f"https://twitter.com/intent/tweet?url={quote(share_url)}&text={quote(title)}"
    linkedin_url = f"https://www.linkedin.com/sharing/share-offsite/?url={quote(share_url)}"
    facebook_url = f"https://www.facebook.com/sharer/sharer.php?u={quote(share_url)}"
    email_url = f"mailto:?subject={quote(title)}&body={quote(description + chr(10) + chr(10) + share_url)}"

    return twitter_url, linkedin_url, facebook_url, email_url


@callback(
    [Output('summary-body', 'className', allow_duplicate=True),
     Output('toggle-summary-btn', 'children', allow_duplicate=True)],
    [Input('toggle-summary-btn', 'n_clicks')],
    [State('summary-body', 'className')],
    prevent_initial_call=True
)
def toggle_summary(n_clicks, current_class):
    if n_clicks:
        if current_class and 'summary-collapsed' in current_class:
            return 'brand-card-body summary-expanded', 'Hide'
        else:
            return 'brand-card-body summary-collapsed', 'Show'
    return no_update, no_update


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


@callback([Output('sample-size', 'children'), Output('auto-summary', 'children'),
           Output('key-stat-heritage', 'children'), Output('key-stat-american', 'children'),
           Output('current-selection-display', 'children'),
           Output('narrative-snapshot', 'children'),
           Output('summary-body', 'className'), Output('toggle-summary-btn', 'children')],
          [Input('mother-dropdown', 'value'), Input('father-dropdown', 'value'), Input('year-dropdown', 'value')])
def update_summary(mother, father, year):
    stats, weighted, unweighted = get_marriage_stats(mother, father, year)

    # Build current selection display
    mother_dem = get_demonym(mother) if mother != 'Any' else 'Any'
    father_dem = get_demonym(father) if father != 'Any' else 'Any'
    year_display = f"{year}" if year != 'All' else "All Years"

    if mother != 'Any' and father != 'Any':
        if mother == father:
            selection_text = f"{mother_dem} × {father_dem}"
        else:
            selection_text = f"{mother_dem} mom × {father_dem} dad"
    elif mother != 'Any':
        selection_text = f"{mother_dem} mother × Any father"
    elif father != 'Any':
        selection_text = f"Any mother × {father_dem} father"
    else:
        selection_text = "All Second-Generation Americans"

    current_selection = html.Div([
        html.Span("Currently viewing: ", className='current-selection-label'),
        html.Span(f"{selection_text} | {year_display}", className='current-selection-value')
    ], className='current-selection')

    # Auto-expand when a specific origin is selected, collapse for default view
    is_default = (mother == 'Any' and father == 'Any')
    summary_class = 'brand-card-body summary-collapsed' if is_default else 'brand-card-body summary-expanded'
    toggle_label = 'Show' if is_default else 'Hide'

    if stats is None:
        return "0", "No data available", "—", "—", current_selection, "No data available for this selection.", summary_class, toggle_label

    # Sample size
    if unweighted < 30:
        size_text = html.Span([f"{weighted:,.0f}", html.Br(),
                               html.Span(f"Small sample ({unweighted})", style={'fontSize': '0.7rem', 'color': COLORS['orange']})])
    else:
        size_text = f"{weighted:,.0f}"

    # Calculate key percentages
    total = stats['WEIGHTED_COUNT'].sum()
    pcts = dict(zip(stats['MARRIAGE_TYPE'], stats['WEIGHTED_COUNT'] / total * 100))

    # Heritage-based marriages (same origin + mother's + father's + both heritages)
    heritage_pct = sum(v for k, v in pcts.items()
                      if any(x in k for x in ['same origin', "mother's origin", "father's origin", 'both heritages']))
    heritage_text = f"{heritage_pct:.0f}%"

    # 3rd+ gen American marriages
    american_pct = sum(v for k, v in pcts.items() if '3rd+ gen' in k)
    american_text = f"{american_pct:.0f}%"

    # Different origin marriages
    diff_pct = sum(v for k, v in pcts.items() if 'different origin' in k)

    # Generate narrative snapshot (2-3 sentences)
    narrative = _build_narrative_snapshot(
        mother, father, year, mother_dem, father_dem,
        heritage_pct, american_pct, diff_pct, weighted, unweighted
    )

    summary = generate_summary(mother, father, year)
    return size_text, summary, heritage_text, american_text, current_selection, narrative, summary_class, toggle_label


@callback(Output('tab-content', 'children'),
          [Input('viz-tabs', 'active_tab'), Input('mother-dropdown', 'value'),
           Input('father-dropdown', 'value'), Input('year-dropdown', 'value')])
def render_tab_content(active_tab, mother, father, year):
    """Render selection-based tabs that respond to heritage dropdowns."""
    if active_tab == 'tab-main':
        return html.Div([
            dcc.Loading(type='circle', color=COLORS['medium_teal'],
                       children=[html.Div(dcc.Graph(id='main-chart', figure=create_main_chart(mother, father, year),
                                          config={'displayModeBar': True, 'scrollZoom': False, 'responsive': True}), className='chart-scroll chart-scroll-wide')]),
            _methodology_block('main-chart', [
                "This chart shows the marriage patterns of second-generation Americans (U.S.-born individuals with at least one immigrant parent) based on the selected parental origins and census year.",
                "Each bar represents a marriage outcome category. \"Same origin\" means the spouse shares the subject's parental heritage (e.g., a child of Irish parents married to an Irish immigrant or another child of Irish parents). \"3rd+ gen American\" means the spouse's parents were both U.S.-born. \"Different origin\" means the spouse has a different immigrant background.",
                "Percentages are computed from weighted census microdata (IPUMS USA). Person weights (PERWT) are applied to produce population-representative estimates. Spouse information comes from IPUMS's \"Attach Characteristics\" feature, which links married individuals to their co-resident spouse's birthplace and parental birthplace data.",
            ]),
        ], style={'padding': '1rem'})
    elif active_tab == 'tab-trends':
        return html.Div([
            dcc.Loading(type='circle', color=COLORS['medium_teal'],
                       children=[html.Div(dcc.Graph(id='time-chart', figure=create_time_chart(mother, father),
                                          config={'displayModeBar': True, 'scrollZoom': False, 'responsive': True}), className='chart-scroll chart-scroll-medium')]),
            _methodology_block('trends-chart', [
                "This chart tracks how marriage patterns changed across census years (1880, 1900, 1910, 1920, 1930) for the selected parental origins.",
                "Each line shows the percentage of second-generation Americans who married a particular category of spouse. The data is cross-sectional: each census year captures everyone currently married at that time, not new marriages formed that year.",
                "Caution: individuals who remained married across multiple census years may appear in more than one cross-section. Historical census microdata lacks longitudinal identifiers, so individual-level deduplication is not possible. Trends should be interpreted as shifts in the stock of existing marriages, not strictly as changes in marriage formation behavior.",
            ]),
        ], style={'padding': '1rem'})
    elif active_tab == 'tab-spouse-gen':
        return html.Div([
            html.P("Distribution of spouse generations for the current selection. Shows whether children of immigrants "
                   "married recent immigrants (1st gen), fellow second-generation Americans (2nd gen), or established Americans (3rd+ gen).",
                   style={'color': COLORS['muted_teal'], 'fontSize': '0.9rem', 'marginBottom': '1rem'}),
            dcc.Loading(type='circle', color=COLORS['medium_teal'],
                       children=[html.Div(dcc.Graph(id='spouse-gen-chart', figure=create_spouse_gen_chart(mother, father, year),
                                          config={'displayModeBar': True, 'scrollZoom': False, 'responsive': True}), className='chart-scroll chart-scroll-medium')]),
            _methodology_block('spouse-gen-chart', [
                "This chart breaks down spouses by their immigrant generation. \"1st generation\" spouses were themselves born outside the U.S. \"2nd generation\" spouses were U.S.-born with at least one foreign-born parent. \"3rd+ generation\" spouses had both parents born in the U.S.",
                "Generation is determined from the spouse's birthplace (BPL_SP) and the spouse's parents' birthplaces (FBPL_SP, MBPL_SP) as recorded in the census. A spouse is classified as 2nd generation if either parent was foreign-born, and 3rd+ generation only if both parents were U.S.-born.",
                "This classification means \"3rd+ generation\" is a residual category\u2014it includes anyone whose immigrant ancestry cannot be traced through parental birthplace, regardless of how many generations their family has been in the U.S.",
            ]),
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
                        {'label': 'Outmarriage into Different Recent-Immigrant Communities', 'value': 'other_immigrant'},
                        {'label': 'Geography-Adjusted Rate', 'value': 'geo_adjusted',
                         'disabled': DATA.get('geographic') is None},
                    ],
                    value='total',
                    style={'maxWidth': '400px', 'width': '100%', 'display': 'inline-block', 'verticalAlign': 'middle'},
                    clearable=False
                ),
            ], style={'marginBottom': '1rem'}),
            dcc.Loading(type='circle', color=COLORS['medium_teal'],
                       children=[html.Div(id='outmarriage-chart-container')]),
            _methodology_block('outmarriage', [
                "Outmarriage rate measures the percentage of second-generation Americans (with same-origin parents) who married someone outside their own ethnic group. Only groups with at least 20,000 weighted individuals are included.",
                "\"Total outmarriage\" combines marriages to 3rd+ generation Americans and marriages into different recent-immigrant communities. \"3rd+ gen American\" counts spouses whose parents were both U.S.-born. \"Different recent-immigrant communities\" counts spouses from a different immigrant heritage.",
                "The \"Geography-Adjusted Rate\" removes the effect of geographic concentration. It fits a regression of outmarriage rate on local ethnic concentration across states, computes each group's average residual, and adds it to the overall mean. This shows which groups outmarried more or less than their geographic circumstances would predict. Groups colored teal outmarried more than concentration predicts; gold outmarried less.",
            ]),
        ], style={'padding': '1rem'})
    elif active_tab == 'tab-heatmap':
        has_adjusted = DATA.get('geo_adjusted_affinity') is not None
        return html.Div([
            html.P("Which ethnic groups had marriage affinities with each other? This network shows connections between second-generation Americans "
                   "(children of same-origin immigrant parents) who intermarried at higher rates than population sizes alone would predict.",
                   style={'color': COLORS['muted_teal'], 'fontSize': '0.9rem', 'marginBottom': '1rem'}),
            html.Div([
                html.Label("View:", style={'fontWeight': '500', 'marginRight': '10px',
                                           'color': COLORS['dark_teal']}),
                dcc.Dropdown(
                    id='network-view-dropdown',
                    options=[
                        {'label': 'National Affinities', 'value': 'raw'},
                        {'label': 'Geography-Adjusted', 'value': 'adjusted',
                         'disabled': not has_adjusted},
                    ],
                    value='adjusted' if has_adjusted else 'raw',
                    style={'maxWidth': '300px', 'width': '100%', 'display': 'inline-block', 'verticalAlign': 'middle'},
                    clearable=False
                ),
            ], style={'marginBottom': '1rem'}),
            dcc.Loading(type='circle', color=COLORS['medium_teal'],
                       children=[html.Div(id='network-chart-container')]),
            _methodology_block('clustering', [
                "This network diagram shows which ethnic groups intermarried at rates higher than random chance would predict. Each node is an ethnic group (sized by population). An edge connects two groups whose intermarriage rate exceeds what population shares alone would predict (affinity > 1.0x). Thicker edges indicate stronger affinities.",
                "Affinity is calculated as: (observed intermarriage rate) / (expected rate based on population shares). An affinity of 2.0x means two groups intermarried at twice the rate you'd expect given their relative population sizes. The layout uses a force-directed algorithm where groups with stronger affinities are pulled closer together.",
                "The \"Geography-Adjusted\" view removes the effect of geographic co-location. Instead of using national population shares, it computes expected intermarriage rates using each group's share within each state, then averages across states. This reveals affinities that persist even after accounting for the fact that some groups lived in the same places. The attraction and avoidance bar charts show the strongest and weakest adjusted pairings. Pairs require at least 500 weighted observations per state-level cell and must appear in both directions to be included.",
            ]),
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
                    style={'maxWidth': '250px', 'width': '100%', 'display': 'inline-block', 'verticalAlign': 'middle'}
                ),
            ], style={'marginBottom': '1rem'}),
            dcc.Loading(type='circle', color=COLORS['medium_teal'],
                       children=[html.Div(id='single-origin-chart-container')]),
            _methodology_block('single-origin', [
                "This chart compares marriage patterns for different parental origin combinations involving the selected heritage. For example, selecting \"Ireland\" shows children of Irish-Irish parents alongside children of Irish-German parents, Irish-English parents, etc.",
                "Each row shows the marriage outcome breakdown for that parental combination: what percentage married within heritage, married a 3rd+ generation American, or married into a different immigrant community. Only the most common parental combinations are shown (those with sufficient sample sizes). US-born parents are excluded to focus on immigrant-origin combinations.",
            ]),
        ], style={'padding': '1rem'})

    return html.Div()


@callback(Output('explain-tab-content', 'children'),
          [Input('explain-tabs', 'active_tab'), Input('year-dropdown', 'value')])
def render_explain_tab_content(active_tab, year):
    """Render explain tabs that decompose patterns by geography."""
    geo_df = get_geographic_data()
    if geo_df is None:
        return html.Div([
            html.P("Geographic data not available. Re-run preprocessing with a STATEFIP-enabled IPUMS extract.",
                   style={'color': COLORS['muted_teal'], 'padding': '2rem'})
        ])

    if active_tab == 'tab-geo-scatter':
        return html.Div([
            html.P("Each dot represents one ethnic group in one state. Groups with higher local concentration "
                   "tend to have lower outmarriage rates. The trend line shows the overall relationship.",
                   style={'color': COLORS['muted_teal'], 'fontSize': '0.9rem', 'marginBottom': '1rem'}),
            dcc.Loading(type='circle', color=COLORS['medium_teal'],
                       children=[html.Div(
                           dcc.Graph(id='geo-scatter',
                                     figure=create_geo_scatter_chart(),
                                     config={'displayModeBar': True, 'scrollZoom': False, 'responsive': True}),
                           className='chart-scroll chart-scroll-medium')]),
            _methodology_block('geo-scatter', [
                "Each dot represents one ethnic group in one state. The x-axis shows the group's share of that state's second-generation immigrant population (log scale); the y-axis shows their outmarriage rate.",
                "The trend line is an OLS regression of outmarriage rate on log\u2081\u2080(concentration). R\u00b2 indicates how much of the variation in outmarriage is explained by concentration alone. All census years are pooled for statistical power.",
            ]),
        ], style={'padding': '1rem'})

    elif active_tab == 'tab-geo-state':
        geo_origins = sorted(geo_df['ORIGIN_GROUP'].unique().tolist())
        default_geo = 'Italy' if 'Italy' in geo_origins else geo_origins[0]
        return html.Div([
            html.P("Select an ethnic group to see their outmarriage rate in each state. "
                   "Annotations show the group's local concentration.",
                   style={'color': COLORS['muted_teal'], 'fontSize': '0.9rem', 'marginBottom': '1rem'}),
            html.Div([
                html.Label("Select group:", style={'fontWeight': '500', 'marginRight': '10px',
                                                    'color': COLORS['dark_teal']}),
                dcc.Dropdown(
                    id='geo-group-dropdown',
                    options=[{'label': get_demonym(o) + '-Americans', 'value': o} for o in geo_origins],
                    value=default_geo,
                    style={'maxWidth': '300px', 'width': '100%', 'display': 'inline-block', 'verticalAlign': 'middle'},
                    clearable=False
                ),
            ], style={'marginBottom': '1rem'}),
            dcc.Loading(type='circle', color=COLORS['medium_teal'],
                       children=[html.Div(id='geo-group-chart-container')]),
            _methodology_block('geo-state', [
                "For the selected ethnic group, shows their outmarriage rate in each state where at least 5,000 weighted individuals are present.",
                "The annotation on each bar shows the group's local concentration (share of the state's second-generation population). States where the group was more concentrated tend to show lower outmarriage rates.",
            ]),
        ], style={'padding': '1rem'})

    elif active_tab == 'tab-geo-residuals':
        return html.Div([
            html.P("After accounting for geographic concentration, which groups outmarried more or less than predicted? "
                   "Teal bars indicate groups that outmarried above expected rates; gold bars indicate groups that outmarried below.",
                   style={'color': COLORS['muted_teal'], 'fontSize': '0.9rem', 'marginBottom': '1rem'}),
            dcc.Loading(type='circle', color=COLORS['medium_teal'],
                       children=[html.Div(
                           dcc.Graph(id='geo-residuals', figure=create_geo_residuals_chart(),
                                     config={'displayModeBar': True, 'scrollZoom': False, 'responsive': True}),
                           className='chart-scroll chart-scroll-medium')]),
            _methodology_block('geo-residuals', [
                "For each ethnic group, computes the difference between their actual average outmarriage rate and what the concentration regression predicts.",
                "Positive residuals (teal) indicate groups that outmarried more than their geographic concentration would predict. Negative residuals (gold) indicate groups that outmarried less than predicted. Residuals are weighted by state-level sample size.",
            ]),
        ], style={'padding': '1rem'})

    return html.Div()


@callback(Output('network-chart-container', 'children'),
          [Input('network-view-dropdown', 'value'), Input('year-dropdown', 'value')])
def update_network_chart(view_mode, year):
    """Update the clustering network based on view mode selection."""
    if view_mode == 'adjusted':
        network_fig = create_adjusted_network_chart()
        attraction_fig = create_attraction_chart()
        avoidance_fig = create_avoidance_chart()
        return html.Div([
            html.Div(dcc.Graph(id='heatmap-chart', figure=network_fig,
                     config={'displayModeBar': True, 'scrollZoom': False, 'responsive': True}),
                     className='chart-scroll chart-scroll-medium'),
            dbc.Row([
                dbc.Col(html.Div(dcc.Graph(id='attraction-chart', figure=attraction_fig,
                         config={'displayModeBar': True, 'scrollZoom': False, 'responsive': True}),
                         className='chart-scroll chart-scroll-medium'),
                         md=6),
                dbc.Col(html.Div(dcc.Graph(id='avoidance-chart', figure=avoidance_fig,
                         config={'displayModeBar': True, 'scrollZoom': False, 'responsive': True}),
                         className='chart-scroll chart-scroll-medium'),
                         md=6),
            ], style={'marginTop': '1.5rem'}),
        ])
    else:
        fig = create_heatmap_chart(year)
        return html.Div(dcc.Graph(id='heatmap-chart', figure=fig,
                         config={'displayModeBar': True, 'scrollZoom': False, 'responsive': True}),
                         className='chart-scroll chart-scroll-medium')


@callback(Output('outmarriage-chart-container', 'children'),
          [Input('outmarriage-sort-dropdown', 'value'), Input('year-dropdown', 'value')])
def update_outmarriage_chart(sort_by, year):
    """Update the outmarriage rates chart based on sorting selection."""
    return html.Div(dcc.Graph(id='outmarriage-chart', figure=create_outmarriage_chart(year, sort_by),
                     config={'displayModeBar': True, 'scrollZoom': False, 'responsive': True}), className='chart-scroll chart-scroll-medium')


@callback(Output('geo-group-chart-container', 'children'),
          [Input('geo-group-dropdown', 'value')])
def update_geo_group_chart(origin):
    """Update the single-group geographic chart."""
    if not origin:
        return html.P("Please select a group", style={'color': COLORS['muted_teal']})
    return html.Div(dcc.Graph(id='geo-group-chart', figure=create_geo_group_chart(origin),
                     config={'displayModeBar': True, 'scrollZoom': False, 'responsive': True}), className='chart-scroll chart-scroll-medium')


@callback(Output('single-origin-chart-container', 'children'),
          [Input('single-origin-dropdown', 'value'), Input('year-dropdown', 'value')])
def update_single_origin_chart(origin, year):
    """Update the single origin overview chart."""
    if not origin:
        return html.P("Please select an origin", style={'color': COLORS['muted_teal']})
    return html.Div(dcc.Graph(id='single-origin-chart', figure=create_single_origin_chart(origin, year),
                     config={'displayModeBar': True, 'scrollZoom': False, 'responsive': True}), className='chart-scroll chart-scroll-wide')


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
    agg['DisplayLabel'] = agg['MARRIAGE_TYPE'].map(MARRIAGE_DISPLAY_LABELS).fillna(agg['MARRIAGE_TYPE'])

    fig = go.Figure(go.Bar(
        x=agg['WEIGHTED_COUNT'], y=agg['DisplayLabel'], orientation='h',
        marker_color=agg['Color'], text=agg['Label'], textposition='auto',
        textfont=dict(family='Hanken Grotesk', size=12, color=COLORS['dark_teal']),
        insidetextanchor='end',
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
        xaxis=dict(title_font=dict(family='Hanken Grotesk', size=12), gridcolor=COLORS['light_gray'], fixedrange=True, automargin=True),
        yaxis=dict(tickfont=dict(family='Hanken Grotesk', size=11), fixedrange=True, automargin=True),
        dragmode=False,
        height=max(400, len(agg) * 45 + 120),
        margin=dict(l=10, r=80, t=80, b=60),
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
        xaxis=dict(gridcolor=COLORS['light_gray'], dtick=10, fixedrange=True, automargin=True),
        yaxis=dict(gridcolor=COLORS['light_gray'], range=[0, max(yearly_agg['Percent'].max() * 1.15, 50)], fixedrange=True, automargin=True),
        dragmode=False,
        height=480, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=80, b=60),
        legend=dict(orientation='h', yanchor='top', y=-0.15, xanchor='center', x=0.5)
    )
    return fig


def _compute_geo_adjusted_rates():
    """Compute geography-adjusted outmarriage rates using residuals from concentration regression."""
    geo_df = get_geographic_data()
    if geo_df is None or len(geo_df) == 0:
        return None

    # Only include groups that appear in the main dashboard (valid_origins),
    # excluding US-born. This prevents groups like Cuba/Japan (which pass the
    # lower geographic threshold but not the main 20k threshold) from appearing.
    valid = set(DATA.get('metadata', {}).get('valid_origins', []))
    valid.discard('US-born')
    if valid:
        geo_df = geo_df[geo_df['ORIGIN_GROUP'].isin(valid)]

    x_all = geo_df['GROUP_SHARE_PCT'].values
    y_all = geo_df['OUTMARRIAGE_RATE'].values
    mask = np.isfinite(x_all) & np.isfinite(y_all) & (x_all > 0)

    if mask.sum() < 3:
        return None

    log_x = np.log10(x_all[mask])
    y_fit = y_all[mask]
    coeffs = np.polyfit(log_x, y_fit, 1)
    overall_mean = np.mean(y_fit)

    # Compute per-group adjusted rates
    geo_df = geo_df.copy()
    geo_df['PREDICTED'] = np.polyval(coeffs, np.log10(geo_df['GROUP_SHARE_PCT'].clip(lower=0.001)))
    geo_df['RESIDUAL'] = geo_df['OUTMARRIAGE_RATE'] - geo_df['PREDICTED']

    results = {}
    for origin, grp in geo_df.groupby('ORIGIN_GROUP'):
        avg_residual = np.average(grp['RESIDUAL'], weights=grp['WEIGHTED_N'])
        raw_rate = np.average(grp['OUTMARRIAGE_RATE'], weights=grp['WEIGHTED_N'])
        adjusted_rate = overall_mean + avg_residual
        results[origin] = {
            'adjusted_rate': adjusted_rate,
            'raw_rate': raw_rate,
            'residual': avg_residual,
        }
    return results


def create_outmarriage_chart(year, sort_by='total'):
    """Create horizontal bar chart showing outmarriage rates with different sorting options."""

    # Geography-adjusted view uses different data path
    if sort_by == 'geo_adjusted':
        adjusted = _compute_geo_adjusted_rates()
        if adjusted is None:
            fig = go.Figure()
            fig.add_annotation(text="Geographic data not available for adjustment",
                               x=0.5, y=0.5, showarrow=False)
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400)
            return fig

        items = [{'origin': k, 'demonym': get_demonym(k), **v} for k, v in adjusted.items()]
        items.sort(key=lambda x: x['adjusted_rate'])  # ascending for horizontal bar

        demonyms = [r['demonym'] + '-Americans' for r in items]
        values = [r['adjusted_rate'] for r in items]

        hover_texts = []
        for r in items:
            hover_texts.append(
                f"<b>{r['demonym']}-Americans</b><br>"
                f"Adjusted rate: {r['adjusted_rate']:.1f}%<br>"
                f"Raw rate: {r['raw_rate']:.1f}%<br>"
                f"Residual: {r['residual']:+.1f}pp"
            )

        fig = go.Figure(go.Bar(
            x=values, y=demonyms, orientation='h',
            marker_color=[COLORS['medium_teal'] if r['residual'] >= 0 else COLORS['gold']
                          for r in items],
            text=[f"{v:.0f}%" for v in values],
            textposition='auto',
            insidetextanchor='end',
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover_texts
        ))

        fig.update_layout(
            title=dict(text="Outmarriage Rates: Geography-Adjusted",
                       font=dict(family='Neuton', size=22, color=COLORS['dark_teal']), x=0),
            xaxis_title="Adjusted Outmarriage Rate (%)",
            xaxis=dict(gridcolor=COLORS['light_gray'],
                       range=[0, max(values) * 1.15 if values else 100], fixedrange=True),
            yaxis=dict(gridcolor=COLORS['light_gray'], fixedrange=True, automargin=True),
            dragmode=False,
            height=max(400, len(items) * 28 + 140),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=60, t=80, b=100),
            annotations=[
                dict(text="Removes effect of geographic concentration · "
                          "<b>Teal</b> = outmarried more than concentration predicts · "
                          "<b>Gold</b> = less",
                     xref='paper', yref='paper', x=0.5, y=-0.12, showarrow=False,
                     font=dict(size=11, color=COLORS['muted_teal'], family='Hanken Grotesk')),
            ]
        )
        return fig

    # Standard (non-adjusted) views
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
        title = "Outmarriage Rates: Into Different Recent-Immigrant Communities"
        xaxis_title = "Outmarriage into Different Recent-Immigrant Communities (%)"
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
            f"  Into different recent-immigrant communities: {r['diff_origin_rate']:.1f}%<br>"
            f"Population: {r['population']:,.0f}"
        )

    fig = go.Figure(go.Bar(
        x=values,
        y=demonyms,
        orientation='h',
        marker_color=COLORS['medium_teal'],
        text=[f"{v:.0f}%" for v in values],
        textposition='auto',
        insidetextanchor='end',
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover_texts
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(family='Neuton', size=22, color=COLORS['dark_teal']), x=0),
        xaxis_title=xaxis_title,
        xaxis=dict(gridcolor=COLORS['light_gray'], range=[0, max(values) * 1.15 if values else 100], fixedrange=True),
        yaxis=dict(gridcolor=COLORS['light_gray'], fixedrange=True, automargin=True),
        dragmode=False,
        height=max(400, len(ranking) * 28 + 100),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=60, t=80)
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
        textposition='auto',
        hovertemplate="<b>%{x}</b><br>%{y:.1f}%<br>Count: %{customdata:,.0f}<extra></extra>",
        customdata=counts
    ))

    fig.update_layout(
        title=dict(text="Spouse Generation Distribution", font=dict(family='Neuton', size=22, color=COLORS['dark_teal']), x=0),
        yaxis_title="% of Spouses",
        xaxis=dict(gridcolor=COLORS['light_gray'], fixedrange=True, automargin=True),
        yaxis=dict(gridcolor=COLORS['light_gray'], range=[0, max(percentages) * 1.2 if percentages else 100], fixedrange=True, automargin=True),
        dragmode=False,
        height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=80, b=60))
    return fig


def _draw_network_fig(nodes, edges, positions, title, footer_lines=None):
    """Shared drawing logic for network visualizations."""
    fig = go.Figure()

    # Draw edges with curved paths
    max_weight = max(e['weight'] for e in edges) if edges else 1
    min_weight = min(e['weight'] for e in edges) if edges else 1

    for edge in edges:
        x0, y0 = positions[edge['source']]
        x1, y1 = positions[edge['target']]

        norm_weight = (edge['weight'] - min_weight) / (max_weight - min_weight) if max_weight > min_weight else 0.5
        width = 1 + norm_weight * 5
        opacity = 0.3 + norm_weight * 0.5

        mid_x = (x0 + x1) / 2
        mid_y = (y0 + y1) / 2
        dx, dy = x1 - x0, y1 - y0
        length = np.sqrt(dx*dx + dy*dy)
        if length > 0:
            offset = 0.05 * length
            mid_x += -dy / length * offset
            mid_y += dx / length * offset

        fig.add_trace(go.Scatter(
            x=[x0, mid_x, x1], y=[y0, mid_y, y1],
            mode='lines',
            line=dict(width=width, color=COLORS['light_teal'], shape='spline'),
            opacity=opacity,
            hoverinfo='skip',
            showlegend=False
        ))

    # Draw nodes
    max_pop = max(n['population'] for n in nodes) if nodes else 1
    node_x = [positions[n['id']][0] for n in nodes]
    node_y = [positions[n['id']][1] for n in nodes]
    node_sizes = [10 + 12 * np.sqrt(n['population'] / max_pop) for n in nodes]
    node_labels = [n['label'] for n in nodes]

    hover_texts = []
    for n in nodes:
        conn_list = n.get('connections', [])
        conn_str = '<br>'.join(conn_list) if conn_list else 'No strong connections'
        hover_texts.append(
            f"<b>{n['label']}-Americans</b><br>"
            f"Population: {n['population']:,.0f}<br>"
            f"<br><b>Affinities:</b><br>{conn_str}"
        )

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        marker=dict(
            size=node_sizes,
            color=COLORS['dark_teal'],
            line=dict(width=2, color=COLORS['white']),
            opacity=0.9
        ),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover_texts,
        showlegend=False
    ))

    # Smart label placement
    center_x = np.mean(node_x)
    center_y = np.mean(node_y)

    all_annotations = []
    for i, n in enumerate(nodes):
        x, y = node_x[i], node_y[i]
        dx = x - center_x
        dy = y - center_y
        dist_from_center = np.sqrt(dx*dx + dy*dy)

        if dist_from_center > 0.01:
            dx_norm = dx / dist_from_center
            dy_norm = dy / dist_from_center
        else:
            dx_norm, dy_norm = 0, 1

        label_offset = 0.12
        if abs(dx_norm) > abs(dy_norm):
            xanchor = 'left' if dx_norm > 0 else 'right'
            yanchor = 'middle'
            label_x = x + dx_norm * label_offset
            label_y = y
        else:
            xanchor = 'center'
            yanchor = 'bottom' if dy_norm > 0 else 'top'
            label_x = x
            label_y = y + dy_norm * label_offset

        all_annotations.append(dict(
            x=label_x, y=label_y, text=node_labels[i],
            showarrow=False, xanchor=xanchor, yanchor=yanchor,
            font=dict(family='Hanken Grotesk', size=9, color=COLORS['dark_teal']),
            bgcolor='rgba(255,255,255,0.85)', borderpad=3
        ))

    # Footer lines
    if footer_lines:
        for idx, line in enumerate(footer_lines):
            all_annotations.append(dict(
                text=line, xref='paper', yref='paper',
                x=0.5, y=-0.04 - idx * 0.04,
                showarrow=False, font=dict(size=10, color=COLORS['muted_teal']),
                xanchor='center'
            ))

    fig.update_layout(
        title=dict(text=title, font=dict(family='Neuton', size=22, color=COLORS['dark_teal']), x=0),
        showlegend=False, hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False, fixedrange=True),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False, fixedrange=True),
        dragmode=False, height=550,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=80, b=60 + (len(footer_lines or []) * 18)),
        annotations=all_annotations
    )
    return fig


def create_heatmap_chart(year):
    """Create network graph showing ethnic clustering patterns (national affinities)."""
    try:
        nodes, edges, positions = get_network_data(year)
    except Exception:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient data for clustering analysis", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400)
        return fig

    if not nodes or not edges:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient data for network visualization", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400)
        return fig

    footer = [
        "Lines connect groups with above-average intermarriage rates · Thicker lines = stronger affinity",
        "Note: Affinities partly reflect geographic co-location (groups in the same states marry more)"
    ]
    return _draw_network_fig(nodes, edges, positions,
                             "Intermarriage Affinities Between Ethnic Groups (National)",
                             footer_lines=footer)


def create_adjusted_network_chart():
    """Create network using geography-adjusted affinities (within-state expected rates)."""
    try:
        nodes, edges, positions = get_adjusted_network_data()
    except Exception:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient data for adjusted network", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400)
        return fig

    if not nodes or not edges:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient data for adjusted network", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400)
        return fig

    footer = [
        "Geography-adjusted: expected rates computed within each state, then averaged",
        "Removes the effect of groups sharing the same states · Shows genuine cultural affinity"
    ]
    return _draw_network_fig(nodes, edges, positions,
                             "Intermarriage Affinities (Geography-Adjusted)",
                             footer_lines=footer)


def create_avoidance_chart():
    """Bar chart showing pairs that intermarried LESS than geographic proximity predicts."""
    adj_df = DATA.get('geo_adjusted_affinity')
    if adj_df is None or len(adj_df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No adjusted affinity data available", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400)
        return fig

    # Pairs below 1.0 = married less than expected
    avoided = adj_df[adj_df['GEO_ADJUSTED_AFFINITY'] < 1.0].copy()
    avoided = avoided.sort_values('GEO_ADJUSTED_AFFINITY', ascending=True).head(20)

    if len(avoided) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No avoided pairs found", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400)
        return fig

    # Invert: show how far below expected (1 - affinity), so most avoided = longest bar
    # Sort so most avoided is at top (highest inverted value at top = last in list for horizontal bars)
    avoided = avoided.sort_values('GEO_ADJUSTED_AFFINITY', ascending=False)

    pair_labels = [
        f"{get_demonym(row['SOURCE'])} — {get_demonym(row['TARGET'])}"
        for _, row in avoided.iterrows()
    ]
    raw_values = avoided['GEO_ADJUSTED_AFFINITY'].values
    display_values = 1.0 - raw_values  # e.g. 0.026x -> 0.974 "shortfall"

    # Color gradient: deeper red/orange for stronger avoidance
    colors = [COLORS['medium_orange'] if rv < 0.2 else COLORS['orange'] if rv < 0.5 else COLORS['light_orange']
              for rv in raw_values]

    fig = go.Figure(go.Bar(
        x=display_values,
        y=pair_labels,
        orientation='h',
        marker_color=colors,
        text=[f"{rv:.2f}x" for rv in raw_values],
        textposition='auto',
        insidetextanchor='end',
        textfont=dict(family='Hanken Grotesk', size=11, color=COLORS['dark_teal']),
        hovertemplate=(
            '<b>%{y}</b><br>'
            'Adjusted affinity: %{customdata:.3f}x expected rate<br>'
            '<i>Below 1.0 = married less than proximity predicts</i><extra></extra>'
        ),
        customdata=raw_values
    ))

    fig.update_layout(
        title=dict(text="Most Avoided Pairs (Despite Geographic Proximity)",
                   font=dict(family='Neuton', size=22, color=COLORS['dark_teal']), x=0),
        xaxis_title="Shortfall below expected (longer bar = stronger avoidance)",
        xaxis=dict(gridcolor=COLORS['light_gray'], fixedrange=True, automargin=True,
                   range=[0, 1.1],
                   title_font=dict(family='Hanken Grotesk', size=12),
                   tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                   ticktext=['1.0x<br>(expected)', '0.75x', '0.50x', '0.25x', '0.0x']),
        yaxis=dict(fixedrange=True, automargin=True),
        dragmode=False,
        height=max(400, len(avoided) * 28 + 180),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=60, t=80, b=120),
        annotations=[
            dict(text="Shared the same states but rarely intermarried — cultural boundaries.",
                 xref='paper', yref='paper', x=0.5, y=-0.18, showarrow=False,
                 font=dict(size=11, color=COLORS['muted_teal'], family='Hanken Grotesk')),
        ]
    )
    return fig


def create_attraction_chart():
    """Bar chart showing pairs that intermarried MORE than geographic proximity predicts."""
    adj_df = DATA.get('geo_adjusted_affinity')
    if adj_df is None or len(adj_df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No adjusted affinity data available", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400)
        return fig

    # Match the count used by avoidance chart
    n_avoided = len(adj_df[adj_df['GEO_ADJUSTED_AFFINITY'] < 1.0].head(20))
    attracted = adj_df[adj_df['GEO_ADJUSTED_AFFINITY'] >= 1.0].copy()
    attracted = attracted.sort_values('GEO_ADJUSTED_AFFINITY', ascending=False).head(n_avoided)

    if len(attracted) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No attracted pairs found", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400)
        return fig

    # Reverse so highest is at top
    attracted = attracted.iloc[::-1]

    pair_labels = [
        f"{get_demonym(row['SOURCE'])} — {get_demonym(row['TARGET'])}"
        for _, row in attracted.iterrows()
    ]
    values = attracted['GEO_ADJUSTED_AFFINITY'].values

    colors = [COLORS['dark_teal'] if v >= 3.0 else COLORS['medium_teal'] if v >= 2.0 else COLORS['light_teal']
              for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=pair_labels,
        orientation='h',
        marker_color=colors,
        text=[f"{v:.1f}x" for v in values],
        textposition='auto',
        insidetextanchor='end',
        textfont=dict(family='Hanken Grotesk', size=11, color=COLORS['dark_teal']),
        hovertemplate=(
            '<b>%{y}</b><br>'
            'Adjusted affinity: %{x:.2f}x expected rate<br>'
            '<i>Above 1.0 = married more than proximity predicts</i><extra></extra>'
        )
    ))

    fig.add_vline(x=1.0, line_width=1, line_dash='dash',
                  line_color=COLORS['muted_teal'], opacity=0.6,
                  annotation_text="expected", annotation_position="top left",
                  annotation_font=dict(size=10, color=COLORS['muted_teal']))

    fig.update_layout(
        title=dict(text="Strongest Affinities (Beyond Geographic Proximity)",
                   font=dict(family='Neuton', size=22, color=COLORS['dark_teal']), x=0),
        xaxis_title="Adjusted Affinity (1.0 = expected from local market share)",
        xaxis=dict(gridcolor=COLORS['light_gray'], fixedrange=True, automargin=True,
                   title_font=dict(family='Hanken Grotesk', size=12)),
        yaxis=dict(fixedrange=True, automargin=True),
        dragmode=False,
        height=max(400, len(attracted) * 28 + 180),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=60, t=80, b=120),
        annotations=[
            dict(text="Married more than local shares predict — genuine cultural affinity.",
                 xref='paper', yref='paper', x=0.5, y=-0.18, showarrow=False,
                 font=dict(size=11, color=COLORS['muted_teal'], family='Hanken Grotesk')),
        ]
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
        xaxis=dict(gridcolor=COLORS['light_gray'], range=[0, 105], ticksuffix='%', fixedrange=True),
        yaxis=dict(tickfont=dict(family='Hanken Grotesk', size=11), fixedrange=True, automargin=True),
        barmode='stack', dragmode=False,
        height=max(400, len(df) * 45 + 120),
        margin=dict(l=10, r=60, t=80, b=60),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation='h', yanchor='top', y=-0.15, xanchor='center', x=0.5)
    )
    return fig


@callback(Output('spouse-table-content', 'children'),
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
# GEOGRAPHIC CHART FUNCTIONS
# =============================================================================

# Color palette for ethnic groups in scatter plot
GEO_GROUP_COLORS = [
    COLORS['dark_teal'], COLORS['medium_teal'], COLORS['orange'],
    COLORS['purple'], COLORS['green'], COLORS['gold'],
    COLORS['light_orange'], COLORS['light_purple'], COLORS['light_teal'],
    COLORS['dark_green'], COLORS['medium_orange'], COLORS['dark_purple'],
    COLORS['dark_gold'], COLORS['light_green'], COLORS['muted_teal'],
]


def create_geo_scatter_chart():
    """Scatter plot: group concentration (%) vs outmarriage rate, colored by group."""
    geo_df = get_geographic_data()
    if geo_df is None or len(geo_df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No geographic data available", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400)
        return fig

    origins = sorted(geo_df['ORIGIN_GROUP'].unique())
    color_map = {o: GEO_GROUP_COLORS[i % len(GEO_GROUP_COLORS)] for i, o in enumerate(origins)}

    fig = go.Figure()
    for origin in origins:
        odf = geo_df[geo_df['ORIGIN_GROUP'] == origin]
        fig.add_trace(go.Scatter(
            x=odf['GROUP_SHARE_PCT'],
            y=odf['OUTMARRIAGE_RATE'],
            mode='markers',
            name=get_demonym(origin),
            marker=dict(
                color=color_map[origin],
                size=np.clip(np.sqrt(odf['WEIGHTED_N'] / 1000) * 2, 6, 30),
                opacity=0.75,
                line=dict(width=1, color='white')
            ),
            hovertemplate=(
                '<b>%{customdata[0]}</b> in %{customdata[1]}<br>'
                'Local concentration: %{x:.1f}%<br>'
                'Outmarriage rate: %{y:.1f}%<br>'
                'Sample size: %{customdata[2]:,.0f}<extra></extra>'
            ),
            customdata=list(zip(
                [get_demonym(origin) + '-Americans'] * len(odf),
                odf['STATE_NAME'],
                odf['WEIGHTED_N']
            ))
        ))

    # OLS trend line across all data points
    x_all = geo_df['GROUP_SHARE_PCT'].values
    y_all = geo_df['OUTMARRIAGE_RATE'].values
    mask = np.isfinite(x_all) & np.isfinite(y_all) & (x_all > 0)
    if mask.sum() > 2:
        log_x = np.log10(x_all[mask])
        y_fit = y_all[mask]
        coeffs = np.polyfit(log_x, y_fit, 1)
        # R-squared
        y_pred = np.polyval(coeffs, log_x)
        ss_res = np.sum((y_fit - y_pred) ** 2)
        ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # Draw trend line
        x_line = np.linspace(log_x.min(), log_x.max(), 50)
        y_line = np.polyval(coeffs, x_line)
        fig.add_trace(go.Scatter(
            x=10 ** x_line, y=y_line,
            mode='lines', name=f'Trend (R²={r_squared:.2f})',
            line=dict(color=COLORS['dark_teal'], width=2, dash='dash'),
            hoverinfo='skip'
        ))

        fig.add_annotation(
            text=f"R² = {r_squared:.2f} (all groups)",
            xref='paper', yref='paper', x=0.98, y=0.98,
            showarrow=False, font=dict(size=12, color=COLORS['dark_teal'], family='Hanken Grotesk'),
            bgcolor='rgba(255,255,255,0.9)', bordercolor=COLORS['light_gray'], borderwidth=1,
            align='right'
        )

    fig.update_layout(
        title=dict(text="Concentration vs. Outmarriage",
                   font=dict(family='Neuton', size=20, color=COLORS['dark_teal']), x=0),
        xaxis_title="Group's Share of State's 2nd-Gen Population (%)",
        yaxis_title="Outmarriage Rate (%)",
        xaxis=dict(type='log', gridcolor=COLORS['light_gray'], fixedrange=True, automargin=True,
                   title_font=dict(family='Hanken Grotesk', size=11)),
        yaxis=dict(gridcolor=COLORS['light_gray'], fixedrange=True, automargin=True,
                   title_font=dict(family='Hanken Grotesk', size=11)),
        dragmode=False,
        height=550,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(font=dict(size=9), itemsizing='constant', orientation='h',
                    yanchor='top', y=-0.25, xanchor='center', x=0.5),
        margin=dict(l=10, t=70, r=10, b=120)
    )
    return fig


def create_geo_group_chart(origin):
    """Stacked horizontal bar chart showing outmarriage breakdown by state."""
    odf = get_group_geographic_data(origin)
    if odf is None or len(odf) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data for this group", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400)
        return fig

    odf = odf.copy()
    odf['DIFF_ORIGIN_RATE'] = odf['OUTMARRIAGE_RATE'] - odf['THIRD_GEN_RATE']
    odf = odf.sort_values('OUTMARRIAGE_RATE', ascending=True)
    demonym = get_demonym(origin)

    fig = go.Figure()

    # 3rd+ gen American component
    fig.add_trace(go.Bar(
        x=odf['THIRD_GEN_RATE'],
        y=odf['STATE_NAME'],
        orientation='h',
        name='3rd+ Gen American',
        marker_color=COLORS['light_teal'],
        text=[f"{r:.0f}%" if r >= 5 else "" for r in odf['THIRD_GEN_RATE']],
        textposition='inside',
        textfont=dict(family='Hanken Grotesk', size=10, color=COLORS['dark_teal']),
        hovertemplate=(
            '<b>%{y}</b><br>'
            '3rd+ gen American: %{x:.1f}%<br>'
            'Local concentration: %{customdata[0]:.1f}%<br>'
            'Sample size: %{customdata[1]:,.0f}<extra></extra>'
        ),
        customdata=list(zip(odf['GROUP_SHARE_PCT'], odf['WEIGHTED_N']))
    ))

    # Different immigrant community component
    fig.add_trace(go.Bar(
        x=odf['DIFF_ORIGIN_RATE'],
        y=odf['STATE_NAME'],
        orientation='h',
        name='Different Immigrant',
        marker_color=COLORS['medium_teal'],
        text=[f"{r:.0f}%" if r >= 5 else "" for r in odf['DIFF_ORIGIN_RATE']],
        textposition='inside',
        textfont=dict(family='Hanken Grotesk', size=10, color='white'),
        hovertemplate=(
            '<b>%{y}</b><br>'
            'Different immigrant: %{x:.1f}%<br>'
            'Local concentration: %{customdata[0]:.1f}%<br>'
            'Sample size: %{customdata[1]:,.0f}<extra></extra>'
        ),
        customdata=list(zip(odf['GROUP_SHARE_PCT'], odf['WEIGHTED_N']))
    ))

    # Add total outmarriage rate and concentration outside bars
    for i, (_, row) in enumerate(odf.iterrows()):
        fig.add_annotation(
            x=row['OUTMARRIAGE_RATE'] + 1.5,
            y=row['STATE_NAME'],
            text=f"{row['OUTMARRIAGE_RATE']:.0f}% ({row['GROUP_SHARE_PCT']:.1f}%)",
            showarrow=False, xanchor='left',
            font=dict(family='Hanken Grotesk', size=10, color=COLORS['dark_teal'])
        )

    fig.update_layout(
        barmode='stack',
        title=dict(text=f"{demonym}-Americans: Outmarriage by State",
                   font=dict(family='Neuton', size=20, color=COLORS['dark_teal']), x=0),
        xaxis_title="Outmarriage Rate (%) — values outside bars: total (concentration)",
        xaxis=dict(gridcolor=COLORS['light_gray'], fixedrange=True,
                   range=[0, min(odf['OUTMARRIAGE_RATE'].max() * 1.3, 110)],
                   ticksuffix='%',
                   title_font=dict(family='Hanken Grotesk', size=10)),
        yaxis=dict(fixedrange=True, automargin=True),
        dragmode=False,
        height=max(400, len(odf) * 28 + 140),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=90, t=70, b=80),
        legend=dict(orientation='h', yanchor='top', y=-0.15, xanchor='center', x=0.5)
    )
    return fig


def create_geo_residuals_chart():
    """Bar chart showing average residuals: actual outmarriage - predicted by concentration."""
    geo_df = get_geographic_data()
    if geo_df is None or len(geo_df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No geographic data available", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400)
        return fig

    # Fit OLS: outmarriage ~ log(concentration) across all group-state pairs
    x_all = geo_df['GROUP_SHARE_PCT'].values
    y_all = geo_df['OUTMARRIAGE_RATE'].values
    mask = np.isfinite(x_all) & np.isfinite(y_all) & (x_all > 0)

    if mask.sum() < 3:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient data for regression", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400)
        return fig

    log_x = np.log10(x_all[mask])
    y_fit = y_all[mask]
    coeffs = np.polyfit(log_x, y_fit, 1)

    # Compute residuals for each row
    geo_df = geo_df.copy()
    geo_df['PREDICTED'] = np.polyval(coeffs, np.log10(geo_df['GROUP_SHARE_PCT'].clip(lower=0.001)))
    geo_df['RESIDUAL'] = geo_df['OUTMARRIAGE_RATE'] - geo_df['PREDICTED']

    # Average residual per group (weighted by sample size)
    group_resid = geo_df.groupby('ORIGIN_GROUP').apply(
        lambda g: np.average(g['RESIDUAL'], weights=g['WEIGHTED_N'])
    ).reset_index()
    group_resid.columns = ['ORIGIN_GROUP', 'AVG_RESIDUAL']
    group_resid = group_resid.sort_values('AVG_RESIDUAL', ascending=True)

    colors = [COLORS['medium_teal'] if r >= 0 else COLORS['gold']
              for r in group_resid['AVG_RESIDUAL']]
    demonyms = [get_demonym(o) + '-Americans' for o in group_resid['ORIGIN_GROUP']]

    fig = go.Figure(go.Bar(
        x=group_resid['AVG_RESIDUAL'],
        y=demonyms,
        orientation='h',
        marker_color=colors,
        text=[f"{r:+.1f}pp" for r in group_resid['AVG_RESIDUAL']],
        textposition='auto',
        insidetextanchor='end',
        textfont=dict(family='Hanken Grotesk', size=11, color=COLORS['dark_teal']),
        hovertemplate=(
            '<b>%{y}</b><br>'
            'Avg residual: %{x:+.1f} percentage points<extra></extra>'
        )
    ))

    # Add a vertical line at 0
    fig.add_vline(x=0, line_width=1, line_color=COLORS['dark_teal'], opacity=0.5)

    fig.update_layout(
        title=dict(text="Beyond Geography: Above or Below Predicted",
                   font=dict(family='Neuton', size=20, color=COLORS['dark_teal']), x=0),
        xaxis=dict(gridcolor=COLORS['light_gray'], fixedrange=True, automargin=True,
                   title=dict(text="Percentage points above/below predicted",
                              font=dict(family='Hanken Grotesk', size=11))),
        yaxis=dict(fixedrange=True, automargin=True),
        dragmode=False,
        height=max(400, len(group_resid) * 28 + 120),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=20, t=70, b=90),
        annotations=[
            dict(text="<b>Teal</b> = more than predicted \u00a0|\u00a0 <b>Gold</b> = less than predicted",
                 xref='paper', yref='paper', x=0.5, y=-0.08, showarrow=False, xanchor='center',
                 font=dict(size=10, color=COLORS['muted_teal'], family='Hanken Grotesk')),
            dict(text="All groups pooled across census years \u2014 independent of filter selections.",
                 xref='paper', yref='paper', x=0.5, y=-0.13, showarrow=False, xanchor='center',
                 font=dict(size=10, color=COLORS['muted_teal'], family='Hanken Grotesk')),
        ]
    )
    return fig


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

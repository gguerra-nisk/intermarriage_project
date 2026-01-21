"""
Immigrant Intermarriage Dashboard v2.0
======================================
With Sankey diagrams showing who marries whom!
"""

import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path

# =============================================================================
# LOAD DATA
# =============================================================================
print("Loading data...")
DATA_DIR = Path("data/processed")

if not (DATA_DIR / "intermarriage_by_year.csv").exists():
    print("\n❌ ERROR: Run 'python process_ipums_v2.py' first.")
    exit(1)

df_year = pd.read_csv(DATA_DIR / "intermarriage_by_year.csv")
df_year_region = pd.read_csv(DATA_DIR / "intermarriage_by_year_region.csv")
df_year_country = pd.read_csv(DATA_DIR / "intermarriage_by_year_country.csv")

# Spouse pairs for Sankey
has_sankey = (DATA_DIR / "spouse_pairs.csv").exists()
if has_sankey:
    df_pairs = pd.read_csv(DATA_DIR / "spouse_pairs.csv")
    print(f"✓ Spouse pairs: {len(df_pairs):,}")
else:
    df_pairs = pd.DataFrame()
    print("⚠️  No spouse_pairs.csv")

# Lists
years = sorted([int(y) for y in df_year['YEAR'].unique()])
regions = sorted([r for r in df_year_region['REGION'].unique() 
                  if r not in ['United States', 'Unknown', 'US Territory']])
countries = sorted([c for c in df_year_country['COUNTRY'].unique() 
                    if 'Code' not in str(c) and c not in ['United States', 'Unknown']])

sankey_countries = []
if has_sankey and len(df_pairs) > 0:
    sankey_countries = sorted([c for c in df_pairs['ORIGIN'].unique() 
                               if 'Code' not in str(c) and c not in ['United States', 'Unknown', 'US-Born']])

print(f"✓ {len(years)} years, {len(countries)} countries, {len(sankey_countries)} Sankey")

# =============================================================================
# COLORS
# =============================================================================
C = {
    'us': '#E74C3C', 'same': '#3498DB', 
    'region': '#2ECC71', 'diff': '#9B59B6', 'accent': '#EC4899'
}

# Sankey colors - solid for nodes, rgba for links
SANKEY_NODE = ['#E74C3C', '#3498DB', '#2ECC71', '#9B59B6', '#F39C12',
               '#1ABC9C', '#E91E63', '#00BCD4', '#FF5722', '#607D8B',
               '#8BC34A', '#FF9800', '#795548', '#9C27B0', '#03A9F4']

# Corresponding rgba with transparency for links
SANKEY_LINK = ['rgba(231,76,60,0.5)', 'rgba(52,152,219,0.5)', 'rgba(46,204,113,0.5)',
               'rgba(155,89,182,0.5)', 'rgba(243,156,18,0.5)', 'rgba(26,188,156,0.5)',
               'rgba(233,30,99,0.5)', 'rgba(0,188,212,0.5)', 'rgba(255,87,34,0.5)',
               'rgba(96,125,139,0.5)', 'rgba(139,195,74,0.5)', 'rgba(255,152,0,0.5)',
               'rgba(121,85,72,0.5)', 'rgba(156,39,176,0.5)', 'rgba(3,169,244,0.5)']

# =============================================================================
# APP
# =============================================================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
                title="💕 Intermarriage Dashboard")
server = app.server

# =============================================================================
# LAYOUT
# =============================================================================
app.layout = dbc.Container([
    html.H2("💕 Immigrant Intermarriage in America", className="text-center mt-3 mb-3"),
    
    dbc.Tabs([
        dbc.Tab(label="📊 Overview", tab_id="t1"),
        dbc.Tab(label="🔀 Who Marries Whom", tab_id="t2"),
    ], id="tabs", active_tab="t1", className="mb-3"),
    
    # === OVERVIEW TAB ===
    html.Div(id="div1", children=[
        dbc.Card([dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("View:"),
                    dbc.RadioItems(id='view', value='all', inline=True,
                                   options=[{'label': 'All', 'value': 'all'},
                                            {'label': 'Region', 'value': 'region'},
                                            {'label': 'Country', 'value': 'country'}])
                ], md=3),
                dbc.Col([
                    html.Label("Origin:"),
                    dcc.Dropdown(id='origin', placeholder="Select...")
                ], md=4),
                dbc.Col([
                    html.Label("Years:"),
                    dcc.RangeSlider(id='yrs', min=min(years), max=max(years),
                                    value=[min(years), max(years)],
                                    marks={y: str(y) for y in years[::3]},
                                    step=None, allowCross=False)
                ], md=5)
            ])
        ])], className="mb-3"),
        dbc.Row([
            dbc.Col([dbc.Card([
                dbc.CardHeader("Trends"),
                dbc.CardBody([dcc.Graph(id='g1', style={'height': '350px'})])
            ])], lg=8),
            dbc.Col([dbc.Card([
                dbc.CardHeader(id='pie-h'),
                dbc.CardBody([dcc.Graph(id='g2', style={'height': '350px'})])
            ])], lg=4),
        ], className="mb-3"),
        dbc.Card([
            dbc.CardHeader(id='bar-h'),
            dbc.CardBody([dcc.Graph(id='g3', style={'height': '380px'})])
        ], className="mb-3"),
        dbc.Row([
            dbc.Col([dbc.Card([dbc.CardBody([
                html.H5(id='s1', className="text-center"), html.Small("Marriages")
            ])], color="light")], md=3),
            dbc.Col([dbc.Card([dbc.CardBody([
                html.H5(id='s2', className="text-center", style={'color': C['us']}), html.Small("% US-Born")
            ])], color="light")], md=3),
            dbc.Col([dbc.Card([dbc.CardBody([
                html.H5(id='s3', className="text-center", style={'color': C['same']}), html.Small("% Same")
            ])], color="light")], md=3),
            dbc.Col([dbc.Card([dbc.CardBody([
                html.H5(id='s4', className="text-center"), html.Small("Years")
            ])], color="light")], md=3),
        ])
    ]),
    
    # === SANKEY TAB ===
    html.Div(id="div2", style={'display': 'none'}, children=[
        dbc.Card([dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Country:"),
                    dcc.Dropdown(id='sk-origin', clearable=False,
                                 options=[{'label': c, 'value': c} for c in sankey_countries],
                                 value=sankey_countries[0] if sankey_countries else None)
                ], md=4),
                dbc.Col([
                    html.Label("Year:"),
                    dcc.Dropdown(id='sk-year', clearable=False,
                                 options=[{'label': str(y), 'value': y} for y in years],
                                 value=years[-1] if years else None)
                ], md=3),
                dbc.Col([
                    html.Label("Top N:"),
                    dcc.Slider(id='sk-n', min=5, max=15, value=10, step=1,
                               marks={5: '5', 10: '10', 15: '15'})
                ], md=5)
            ])
        ])], className="mb-3"),
        dbc.Card([
            dbc.CardHeader(id='sk-title'),
            dbc.CardBody([dcc.Graph(id='sk-chart', style={'height': '500px'})])
        ], className="mb-3"),
        dbc.Alert("Width = proportion of marriages. 'US-Born' = spouse born in USA.", color="info"),
        dbc.Card([
            dbc.CardHeader("Breakdown"),
            dbc.CardBody([html.Div(id='sk-table')])
        ])
    ]),
    
    html.Hr(), html.P("Data: IPUMS USA", className="text-center text-muted small")
], fluid=True)

# =============================================================================
# CALLBACKS
# =============================================================================

@callback(Output('div1', 'style'), Output('div2', 'style'), Input('tabs', 'active_tab'))
def switch_tab(t):
    if t == 't1':
        return {'display': 'block'}, {'display': 'none'}
    return {'display': 'none'}, {'display': 'block'}

@callback(Output('origin', 'options'), Output('origin', 'value'), 
          Output('origin', 'disabled'), Input('view', 'value'))
def update_dd(v):
    if v == 'region':
        return [{'label': r, 'value': r} for r in regions], regions[0] if regions else None, False
    if v == 'country':
        return [{'label': c, 'value': c} for c in countries], countries[0] if countries else None, False
    return [], None, True

@callback(
    Output('g1', 'figure'), Output('g2', 'figure'), Output('pie-h', 'children'),
    Output('g3', 'figure'), Output('bar-h', 'children'),
    Output('s1', 'children'), Output('s2', 'children'), Output('s3', 'children'), Output('s4', 'children'),
    Input('view', 'value'), Input('origin', 'value'), Input('yrs', 'value'))
def update_overview(view, origin, yr):
    y1, y2 = yr
    if view == 'region' and origin:
        df = df_year_region[(df_year_region['REGION'] == origin) & 
                            (df_year_region['YEAR'] >= y1) & (df_year_region['YEAR'] <= y2)]
    elif view == 'country' and origin:
        df = df_year_country[(df_year_country['COUNTRY'] == origin) & 
                              (df_year_country['YEAR'] >= y1) & (df_year_country['YEAR'] <= y2)]
    else:
        df = df_year[(df_year['YEAR'] >= y1) & (df_year['YEAR'] <= y2)]
    
    # Line chart
    f1 = go.Figure()
    if len(df) > 0:
        for col, nm, clr in [('pct_married_us_born', 'US-Born', C['us']),
                              ('pct_same_country', 'Same', C['same']),
                              ('pct_same_region', 'Region', C['region']),
                              ('pct_different_region', 'Diff', C['diff'])]:
            f1.add_trace(go.Scatter(x=df['YEAR'], y=df[col], name=nm, mode='lines+markers',
                                    line=dict(color=clr, width=2)))
    f1.update_layout(yaxis=dict(range=[0, 100], ticksuffix='%'), 
                     legend=dict(orientation='h', y=1.1), margin=dict(t=30), plot_bgcolor='white')
    
    # Pie
    if len(df) > 0:
        lyr = int(df['YEAR'].max())
        r = df[df['YEAR'] == lyr].iloc[0]
        vals = [r['pct_married_us_born'], r['pct_same_country'], r['pct_same_region'], r['pct_different_region']]
    else:
        lyr, vals = y2, [25]*4
    f2 = go.Figure(go.Pie(values=vals, labels=['US', 'Same', 'Region', 'Diff'],
                          marker_colors=[C['us'], C['same'], C['region'], C['diff']], hole=0.4))
    f2.update_layout(margin=dict(t=10, b=10), legend=dict(orientation='h', y=-0.1))
    
    # Bar
    my = int(df_year_country['YEAR'].max())
    cdf = df_year_country[(df_year_country['YEAR'] == my) & 
                          (~df_year_country['COUNTRY'].str.contains('Code|Unknown', na=False)) &
                          (df_year_country['n_unweighted'] >= 100)]
    cdf = cdf.nlargest(12, 'n_unweighted').sort_values('pct_married_us_born')
    f3 = go.Figure()
    for col, nm, clr in [('pct_married_us_born', 'US', C['us']), ('pct_same_country', 'Same', C['same']),
                          ('pct_same_region', 'Reg', C['region']), ('pct_different_region', 'Diff', C['diff'])]:
        f3.add_trace(go.Bar(y=cdf['COUNTRY'], x=cdf[col], name=nm, orientation='h', marker_color=clr))
    f3.update_layout(barmode='stack', xaxis=dict(range=[0, 100], ticksuffix='%'),
                     legend=dict(orientation='h', y=1.05), margin=dict(l=100, t=30), plot_bgcolor='white')
    
    # Stats
    if len(df) > 0:
        s1 = f"{int(df['n_unweighted'].sum()):,}"
        lt = df[df['YEAR'] == df['YEAR'].max()].iloc[0]
        s2 = f"{lt['pct_married_us_born']:.1f}%"
        s3 = f"{lt['pct_same_country']:.1f}%"
        s4 = str(len(df))
    else:
        s1, s2, s3, s4 = "0", "-", "-", "0"
    
    return f1, f2, f"In {lyr}", f3, f"Top Origins ({my})", s1, s2, s3, s4

@callback(
    Output('sk-chart', 'figure'), Output('sk-title', 'children'), Output('sk-table', 'children'),
    Input('sk-origin', 'value'), Input('sk-year', 'value'), Input('sk-n', 'value'),
    prevent_initial_call=True)
def update_sankey(origin, year, n):
    # Return empty if no data
    empty = go.Figure()
    empty.add_annotation(text="Select country and year", showarrow=False, font_size=14)
    
    if not has_sankey or origin is None or year is None or len(df_pairs) == 0:
        return empty, "No Data", html.P("No data available")
    
    try:
        data = df_pairs[(df_pairs['ORIGIN'] == origin) & (df_pairs['YEAR'] == year)]
        if len(data) == 0:
            empty.update_layout(annotations=[dict(text=f"No data for {origin} in {year}", showarrow=False)])
            return empty, f"{origin} ({year})", html.P("No data for selection")
        
        data = data.sort_values('PERCENTAGE', ascending=False).head(n)
        spouses = data['SPOUSE'].tolist()
        labels = [origin] + spouses
        
        src, tgt, val, node_colors, link_colors = [], [], [], [C['accent']], []
        for i, row in enumerate(data.itertuples()):
            src.append(0)
            tgt.append(i + 1)
            val.append(row.PERCENTAGE)
            node_colors.append(SANKEY_NODE[i % len(SANKEY_NODE)])
            link_colors.append(SANKEY_LINK[i % len(SANKEY_LINK)])
        
        fig = go.Figure(go.Sankey(
            node=dict(pad=15, thickness=20, label=labels, color=node_colors),
            link=dict(source=src, target=tgt, value=val, color=link_colors)
        ))
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10))
        
        tbl = data[['SPOUSE', 'PERCENTAGE', 'UNWEIGHTED_COUNT']].copy()
        tbl.columns = ['Spouse', '%', 'N']
        tbl['%'] = tbl['%'].apply(lambda x: f"{x:.1f}%")
        tbl['N'] = tbl['N'].apply(lambda x: f"{int(x):,}")
        table = dbc.Table.from_dataframe(tbl, striped=True, bordered=True, size='sm')
        
        return fig, f"{origin} → Spouses ({year})", table
    
    except Exception as e:
        print(f"Sankey error: {e}")
        return empty, "Error", html.P(f"Error: {str(e)}")

# =============================================================================
# RUN
# =============================================================================
if __name__ == '__main__':
    print("\n" + "="*40)
    print("💕 DASHBOARD v2.0")
    print("Open: http://127.0.0.1:8050")
    print("="*40 + "\n")
    app.run(debug=False, host='127.0.0.1', port=8050)

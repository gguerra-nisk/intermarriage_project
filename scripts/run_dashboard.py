"""
Immigrant Intermarriage Interactive Dashboard
==============================================
A Valentine's Day exploration of love across borders!

Usage: python scripts/run_dashboard.py
Then open: http://127.0.0.1:8050

Press Ctrl+C to stop the server.
"""

import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path

# =============================================================================
# LOAD DATA
# =============================================================================
print("Loading data...")
DATA_DIR = Path("data/processed")

# Check if data exists
if not (DATA_DIR / "intermarriage_by_year.csv").exists():
    print("\n❌ ERROR: Processed data not found!")
    print("   Run 'python scripts/process_ipums.py' first.")
    exit(1)

# Load aggregated statistics
df_year = pd.read_csv(DATA_DIR / "intermarriage_by_year.csv")
df_year_region = pd.read_csv(DATA_DIR / "intermarriage_by_year_region.csv")
df_year_country = pd.read_csv(DATA_DIR / "intermarriage_by_year_country.csv")

# Get unique values for filters - CONVERT TO PYTHON INTS
years = [int(y) for y in sorted(df_year['YEAR'].unique())]
regions = sorted([r for r in df_year_region['REGION'].unique() if r not in ['United States', 'Unknown', 'US Territory']])
countries = sorted([c for c in df_year_country['COUNTRY'].unique() if 'Code' not in str(c) and c not in ['United States', 'Unknown']])

print(f"✓ Loaded data for {len(years)} census years")
print(f"✓ {len(regions)} origin regions")
print(f"✓ {len(countries)} origin countries")

# =============================================================================
# COLOR SCHEME (Valentine's Day themed!)
# =============================================================================
COLORS = {
    'married_us': '#E74C3C',      # Red - married US-born
    'same_country': '#3498DB',     # Blue - same country
    'same_region': '#2ECC71',      # Green - same region  
    'different_region': '#9B59B6', # Purple - different region
    'background': '#FDF2F8',       # Light pink background
    'text': '#1F2937',             # Dark gray text
    'accent': '#EC4899',           # Pink accent
}

# =============================================================================
# INITIALIZE DASH APP
# =============================================================================
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="💕 Immigrant Intermarriage in America",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
)

server = app.server  # For deployment

# Create marks for slider - ensure all keys are Python ints
year_marks = {int(y): {'label': str(y), 'style': {'fontSize': '10px'}} for y in years[::3]}

# =============================================================================
# LAYOUT
# =============================================================================
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1([
                "💕 Immigrant Intermarriage in America ",
                html.Span("1850-2023", style={'color': COLORS['accent'], 'fontSize': '0.7em'})
            ], className="text-center mt-4 mb-2"),
            html.P(
                "Explore 170+ years of love stories across borders — who married whom, and how it's changed over time",
                className="text-center text-muted mb-4",
                style={'fontSize': '1.1em'}
            )
        ])
    ]),
    
    # Control Panel
    dbc.Card([
        dbc.CardBody([
            dbc.Row([
                # View selector
                dbc.Col([
                    html.Label("View by:", className="fw-bold"),
                    dbc.RadioItems(
                        id='view-type',
                        options=[
                            {'label': ' All Immigrants', 'value': 'all'},
                            {'label': ' By Region', 'value': 'region'},
                            {'label': ' By Country', 'value': 'country'},
                        ],
                        value='all',
                        inline=True
                    )
                ], md=4),
                
                # Region/Country dropdown (conditional)
                dbc.Col([
                    html.Label("Select Origin:", className="fw-bold"),
                    dcc.Dropdown(
                        id='origin-dropdown',
                        placeholder="Select a region or country...",
                        clearable=True
                    )
                ], md=4),
                
                # Year range
                dbc.Col([
                    html.Label("Year Range:", className="fw-bold"),
                    dcc.RangeSlider(
                        id='year-slider',
                        min=int(min(years)),
                        max=int(max(years)),
                        value=[int(min(years)), int(max(years))],
                        marks=year_marks,
                        step=None,
                        allowCross=False,
                        tooltip={"placement": "bottom", "always_visible": False}
                    )
                ], md=4)
            ])
        ])
    ], className="mb-4", style={'backgroundColor': '#f8f9fa'}),
    
    # Main visualization row
    dbc.Row([
        # Time series (left, larger)
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H5("Marriage Patterns Over Time", className="mb-0")
                ]),
                dbc.CardBody([
                    dcc.Graph(id='time-series-chart', style={'height': '400px'})
                ])
            ])
        ], lg=8, md=12, className="mb-4"),
        
        # Pie chart (right, smaller)
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H5(id='pie-title', className="mb-0")
                ]),
                dbc.CardBody([
                    dcc.Graph(id='pie-chart', style={'height': '400px'})
                ])
            ])
        ], lg=4, md=12, className="mb-4")
    ]),
    
    # Comparison row
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H5(id='comparison-title', className="mb-0")
                ]),
                dbc.CardBody([
                    dcc.Graph(id='comparison-chart', style={'height': '450px'})
                ])
            ])
        ], md=12, className="mb-4")
    ]),
    
    # Stats cards
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3(id='stat-total', className="text-center mb-0", style={'color': COLORS['accent']}),
                    html.P("Immigrant Marriages Analyzed", className="text-center text-muted mb-0")
                ])
            ], style={'backgroundColor': COLORS['background']})
        ], md=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3(id='stat-intermarriage', className="text-center mb-0", style={'color': COLORS['married_us']}),
                    html.P("Married US-Born (Latest)", className="text-center text-muted mb-0")
                ])
            ], style={'backgroundColor': COLORS['background']})
        ], md=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3(id='stat-endogamy', className="text-center mb-0", style={'color': COLORS['same_country']}),
                    html.P("Same Country (Latest)", className="text-center text-muted mb-0")
                ])
            ], style={'backgroundColor': COLORS['background']})
        ], md=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3(id='stat-years', className="text-center mb-0", style={'color': COLORS['different_region']}),
                    html.P("Years of Data", className="text-center text-muted mb-0")
                ])
            ], style={'backgroundColor': COLORS['background']})
        ], md=3),
    ], className="mb-4"),
    
    # Data table
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H5("Detailed Statistics", className="mb-0")
                ]),
                dbc.CardBody([
                    html.Div(id='stats-table', style={'overflowX': 'auto'})
                ])
            ])
        ])
    ], className="mb-4"),
    
    # Footer
    dbc.Row([
        dbc.Col([
            html.Hr(),
            html.P([
                "Data: IPUMS USA, University of Minnesota (",
                html.A("ipums.org", href="https://usa.ipums.org", target="_blank"),
                ") • Built with 💕 for Valentine's Day 2025"
            ], className="text-center text-muted small")
        ])
    ])
    
], fluid=True, style={'backgroundColor': '#ffffff', 'minHeight': '100vh'})


# =============================================================================
# CALLBACKS
# =============================================================================

@callback(
    Output('origin-dropdown', 'options'),
    Output('origin-dropdown', 'value'),
    Output('origin-dropdown', 'disabled'),
    Input('view-type', 'value')
)
def update_dropdown_options(view_type):
    """Update dropdown based on view type selection."""
    if view_type == 'all':
        return [], None, True
    elif view_type == 'region':
        options = [{'label': r, 'value': r} for r in regions]
        return options, regions[0] if regions else None, False
    else:  # country
        options = [{'label': c, 'value': c} for c in countries]
        return options, countries[0] if countries else None, False


@callback(
    Output('time-series-chart', 'figure'),
    Output('pie-chart', 'figure'),
    Output('pie-title', 'children'),
    Output('comparison-chart', 'figure'),
    Output('comparison-title', 'children'),
    Output('stats-table', 'children'),
    Output('stat-total', 'children'),
    Output('stat-intermarriage', 'children'),
    Output('stat-endogamy', 'children'),
    Output('stat-years', 'children'),
    Input('view-type', 'value'),
    Input('origin-dropdown', 'value'),
    Input('year-slider', 'value')
)
def update_all_charts(view_type, origin, year_range):
    """Master callback to update all visualizations."""
    
    year_min, year_max = year_range
    
    # Select appropriate dataframe based on view
    if view_type == 'all':
        df_filtered = df_year[
            (df_year['YEAR'] >= year_min) & 
            (df_year['YEAR'] <= year_max)
        ].copy()
        title_suffix = "All Immigrants"
    elif view_type == 'region' and origin:
        df_filtered = df_year_region[
            (df_year_region['REGION'] == origin) &
            (df_year_region['YEAR'] >= year_min) & 
            (df_year_region['YEAR'] <= year_max)
        ].copy()
        title_suffix = f"Immigrants from {origin}"
    elif view_type == 'country' and origin:
        df_filtered = df_year_country[
            (df_year_country['COUNTRY'] == origin) &
            (df_year_country['YEAR'] >= year_min) & 
            (df_year_country['YEAR'] <= year_max)
        ].copy()
        title_suffix = f"Immigrants from {origin}"
    else:
        df_filtered = df_year[
            (df_year['YEAR'] >= year_min) & 
            (df_year['YEAR'] <= year_max)
        ].copy()
        title_suffix = "All Immigrants"
    
    # =========================================================================
    # TIME SERIES CHART
    # =========================================================================
    fig_time = go.Figure()
    
    if len(df_filtered) > 0:
        # Add traces for each marriage type
        fig_time.add_trace(go.Scatter(
            x=df_filtered['YEAR'],
            y=df_filtered['pct_married_us_born'],
            name='Married US-Born',
            mode='lines+markers',
            line=dict(color=COLORS['married_us'], width=3),
            marker=dict(size=8),
            hovertemplate='%{y:.1f}%<extra>Married US-Born</extra>'
        ))
        
        fig_time.add_trace(go.Scatter(
            x=df_filtered['YEAR'],
            y=df_filtered['pct_same_country'],
            name='Same Country',
            mode='lines+markers',
            line=dict(color=COLORS['same_country'], width=3),
            marker=dict(size=8),
            hovertemplate='%{y:.1f}%<extra>Same Country</extra>'
        ))
        
        fig_time.add_trace(go.Scatter(
            x=df_filtered['YEAR'],
            y=df_filtered['pct_same_region'],
            name='Same Region',
            mode='lines+markers',
            line=dict(color=COLORS['same_region'], width=3),
            marker=dict(size=8),
            hovertemplate='%{y:.1f}%<extra>Same Region</extra>'
        ))
        
        fig_time.add_trace(go.Scatter(
            x=df_filtered['YEAR'],
            y=df_filtered['pct_different_region'],
            name='Different Region',
            mode='lines+markers',
            line=dict(color=COLORS['different_region'], width=3),
            marker=dict(size=8),
            hovertemplate='%{y:.1f}%<extra>Different Region</extra>'
        ))
    
    fig_time.update_layout(
        title=dict(text=f"<b>{title_suffix}</b>", x=0.5),
        xaxis_title="Census Year",
        yaxis_title="Percentage of Married Immigrants",
        yaxis=dict(range=[0, 100], ticksuffix='%'),
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        margin=dict(t=80, b=50),
        plot_bgcolor='white'
    )
    fig_time.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
    fig_time.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
    
    # =========================================================================
    # PIE CHART (Latest Year)
    # =========================================================================
    if len(df_filtered) > 0:
        latest_year = int(df_filtered['YEAR'].max())
        latest_data = df_filtered[df_filtered['YEAR'] == latest_year].iloc[0]
        
        pie_values = [
            latest_data['pct_married_us_born'],
            latest_data['pct_same_country'],
            latest_data['pct_same_region'],
            latest_data['pct_different_region']
        ]
        pie_labels = ['Married US-Born', 'Same Country', 'Same Region', 'Different Region']
        pie_colors = [COLORS['married_us'], COLORS['same_country'], 
                      COLORS['same_region'], COLORS['different_region']]
    else:
        latest_year = year_max
        pie_values = [25, 25, 25, 25]
        pie_labels = ['No Data'] * 4
        pie_colors = ['#cccccc'] * 4
    
    fig_pie = go.Figure(data=[go.Pie(
        labels=pie_labels,
        values=pie_values,
        marker_colors=pie_colors,
        hole=0.4,
        textinfo='percent',
        textfont_size=12,
        hovertemplate='%{label}<br>%{percent}<extra></extra>'
    )])
    
    fig_pie.update_layout(
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.15,
            xanchor='center',
            x=0.5,
            font=dict(size=10)
        ),
        margin=dict(t=30, b=60, l=20, r=20)
    )
    
    pie_title = f"Distribution in {latest_year}"
    
    # =========================================================================
    # COMPARISON CHART (Stacked Bar by Region or Top Countries)
    # =========================================================================
    if view_type == 'all':
        # Show all regions for the latest year
        max_year = int(df_year_region['YEAR'].max())
        comp_data = df_year_region[
            (df_year_region['YEAR'] == max_year) &
            (~df_year_region['REGION'].isin(['United States', 'Unknown', 'US Territory', 'Other/Unknown']))
        ].copy()
        comp_data = comp_data[comp_data['n_unweighted'] >= 100]  # Minimum sample
        comp_data = comp_data.sort_values('pct_married_us_born', ascending=True)
        x_col = 'REGION'
        comparison_title = f"Intermarriage by Origin Region ({max_year})"
    else:
        # Show top origins for comparison
        if view_type == 'region':
            max_year = int(df_year_region['YEAR'].max())
            comp_data = df_year_region[
                (df_year_region['YEAR'] == max_year) &
                (~df_year_region['REGION'].isin(['United States', 'Unknown', 'US Territory', 'Other/Unknown']))
            ].copy()
            x_col = 'REGION'
        else:
            max_year = int(df_year_country['YEAR'].max())
            comp_data = df_year_country[
                (df_year_country['YEAR'] == max_year) &
                (~df_year_country['COUNTRY'].str.contains('Code|Unknown|United States', na=False))
            ].copy()
            x_col = 'COUNTRY'
        
        comp_data = comp_data[comp_data['n_unweighted'] >= 100]
        comp_data = comp_data.nlargest(15, 'n_unweighted')
        comp_data = comp_data.sort_values('pct_married_us_born', ascending=True)
        comparison_title = f"Top Origins Comparison ({max_year})"
    
    fig_comp = go.Figure()
    
    if len(comp_data) > 0:
        fig_comp.add_trace(go.Bar(
            y=comp_data[x_col],
            x=comp_data['pct_married_us_born'],
            name='Married US-Born',
            orientation='h',
            marker_color=COLORS['married_us'],
            hovertemplate='%{x:.1f}%<extra>Married US-Born</extra>'
        ))
        
        fig_comp.add_trace(go.Bar(
            y=comp_data[x_col],
            x=comp_data['pct_same_country'],
            name='Same Country',
            orientation='h',
            marker_color=COLORS['same_country'],
            hovertemplate='%{x:.1f}%<extra>Same Country</extra>'
        ))
        
        fig_comp.add_trace(go.Bar(
            y=comp_data[x_col],
            x=comp_data['pct_same_region'],
            name='Same Region',
            orientation='h',
            marker_color=COLORS['same_region'],
            hovertemplate='%{x:.1f}%<extra>Same Region</extra>'
        ))
        
        fig_comp.add_trace(go.Bar(
            y=comp_data[x_col],
            x=comp_data['pct_different_region'],
            name='Different Region',
            orientation='h',
            marker_color=COLORS['different_region'],
            hovertemplate='%{x:.1f}%<extra>Different Region</extra>'
        ))
    
    fig_comp.update_layout(
        barmode='stack',
        xaxis_title="Percentage",
        xaxis=dict(range=[0, 100], ticksuffix='%'),
        yaxis_title="",
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        margin=dict(t=60, b=50, l=150),
        plot_bgcolor='white',
        height=450
    )
    fig_comp.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
    
    # =========================================================================
    # STATS TABLE
    # =========================================================================
    if len(df_filtered) > 0:
        table_data = df_filtered[['YEAR', 'n_unweighted', 'pct_married_us_born', 
                                   'pct_same_country', 'pct_same_region', 
                                   'pct_different_region']].copy()
        table_data.columns = ['Year', 'Sample Size', '% Married US-Born', 
                              '% Same Country', '% Same Region', '% Different Region']
        
        # Format numbers
        table_data['Year'] = table_data['Year'].astype(int)
        table_data['Sample Size'] = table_data['Sample Size'].apply(lambda x: f"{int(x):,}")
        for col in ['% Married US-Born', '% Same Country', '% Same Region', '% Different Region']:
            table_data[col] = table_data[col].apply(lambda x: f"{x:.1f}%")
        
        stats_table = dbc.Table.from_dataframe(
            table_data.sort_values('Year', ascending=False),
            striped=True,
            bordered=True,
            hover=True,
            size='sm',
            className='mb-0'
        )
    else:
        stats_table = html.P("No data available for selected filters.", className="text-muted")
    
    # =========================================================================
    # STAT CARDS
    # =========================================================================
    if len(df_filtered) > 0:
        total_n = f"{int(df_filtered['n_unweighted'].sum()):,}"
        latest = df_filtered[df_filtered['YEAR'] == df_filtered['YEAR'].max()].iloc[0]
        intermarriage_pct = f"{latest['pct_married_us_born']:.1f}%"
        endogamy_pct = f"{latest['pct_same_country']:.1f}%"
        n_years = str(len(df_filtered))
    else:
        total_n = "0"
        intermarriage_pct = "N/A"
        endogamy_pct = "N/A"
        n_years = "0"
    
    return (fig_time, fig_pie, pie_title, fig_comp, comparison_title, 
            stats_table, total_n, intermarriage_pct, endogamy_pct, n_years)


# =============================================================================
# RUN SERVER
# =============================================================================
if __name__ == '__main__':
    print("\n" + "="*60)
    print("💕 IMMIGRANT INTERMARRIAGE DASHBOARD 💕")
    print("="*60)
    print("\n🚀 Starting server...")
    print("📊 Open your browser to: http://127.0.0.1:8050")
    print("\nPress Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    app.run(debug=True, host='127.0.0.1', port=8050)

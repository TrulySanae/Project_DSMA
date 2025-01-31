import dash
from dash import dcc, html, Input, Output
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
model = joblib.load('1b) best_revenue_prediction_model.pkl')
scaler = joblib.load('1c) scaler.pkl')

# Variable groups
binary_vars = ['ambiance_intimate', 'ambiance_touristy', 'ambiance_hipster', 'ambiance_divey', 'ambiance_classy',
               'ambiance_upscale', 'ambiance_casual', 'ambiance_trendy', 'ambiance_romantic', 'meals_breakfast',
               'meals_brunch', 'meals_lunch', 'meals_dinner', 'meals_dessert', 'meals_latenight', 'attr_parking',
               'attr_credit_cards', 'attr_outdoor_seating', 'attr_tv', 'reservations', 'service_table_service',
               'service_caters', 'service_good_for_kids', 'service_good_for_groups', 'collect_takeout',
               'collect_delivery', 'alcohol', 'wifi']
continuous_vars = ['rating_review_count', 'rating_stars', 'rating_popularity', 'week_hours']

# Initialize the Dash app
app = dash.Dash(__name__)
app.title = "Revenue Prediction Dashboard"

# App Layout
app.layout = html.Div([
    html.Div([
        html.H1('Revenue Prediction Dashboard', style={
            'textAlign': 'center', 'marginBottom': '20px',
            'color': '#007BFF', 'fontWeight': 'bold'
        }),
        html.P('Predict restaurant revenue for each day of the week based on various attributes.',
               style={'textAlign': 'center', 'color': '#555', 'fontSize': '16px'}),
    ], style={'padding': '20px', 'backgroundColor': '#f4f4f4', 'borderBottom': '2px solid #ddd'}),

    # Input Section
    html.Div([
        html.Div([
            html.H3('Continuous Variables', style={'color': '#007BFF'}),
            *[
                html.Div([
                    html.Label(var, style={'fontWeight': 'bold', 'color': '#333'}),
                    dcc.Input(id=var, type='number', value=5, step=0.1,
                              style={'width': '100%', 'padding': '8px', 'marginBottom': '10px', 'border': '1px solid #ddd',
                                     'borderRadius': '5px'})
                ]) for var in continuous_vars
            ]
        ], style={'flex': '1', 'padding': '10px'}),

        html.Div([
            html.H3('Binary Variables', style={'color': '#007BFF'}),
            *[
                html.Div([
                    html.Label(var, style={'fontWeight': 'bold', 'color': '#333'}),
                    dcc.RadioItems(
                        id=var,
                        options=[{'label': 'No', 'value': 0}, {'label': 'Yes', 'value': 1}],
                        value=0,
                        labelStyle={'display': 'inline-block', 'marginRight': '10px', 'color': '#555'}
                    )
                ], style={'marginBottom': '10px'}) for var in binary_vars
            ]
        ], style={'flex': '1', 'padding': '10px', 'overflowY': 'scroll', 'maxHeight': '400px',
                  'borderLeft': '1px solid #ddd', 'backgroundColor': '#fafafa'}),
    ], style={'display': 'flex', 'margin': '20px 0'}),

    # Dropdown and Graphs
    html.Div([
        html.Div([
            html.H3('Social Media Presence', style={'color': '#007BFF'}),
            dcc.RadioItems(
                id='social-media-radio',
                options=[
                    {'label': 'No Social Media (0)', 'value': 0},
                    {'label': 'Some Social Media (1)', 'value': 1},
                    {'label': 'Active Social Media (2)', 'value': 2}
                ],
                value=0,
                labelStyle={'display': 'block', 'margin': '5px 0', 'color': '#555'}
            )
        ], style={'marginBottom': '20px', 'padding': '10px', 'backgroundColor': '#f9f9f9', 'border': '1px solid #ddd',
                  'borderRadius': '10px'}),
    ], style={'display': 'flex', 'flexDirection': 'column', 'padding': '10px', 'backgroundColor': '#fafafa',
              'borderTop': '2px solid #ddd'}),

    # Total Weekly Revenue
    html.Div([
        html.Div([
            html.H3('Total Weekly Revenue', style={'textAlign': 'center', 'color': '#28A745'}),
            html.P(id='total-revenue', style={
                'fontSize': '24px', 'fontWeight': 'bold', 'textAlign': 'center',
                'color': '#28A745', 'marginTop': '10px'
            }),
        ], style={
            'padding': '20px', 'margin': '20px auto', 'border': '1px solid #ddd',
            'borderRadius': '10px', 'backgroundColor': '#f9f9f9', 'width': '50%',
            'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.1)'
        }),
    ], style={'display': 'flex', 'justifyContent': 'center'}),

    # Graphs
    html.Div([
        dcc.Graph(id='revenue-barplot'),
        dcc.Graph(id='revenue-boxplot'),
    ], style={'margin': '20px 0'}),
], style={'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#ffffff'})


@app.callback(
    [Output('revenue-barplot', 'figure'),
     Output('revenue-boxplot', 'figure'),
     Output('total-revenue', 'children')],  # Add an output for total revenue
    [
        Input(var, 'value') for var in continuous_vars + binary_vars
    ] + [Input('social-media-radio', 'value')]
)
def update_revenue(*inputs):
    continuous_values = inputs[:len(continuous_vars)]
    binary_values = inputs[len(continuous_vars):-1]
    social_media_value = inputs[-1]

    feature_dict = {
        **{var: [val] for var, val in zip(continuous_vars, continuous_values)},
        **{var: [val] for var, val in zip(binary_vars, binary_values)},
        'social_media': [social_media_value]
    }

    feature_df = pd.DataFrame(feature_dict)
    feature_df = feature_df[scaler.feature_names_in_]
    feature_vector_scaled = scaler.transform(feature_df)
    revenue_predictions = model.predict(feature_vector_scaled).flatten()

    revenue_df = pd.DataFrame({
        'Day of the Week': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
        'Predicted Revenue': revenue_predictions
    })

    barplot_fig = px.bar(revenue_df, x='Day of the Week', y='Predicted Revenue',
                         title='Predicted Revenue for Each Day',
                         text_auto='.2s')
    barplot_fig.update_layout(yaxis_title='Revenue (€)', xaxis_title='Day',
                               title_font_size=20, template='simple_white')

    weekday_revenue = revenue_predictions[:5]
    weekend_revenue = revenue_predictions[5:]
    boxplot_df = pd.DataFrame({
        'Type': ['Weekday'] * len(weekday_revenue) + ['Weekend'] * len(weekend_revenue),
        'Revenue': np.concatenate([weekday_revenue, weekend_revenue])
    })

    boxplot_fig = px.box(boxplot_df, x='Type', y='Revenue', title='Weekday vs Weekend Revenue',
                         points='all', template='simple_white')
    boxplot_fig.update_layout(yaxis_title='Revenue (€)', xaxis_title='Time Period',
                               title_font_size=20)

    # Calculate total weekly revenue
    total_revenue = np.sum(revenue_predictions)
    total_revenue_text = f"€{total_revenue:,.2f}"  # Format as currency

    return barplot_fig, boxplot_fig, total_revenue_text


if __name__ == '__main__':
    app.run_server(debug=True)

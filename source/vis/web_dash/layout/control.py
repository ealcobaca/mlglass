import dash_core_components as dcc
import dash_html_components as html

options_metrics = [
    'RRMSE'
    , 'RMSE'
    , 'MARE'
    , 'R2'
]
options_methos = [
     'MLP'
    , 'DT'
    , 'RF'
]


def layout():
    return [
        'Select Metric:'
        , dcc.Dropdown(
            id='drp_options_metrics'
            # , style={'width': '30%', 'display': 'inline-block'}
            , options=[dict(label=e, value=e) for e in options_metrics]
            , multi=False
            , value=options_metrics[0]
        )
        , 'Select Methods:'
        , dcc.Dropdown(
            id='drp_options_methods_1'
            , style={'width': '25%', 'display': 'inline-block'}
            , options=[dict(label=e, value=e) for e in options_methos]
            , multi=False
            , value=options_methos[0]
        )
        , dcc.Dropdown(
            id='drp_options_methods_2'
            , style={'width': '25%', 'display': 'inline-block'}
            , options=[dict(label=e, value=e) for e in options_methos]
            , multi=False
            , value=options_methos[0]
        )
        , dcc.Dropdown(
            id='drp_options_methods_3'
            , style={'width': '25%', 'display': 'inline-block'}
            , options=[dict(label=e, value=e) for e in options_methos]
            , multi=False
            , value=options_methos[0]
        )
        , html.Button('Plot', id='bnt_plot')
    ]

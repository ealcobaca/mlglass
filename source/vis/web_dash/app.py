from pprint import pprint

import dash
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc

import json
import numpy as np
from source.vis.web_dash.data import Data
from source.vis.web_dash.layout import control
from source.vis.web_dash.layout import plot_heat


data_glass = Data()
x, y, z = data_glass.map_data(method=['MLP', 'MLP', 'MLP'], metric='Global_mean_RRMSE')
title_scale = 'RRMSE'

app = dash.Dash()
app.config['suppress_callback_exceptions'] = True
app.layout = html.Div(control.layout() + [
    html.Div(id='div_plot_heat')
    , 'hovermode:'
    ,  dcc.RadioItems(
        id='radio_hovermode',
        options=[
            {'label': 'x', 'value': 'x'},
            {'label': 'y', 'value': 'y'},
            {'label': 'closest', 'value': 'closest'}
        ],
        value='x'
    )
    , html.Div(id='output')
    , html.Div(id='output_local_metrics')

])

@app.callback(
    Output('div_plot_heat', 'children'),
    [
     Input('bnt_plot', 'n_clicks')
    ]
    , [
        State('drp_options_metrics', 'value')
        , State('drp_options_methods_1', 'value')
        , State('drp_options_methods_2', 'value')
        , State('drp_options_methods_3', 'value')
    ])
def plot_heat_map(n_clicks, metric, method1, method2, method3):
    if n_clicks is None or metric == '' or method1 == '' or method2 == '' or method3 == '':
        return ['Select metric and methods.']
    x, y, z = data_glass.map_data(method=[method1, method2, method3], metric='Global_mean_{:}'.format(metric))
    title_scale = metric
    return plot_heat.plot(x, y, z, title_scale)

@app.callback(
    Output('plot_heatmap', 'figure')
    , [Input('radio_hovermode', 'value')
       , Input('plot_heatmap', 'hoverData')]
    , [State('plot_heatmap', 'figure')]
)
def change_hovermode(value, hoverData, figure):
    figure['layout']['hovermode'] = value
    if hoverData is None:
        return figure
    if value == 'x':
        x = hoverData['points'][0]['x']
        figure['data'][-1] = dict(
            name='x'
            , x=[x, x]
            , y=[0, 40]
            , opacity=0.3
            , mode='lines'
            , hovermode='false'
            , hovertext=''
            , line=dict(color='(rgb(0, 0, 0))')
        )
    else:
        y = hoverData['points'][0]['y']
        figure['data'][-1] = dict(
            name='y'
            , x=[0, 40]
            , y=[y, y]
            , opacity=0.3
            , mode='lines'
            , line=dict(color='(rgb(0, 0, 0))')
        )
    return figure


@app.callback(
    Output('output', 'children'),
    [
        Input('div_plot_heat', 'children'),
        Input('radio_hovermode', 'value')
    ]
    , [
        State('drp_options_metrics', 'value')
    ]
)
def display_graph2d(chieldren, hover_mode, metric):
    data_plot = []

    for e in [1.5, 2.5, 3.5, 5, 10, 15, 20, 25, 30]:
        x, y, dy, indices = data_glass.data_2d(value_fixed=e, metric=metric, variable_fixed=hover_mode)
        data_plot += plot_heat.scatter_2d(x, y, e, dy=dy, custom_data=indices)

    return plot_heat.plot_2d(data_plot) + [
    ]

@app.callback(
    Output('output_local_metrics', 'children'),
    [
        Input('plot_data_2d', 'hoverData'),
    ]
    , [
        State('drp_options_metrics', 'value')
    ]
)
def display_graph2d(data, metric):
    index = data['points'][0]['customdata']
    print(index)
    print(data_glass.data.loc[index])

    return json.dumps(data)

@app.callback(
    Output('plot_data_2d', 'figure'),
    [Input('plot_heatmap', 'hoverData'),
     Input('plot_heatmap', 'clickData')]
    , [
        State('drp_options_metrics', 'value')
        , State('plot_data_2d', 'figure')
        , State('radio_hovermode', 'value')
    ]
)
def display_hoverdata(hoverData, clickData, metric, figure, hovermode):
    if hoverData is None:
        return figure
    if hovermode == 'x':
        x_fixed = hoverData['points'][0]['x']
    else:
        x_fixed = hoverData['points'][0]['y']
    data = figure['data']
    for sdata in data:
        pprint(sdata['name'])
        if sdata['name'] == x_fixed:
            sdata['opacity'] = 1
            # sdata['lines']['opacity'] = 1
        else:
            sdata['opacity'] = 0.2
            # sdata['lines']['opacity'] = 0.2
    return figure


# if __name__ == '__main__':
def main1():
    app.run_server(debug=True)

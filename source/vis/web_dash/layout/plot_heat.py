import dash_core_components as dcc
import dash_html_components as html
import numpy as np
from definitions import ROOT_DIR

layout = {
    'xgap': 10,
    'ygap': 10,
    'margin': dict(r=20, t=20, l=40, b=40),
    'yaxis': {'title': "X final",
              'showline': True,
              'linewidth': 2,
              'mirror': 'ticks'
        , 'range': [-1, 35]
              }
    , 'xaxis': {
        'title': 'X initial',
        'showline': True,
        'linewidth': 2,
        'mirror': 'ticks'
        , 'range': [-1, 35]
    }
    , 'hovermode': 'x'
}

layout_2d = {
    'xgap': 10,
    'ygap': 10,
    'margin': dict(r=20, t=20, l=40, b=40),
    'yaxis': {'title': "metric",
              'showline': True,
              'linewidth': 2,
              'mirror': 'ticks'
              # , 'range': [0, 35]
              }
    , 'xaxis': {
        'title': 'x',
        'showline': True,
        'linewidth': 2,
        'mirror': 'ticks'
        # , 'range': [0, 35]
    }
    , 'hovermode': 'closest'
}

layout_2d_errors_range = {
    'xgap': 10,
    'ygap': 10,
    'margin': dict(r=20, t=20, l=40, b=40),
    'yaxis': {'title': "metric",
              'showline': True,
              'linewidth': 2,
              'mirror': 'ticks'
              # , 'range': [0, 35]
              }
    , 'xaxis': {
        'title': 'x',
        'showline': True,
        'linewidth': 2,
        'mirror': 'ticks'
        # , 'range': [0, 35]
    }
    , 'hovermode': 'closest'
}


def scatter_color(x, y, z, title_scale):
    data = [dict(
        x=x,
        y=y
        , mode='markers',
        marker=dict(
            opacity=1,
            size=15
            , symbol='square'
            , cmin=np.min(z),
            cmax=np.max(z),
            color=z,
            colorbar=dict(
                title=title_scale
            ),
            colorscale='Jet'
        ),
        # hoverinfo='text',
        # text=hover_text,
        textposition='top left',
        showlegend=False
        , type='scatter'
    )]
    return data


def scatter_2d(x, y, name, dy=[], custom_data=[]):
    data_s = dict(
        x=x
        , y=y
        , error_y=dict(
            type='data',
            array=dy,
            visible=True
        )
        , customdata=custom_data
        , name=name
        , opacity=.5
        , mode='lines+markers'
        , lines=dict(
            opacity=0.1,
        )
        , marker=dict(
            opacity=1,
            size=15
            , symbol='square'
            # , cmin=np.min(z),
            # cmax=np.max(z),
            # color=z,
            # colorbar=dict(
            #     title=title_scale
            # ),
            # colorscale='Jet'
        ),
        hoverinfo='text',
        # text=hover_text,
        textposition='top left',
        showlegend=True
        , type='scatter'
    )
    data = [data_s]
    return data


def plot(x, y, z, title_scale):
    data_plot = [dict(
        x=[1, 1]
        , y=[0, 35]
        , mode='lines'
        , color='(rgb(0, 0, 0))'
        , opacity=0.3
        , line=dict(color='(rgb(0, 0, 0))')
        , hovermode='false'
        , hovertext=''
        , showlegend=False
    )]
    return [dcc.Graph(
        id='plot_heatmap',
        figure={
            'data': scatter_color(x, y, z, title_scale) + data_plot
            , 'layout': layout
            , 'config': {'editable': True, 'modeBarButtonsToRemove': ['pan2d', 'lasso2d']}
        }
    )]


def plot_2d(data, metric):
    layout_2d['yaxis']['title'] = metric
    return [dcc.Graph(
        id='plot_data_2d',
        figure={
            'data': data
            , 'layout': layout_2d
            , 'config': {'editable': True, 'modeBarButtonsToRemove': ['pan2d', 'lasso2d']}
        }
    )]


def plot_2d_errors_range(data, metric, x_ticks=None):
    layout_2d_errors_range['yaxis']['title'] = metric
    if x_ticks is not None:
        layout_2d_errors_range['xaxis']['ticktext'] = x_ticks
    return [dcc.Graph(
        id='plot_data_2d_errors_range',
        figure={
            'data': data
            , 'layout': layout_2d_errors_range
            , 'config': {'editable': True, 'modeBarButtonsToRemove': ['pan2d', 'lasso2d']}
        }
    )]


def scatter_2d_errors_range(x, y, name, dy=[], custom_data=[], x_ticks=[]):
    data_s = dict(
        x=x
        , y=y
        , error_y=dict(
            type='data',
            array=dy,
            visible=True
        )
        , customdata=custom_data
        , name=name
        , opacity=.5
        , mode='lines+markers'
        , lines=dict(
            opacity=0.1,
        )
        , marker=dict(
            opacity=1,
            size=15
            , symbol='square'
            # , cmin=np.min(z),
            # cmax=np.max(z),
            # color=z,
            # colorbar=dict(
            #     title=title_scale
            # ),
            # colorscale='Jet'
        ),
        hoverinfo='text',
        # text=hover_text,
        textposition='top left',
        showlegend=False
        , type='scatter'
    )
    data = [data_s]
    return data


def data_histogram(x_initial, x_final):
    hist = []
    bins = []
    start2tg = {
        0: 350.00
        , 1.5: 468.15,
        2.5: 493.15,
        3.5: 512.15,
        5: 530.15,
        10: 572.15,
        15: 604.15,
        20: 631.15,
        25: 658.15,
        30: 684.15,
    }

    end2tg = {
        0: 1451.0
        , 1.5: 1116.15
        , 2.5: 1082.43,
        3.5: 1061.15,
        5: 1036.15,
        10: 975.15,
        15: 929.15,
        20: 891.15,
        25: 863.15,
        30: 838.15,
    }
    print('----------------')
    print(x_initial, x_final)
    start_value = start2tg[x_initial]
    end_value = end2tg[x_final]

    with open('{:}/source/vis/web_dash/data/histogram.dat'.format(ROOT_DIR), 'r') as fd:
        for line in fd:
            words = line.split()
            hist += [float(words[1])]
            bins += [float(words[0])]
    # print(bins)
    # print(hist)
    x_e = []
    y_e = []
    x_m = []
    y_m = []
    x_f = []
    y_f = []
    for (b, h) in zip(bins, hist):
        if b < start_value:
            x_e += [b]
            y_e += [h]
        if b > start_value and b < end_value:
            x_m += [b]
            y_m += [h]
        if b > end_value:
            x_f += [b]
            y_f += [h]

    data_e = dict(
        x=x_e
        , y=y_e
        , opacity=1.
        , mode='marker'
        , hoverinfo='text',
        # text=hover_text,
        textposition='top left',
        showlegend=False
        , type='bar'
    )

    data_f = dict(
        x=x_f
        , y=y_f
        , opacity=1.
        , mode='marker'
        , hoverinfo='text',
        # text=hover_text,
        textposition='top left',
        showlegend=False
        , type='bar'
    )
    data_m = dict(
        x=x_m
        , y=y_m
        , opacity=1.
        , mode='marker'
        , hoverinfo='text',
        # text=hover_text,
        textposition='top left',
        showlegend=False
        , type='bar'
    )

    data = [data_e, data_m, data_f]
    return data


def histogram(x_initial, x_final):
    data = data_histogram(x_initial, x_final)
    return [dcc.Graph(
        id='plot_histogram',
        figure={
            'data': data
            , 'layout': layout_2d_errors_range
            # , 'config': {'editable': True, 'modeBarButtonsToRemove': ['pan2d', 'lasso2d']}
        }
    )]
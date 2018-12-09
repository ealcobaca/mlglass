import dash_core_components as dcc
import dash_html_components as html
import numpy as np

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

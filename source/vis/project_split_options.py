import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from plotly.offline import plot
import plotly.graph_objs as go


def project_data_points(data, method='PCA'):
    """ Projects data matrix using a Multidimensional Projection algorithm.

    The points in 'data' are mapped to bidimensional representations, which are
    returned by the function. The supported methods are 'PCA', 't-SNE', 'MDS'
    and 'Isomap'.
    """
    projected = None
    data = StandardScaler().fit_transform(data)
    if method == 't-SNE':
        projection = TSNE(n_components=2)
    else:
        projection = PCA(n_components=2)

    projected = projection.fit_transform(data)
    return projected


def subset_and_transform(data, metric, start2tg, end2tg):
    plot_data = np.zeros((data.shape[0], 12))
    for i in range(data.shape[0]):
        plot_data[i, 0] = start2tg[data.iloc[i, 0]]
        plot_data[i, 1] = end2tg[data.iloc[i, 2]]
    plot_data[:, 2:11] = data.iloc[:, 3:12]
    plot_data[:, 11] = data.loc[:, metric]
    return plot_data


def subset_and_transform2(data, metric, start2tg, end2tg):
    plot_data = np.zeros((data.shape[0], 25))
    for i in range(data.shape[0]):
        plot_data[i, 0] = start2tg[data.iloc[i, 0]]
        plot_data[i, 1] = end2tg[data.iloc[i, 2]]
    plot_data[:, 2:] = data.iloc[:, 3:26]
    return plot_data


def subset_and_transform3(data, metric, start2tg, end2tg):
    plot_data = np.zeros((data.shape[0], 67))
    for i in range(data.shape[0]):
        plot_data[i, 0] = start2tg[data.iloc[i, 0]]
        plot_data[i, 1] = end2tg[data.iloc[i, 2]]
    plot_data[:, 2:] = data.iloc[:, 3:]
    return plot_data


def subset_and_transform_local(data, metric, start2tg, end2tg):
    cols = ['Local_S_mean_{}'.format(metric)]
    cols.extend(['Local_M_mean_{}'.format(metric)])
    cols.extend(['Local_E_mean_{}'.format(metric)])
    cols.extend(['Local_S_sd_{}'.format(metric)])
    cols.extend(['Local_M_sd_{}'.format(metric)])
    cols.extend(['Local_E_sd_{}'.format(metric)])
    plot_data = np.zeros((data.shape[0], 17))
    for i in range(data.shape[0]):
        plot_data[i, 0] = start2tg[data.iloc[i, 0]]
        plot_data[i, 1] = end2tg[data.iloc[i, 2]]
    plot_data[:, 2:11] = data.iloc[:, 3:12]
    plot_data[:, 11:] = data.loc[:, cols]
    return plot_data


def subset_and_transform_local2(data, metric, start2tg, end2tg):
    plot_data = np.zeros((data.shape[0], 53))
    for i in range(data.shape[0]):
        plot_data[i, 0] = start2tg[data.iloc[i, 0]]
        plot_data[i, 1] = end2tg[data.iloc[i, 2]]
    plot_data[:, 2:11] = data.iloc[:, 3:12]
    plot_data[:, 11:] = data.iloc[:, 26:]
    return plot_data


def get_hover_text(data, start2tg, end2tg, code2method, metric):
    hover_text = []
    for index, line in data.iterrows():
        m_s = code2method['{}{}{}'.format(int(line['S_MLP']),
                                          int(line['S_RF']),
                                          int(line['S_DT']))]
        m_m = code2method['{}{}{}'.format(int(line['M_MLP']),
                                          int(line['M_RF']),
                                          int(line['M_DT']))]
        m_e = code2method['{}{}{}'.format(int(line['E_MLP']),
                                          int(line['E_RF']),
                                          int(line['E_DT']))]
        ht = '{}: {:0.2f}<br>'.format(metric, line[metric]) + \
            '({:0.1f}%, {:0.1f}%, {:0.1f}%)<br>'.format(line['S'], line['M'],
                                                        line['E']) + \
            '({}, {})<br>'.format(start2tg[line['S']], end2tg[line['E']]) + \
            '({}, {}, {})'.format(m_s, m_m, m_e)
        hover_text.append(ht)
    return hover_text


def get_hover_text_local(data, start2tg, end2tg, code2method, metric):
    hover_text = []
    for index, line in data.iterrows():
        m_s = code2method['{}{}{}'.format(int(line['S_MLP']),
                                          int(line['S_RF']),
                                          int(line['S_DT']))]
        m_m = code2method['{}{}{}'.format(int(line['M_MLP']),
                                          int(line['M_RF']),
                                          int(line['M_DT']))]
        m_e = code2method['{}{}{}'.format(int(line['E_MLP']),
                                          int(line['E_RF']),
                                          int(line['E_DT']))]
        ht = '{}: ({:0.2f}, {:0.2f}, {:0.2f})<br>'.format(
                metric,
                line['Local_S_mean_{}'.format(metric)],
                line['Local_M_mean_{}'.format(metric)],
                line['Local_E_mean_{}'.format(metric)]
            ) + \
            '({:0.1f}%, {:0.1f}%, {:0.1f}%)<br>'.format(line['S'], line['M'],
                                                        line['E']) + \
            '({}, {})<br>'.format(start2tg[line['S']], end2tg[line['E']]) + \
            '({}, {}, {})'.format(m_s, m_m, m_e)
        hover_text.append(ht)
    return hover_text


def plot_projections(projections, metric_data, metric, hover_text, out_path):
    """ Creates an interative scatter plot of the projections.
    """
    trace = go.Scatter(
        x=projections[:, 0],
        y=projections[:, 1],
        mode='markers',
        marker=dict(
            opacity=1,
            size=12,
            cmin=np.min(metric_data),
            cmax=np.max(metric_data),
            color=[v for v in metric_data],
            colorbar=dict(
                title=metric
            ),
            colorscale='Bluered'
        ),
        hoverinfo='text',
        text=hover_text,
        textposition='top left',
        showlegend=False
    )

    layout = go.Layout(
        xaxis=dict(
            title='Component 1'
        ),
        yaxis=dict(
            title='Component 2'
        )
    )

    plot_data = {
        'data': [trace],
        'layout': layout
    }

    plot(
        plot_data,
        filename=out_path + '.html',
        auto_open=False
    )


if __name__ == '__main__':
    start2tg = {
        1.5: 468.15,
        2.5: 493.15,
        3.5: 512.15,
        5.0: 530.15,
        10.0: 572.15,
        15.0: 604.15,
        20.0: 631.15,
        25.0: 658.15,
        30.0: 684.15,
    }

    end2tg = {
        1.5: 838.15,
        2.5: 863.15,
        3.5: 891.15,
        5.0: 929.15,
        10.0: 975.15,
        15.0: 1036.15,
        20.0: 1061.15,
        25.0: 1082.43,
        30.0: 1116.15
    }

    code2method = {
        '100': 'MLP',
        '010': 'RF',
        '001': 'DT'
    }

    data = pd.read_csv('../../result/evaluating_range/ranges_2.0.csv')
    data = data.iloc[:, 1:]
    data = data.dropna()
    metric = 'Global_mean_RMSE'
    metric_sufix = 'RMSE'
    method = 't-SNE'
    out_path = './xcomposed_RMSE_t-SNE'
    data_t = subset_and_transform3(data, metric, start2tg, end2tg)
    hover_text = get_hover_text(data, start2tg, end2tg, code2method,
                                metric)
    # data_t = subset_and_transform_local2(data, metric_sufix, start2tg, end2tg)
    # hover_text = get_hover_text_local(data, start2tg, end2tg, code2method,
    #                                   metric_sufix)
    projected = project_data_points(data_t, method)
    plot_projections(projected, data.loc[:, metric], metric,
                     hover_text, out_path)

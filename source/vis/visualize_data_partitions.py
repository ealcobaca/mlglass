import numpy as np
import pandas as pd
import pickle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns


def load_class_files(class_path, strategy):
    log_ts = log_ts_rem = None

    with open('{}/{}_test_root_.list'.format(class_path, strategy), 'rb') as f:
        log_ts = pickle.load(f)
    with open('{}/{}_test_rem_root_.list'.format(class_path, strategy), 'rb') as f:
        log_ts_rem = pickle.load(f)
    return log_ts, log_ts_rem


def read_xy(data_path):
    data = pd.read_csv(data_path)
    X, y = data.values[:, :-1], data.values[:, -1:]
    return X, y


def visualize_pca(output_path, strategy, data_path, X, y, t_labels, p_labels):
    fontP = FontProperties()
    fontP.set_size('small')

    p_labels = p_labels.astype(int)
    file_sufix = data_path.split('/')[-1].split('.')[0]
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)

    pca = PCA(n_components=2)
    pca.fit(X)
    X_proj = pca.transform(X)

    # Plot 1: color gradient
    print('PCA - color gradient: {}-{}'.format(file_sufix, strategy))
    # f = plt.figure(figsize=(10, 3.7), dpi=500)
    f = plt.figure(dpi=500)
    plt.scatter(X_proj[:, 0], X_proj[:, 1], c=y[:, 0], s=10,
                cmap=sns.cubehelix_palette(light=0.75, start=.5, rot=-.75,
                                           as_cmap=True))

    plt.xlabel('PC1')
    plt.ylabel('PC2')

    cbar = plt.colorbar()
    cbar.set_label('Tg')

    f.savefig('{}/pca_gradient_{}_{}.png'.format(output_path, file_sufix,
                                                 strategy),
              bbox_inches='tight', additional_artists=[cbar])

    plt.close()

    # Colors for generating the plots
    colors = sns.color_palette('colorblind', n_colors=3).as_hex()
    categories = {1: 'Low', 2: 'Medium', 3: 'High'}

    # Plot 2: true labels
    print('PCA - true labels: {}-{}'.format(file_sufix, strategy))
    uniques = np.unique(t_labels).tolist()
    # f = plt.figure(figsize=(10, 3.7), dpi=500)
    f = plt.figure(dpi=500)

    for u in uniques:
        sel_sampl = np.array(t_labels) == u
        plt.scatter(X_proj[sel_sampl, 0], X_proj[sel_sampl, 1],
                    c=colors[u-1], s=10, label=categories[u])

    plt.xlabel('PC1')
    plt.ylabel('PC2')

    lgn = plt.legend(title='Category', loc=9, bbox_to_anchor=(0.5, 1.15),
                     ncol=3, prop=fontP)

    f.savefig('{}/pca_true_labels_{}_{}.png'.format(output_path, file_sufix,
                                                    strategy),
              bbox_inches='tight', additional_artists=[lgn])

    plt.close()

    # Plot 3: predicted labels
    print('PCA - predicted labels: {}-{}'.format(file_sufix, strategy))
    uniques = np.unique(p_labels).tolist()
    # f = plt.figure(figsize=(10, 3.7), dpi=500)
    f = plt.figure(dpi=500)

    for u in uniques:
        sel_sampl = np.array(p_labels) == u
        plt.scatter(X_proj[sel_sampl, 0], X_proj[sel_sampl, 1],
                    c=colors[u-1], s=10, label=categories[u])

    plt.xlabel('PC1')
    plt.ylabel('PC2')

    lgn = plt.legend(title='Category', loc=9, bbox_to_anchor=(0.5, 1.15),
                     ncol=3, prop=fontP)

    f.savefig('{}/pca_predicted_labels_{}_{}.png'.format(output_path,
                                                         file_sufix,
                                                         strategy),
              bbox_inches='tight', additional_artists=[lgn])

    plt.close()


def visualize_tsne(output_path, strategy, data_path, X, y, t_labels, p_labels):
    fontP = FontProperties()
    fontP.set_size('small')

    p_labels = p_labels.astype(int)
    file_sufix = data_path.split('/')[-1].split('.')[0]

    tsne = TSNE(n_components=2)
    X_proj = tsne.fit_transform(X)

    # Plot 1: color gradient
    print('T-SNE - color gradient: {}-{}'.format(file_sufix, strategy))
    # f = plt.figure(figsize=(10, 3.7), dpi=500)
    f = plt.figure(dpi=500)
    plt.scatter(X_proj[:, 0], X_proj[:, 1], c=y[:, 0], s=10,
                cmap=sns.cubehelix_palette(light=0.75, start=.5, rot=-.75,
                                           as_cmap=True))

    plt.xlabel('')
    plt.ylabel('')

    cbar = plt.colorbar()
    cbar.set_label('Tg')

    f.savefig('{}/tsne_gradient_{}_{}.png'.format(output_path, file_sufix,
                                                  strategy),
              bbox_inches='tight', additional_artists=[cbar])

    plt.close()

    # Colors for generating the plots
    colors = sns.color_palette('colorblind', n_colors=3).as_hex()
    categories = {1: 'Low', 2: 'Medium', 3: 'High'}

    # Plot 2: true labels
    print('T-SNE - true labels: {}-{}'.format(file_sufix, strategy))
    uniques = np.unique(t_labels).tolist()
    # f = plt.figure(figsize=(10, 3.7), dpi=500)
    f = plt.figure(dpi=500)

    for u in uniques:
        sel_sampl = np.array(t_labels) == u
        plt.scatter(X_proj[sel_sampl, 0], X_proj[sel_sampl, 1],
                    c=colors[u-1], s=10, label=categories[u])

    plt.xlabel('')
    plt.ylabel('')

    lgn = plt.legend(title='Category', loc=9, bbox_to_anchor=(0.5, 1.15),
                     ncol=3, prop=fontP)

    f.savefig('{}/tsne_true_labels_{}_{}.png'.format(output_path, file_sufix,
                                                     strategy),
              bbox_inches='tight', additional_artists=[lgn])

    plt.close()

    # Plot 3: predicted labels
    print('T-SNE - predicted labels: {}-{}'.format(file_sufix, strategy))
    uniques = np.unique(p_labels).tolist()
    # f = plt.figure(figsize=(10, 3.7), dpi=500)
    f = plt.figure(dpi=500)

    for u in uniques:
        sel_sampl = np.array(p_labels) == u
        plt.scatter(X_proj[sel_sampl, 0], X_proj[sel_sampl, 1],
                    c=colors[u-1], s=10, label=categories[u])

    plt.xlabel('')
    plt.ylabel('')

    lgn = plt.legend(title='Category', loc=9, bbox_to_anchor=(0.5, 1.15),
                     ncol=3, prop=fontP)

    f.savefig('{}/tsne_predicted_labels_{}_{}.png'.format(output_path,
                                                          file_sufix,
                                                          strategy),
              bbox_inches='tight', additional_artists=[lgn])

    plt.close()


if __name__ == '__main__':
    class_path = '../../result/result_oracle/default-model'
    ts_path = '../../data/clean/oxides_Tg_test.csv'
    ts_rem_path = '../../data/clean/oxides_Tg_test_rem.csv'
    output_path = '../../result/vis'

    strategies = {'mean': None, 'mode': None}

    for strategy in strategies.keys():
        strategies[strategy] = load_class_files(class_path, strategy)

    for i, data_path in enumerate([ts_path, ts_rem_path]):
        X, y = read_xy(data_path)
        for strategy in strategies.keys():
            visualize_pca(output_path, strategy, data_path, X, y,
                          *strategies[strategy][i])
            visualize_tsne(output_path, strategy, data_path, X, y,
                           *strategies[strategy][i])

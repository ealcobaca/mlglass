import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from constants import DATA_PATH as input_path
from constants import OUTPUT_PATH as output_path


def get_tree_paths(tree, min, max):
    n_nodes = tree.node_count
    children_left = tree.children_left
    children_right = tree.children_right
    feature = tree.feature
    threshold = tree.threshold
    values = tree.value.ravel()

    # Visits all nodes and select those which are leaves
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [0]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id = stack.pop()

        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append(children_left[node_id])
            stack.append(children_right[node_id])
        else:
            is_leaves[node_id] = True

    # Leaves with values within the desired range
    interest_ids = []
    for i, is_leaf in enumerate(is_leaves):
        if is_leaf and values[i] >= min and values[i] <= max:
            interest_ids.append(i)

    # Corresponding paths for those leaves
    paths = []
    for i in interest_ids:
        path = [(None, None, values[i])]

        current = i
        while True:
            aux1 = np.where(children_left == current)[0]
            aux2 = np.where(children_right == current)[0]
            if len(aux1) > 0:
                current = aux1[0]
                path.insert(0, (True, feature[current], threshold[current]))
            else:
                current = aux2[0]
                path.insert(0, (False, feature[current], threshold[current]))
            if current == 0:
                break
        paths.append(path)
    return paths


def extract_intervals(rf, min_r, max_r, features_names, resolution=100,
                      range_features=(0, 1)):
    decay = 0.9
    incr = (range_features[1] - range_features[0]) / resolution
    n_features = len(features_names) - 1
    paths = []
    for model in rf.estimators_:
        paths.extend(get_tree_paths(model.tree_, min_r, max_r))

    intervals = [np.zeros(resolution + 1) for i in range(n_features)]
    for path in paths:
        for elem in path:
            if elem[0] is not None:
                point = int(round(elem[2] / incr))

                v = 1
                if elem[0]:
                    for i in range(point - 1, -1, -1):
                        intervals[elem[1]][i] += v
                        v *= decay
                else:
                    for i in range(point, resolution + 1):
                        intervals[elem[1]][i] += v
                        v *= decay
    plot_data = {
        'label': [],
        'dist': []
    }

    relevances = []
    for feat, interval in enumerate(intervals):
        relevances.append(max(interval))
        interval_norm = interval.copy()

        max_ = max(interval_norm)
        min_ = min(interval_norm)

        interval_norm = [
            int(round(100 * (x - min_)/(max_ - min_))) if max_ > 0 else 0
            for x in interval_norm
        ]

        for i, elem in enumerate(interval_norm):
            plot_data['label'].extend(
                [features_names[feat] for j in range(int(elem))]
            )
            plot_data['dist'].extend(np.repeat(100 * i * incr, elem).tolist())

    max_ = max(relevances)
    min_ = min(relevances)
    # relevances = {e: (x - min_)/(max_ - min_)
    #               for e, x in zip(features_names, relevances)}
    relevances = [(x - min_)/(max_ - min_)
                  for x in relevances]

    return plot_data, relevances


def plot_violins(plot_data, relevances, filename):
    dt_plot = pd.DataFrame.from_dict(plot_data)
    color_p = sns.cubehelix_palette(start=2, rot=0, dark=0.2, light=1,
                                    reverse=False, as_cmap=True)
    colors = [color_p(r) for r in relevances]
    sns.set(style='whitegrid')
    f, ax = plt.subplots(figsize=(12, 4))
    # sns.stripplot(x='label', y='dist', data=dt_plot,
    #               palette=colors, linewidth=0.5, size=0.5, jitter=0.05)
    sns.violinplot(
        x='label', y='dist', data=dt_plot,
        inner=None, palette=colors, linewidth=0.7, cut=0,
        scale='width', bw=1, gridsize=1000
    )
    # sns.despine(left=True)

    ##########################################################################
    # Tiago e Edesio, o parametro bw acima controla a "riqueza de detalhes" na
    # interpolacao
    ##########################################################################

    ax.set_title('Composition visualization')
    ax.set_xlabel('Features')
    ax.set_ylabel('Percent')
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(10)
    norm = plt.Normalize(0, 1)
    sm = plt.cm.ScalarMappable(cmap=color_p, norm=norm)
    sm.set_array([])
    cbar = ax.figure.colorbar(sm, fraction=0.012, pad=0.04)
    cbar.ax.set_title('Frequency', size=10)
    plt.tight_layout()
    plt.savefig(filename, dpi=700)


if __name__ == '__main__':
    model_name = '{0}/rf/best_rf_tg_fold04.model'.format(output_path)

    with open(model_name, 'rb') as f:
        rf = pickle.load(f)

    data = pd.read_csv('{0}/data_tg_dupl_rem.csv'.format(input_path))
    features_names = list(data)

    plot_data, relevances = extract_intervals(rf, 1200, 1500, features_names)
    filename = '{0}/interpretation/rf_vis_high_tg.png'
    plot_violins(plot_data, relevances, filename)

    plot_data, relevances = extract_intervals(rf, 0, 400, features_names)
    filename = '{0}/interpretation/rf_vis_low_tg.png'
    plot_violins(plot_data, relevances, filename)

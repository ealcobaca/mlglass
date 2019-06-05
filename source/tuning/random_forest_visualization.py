import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from constants import DATA_PATH as input_path
from constants import OUTPUT_PATH as output_path


def get_tree_paths(tree, min, max, predicates):
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
    for i in interest_ids:
        current = i
        while True:
            aux1 = np.where(children_left == current)[0]
            aux2 = np.where(children_right == current)[0]
            # np.where returns an array of indexes that match the query
            # Only one branch is supposed to have a valid test
            if len(aux1) > 0:
                current = aux1[0]
                if not feature[current] in predicates:
                    predicates[feature[current]] = []
                predicates[feature[current]].insert(
                    0, (True, threshold[current])
                )
            else:
                current = aux2[0]
                if not feature[current] in predicates:
                    predicates[feature[current]] = []
                predicates[feature[current]].insert(
                    0, (False, threshold[current])
                )
            if current == 0:
                break
    return predicates


def extract_intervals_with_data(rf, data, min_r, max_r, features_names,
                                resolution=100, range_features=(0, 1)):
    # TODO: verificar conectivo logico
    filter1 = data.iloc[:, -1] >= min_r
    filter2 = data.iloc[:, -1] <= max_r
    selected_s = [i for i in range(len(data)) if filter1[i] and filter2[i]]

    incr = (range_features[1] - range_features[0]) / resolution
    n_features = len(features_names) - 1
    predicates = {}
    for model in rf.estimators_:
        # paths.extend(get_tree_paths(model.tree_, min_r, max_r, predicates))
        predicates = get_tree_paths(model.tree_, min_r, max_r, predicates)

    # TODO: continuar
    intervals = [np.zeros(resolution + 1) for i in range(n_features)]
    for s in selected_s:
        for f, f_v in enumerate(data.iloc[s, :-1].values):
            if not f_v > 0.0:
                continue
            for rtest in predicates[f]:
                stsfs = (f_v <= rtest[1]) == rtest[0]

                if stsfs:
                    point = int(round(f_v / incr))
                    intervals[f][point] += 1
    plot_data = {
        'label': [],
        'dist': []
    }

    relevances = []
    for feat, interval in enumerate(intervals):
        # Colorscale
        relevances.append(np.sum(interval))
        interval_norm = interval.copy()

        max_ = max(interval_norm)
        # Avoid transforming the minimum element to zero
        interval_norm = [
            int(round(100 * x/max_)) if max_ > 0.0 else 0.0
            for x in interval_norm
        ]

        for i, elem in enumerate(interval_norm):
            if elem > 0:
                plot_data['label'].extend(
                    [features_names[feat] for j in range(int(elem))]
                )
                plot_data['dist'].extend(
                    np.repeat(100 * i * incr, int(elem)).tolist()
                )
            else:
                plot_data['label'].append(
                    features_names[feat]
                )
                plot_data['dist'].append(
                    None
                )

    sum_norm = np.sum(relevances)
    relevances = [x / sum_norm for x in relevances]

    return plot_data, relevances


def plot_violins(plot_data, relevances, filename):
    dt_plot = pd.DataFrame.from_dict(plot_data)
    color_p = sns.cubehelix_palette(start=2, rot=0, dark=0.0, light=1,
                                    reverse=False, as_cmap=True)
    colors = [color_p(r) for r in relevances]
    sns.set(style='whitegrid')
    f, ax = plt.subplots(figsize=(12, 4))
    sns.violinplot(
        x='label', y='dist', data=dt_plot,
        inner=None, palette=colors, linewidth=0.7, cut=0,
        scale='width', bw='silverman', gridsize=1000
    )
    # sns.despine(left=True)

    ax.set_title('Composition visualization')
    ax.set_xlabel('Features')
    ax.set_ylabel('Amount (%)')
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(10)
    norm = plt.Normalize(0, 1)
    sm = plt.cm.ScalarMappable(cmap=color_p, norm=norm)
    sm.set_array([])
    ticks = np.linspace(0, 1, 11)
    cbar = ax.figure.colorbar(
        sm, fraction=0.012, pad=0.04, ticks=ticks
    )
    cbar.ax.set_yticklabels(
        ['{}%'.format(int(100 * v)) for v in ticks], ha='right'
    )
    cbar.ax.yaxis.set_tick_params(pad=30)
    cbar.ax.set_title('Frequency', size=10, pad=10)
    plt.setp(ax.get_xticklabels(), rotation=90, ha='center')
    plt.tight_layout()
    plt.savefig(filename, dpi=700)


if __name__ == '__main__':
    model_name = '{0}/rf/best_rf_tg_fold04.model'.format(output_path)

    with open(model_name, 'rb') as f:
        rf = pickle.load(f)

    data = pd.read_csv(
        '{0}/train_test_split/tg_train_fold04.csv'.format(input_path)
    )
    features_names = list(data)

    dp_h = '{0}/interpretation/rf_data_plot_high.pic'.format(output_path)
    dp_l = '{0}/interpretation/rf_data_plot_low.pic'.format(output_path)
    dp_m = '{0}/interpretation/rf_data_plot_mid.pic'.format(output_path)

    filename_high = '{0}/interpretation/rf_vis_high_tg.png'.format(output_path)
    filename_low = '{0}/interpretation/rf_vis_low_tg.png'.format(output_path)
    filename_mid = '{0}/interpretation/rf_vis_mid_tg.png'.format(output_path)

    # High Tg plot
    if os.path.isfile(dp_h):
        with open(dp_h, 'rb') as f:
            plot_data, relevances = pickle.load(f)
    else:
        plot_data, relevances = extract_intervals_with_data(
            rf, data, 1200, 2000, features_names
        )
        with open(dp_h, 'wb') as f:
            pickle.dump(file=f, obj=(plot_data, relevances), protocol=-1)
    plot_violins(plot_data, relevances, filename_high)

    # Low Tg plot
    if os.path.isfile(dp_l):
        with open(dp_l, 'rb') as f:
            plot_data, relevances = pickle.load(f)
    else:
        plot_data, relevances = extract_intervals_with_data(
            rf, data, 0, 400, features_names
        )
        with open(dp_l, 'wb') as f:
            pickle.dump(file=f, obj=(plot_data, relevances), protocol=-1)

    plot_violins(plot_data, relevances, filename_low)

    # Middle Tg plot
    if os.path.isfile(dp_m):
        with open(dp_m, 'rb') as f:
            plot_data, relevances = pickle.load(f)
    else:
        plot_data, relevances = extract_intervals_with_data(
            rf, data, 400, 1200, features_names
        )
        with open(dp_m, 'wb') as f:
            pickle.dump(file=f, obj=(plot_data, relevances), protocol=-1)

    plot_violins(plot_data, relevances, filename_mid)

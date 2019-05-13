import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from constants import DATA_PATH as input_path


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


def extract_intervals(rf, min, max, features_names, resolution=100,
                      range_features=(0, 1)):
    incr = (range_features[1] - range_features[0]) / resolution
    n_features = len(features_names) - 1
    paths = []
    for model in rf.estimators_:
        paths.extend(get_tree_paths(model.tree_, min, max))

    intervals = [np.zeros(resolution) for i in range(n_features)]
    for path in paths:
        for elem in path:
            if elem[0] is not None:
                point = int(round(elem[2] / incr))
                # TODO: verify whether this is necessary
                if point > resolution:
                    point = resolution

                if elem[0]:
                    for i in range(point):
                        intervals[elem[1]][i] += 1
                else:
                    for i in range(point, resolution):
                        intervals[elem[1]][i] += 1
    plot_data = {
        'label': [],
        'dist': []
    }
    for feat, interval in enumerate(intervals):
        for i, elem in enumerate(interval):
            plot_data['label'].extend(
                [features_names[feat] for j in range(int(elem))]
            )
            plot_data['dist'].extend(np.repeat(i * incr, elem).tolist())

    return plot_data


def plot_violins(plot_data):
    dt_plot = pd.DataFrame.from_dict(plot_data)
    sns.set(style='whitegrid')
    f, ax = plt.subplots(figsize=(10, 5))
    sns.violinplot(
        x='label', y='dist', data=dt_plot,
        inner='box', palette='Set3', cut=2, linewidth=3,
        scale='count'
    )
    sns.despine(left=True)

    f.suptitle('Composition visualization', fontsize=18, fontweight='bold')
    ax.set_xlabel('Features', size=16, alpha=0.7)
    ax.set_ylabel('Amount', size=16, alpha=0.7)
    plt.show()


with open('../../../predicting_high_low_TG/result/rf/default_rf_tg_fold01.model', 'rb') as f:
    rf = pickle.load(f)

data = pd.read_csv('{0}/data_tg_dupl_rem.csv'.format(input_path))
features_names = list(data)

plot_data = extract_intervals(rf, 0, 400, features_names)
plot_violins(plot_data)

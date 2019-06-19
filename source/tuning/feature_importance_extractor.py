import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from constants import TARGETS_FORMATTED as targets
from constants import OUTPUT_PATH as output_prefix
from constants import DATA_PATH as data_path_prefix
from constants import N_FOLDS_OUTER as n_folds


output_path = '{}/interpretation'.format(output_prefix)

if not os.path.exists(output_path):
    os.makedirs(output_path)


for target, ftarget in targets.items():
    data_path = '{}/data_{}_dupl_rem.csv'.format(data_path_prefix, target)
    data = pd.read_csv(data_path)
    col_names = list(data)

    importances = []
    for k in range(1, n_folds + 1):
        model_path = '{}/rf/best_rf_{}_fold{:02d}.model'.format(
            output_prefix, target, k
        )

        with open(model_path, 'rb') as f:
            forest = pickle.load(f)

        importances.append(forest.feature_importances_)
    mean_imp = np.mean(importances, axis=0)
    std_imp = np.std(importances, axis=0)
    indices = np.argsort(mean_imp)[::-1]

    ordered_feat = [col_names[i] for i in indices]
    # Plot the feature importances of the forest
    plt.figure(figsize=(10, 4))
    plt.style.use('seaborn-whitegrid')
    plt.title("Feature importances", fontsize=12)
    plt.bar(range(len(mean_imp)), mean_imp[indices],
            color='darkgreen', yerr=std_imp[indices], align='center', log=True,
            capsize=1.5)
    plt.xticks(range(len(mean_imp)), ordered_feat, rotation=90, fontsize=8)
    plt.xlim([-1, len(mean_imp)])
    plt.ylabel('log(Importance)', fontsize=10)

    x_tick_pos, _ = plt.xticks()
    plt.bar(
        x_tick_pos, [max(plt.yticks()[0])] * len(x_tick_pos),
        width=(x_tick_pos[1] - x_tick_pos[0]),
        color=['lightgray', 'white']
    )

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'importance_{}.eps'.format(target)),
                dpi=500)

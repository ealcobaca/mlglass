import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

output_path = '../../result/interpretation'
targets = {
    'tg': 'Tg',
    'nd300': 'ND300',
    'tl': 'Tliquidus'
}

for target, ftarget in targets.items():
    data_path = '../../data/clean/oxides_{}_test.csv'.format(ftarget)
    model_path = '../../result/rf/default_rf_{}.model'.format(target)

    data = pd.read_csv(data_path)
    col_names = list(data)

    with open(model_path, 'rb') as f:
        forest = pickle.load(f)

    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    ordered_feat = [col_names[i] for i in indices]
    # Plot the feature importances of the forest
    plt.figure(figsize=(10, 4))
    plt.style.use('seaborn-whitegrid')
    plt.title("Feature importances", fontsize=12)
    plt.bar(range(len(importances)), importances[indices],
            color='darkgreen', yerr=std[indices], align='center', log=True,
            capsize=1.5)
    plt.xticks(range(len(importances)), ordered_feat, rotation=90, fontsize=8)
    plt.xlim([-1, len(importances)])
    plt.ylabel('log(Importance)', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'importance_{}.eps'.format(target)),
                dpi=500)
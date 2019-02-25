import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


input_prefix = '../../result'
output_folder = '../../result/line_plots'
target = 'Tg'

standard = pd.read_csv(os.path.join(input_prefix, 'predictions_standard_models.csv'))
best = pd.read_csv(os.path.join(input_prefix, 'predictions_best_models.csv'))

regressors = {
    'dt': 'DT',
    'knn': 'k-NN',
    'mlp': 'MLP',
    'svr': 'SVR',
    'rf': 'RF',
    # 'dt': 'DT',
}

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for reg, freg in regressors.items():
    plt.subplot(2, 1, 1)
    order = np.argsort(best.loc[:, reg])
    x = standard.loc[:, reg]
    x_min, x_max = (min(x), max(x))
    y = standard.loc[:, '{}_pred'.format(reg)]
    y_min, y_max = (min(y), max(y))

    plt.scatter(x[order], y[order], s=3)
    plt.plot([x_min, x_max], [y_min, y_max], c='black')
    plt.xlabel('Observed $T_g$')
    plt.ylabel('Predicted $T_g$')
    plt.title('{} - Default model'.format(freg))
    plt.grid(True)

    plt.subplot(2, 1, 2)
    order = np.argsort(best.loc[:, reg])
    x = best.loc[:, reg]
    x_min, x_max = (min(x), max(x))
    y = best.loc[:, '{}_pred'.format(reg)]
    y_min, y_max = (min(y), max(y))

    plt.scatter(x[order], y[order], s=3)
    plt.plot([x_min, x_max], [y_min, y_max], c='black')
    plt.xlabel('Observed $T_g$')
    plt.ylabel('Predicted $T_g$')
    plt.title('{} - Tuned model'.format(freg))
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_folder, 'scatter_{}_{}.png'.format(reg, target)),
        dpi=500
    )
    plt.clf()

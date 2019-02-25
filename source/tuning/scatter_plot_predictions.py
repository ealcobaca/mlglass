import os
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def roundup(x, fractionary=False):
    if fractionary:
        return int(math.ceil(x * 100.0)) / 100 + 0.1
    else:
        return int(math.ceil(x / 100.0)) * 100 + 100


def rounddown(x, fractionary=False):
    if fractionary:
        return int(math.floor(x * 100.0)) / 100 - 0.1
    else:
        return int(math.floor(x / 100.0)) * 100 - 100


input_prefix = '../../result'
output_folder = '../../result/line_plots'
targets = {
    'tg': '$T_g$',
    'nd300': 'ND300',
    'tl': 'Tliquidus'
}

regressors = {
    'dt': 'DT',
    'knn': 'k-NN',
    'mlp': 'MLP',
    'svr': 'SVR',
    'rf': 'RF',
}

cmap = 'gist_heat_r'


def relative_deviation(obs, pred):
    return np.abs(obs-pred)/obs


if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for target, ftarget in targets.items():
    fract = target == 'nd300'
    incr = 0.1 if fract else 100
    standard = pd.read_csv(
        os.path.join(
            input_prefix,
            'predictions_standard_models_{}.csv'.format(target)
        )
    )
    best = pd.read_csv(
        os.path.join(
            input_prefix,
            'predictions_best_models_{}.csv'.format(target)
        )
    )
    for reg, freg in regressors.items():
        colors_standard = relative_deviation(
            standard.loc[:, reg],
            standard.loc[:, '{}_pred'.format(reg)]
        )
        colors_best = relative_deviation(
            best.loc[:, reg],
            best.loc[:, '{}_pred'.format(reg)]
        )
        min_color = np.min(
            [np.min(colors_standard), np.min(colors_best)]
        )
        max_color = np.max(
            [np.max(colors_standard), np.max(colors_best)]
        )
        colors_standard = np.array([
            (c - min_color)/(max_color - min_color) for c in colors_standard
        ])
        colors_best = np.array([
            (c - min_color)/(max_color - min_color) for c in colors_best
        ])

        fig, axes = fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True,
                                           sharey=True, figsize=(10, 5))
        ax = plt.subplot(121, aspect='equal')
        order = np.argsort(best.loc[:, reg])
        x = standard.loc[:, reg]
        x_min, x_max = (min(x), max(x))
        y = standard.loc[:, '{}_pred'.format(reg)]

        plt.scatter(x[order], y[order], s=3, c=colors_standard[order],
                    cmap=cmap, vmin=0, vmax=1)
        plt.plot([x_min, x_max], [x_min, x_max], c='black')
        plt.xlabel(None)
        plt.ylabel(None)
        plt.title('Default model', fontsize=8)
        plt.xlim(rounddown(x_min, fract), roundup(x_max, fract))
        plt.ylim(rounddown(x_min, fract), roundup(x_max, fract))
        ax.set_xticks(
            np.arange(rounddown(x_min, fract), roundup(x_max, fract), incr)
        )
        ax.set_yticks(
            np.arange(rounddown(x_min, fract), roundup(x_max, fract), incr)
        )
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=8)
        ax.tick_params(axis='x', labelrotation=45)
        plt.grid(True)

        ax = plt.subplot(122, aspect='equal')
        order = np.argsort(best.loc[:, reg])
        x = best.loc[:, reg]
        x_min, x_max = (min(x), max(x))
        y = best.loc[:, '{}_pred'.format(reg)]

        aux_cb = plt.scatter(x[order], y[order], s=3, c=colors_best[order],
                             cmap=cmap, vmin=0, vmax=1)
        plt.plot([x_min, x_max], [x_min, x_max], c='black')
        plt.xlabel(None)
        plt.ylabel(None)
        plt.title('Tuned model', fontsize=8)
        plt.xlim(rounddown(x_min, fract), roundup(x_max, fract))
        plt.ylim(rounddown(x_min, fract), roundup(x_max, fract))
        ax.set_xticks(
            np.arange(rounddown(x_min, fract), roundup(x_max, fract), incr)
        )
        ax.set_yticks(
            np.arange(rounddown(x_min, fract), roundup(x_max, fract), incr)
        )
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=8)
        ax.tick_params(axis='x', labelrotation=45)
        plt.grid(True)

        fig.text(0.5, 0.9, '{}'.format(freg),
                 ha='center', va='center', fontsize=12)
        fig.text(0.5, 0.05, 'Observed {}'.format(ftarget),
                 ha='center', va='center')
        fig.text(0.07, 0.5, 'Predicted {}'.format(ftarget),
                 ha='center', va='center', rotation='vertical')

        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.91, 0.27, 0.01, 0.4])
        cbar = fig.colorbar(aux_cb, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=6)
        cbar.ax.set_title('RD', fontsize=8)

        plt.savefig(
            os.path.join(
                output_folder, 'scatter_{}_{}.png'.format(reg, target)
            ),
            dpi=500, bbox_inches='tight', pad_inches=0
        )
        plt.clf()

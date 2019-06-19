import os
import math
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import ColorbarBase
import numpy as np
import pandas as pd
from constants import TARGETS_FORMATTED as targets
from constants import REGRESSORS_FORMATTED as regressors
from constants import OUTPUT_PATH as folder_prefix
from constants import N_FOLDS_OUTER as n_folds


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


output_folder = '{}/line_plots'.format(folder_prefix)
input_prefix = '{}/logs'.format(folder_prefix)

cmap = 'gist_heat_r'


def relative_deviation(obs, pred):
    return np.abs(obs-pred)/obs * 100


if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for target, ftarget in targets.items():
    fract = target == 'nd300'
    incr = 0.1 if fract else 100
    df_standard = []
    df_tuned = []

    for k in range(1, n_folds + 1):
        standard = pd.read_csv(
            os.path.join(
                input_prefix,
                'predictions_standard_models_{}_fold{:02d}.csv'.format(
                    target, k
                )
            )
        )
        standard = standard.iloc[:, 1:]

        best = pd.read_csv(
            os.path.join(
                input_prefix,
                'predictions_best_models_{}_fold{:02d}.csv'.format(
                    target, k
                )
            )
        )
        best = best.iloc[:, 1:]

        df_standard.append(standard)
        df_tuned.append(best)
    standard = pd.concat(df_standard)
    best = pd.concat(df_tuned)

    for reg, freg in regressors.items():
        colors_standard = relative_deviation(
            standard.loc[:, reg],
            standard.loc[:, '{}_pred'.format(reg)]
        )
        colors_best = relative_deviation(
            best.loc[:, reg],
            best.loc[:, '{}_pred'.format(reg)]
        )

        fig, axes = fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True,
                                           sharey=True, figsize=(10, 5))
        ax = plt.subplot(121, aspect='equal')
        order = np.argsort(best.loc[:, reg])
        x = standard.loc[:, reg]
        x_min, x_max = (min(x), max(x))
        y = standard.loc[:, '{}_pred'.format(reg)]

        plt.scatter(x[order], y[order], s=3, c=colors_standard[order],
                    cmap=cmap, vmin=0, vmax=np.max(colors_standard))
        plt.plot([x_min, x_max], [x_min, x_max], c='black')
        plt.xlabel(None)
        plt.ylabel(None)
        plt.title('Default models', fontsize=10)
        plt.xlim(rounddown(x_min, fract), roundup(x_max, fract))
        plt.ylim(rounddown(x_min, fract), roundup(x_max, fract))
        ax.set_xticks(
            np.arange(rounddown(x_min, fract), roundup(x_max, fract), incr)
        )
        ax.set_yticks(
            np.arange(rounddown(x_min, fract), roundup(x_max, fract), incr)
        )
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=10)
        ax.tick_params(axis='x', labelrotation=45)
        plt.grid(True, linewidth=0.1)

        ax = plt.subplot(122, aspect='equal')
        order = np.argsort(best.loc[:, reg])
        x = best.loc[:, reg]
        x_min, x_max = (min(x), max(x))
        y = best.loc[:, '{}_pred'.format(reg)]

        plt.scatter(x[order], y[order], s=3, c=colors_best[order],
                    cmap=cmap, vmin=0, vmax=np.max(colors_best))
        plt.plot([x_min, x_max], [x_min, x_max], c='black')
        plt.xlabel(None)
        plt.ylabel(None)
        plt.title('Tuned models', fontsize=10)
        plt.xlim(rounddown(x_min, fract), roundup(x_max, fract))
        plt.ylim(rounddown(x_min, fract), roundup(x_max, fract))
        ax.set_xticks(
            np.arange(rounddown(x_min, fract), roundup(x_max, fract), incr)
        )
        ax.set_yticks(
            np.arange(rounddown(x_min, fract), roundup(x_max, fract), incr)
        )
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=10)
        ax.tick_params(axis='x', labelrotation=45)
        plt.grid(True, linewidth=0.1)

        fig.text(0.5, 0.9, '{}'.format(freg),
                 ha='center', va='center', fontsize=12)
        fig.text(0.5, 0.05, 'Observed {}'.format(ftarget),
                 ha='center', va='center')
        fig.text(0.07, 0.5, 'Predicted {}'.format(ftarget),
                 ha='center', va='center', rotation='vertical')

        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.91, 0.27, 0.01, 0.4])
        cbar = ColorbarBase(
            cbar_ax, norm=Normalize(vmin=0, vmax=100), cmap=cmap
        )
        cbar.ax.set_yticklabels(['{}%'.format(tick) for tick in
                                 np.arange(0, 110, 20)])
        cbar.ax.tick_params(labelsize=8)
        cbar.ax.set_title('RD', fontsize=10)

        plt.savefig(
            os.path.join(
                output_folder, 'scatter_{}_{}.png'.format(reg, target)
            ),
            dpi=500, bbox_inches='tight', pad_inches=0
        )
        plt.clf()

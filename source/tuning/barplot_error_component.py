import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import OrderedDict


def relative_deviation(obs, pred):
    return np.sum(np.abs(obs-pred)/obs)/len(obs) * 100


def get_predictions(model_path, data):
    with open(os.path.join(model_path), 'rb') as f:
        regressor = pickle.load(f)
        predictions = regressor.predict(data)

    return predictions


def main(targets, regressors, output_path):
    colors = ['#d7191c', '#fdae61', '#ffffbf', '#abd9e9', '#2c7bb6']
    bar_width = 2.2/len(regressors)
    offset = len(regressors) * bar_width / 0.8
    for target, ftarget in targets.items():
        data_path = '../../data/clean/oxides_{}_test.csv'.format(ftarget)
        data = pd.read_csv(data_path)
        elements = list(data)[:-1]
        data = data.values
        n_examples = []
        for j, elem in enumerate(elements):
            n_examples.append(len(np.nonzero(data[:, j])[0]))

        plt.figure(figsize=(10, 3.5))
        plt.style.use('seaborn-whitegrid')
        plt.title("Error per element", fontsize=12)
        bar_heights = OrderedDict()
        for k, (regressor, (type, _)) in enumerate(regressors.items()):
            model_path = '{0}/{1}/{2}_{1}_{3}.model'.format(
                output_path, regressor, type, target
            )
            errors = []
            for j, elem in enumerate(elements):
                idx_test = np.nonzero(data[:, j])[0]
                if len(idx_test) > 0:
                    data_test = data[idx_test, :]
                    preds = get_predictions(model_path, data_test[:, :-1])
                    error = relative_deviation(data_test[:, -1], preds)
                    errors.append(error)
                else:
                    errors.append(0.0)
            x_values = [b + k * bar_width + offset * b
                        for b in range(len(elements))]
            barplot = plt.bar(x_values, errors,
                              color=colors[k], align='center', width=bar_width,
                              capsize=1.5, edgecolor='black', linewidth=0.1)
            for elem, bar in zip(elements, barplot):
                if elem not in bar_heights:
                    bar_heights[elem] = [bar.get_height()]
                else:
                    bar_heights[elem].append(bar.get_height())
        x_values = [b + (len(regressors) * bar_width)/2.0 + b * offset
                    for b in range(len(elements))]
        plt.xticks(x_values, elements,
                   rotation=90, fontsize=8)
        plt.yticks(np.arange(0, 22.5, 2.5),
                   ['0.0%', '2.5%', '5.0%', '7.5%', '10.0%', '12.5%',
                    '15.0%', '17.5%', '20.0%'],
                   fontsize=8)
        plt.xlim([-1 - offset, len(elements) * (1 + offset) + (len(regressors)
                  * bar_width)/2.0])
        plt.ylim([0, 22.5])
        plt.ylabel('RD', fontsize=10)
        for j, heights in enumerate(bar_heights.values()):
            height = np.max(heights)
            plt.text(j + (len(regressors) * bar_width)/2.0 + j * offset -
                     bar_width/2.0,
                     height + 0.2,
                     n_examples[j], ha='center', va='bottom',
                     fontsize=6, rotation=90)
        handles = [mpatches.Patch(color=colors[k], label=regressors[reg][1])
                   for k, reg in enumerate(regressors.keys())]
        labels = [val[1] for val in regressors.values()]
        plt.legend(handles=handles, labels=labels, prop={'size': 6})
        ax = plt.gca()
        ax.xaxis.grid(False)
        ax.yaxis.grid(True)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                output_path,
                'errors_per_element/errors_{}.eps'.
                format(target)
            ),
            dpi=500
        )
        plt.close()


targets = {
    'tg': 'Tg',
    # 'nd300': 'ND300',
    # 'tl': 'Tliquidus'
}

regressors = {
    'dt': ('default', 'DT'),
    'knn': ('best', 'k-NN'),
    'mlp': ('best', 'MLP'),
    'rf': ('default', 'RF'),
    'svr': ('best', 'SVR')
}
output_path = '../../result'


if __name__ == '__main__':
    main(targets, regressors, output_path)

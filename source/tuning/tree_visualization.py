import os
import pickle
import numpy as np
import pandas as pd
from collections import OrderedDict
from constants import N_FOLDS_OUTER as n_folds
from constants import TARGETS_LIST as targets
from constants import OUTPUT_PATH as output_path
from constants import SPLIT_DATA_PATH as dataset_path


def relative_deviation(obs, pred):
    return np.abs(obs-pred)/obs * 100


def simplify_rules(tests):
    hash = OrderedDict()
    for test in tests:
        aux = test.split(' ')
        if not (aux[0], aux[1]) in hash:
            hash[(aux[0], aux[1])] = [float(aux[2])]
        else:
            hash[(aux[0], aux[1])].append(float(aux[2]))
    rule_set = []
    for (elem, signal), values in hash.items():
        if signal == '$>$':
            value = np.max(values)
        else:
            value = np.min(values)
        rule = '{0} {1} {2:.3f}'.format(elem, signal, value)
        rule_set.append(rule)
    return rule_set


def path2latex_formula(estimator, features, sample):
    if len(sample.shape) > 1 and sample.shape[1] > 1:
        print('Subject one sample at a time.')
        return
    if len(sample.shape) == 1:
        sample = sample.reshape(1, -1)

    feature = estimator.tree_.feature
    threshold = estimator.tree_.threshold

    node_indicator = estimator.decision_path(sample)
    leave_id = estimator.apply(sample)

    node_index = node_indicator.indices[node_indicator.indptr[0]:
                                        node_indicator.indptr[1]]
    decisions = []
    for node_id in node_index:
        if leave_id[0] == node_id:
            continue

        if (sample[0, feature[node_id]] <= threshold[node_id]):
            threshold_sign = '$\\le$'
        else:
            threshold_sign = '$>$'

        decisions.append(
            '\\textit{{{0}}} {1} {2}'.format(
                features[feature[node_id]], threshold_sign, threshold[node_id]
            )
        )
    decisions = simplify_rules(decisions)
    rule = ' $\\wedge$ '.join(decisions)
    return rule


def generate_path_table(obs, preds, rds, paths, target, output_dir, type):
    with open(
            os.path.join(output_dir, 'path_table_{0}_{1}.tex'.format(
                    target, type
                )
            ),
            'w'
         ) as f:
        f.write('% Add to the document preamble: \\usepackage{array}\n')
        f.write('\\begin{table}[!htbp]\n')
        f.write('\t\\setlength{\\tabcolsep}{3pt}\n')
        f.write('\t\\begin{tabular}{ccccm{0.7\\textwidth}}\n')
        f.write('\t\t\\toprule\n')
        f.write('\t\tid & $T_g$ & $\\hat{T_g}$ & RD (\\%) & Tree branch\\\\\n')
        f.write('\t\t\\midrule\n')

        for i in range(len(obs)):
            line = '{:02d} & {:.2f} & {:.2f} & {:.2f} & {}\\\\'.format(
                i + 1, obs[i], preds[i], rds[i], paths[i]
            )
            f.write('\t\t' + line + '\n')
            if i < len(obs) - 1:
                f.write('\t\t\\hline\n')
        f.write('\t\t\\bottomrule\n')
        f.write('\t\\end{tabular}\n')
        f.write('\\end{table}\n')


def main(dataset_path, model_path, output_dir, targets, type='default'):
    obs = []
    preds = []
    rds = []
    paths = []

    for target in targets:
        for k in range(1, n_folds + 1):
            with open(
                    '{0}/dt/{1}_dt_{2}_fold{3:02d}.model'.format(
                        model_path, type, target, k
                    ), 'rb'
                 ) as f:
                estimator = pickle.load(f)
            data = pd.read_csv('{0}/{1}_test_fold{2:02d}_extreme.csv'.format(
                    dataset_path, target, k
                )
            )
            X = data.values[:, :-1]
            y = data.values[:, -1]
            y_pred = estimator.predict(X)
            obs.extend(y.tolist())
            preds.extend(y_pred.tolist())
            rds.extend(relative_deviation(y, y_pred).tolist())
            paths.extend([
                path2latex_formula(
                    estimator=estimator, features=list(data), sample=X[i]
                ) for i in range(len(X))
            ])

        generate_path_table(obs, preds, rds, paths, target, output_dir, type)


output_dir = os.path.join(output_path, 'interpretation')

if __name__ == '__main__':
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    main(dataset_path, output_path, output_dir, targets)
    main(dataset_path, output_path, output_dir, targets, 'best')

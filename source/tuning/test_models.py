import numpy as np
import pandas as pd
import pickle
import os
from collections import OrderedDict


def relative_deviation(obs, pred):
    return np.sum(np.abs(obs-pred)/obs)/len(obs) * 100


def R2(target, pred):
    return np.corrcoef(target, pred)[0, 1] ** 2


def RRMSE(target, pred):
    num = np.sum((target - pred) ** 2)
    den = np.sum((np.mean(target) - target) ** 2)
    return np.sqrt(num/den)


def RMSE(target, pred):
    N = len(target)
    return np.sqrt(np.sum((target-pred)**2)/N)


def evaluate_models(data_path, output_path, regressors, target, metrics,
                    type='default'):
    errors = np.zeros((len(metrics), len(regressors)))
    test_data = pd.read_csv(data_path).values
    for j, regressor in enumerate(regressors):
        with open(
            os.path.join(
                output_path, regressor,
                '{}_{}_{}.model'.format(type, regressor, target)
            ), 'rb'
        ) as f:
            regressor = pickle.load(f)
            predictions = regressor.predict(test_data[:, :-1])
        for i, metric in enumerate(metrics.values()):
            errors[i, j] = metric(test_data[:, -1], predictions)
    df = pd.DataFrame(
        index=[m for m in metrics.keys()],
        columns=regressors,
        data=errors
    )
    return df


def get_predictions(data_path, output_path, regressors, target,
                    type='default'):
    test_data = pd.read_csv(data_path).values
    log = np.zeros((len(test_data), 2 * len(regressors)))
    for j, regressor in enumerate(regressors):
        with open(
            os.path.join(
                output_path, regressor,
                '{}_{}_{}.model'.format(type, regressor, target)
            ), 'rb'
        ) as f:
            regressor = pickle.load(f)
            log[:, 2*j] = test_data[:, -1]
            log[:, 2*j+1] = regressor.predict(test_data[:, :-1])
    columns = [[r, '{}_pred'.format(r)] for r in regressors]
    columns = [n for subset in columns for n in subset]
    df = pd.DataFrame(
        columns=columns,
        data=log
    )
    return df


def main(data_path, output_path, regressors, target, metrics):
    test_path = '{}.csv'.format(data_path)
    errors_standard = evaluate_models(test_path, output_path, regressors,
                                      target, metrics)
    errors_best = evaluate_models(test_path, output_path, regressors,
                                  target, metrics, 'best')
    errors_standard.to_csv(
        os.path.join(
            output_path, 'performance_standard_models_{}.csv'.format(target)
        )
    )
    errors_best.to_csv(
        os.path.join(
            output_path, 'performance_best_models_{}.csv'.format(target)
        )
    )

    pred_standard = get_predictions(test_path, output_path, regressors,
                                    target)
    pred_best = get_predictions(test_path, output_path, regressors,
                                target, 'best')
    pred_standard.to_csv(
        os.path.join(
            output_path, 'predictions_standard_models_{}.csv'.format(target)
        )
    )
    pred_best.to_csv(
        os.path.join(
            output_path, 'predictions_best_models_{}.csv'.format(target)
        )
    )

    ext_test_path = '{}_rem.csv'.format(data_path)
    pred_standard = get_predictions(ext_test_path, output_path, regressors,
                                    target)
    pred_best = get_predictions(ext_test_path, output_path, regressors,
                                target, 'best')
    pred_standard.to_csv(
        os.path.join(
            output_path,
            'predictions_extremes_standard_models_{}.csv'.format(target)
        )
    )
    pred_best.to_csv(
        os.path.join(
            output_path,
            'predictions_extremes_best_models_{}.csv'.format(target)
        )
    )


targets = {
    'tg': 'Tg',
    'nd300': 'ND300',
    'tl': 'Tliquidus'
}

regressors = ['dt', 'knn', 'mlp', 'rf', 'svr']
metrics = OrderedDict(
    {'relative_deviation': relative_deviation, 'R2': R2, 'RMSE': RMSE,
     'RRMSE': RRMSE}
)
output_path = '../../result'


if __name__ == '__main__':
    for target, ftarget in targets.items():
        data_path = '../../data/clean/oxides_{}_test'.format(ftarget)
        main(data_path, output_path, regressors, target, metrics)

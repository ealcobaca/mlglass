import os
import pickle
import numpy as np
import pandas as pd
from collections import OrderedDict
from constants import TARGETS_LIST as targets
from constants import REGRESSORS_LIST as regressors
from constants import OUTPUT_PATH as output_path
from constants import SPLIT_DATA_PATH as data_path
from constants import N_FOLDS_OUTER as n_folds


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


def evaluate_models(input_path, output_path, regressors, target, metrics,
                    fold, type='default'):
    errors = np.zeros((len(metrics), len(regressors)))
    test_data = pd.read_csv(input_path).values
    for j, regressor in enumerate(regressors):
        with open(
            os.path.join(
                output_path, regressor,
                '{0}_{1}_{2}_fold{3:02d}.model'.format(
                    type, regressor, target, fold
                )
            ), 'rb'
        ) as f:
            regressor = pickle.load(f)
            if not isinstance(regressor, dict):
                predictions = regressor.predict(test_data[:, :-1])
            elif regressor['scaler'] is None:
                predictions = regressor['model'].predict(test_data[:, :-1])
            else:
                predictions = regressor['scaler'].inverse_transform(
                    regressor['model'].predict(
                        regressor['scaler'].transform(
                            test_data[:, :-1]
                        )
                    ).reshape(-1, 1)
                )[:, 0]

        for i, metric in enumerate(metrics.values()):
            errors[i, j] = metric(test_data[:, -1], predictions)
    df = pd.DataFrame(
        index=[m for m in metrics.keys()],
        columns=regressors,
        data=errors
    )
    return df


def get_predictions(input_path, output_path, regressors, target, fold,
                    type='default'):
    test_data = pd.read_csv(input_path).values
    log = np.zeros((len(test_data), 2 * len(regressors)))
    for j, regressor in enumerate(regressors):
        with open(
            os.path.join(
                output_path, regressor,
                '{0}_{1}_{2}_fold{3:02d}.model'.format(
                    type, regressor, target, fold
                )
            ), 'rb'
        ) as f:
            regressor = pickle.load(f)
            log[:, 2*j] = test_data[:, -1]
            if not isinstance(regressor, dict):
                log[:, 2*j+1] = regressor.predict(test_data[:, :-1])
            elif regressor['scaler'] is None:
                log[:, 2*j+1] = regressor['model'].predict(test_data[:, :-1])
            else:
                log[:, 2*j+1] = regressor['scaler'].inverse_transform(
                    regressor['model'].predict(
                        regressor['scaler'].transform(
                            test_data[:, :-1]
                        )
                    ).reshape(-1, 1)
                )[:, 0]
    columns = [[r, '{}_pred'.format(r)] for r in regressors]
    columns = [n for subset in columns for n in subset]
    df = pd.DataFrame(
        columns=columns,
        data=log
    )
    return df


def generate4fold(input_path, output_path, log_path, regressors, target,
                  metrics, fold):
    print('Fold {}'.format(fold))
    test_path = '{}fold{:02d}.csv'.format(input_path, fold)
    errors_standard = evaluate_models(test_path, output_path, regressors,
                                      target, metrics, fold)
    errors_best = evaluate_models(test_path, output_path, regressors,
                                  target, metrics, fold, 'best')
    errors_standard.to_csv(
        os.path.join(
            log_path,
            'performance_standard_models_{0}_fold{1:02d}.csv'.format(
                target, fold
            )
        )
    )
    errors_best.to_csv(
        os.path.join(
            log_path,
            'performance_best_models_{0}_fold{1:02d}.csv'.format(
                target, fold
            )
        )
    )

    pred_standard = get_predictions(test_path, output_path, regressors,
                                    target, fold)
    pred_best = get_predictions(test_path, output_path, regressors,
                                target, fold, 'best')
    pred_standard.to_csv(
        os.path.join(
            log_path,
            'predictions_standard_models_{0}_fold{1:02d}.csv'.format(
                target, fold
            )
        )
    )
    pred_best.to_csv(
        os.path.join(
            log_path,
            'predictions_best_models_{0}_fold{1:02d}.csv'.format(
                target, fold
            )
        )
    )

    # ext_test_path = '{0}extreme.csv'.format(input_path)
    # pred_standard = get_predictions(ext_test_path, output_path, regressors,
    #                                 target, fold)
    # pred_best = get_predictions(ext_test_path, output_path, regressors,
    #                             target, fold, 'best')
    # pred_standard.to_csv(
    #     os.path.join(
    #         log_path,
    #         'predictions_extremes_standard_models_{0}_fold{1:02d}.csv'.format(
    #             target, fold
    #         )
    #     )
    # )
    # pred_best.to_csv(
    #     os.path.join(
    #         log_path,
    #         'predictions_extremes_best_models_{0}_fold{1:02d}.csv'.format(
    #             target, fold
    #         )
    #     )
    # )


def merge_errors(target, output_path, log_path, type='standard'):
    dfs = []

    for k in range(1, n_folds + 1):
        df = pd.read_csv(
            os.path.join(
                log_path,
                'performance_{0}_models_{1}_fold{2:02d}.csv'.format(
                    type, target, k
                )
            )
        )
        col_names = list(df)
        col_names[0] = 'metric'
        df.columns = col_names
        df = df.set_index('metric')
        df = df.assign(fold='fold{:02d}'.format(k))
        dfs.append(df)

    p = pd.concat(dfs)
    means = p.groupby(['metric']).mean()
    means.to_csv(
        os.path.join(
            output_path, 'mean_performance_{0}_{1}_all.csv'.format(
                type, target
            )
        )
    )
    stds = p.groupby('metric').std()
    stds.to_csv(
        os.path.join(
            output_path, 'std_performance_{0}_{1}_all.csv'.format(
                type, target
            )
        )
    )


metrics = OrderedDict(
    {'relative_deviation': relative_deviation, 'R2': R2, 'RMSE': RMSE,
     'RRMSE': RRMSE}
)


if __name__ == '__main__':
    print()
    print('Testing trained models')
    print()
    log_path = '{0}/logs'.format(output_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    for target in targets:
        for k in range(1, n_folds + 1):
            input_path = '{0}/{1}_test_'.format(
                data_path, target
            )
            generate4fold(
                input_path, output_path, log_path, regressors, target,
                metrics, k
            )
        merge_errors(target, output_path, log_path)
        merge_errors(target, output_path, log_path, 'best')

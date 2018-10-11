import numpy as np
import pandas as pd
import os
from sklearn import preprocessing
from sklearn.model_selection import KFold
from stacked_single_target import StackedSingleTarget
from utils import get_regressor, RMSE, RRMSE, aRMSE, aRRMSE


def run(data_path, out_path, datasets, regressor, n_parts=10,
        sst_method='predictions', sst_n_part=None, seed=2018):
    idx_column = ['fold_{:02d}'.format(k + 1) for k in range(n_parts)]
    idx_column.append('mean')
    for d, nt in datasets.items():
        print(d)
        dataset_file = os.path.join(data_path, '{}.csv'.format(d))
        data = pd.read_csv(dataset_file)
        target_names = list(data)[-nt:]
        # Transform data in numpy.ndarray
        data = data.values

        log_columns = ['armse', 'arrmse']
        log_columns.extend(['rmse_{}'.format(tn) for tn in target_names])
        log_columns.extend(['rrmse_{}'.format(tn) for tn in target_names])
        out_log = pd.DataFrame(
            np.zeros((n_parts + 1, 2 * nt + 2)),
            columns=log_columns
        )

        kf = KFold(n_splits=n_parts, shuffle=True, random_state=seed)
        for k, (train_index, test_index) in enumerate(kf.split(data)):
            print('Fold {:02d}'.format(k + 1))
            X_train, X_test = data[train_index, :-nt], data[test_index, :-nt]
            Y_train, Y_test = data[train_index, -nt:], data[test_index, -nt:]

            scaler = preprocessing.StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

            regr, params = get_regressor(regressor, seed=seed)
            sst = StackedSingleTarget(
                regressor=regr,
                regressor_params=params,
                method=sst_method,
                n_part=sst_n_part
            )

            sst.fit(X_train, Y_train)
            predictions = sst.predict(X_test)

            for t in range(nt):
                out_log.loc[k, 'rrmse_{}'.format(target_names[t])] = \
                    RRMSE(Y_test[:, t], predictions[:, t])
                out_log.loc[k, 'rmse_{}'.format(target_names[t])] = \
                    RMSE(Y_test[:, t], predictions[:, t])
            out_log.loc[k, 'arrmse'] = aRRMSE(Y_test, predictions)
            out_log.loc[k, 'armse'] = aRMSE(Y_test, predictions)

        for c in range(2*nt + 2):
            out_log.iloc[-1:, c] = np.mean(out_log.iloc[:-1, c])

        out_log.insert(0, 'partition', idx_column)

        if sst_method == 'predictions':
            log_name = 'results_sst_{}_{}_predictions.csv'.format(d, regressor)
        elif sst_method == 'internal_cv':
            log_name = 'results_sst_{}_{}_internal_cv_{}.csv'.\
                       format(d, regressor, sst_n_part)
        elif sst_method == 'targets_values':
            log_name = 'results_sst_{}_{}_targets_values.csv'.\
                       format(d, regressor)
        out_log.to_csv(os.path.join(out_path, log_name), index=False)


if __name__ == '__main__':
    data_path = '../../data/mtr'
    out_path = '../../result/mtr'

    datasets = {
        'oxides_Tg_Tliquidus_above_Tg_900': 2
    }
    regressor = 'RF'
    n_parts = 10
    sst_method = 'targets_values'
    sst_n_part = 10

    run(data_path=data_path,
        out_path=out_path,
        datasets=datasets,
        regressor=regressor,
        n_parts=n_parts,
        sst_method=sst_method,
        sst_n_part=sst_n_part,
        seed=2018)

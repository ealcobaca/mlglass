import numpy as np
import pandas as pd
import os
from sklearn import preprocessing
from stacked_single_target import StackedSingleTarget
from utils import get_regressor, RMSE, RRMSE


def run(data_path, out_path, training_dataset, n_targets, testing_datasets,
        regressor, out_log, sst_method='predictions', sst_n_part=None,
        seed=2018):

    print('Regressor: {}'.format(regressor))
    training_name = os.path.join(data_path, '{}.csv'.format(training_dataset))
    training_data = pd.read_csv(training_name)
    training_data = training_data.values
    X_train, Y_train = training_data[:, :-n_targets], \
        training_data[:, -n_targets:]

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)

    regr, params = get_regressor(regressor, seed=seed)
    sst = StackedSingleTarget(
        regressor=regr,
        regressor_params=params,
        method=sst_method,
        n_part=sst_n_part
    )
    sst.fit(X_train, Y_train)
    print('SST-{} done'.format(sst_method))

    for pos, d in enumerate(testing_datasets):
        regr, params = get_regressor(regressor, seed=seed)
        st = regr(**params)
        st.fit(X_train, Y_train[:, pos])
        print('ST done')

        testing_name = os.path.join(data_path, '{}.csv'.format(d))
        testing_data = pd.read_csv(testing_name)
        testing_data = testing_data.values

        X_test, y_test = testing_data[:, :-1], testing_data[:, -1:]

        X_test = scaler.transform(X_test)
        y_test = y_test[:, 0]
        st_predictions = st.predict(X_test)
        sst_predictions = sst.predict(X_test)

        out_log.loc[pos, 'st_rmse'] = \
            RMSE(y_test, st_predictions)
        out_log.loc[pos, 'st_rrmse'] = \
            RRMSE(y_test, st_predictions)
        out_log.loc[pos, 'sst_rmse'] = \
            RMSE(y_test, sst_predictions[:, pos])
        out_log.loc[pos, 'sst_rrmse'] = \
            RRMSE(y_test, sst_predictions[:, pos])

    if sst_method == 'predictions':
        log_name = 'results_sst_train_test_TgTliq_{}_predictions.csv'.\
                   format(regressor)
    elif sst_method == 'internal_cv':
        log_name = 'results_sst_train_test_TgTliq_{}_internal_cv_{}.csv'.\
                   format(regressor, sst_n_part)
    elif sst_method == 'targets_values':
        log_name = 'results_sst_train_test_TgTliq_{}_targets_values.csv'.\
                   format(regressor)
    out_log.to_csv(os.path.join(out_path, log_name), index=False)


if __name__ == '__main__':
    data_path = '../../data/mtr'
    out_path = '../../result/mtr'

    training_dataset = 'train_oxides_Tg_Tliquidus'
    n_targets = 2

    sst_n_part = 10

    testing_datasets = ['test_oxides_Tg', 'test_oxides_Tliquidus']

    out_log = pd.DataFrame(
        np.zeros((len(testing_datasets), 4)),
        columns=['st_rmse', 'sst_rmse', 'st_rrmse', 'sst_rrmse']
    )
    regressors = ['RF', 'DT', 'MLP']
    sst_methods = ['predictions', 'internal_cv', 'targets_values']

    out_log.insert(0, 'target', ['Tg', 'Tliquidus'])
    for regressor in regressors:
        for sst_method in sst_methods:
            run(data_path=data_path,
                out_path=out_path,
                training_dataset=training_dataset,
                n_targets=n_targets,
                testing_datasets=testing_datasets,
                regressor=regressor,
                out_log=out_log,
                sst_method=sst_method,
                sst_n_part=sst_n_part,
                seed=2018)

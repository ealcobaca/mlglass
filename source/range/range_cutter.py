from sklearn.model_selection import RepeatedKFold
import pandas as pd
import numpy as np
from regressors import train_regressors, apply_regressors, compute_performance

SEED=123

def fill_data(X, y, value, grather=True):
    if grather:
        return X[y >= value], y[y>=value]
    return X[y <= value], y[y<=value]


def evaluate(X, y, rcv, range_high_TG, range_low_TG):

    results_perf = []
    results_perf_high = []
    results_perf_low = []

    for train_index, test_index in rcv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        train_regr = train_regressors(X_train, y_train)
        preds = apply_regressors(X_test)
        result_perf = compute_performance(preds, y_test)

        results_perf.append(result_perf)

        aux = []
        for data_range in range_high_TG:
            X_range_test, y_range_test = fill_data(X_test, y_test, data_range, True)
            preds = apply_regressors(X_range_test)
            result_perf = compute_performance(preds, y_range_test)
            aux.append(result_perf)
        results_perf_high.append(aux)

        aux = []
        for data_range in range_low_TG:
            X_range_test, y_range_test = fill_data(X_test, y_test, data_range, False)
            preds = apply_regressors(X_range_test)
            result_perf = compute_performance(preds, y_range_test)
            aux.append(result_perf)
        results_perf_low.append(aux)


def evaluate_range(X, y, rcv, value, grather):

    results_perf = []
    for train_index, test_index in rcv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_range_train, y_range_train = fill_data(X_train, y_train, value, grather)
        X_range_test, y_range_test = fill_data(X_test, y_test, value, grather)

        train_regr = train_regressors(X_range_train, y_range_train)
        preds = apply_regressors(X_range_test)
        result_perf = compute_performance(preds, y_range_test)

        results_perf.append(result_perf)

    return results_perf


def run(data_path, str_class):
    data = pd.read_csv(data_path)
    rcv = RepeatedKFold(n_splits=10, n_repeats=100, random_state=SEED)

    range_high_TG = np.arange(start=900, stop=1150, step=25)
    range_low_TG = np.arange(start=350, stop=650+1, step=25)

    X = data.drop([str_class], axis=1).values
    y = data[str_class].values

    evaluate(X, y, rcv, range_high_TG, range_low_TG)

    for data_range in range_high_TG:
        evaluate_range(X, y, rcv, data_range, True)

    for data_range in range_low_TG:
        evaluate_range(X, y, rcv, data_range, True)



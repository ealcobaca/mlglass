from sklearn.model_selection import RepeatedKFold
import pandas as pd
import numpy as np
import os
import pickle
from multiprocessing import Pool
from regressors import train_regressors, apply_regressors, compute_performance

SEED=123

def fill_data(X, y, value, grather=True):
    if grather:
        return X[y >= value], y[y>=value]
    return X[y <= value], y[y<=value]


def evaluate(X, y, alg, rcv, range_high_TG, range_low_TG, file_names):

    results_perf = []
    results_perf_high = []
    results_perf_low = []

    for train_index, test_index in rcv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        train_regr = train_regressors(X_train, y_train, alg, np.random.randint(1,10000))
        preds = apply_regressors(train_regr, X_test)
        result_perf = compute_performance(preds, y_test)

        results_perf.append(result_perf)

        aux = []
        for data_range in range_high_TG:
            X_range_test, y_range_test = fill_data(X_test, y_test, data_range, True)
            preds = apply_regressors(train_regr, X_range_test)
            result_perf = compute_performance(preds, y_range_test)
            aux.append(result_perf)
        results_perf_high.append(aux)

        aux = []
        for data_range in range_low_TG:
            X_range_test, y_range_test = fill_data(X_test, y_test, data_range, False)
            preds = apply_regressors(train_regr, X_range_test)
            result_perf = compute_performance(preds, y_range_test)
            aux.append(result_perf)
        results_perf_low.append(aux)

    pickle.dump(results_perf, open(file_name[0], "wb" ))
    pickle.dump(results_perf_high, open(file_name[1], "wb" ))
    pickle.dump(results_perf_low, open(file_name[2], "wb" ))
    return

def evaluate_range(X, y, alg, rcv, value, grather, file_name):

    results_perf = []
    for train_index, test_index in rcv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_range_train, y_range_train = fill_data(X_train, y_train, value, grather)
        X_range_test, y_range_test = fill_data(X_test, y_test, value, grather)

        train_regr = train_regressors(X_range_train, y_range_train, alg, np.random.randint(1,10000))
        preds = apply_regressors(train_regr, X_range_test)
        result_perf = compute_performance(preds, y_range_test)

        results_perf.append(result_perf)

    pickle.dump(results_perf, open(file_name, "wb"))


def run(data_path, str_class, n_cpus):
    # algs = ["DT", "RF", "XG", "SVM", "MLP"]
    np.random.seed(SEED)
    algs = ["MLP", "SVM", "DT", "RF"]
    data = pd.read_csv(data_path)
    rcv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=SEED)

    #range_high_TG = np.arange(start=900, stop=1150+1, step=25)
    #range_low_TG = np.arange(start=400, stop=650+1, step=25)

    X = data.drop([str_class], axis=1).values
    y = data[str_class].values

    perceltil_inf = [5*i for i in range(1,9)]
    perceltil_sup = [(100-perceltil_inf[i]) for i in range(len(perceltil_inf))]
    range_high_TG = [np.percentile(y, perceltil_sup[i]) for i in range(len(perceltil_sup))]
    range_low_TG = [np.percentile(y, perceltil_inf[i]) for i in range(len(perceltil_inf))]
    

    # mean = np.mean(yy)
    # sd = np.std(yy, ddof=0)
    # y = (yy - np.mean(yy))/np.std(yy,ddof=0)
    # evaluate(X, y, rcv, range_high_TG, range_low_TG)

    pool = Pool(processes=n_cpus)
    results = []

    result_path = "../../result/result_high/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    for data_range in range_high_TG:
        for alg in algs:
            print("High - {0} - {1}".format(alg, data_range))
            file_name = result_path+"result_"+str(data_range)+"_"+alg+".csv"
            results.append(pool.apply_async(evaluate_range, (X, y, alg, rcv, data_range, True, file_name)))

    result_path = "../../result/result_low/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    for data_range in range_low_TG:
        for alg in algs:
            print("Low - {0} - {1}".format(alg, data_range))
            file_name = result_path+"result_"+str(data_range)+"_"+alg+".csv"
            # evaluate_range(X, y, rcv, data_range, False, file_name)
            results.append(pool.apply_async(evaluate_range, (X, y, alg, rcv, data_range, False, file_name)))


    result_path = "../../result/result_all/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    file_names = [None, None, None]
    for alg in algs:
        print("all - {0}".format(alg))
        file_names[0] = result_path+"result_all_"+alg+".csv"
        file_names[1] = result_path+"result_all_high_"+alg+".csv"
        file_names[2] = result_path+"result_all_low"+alg+".csv"
        results.append(pool.apply_async(evaluate, (X, y, alg, rcv, range_high_TG, range_low_TG, file_names)))

    pool.close()
    pool.join()

from sklearn.model_selection import RepeatedKFold
import pandas as pd
import numpy as np
import os
import pickle
from multiprocessing import Pool, Manager
from regressors import train_regressors, apply_regressors, compute_performance
# import line_profiler

SEED=123

def fill_data(X, y, low, high):
    X_high, y_high = X[y >= high], y[y >= high]
    X_low, y_low = X[y <= low], y[y <= low]

    aux = [i > low and i < high for i in y]
    X_middle, y_middle = X[aux], y[aux]

    return X_low, y_low, X_middle, y_middle, X_high, y_high


def get_train_regressors(X, y, alg, key, d, seed):
    if key not in d:
        d[key] = train_regressors(X, y, alg, seed)
    return d[key]


def apply_oracle(reg_high, reg_middle, reg_low, low, high, X_test, y_test):
    y_pred = np.zeros_like(y_test)
    for i in range(len(y_test)):
        if y_test[i] >= high:
            y_pred[i] = reg_high.predict([X_test[i]])
        elif y_test[i] <= low:
            y_pred[i] = reg_low.predict([X_test[i]])
        else:
            y_pred[i] = reg_middle.predict([X_test[i]])
    return np.array(y_pred)

# @profile
def evaluate(X, y, alg, rcv, ranges, file_path, d, folds=10):
    np.random.seed(SEED)
    alg_low, alg_middle, alg_high = alg
    low, high = ranges
    results_perf = []
    file_name = file_path+str(alg_low)+"_"+str(alg_middle)+"_"+str(alg_high)+"_"+str(low)+"_"+str(high)+"_.csv"

    # 10-fold CV
    k=0
    for (train_index, test_index), seed in zip(rcv.split(X), np.random.randint(1,10000, folds)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_low_train, y_low_train, X_middle_train, y_middle_train, X_high_train, y_high_train = fill_data(X_train, y_train, low, high)

        key = "{0}_{1}_{2}".format(alg_low, low, k)
        reg_low = get_train_regressors(X_low_train, y_low_train, alg_low, key, d, seed)
        key = "{0}_{1}-{2}_{3}".format(alg_middle, low, high, k)
        reg_middle = get_train_regressors(X_middle_train, y_middle_train, alg_middle, key, d, seed)
        key = "{0}_{1}_{2}".format(alg_high, high, k)
        reg_high = get_train_regressors(X_high_train, y_high_train, alg_high, key, d, seed)

        # Oracle
        X_low_test, y_low_test, X_middle_test, y_middle_test, X_high_test, y_high_test = fill_data(X_test, y_test, low, high)

        preds = apply_oracle(reg_high, reg_middle, reg_low, low, high, X_test, y_test)
        result_oracle = compute_performance(y_test, preds)
        result_oracle.append("oracle-all")
        results_perf.append(result_oracle)

        preds = apply_regressors(reg_low, X_low_test)
        result_low = compute_performance(y_low_test, preds)
        result_low.append("oracle-low")
        results_perf.append(result_low)

        preds = apply_regressors(reg_high, X_high_test)
        result_high = compute_performance(y_high_test, preds)
        result_high.append("oracle-high")
        results_perf.append(result_high)

        preds = apply_regressors(reg_middle, X_middle_test)
        result_middle = compute_performance(y_middle_test, preds)
        result_middle.append("oracle-middle")
        results_perf.append(result_middle)
        k += 1

    df = pd.DataFrame(data=results_perf,
                      columns=["mean_absolute_error", "mean_squared_error",
                               "r2_score", "RRMSE", "RMSE", "MARE", "R2", "type"])
    df.to_csv(file_name)


def run_eval(data_path, str_class, n_cpus):
    np.random.seed(SEED)
    algs = ["DT", "MLP", "RF"]
    # algs = ["MLP", "RF"]
    data = pd.read_csv(data_path)
    folds=10
    rep=1
    rcv = RepeatedKFold(n_splits=folds, n_repeats=rep, random_state=SEED)

    X = data.drop([str_class], axis=1).values
    y = data[str_class].values

    perceltil_inf = [1.5, 2.5, 3.5] +[5*i for i in range(1,8)]
    perceltil_sup = [(100-perceltil_inf[i]) for i in range(len(perceltil_inf))]
    range_high_TG = [np.percentile(y, perceltil_sup[i]) for i in range(len(perceltil_sup))]
    range_high_TG.reverse()
    range_low_TG = [np.percentile(y, perceltil_inf[i]) for i in range(len(perceltil_inf))]

    range_high_TG = np.round(range_high_TG, 2)
    range_low_TG = np.round(range_low_TG,2)

    result_path = "../../result/result_oracle/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    manager = Manager()
    d = manager.dict()
    counter = 0

    for low in range_low_TG:
        for high in range_high_TG:
            pool = Pool(processes=n_cpus)
            for alg_low in algs:
                for alg_middle in algs:
                    for alg_high in algs:
                       ranges = (low, high)
                       alg = (alg_low, alg_middle, alg_high)
                       # evaluate(X, y, alg, rcv, ranges, result_path, d, folds*rep)
                       pool.apply_async(evaluate, (X, y, alg, rcv, ranges, result_path, d, folds*rep))
                       counter = counter + 1
                       print(counter)
                       print()
            pool.close()
            pool.join()
    pickle.dump(d, open(result_path+"dic_regressors.data", "wb"))


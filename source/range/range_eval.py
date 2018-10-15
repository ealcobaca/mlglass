from sklearn.model_selection import RepeatedKFold
import pandas as pd
import numpy as np
import os
import pickle
from multiprocessing import Pool
from regressors import train_regressors, apply_regressors, compute_performance

SEED=123

def fill_data(X, y, low, high):
    X_high, y_high = X[y >= high], y[y >= high]
    X_low, y_low = X[y <= low], y[y <= low]

    aux = [i[0] > low and i[0] < high for i in y]
    X_middle, y_middle = X[aux], y[aux]

    return X_low, y_low, X_middle, y_middle, X_high, y_high


def apply_oracle(reg_high, reg_middle, reg_low, low, high, X_test, y_test):
    y_pred = np.zeros_like(y_test)
    for i in range(len(y_test)):
        if y_test[i] >= high:
            y_pred[i] = reg_high.predict(X_test[i])
        elif y_test[i] <= low:
            y_pred[i] = reg_low.predict(X_test[i])
        else:
            y_pred[i] = reg_middle.predict(X_test[i])
    return y_pred


def evaluate(X, y, alg, rcv, value, grather, file_name):
    results_perf = []

    # 10-fold CV
    for train_index, test_index in rcv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_low_train, y_low_train,
        X_middle_train, y_middle_train,
        X_high_train, y_high_train = fill_data(X_train, y_train, low, high)

        reg_high = train_regressors(X_high_train, y_high_train, alg, np.random.randint(1,10000))
        reg_middle = train_regressors(X_middle_train, y_middle_train, alg, np.random.randint(1,10000))
        reg_low = train_regressors(X_low_train, y_low_train, alg, np.random.randint(1,10000))

        # Oracle
        X_low_test, y_low_test,
        X_middle_test, y_middle_test,
        X_high_test, y_high_test = fill_data(X_test, y_test, low, high)

        preds = apply_oracle(reg_high, reg_middle, reg_low, low, high, X_test, y_test)
        result_oracle = compute_performance(preds, y_range_test)
        result_oracle.append("oracle-all")
        results_perf.append(result_oracle)

        preds = apply_regressors(reg_low, X_low_test)
        result_low = compute_performance(preds, y_low_test)
        result_low.append("oracle-low")
        results_perf.append(result_low)

        preds = apply_regressors(reg_high, X_high_test)
        result_high = compute_performance(preds, y_high_test)
        result_high.append("oracle-high")
        results_perf.append(result_high)

        preds = apply_regressors(reg_middle, X_middle_test)
        result_middle = compute_performance(preds, y_middle_test)
        result_middle.append("oracle-middle")
        results_perf.append(result_middle)

    pd.DataFrame(data=results_perf,
                 columns=["mean_absolute_error", "mean_squared_error",
                          "r2_score", "RRMSE", "RMSE", "type"])
    pd.to_csv(file_name)


def run_eval(data_path, str_class, n_cpus):
    np.random.seed(SEED)
    algs = ["DT", "MLP", "RF"]
    data = pd.read_csv(data_path)
    rcv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=SEED)

    range_high_TG = np.arange(start=900, stop=1150+1, step=25).tolist()
    range_low_TG = np.arange(start=425, stop=650+1, step=25).tolist()

    X = data.drop([str_class], axis=1).values
    y = data[str_class].values

    perceltil_inf = [5*i for i in range(1,9)]
    perceltil_sup = [(100-perceltil_inf[i]) for i in range(len(perceltil_inf))]
    aux_high_TG = [np.percentile(y, perceltil_sup[i]) for i in range(len(perceltil_sup))]
    aux_low_TG = [np.percentile(y, perceltil_inf[i]) for i in range(len(perceltil_inf))]

    range_high_TG = range_high_TG + aux_high_TG
    range_low_TG = range_low_TG + aux_low_TG

    result_path = "../../result/result_oracle/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)


    # pool = Pool(processes=n_cpus)
    # results = []
    #
    # result_path = "../../result/result_high/"
    # if not os.path.exists(result_path):
    #     os.makedirs(result_path)
    #
    # for data_range in range_high_TG:
    #     for alg in algs:
    #         print("High - {0} - {1}".format(alg, data_range))
    #         file_name = result_path+"result_"+str(data_range)+"_"+alg+".csv"
    #         results.append(pool.apply_async(evaluate_range, (X, y, alg, rcv, data_range, True, file_name)))
    #
    # result_path = "../../result/result_low/"
    # if not os.path.exists(result_path):
    #     os.makedirs(result_path)
    #
    # for data_range in range_low_TG:
    #     for alg in algs:
    #         print("Low - {0} - {1}".format(alg, data_range))
    #         file_name = result_path+"result_"+str(data_range)+"_"+alg+".csv"
    #         # evaluate_range(X, y, rcv, data_range, False, file_name)
    #         results.append(pool.apply_async(evaluate_range, (X, y, alg, rcv, data_range, False, file_name)))
    #
    #
    # result_path = "../../result/result_all/"
    # if not os.path.exists(result_path):
    #     os.makedirs(result_path)
    #
    # file_names = [None, None, None]
    # for alg in algs:
    #     print("all - {0}".format(alg))
    #     file_names[0] = result_path+"result_all_"+alg+".csv"
    #     file_names[1] = result_path+"result_all_high_"+alg+".csv"
    #     file_names[2] = result_path+"result_all_low_"+alg+".csv"
    #     results.append(pool.apply_async(evaluate, (X, y, alg, rcv, range_high_TG, range_low_TG, file_names)))
    #
    # pool.close()
    # pool.join()

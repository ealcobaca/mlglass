from sklearn.model_selection import RepeatedKFold
import pandas as pd
import numpy as np
import os
import pickle
from multiprocessing import Pool, Manager
from regressors import train_regressors, apply_regressors, compute_performance
# import line_profiler

SEED=123

def fill_data(X, y, range_type, range_size):
    if(range_type == "low"):
        range_size = float(range_size)
        return X[y <= range_size], y[y <= range_size]
    elif(range_type == "middle"):
        low, high = range_size.split("-")
        low = float(low)
        high = float(high)
        aux = [i > low and i < high for i in y]
        X_middle, y_middle = X[aux], y[aux]
        return X_middle, y_middle
    elif(range_type == "high"):
        range_size = float(range_size)
        return X[y >= range_size], y[y >= range_size]
    elif(range_type == "all"):
        return X, y
    else:
        print("Error")
        return None, None


def save_train_regressors(X, y, alg, file_name, seed):
    alg = train_regressors(X, y, alg, seed)
    # pickle.dump(obj=alg, file=open(file_name, "wb"), protocol=4)
    return alg


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


def f(X_train, y_train, X_test, y_test, index, alg, file_name, seed):
    model = save_train_regressors(X_train, y_train, alg, file_name, seed)
    preds = apply_regressors(model, X_test)
    return [index, y_test, preds]


def evaluate(X, y, rcv, conf, folds):
    np.random.seed(SEED)
    results = []
    map_input = []
    alg, range_type, range_size, result_path, alg_path  = conf

    # 10-fold CV
    k=0
    for (train_index, test_index), seed in zip(rcv.split(X), np.random.randint(1,10000, folds)):
        X_train_cv, X_test_cv = X[train_index], X[test_index]
        y_train_cv, y_test_cv = y[train_index], y[test_index]

        index,_ = fill_data(test_index, y_test_cv, range_type, range_size)
        X_train, y_train = fill_data(X_train_cv, y_train_cv, range_type, range_size)
        X_test, y_test = fill_data(X_test_cv, y_test_cv, range_type, range_size)

        key = "{0}_{1}_{2}_{3}_.data".format(alg, range_type, range_size, k)
        file_name = alg_path+key
        k += 1
        map_input.append((X_train, y_train, X_test, y_test, index, alg, file_name, seed))

    with Pool(processes = 10) as p:
        results = p.starmap(f, map_input)

    file_name = "{0}{1}_{2}_{3}_.list".format(result_path, alg, range_type, range_size).format()
    pickle.dump(obj=results, file=open(file_name, "wb"), protocol=4)


def run_eval_pbs(data_path, str_class, alg, range_type, range_size):
    np.random.seed(SEED)
    data = pd.read_csv(data_path)

    # for all execution
    folds = 10
    rep = 1
    rcv = RepeatedKFold(n_splits=folds, n_repeats=rep, random_state=SEED)

    X = data.drop([str_class], axis=1).values
    y = data[str_class].values

    alg_path = "../../result/result_oracle/regressors_obj/"
    if not os.path.exists(alg_path):
        os.makedirs(alg_path)

    result_path = "../../result/result_oracle/result_list/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    conf = (alg, range_type, range_size, result_path, alg_path)
    evaluate(X, y, rcv, conf, folds)


def experiment_conf_id(data_path, str_class):

    conf = []
    data = pd.read_csv(data_path)
    X = data.drop([str_class], axis=1).values
    y = data[str_class].values

    perceltil_inf = [0, 1.5, 2.5, 3.5] +[5*i for i in range(1,7)]
    perceltil_sup = [(100-perceltil_inf[i]) for i in range(len(perceltil_inf))]
    range_high_TG = [np.percentile(y, perceltil_sup[i]) for i in range(len(perceltil_sup))]
    range_high_TG.reverse()
    range_low_TG = [np.percentile(y, perceltil_inf[i]) for i in range(len(perceltil_inf))]

    range_high_TG = np.round(range_high_TG, 2)
    range_low_TG = np.round(range_low_TG,2)

    algs = ["DT", "RF", "MLP"]
    range_types = ["low", "middle", "high", "all"]
    for alg in algs:
        for range_type in range_types:
            if(range_type == "low"):
                for range_size in range_low_TG:
                    conf.append((data_path, str_class, alg, range_type, str(range_size)))
            elif(range_type == "middle"):
                for low in range_low_TG:
                    for high in range_high_TG:
                        conf.append((data_path, str_class, alg, range_type, "{0}-{1}".format(low, high)))
            elif(range_type == "high"):
                for range_size in range_high_TG:
                    conf.append((data_path, str_class, alg, range_type, str(range_size)))
            elif(range_type == "all"):
                conf.append((data_path, str_class, alg, range_type, "all"))
    return conf

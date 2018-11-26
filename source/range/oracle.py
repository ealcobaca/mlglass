import pandas as pd
import numpy as np
import pickle
from range.regressors import compute_performance


path_result = "../result/result_oracle/result_list/"

def oracle(low, high, alg_low, alg_middle, alg_high):

    res_low_name = "{0}{1}_low_{2}_.list".format(path_result, alg_low, low)
    res_middle_name = "{0}{1}_middle_{2}-{3}_.list".format(path_result, alg_middle, low, high)
    res_high_name = "{0}{1}_high_{2}_.list".format(path_result, alg_high, high)

    res_low = pickle.load(open(res_low_name, "rb"))
    res_middle = pickle.load(open(res_middle_name, "rb"))
    res_high = pickle.load(open(res_high_name, "rb"))

    res = [[np.concatenate((i[1], j[1], k[1])), np.concatenate((i[2], j[2], k[2]))] for i,j,k in zip(res_low, res_middle, res_high)]
    perf = [compute_performance(r[0], r[1]) for r in res]

    return np.mean(perf, axis=0), np.std(perf, axis=0)


def order(dic, by="RRMSE"):
    dic_measure = {"MAE":0, "MSE":1, "R2_S":2, "RRMSE":3, "RMSE":4, "MARE":5, "R2":6}
    index = dic_measure[by]

    if by in ("R2_S", "R2"):
        reverse=True
    else:
        reverse = False

    return sorted(dic.items(), key=lambda x: x[1][0][index], reverse=reverse)


def run(data_path, str_class, algs=["DT", "MLP", "RF"]):

    data = pd.read_csv(data_path)
    folds=10
    rep=1

    X = data.drop([str_class], axis=1).values
    y = data[str_class].values

    perceltil_inf = [1.5, 2.5, 3.5] +[5*i for i in range(1,7)]
    perceltil_sup = [(100-perceltil_inf[i]) for i in range(len(perceltil_inf))]
    range_high_TG = [np.percentile(y, perceltil_sup[i]) for i in range(len(perceltil_sup))]
    range_high_TG.reverse()
    range_low_TG = [np.percentile(y, perceltil_inf[i]) for i in range(len(perceltil_inf))]

    range_high_TG = np.round(range_high_TG, 2)
    range_low_TG = np.round(range_low_TG,2)

    print(range_high_TG)
    print(range_low_TG)

    dic_oracle = {}
    for alg_low in algs:
        for alg_middle in algs:
            for alg_high in algs:
                for low in range_low_TG:
                    for high in range_high_TG:
                        res = oracle(low, high, alg_low, alg_middle, alg_high)
                        dic_oracle["{0}-{1}_{2}_{3}_{4}".format(low, high, alg_low, alg_middle, alg_high)] = res

    return dic_oracle

def summary(max_value=3, algs=None):
    if algs != None:
        dic = run("../data/clean/oxides_Tg_train.csv", "Tg", algs)
    else:
        dic = run("../data/clean/oxides_Tg_train.csv", "Tg")
    dic_measure = {"MAE":0, "MSE":1, "R2_S":2, "RRMSE":3, "RMSE":4, "MARE":5, "R2":6}

    result = []
    for measure in dic_measure.keys():
        dic_ord = order(dic, by=measure)
        print("------ "+measure+" ------")
        for i in range(0,max_value):
            result.append(dic_ord[i][0])
            print("{0} -- {1} +/- {2}".format(
                dic_ord[i][0],
                round(dic_ord[i][1][0][dic_measure[measure]],4),
                round(dic_ord[i][1][1][dic_measure[measure]],4)))
        print("-------------------------")
    return result

# summary(algs=["MLP"])
# summary()
#

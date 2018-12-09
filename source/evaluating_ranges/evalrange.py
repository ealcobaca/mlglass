import pandas as pd
import numpy as np
import pickle
import os.path
from range.regressors import compute_performance


path_result = "../result/result_oracle/result_list/"

def oracle(low, high, alg_low, alg_middle, alg_high):

    res_low_name = "{0}{1}_low_{2}_.list".format(path_result, alg_low, low)
    res_middle_name = "{0}{1}_middle_{2}-{3}_.list".format(path_result, alg_middle, low, high)
    res_high_name = "{0}{1}_high_{2}_.list".format(path_result, alg_high, high)
    # print(res_low_name)
    # print(res_middle_name)
    # print(res_high_name)

    res_low = res_middle = res_high = None
    if os.path.isfile(res_low_name):
        res_low = pickle.load(open(res_low_name, "rb"))
    if os.path.isfile(res_middle_name):
        res_middle = pickle.load(open(res_middle_name, "rb"))
    if os.path.isfile(res_high_name):
        res_high = pickle.load(open(res_high_name, "rb"))

    # print(res_low)
    # print(res_middle)
    # print(res_high)

    if res_low == None and res_high == None:
        res = [[i[1], i[2]] for i in res_middle]
    elif res_low == None:
        res = [[np.concatenate((i[1], j[1])), np.concatenate((i[2], j[2]))] for i,j in zip(res_middle, res_high)]
    elif res_high == None:
        res = [[np.concatenate((i[1], j[1])), np.concatenate((i[2], j[2]))] for i,j in zip(res_low, res_middle)]
    else:
        res = [[np.concatenate((i[1], j[1], k[1])), np.concatenate((i[2], j[2], k[2]))] for i,j,k in zip(res_low, res_middle, res_high)]

    perf = [compute_performance(r[0], r[1]) for r in res]
    evaluation = np.concatenate([np.mean(perf, axis=0),np.std(perf, axis=0)])

    for res_range in [res_low, res_middle, res_high]:
        if res_range == None:
            a =np.array([None, None, None, None, None, None, None])
            evaluation = np.concatenate([evaluation, a, a])
        else:
            res = [[i[1],i[2]] for i in res_range]
            perf = [compute_performance(r[0], r[1]) for r in res]
            evaluation = np.concatenate([evaluation, np.mean(perf, axis=0),np.std(perf, axis=0)])

    return evaluation


def run(data_path, str_class, algs=["DT", "MLP", "RF"]):

    data = pd.read_csv(data_path)
    folds=10
    rep=1

    X = data.drop([str_class], axis=1).values
    y = data[str_class].values

    perceltil_inf = [0, 1.5, 2.5, 3.5] +[5*i for i in range(1,7)]
    perceltil_sup = [(100-perceltil_inf[i]) for i in range(len(perceltil_inf))]
    range_high_TG = [np.percentile(y, perceltil_sup[i]) for i in range(len(perceltil_sup))]
    # range_high_TG.reverse()
    range_low_TG = [np.percentile(y, perceltil_inf[i]) for i in range(len(perceltil_inf))]

    range_high_TG = np.round(range_high_TG, 2)
    range_low_TG = np.round(range_low_TG,2)

    print(range_low_TG)
    print(perceltil_inf)
    print(range_high_TG)
    print(perceltil_sup)

    dic_oracle = {}
    data = []
    for low, low_perc in zip(range_low_TG, perceltil_inf):
        for high, high_perc in zip(range_high_TG, perceltil_sup):
            for alg_low in algs:
                for alg_middle in algs:
                    for alg_high in algs:
                        line = []
                        line = [low_perc, 100-(low_perc+100-high_perc), 100-high_perc]

                        for i in [alg_low, alg_middle, alg_high]:
                            line = line + [1 if i == "MLP" else 0,
                                           1 if i == "RF"  else 0,
                                           1 if i == "DT"  else 0]
                        res = oracle(low, high, alg_low, alg_middle, alg_high)
                        line = line + res.tolist()
                        dic_oracle["{0}-{1}_{2}_{3}_{4}".format(low, high, alg_low, alg_middle, alg_high)] = line
                        data.append(line)

    cols_name = ["S","M","E",
                 "S_MLP","S_RF","S_DT",
                 "M_MLP","M_RF","M_DT",
                 "E_MLP","E_RF","E_DT",
                 "Global_mean_MAE", "Global_mean_MSE", "Global_mean_R2_S",
                 "Global_mean_RRMSE", "Global_mean_RMSE", "Global_mean_MARE", "Global_mean_R2",
                 "Global_sd_MAE", "Global_sd_MSE", "Global_sd_R2_S",
                 "Global_sd_RRMSE", "Global_sd_RMSE", "Global_sd_MARE", "Global_sd_R2",
                 "Local_S_mean_MAE", "Local_S_mean_MSE", "Local_S_mean_R2_S",
                 "Local_S_mean_RRMSE", "Local_S_mean_RMSE", "Local_S_mean_MARE", "Local_S_mean_R2",
                 "Local_S_sd_MAE", "Local_S_sd_MSE", "Local_S_sd_R2_S",
                 "Local_S_sd_RRMSE", "Local_S_sd_RMSE", "Local_S_sd_MARE", "Local_S_sd_R2",
                 "Local_M_mean_MAE", "Local_M_mean_MSE", "Local_M_mean_R2_S",
                 "Local_M_mean_RRMSE", "Local_M_mean_RMSE", "Local_M_mean_MARE", "Local_M_mean_R2",
                 "Local_M_sd_MAE", "Local_M_sd_MSE", "Local_M_sd_R2_S",
                 "Local_M_sd_RRMSE", "Local_M_sd_RMSE", "Local_M_sd_MARE", "Local_M_sd_R2",
                 "Local_E_mean_MAE", "Local_E_mean_MSE", "Local_E_mean_R2_S",
                 "Local_E_mean_RRMSE", "Local_E_mean_RMSE", "Local_E_mean_MARE", "Local_E_mean_R2",
                 "Local_E_sd_MAE", "Local_E_sd_MSE", "Local_E_sd_R2_S",
                 "Local_E_sd_RRMSE", "Local_E_sd_RMSE", "Local_E_sd_MARE", "Local_E_sd_R2"
                 ]
    df = pd.DataFrame(data, columns = cols_name)
    df.to_csv('../result/evaluating_range/ranges2.csv')

    return df

a = run('../data/clean/oxides_Tg_train.csv', 'Tg')

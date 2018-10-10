import pickle
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np


def aggr(read_path, save_path):
    files = [f for f in listdir(read_path) if isfile(join(read_path, f))]
    path_files = [join(read_path,f) for f in listdir(read_path) if isfile(join(read_path, f))]

    data=[]
    for f, path in zip(files, path_files):
        exp_range = float(f.split('_')[1])
        exp_alg = f.split('_')[2].split('.')[0]
        result = pickle.load( open(path, "rb" ))
        [(r.append(exp_range),r.append(exp_alg), data.append(r)) for r in result]

    df = pd.DataFrame(data=data, columns=["mean_absolute_error",
                                          "mean_absolute_error",
                                          "r2_score",
                                          "RRMSE",
                                          "RMSE",
                                          "range",
                                          "alg"])

    if save_path != None:
        return df.to_csv(save_path)

    return df


def aggr_all(read_path, save_path, data_path, str_class):
    files = [f for f in listdir(read_path) if isfile(join(read_path, f))]
    path_files = [join(read_path,f) for f in listdir(read_path) if isfile(join(read_path, f))]

    tot_cv = 10*5
    dt = pd.read_csv(data_path)
    X = dt.drop([str_class], axis=1).values
    y = dt[str_class].values

    perceltil_inf = [5*i for i in range(1,9)]
    perceltil_sup = [(100-perceltil_inf[i]) for i in range(len(perceltil_inf))]
    range_high_TG_per = [np.percentile(y, perceltil_sup[i]) for i in range(len(perceltil_sup))]
    range_low_TG_per = [np.percentile(y, perceltil_inf[i]) for i in range(len(perceltil_inf))]
    range_low_TG = np.arange(start=400, stop=650+1, step=25)
    range_high_TG = np.arange(start=900, stop=1150+1, step=25)

    data=[]
    for f, path in zip(files, path_files):
        exp_range = f.split('_')[1]
        exp_alg = f.split('_')[2].split('.')[0]
        if exp_alg in ("high", "low"):
            alg = f.split('_')[3].split('.')[0]
            result = pickle.load(open(path, "rb" ))
            if(exp_alg == "high"):
                if(["percentil"] in read_path.split("/")[3].split("_")):
                    ran = range_high_TG_per.tolist() * tot_cv
                else:
                    ran = range_high_TG.tolist() * tot_cv
            else:
                if(["percentil"] in read_path.split("/")[3].split("_")):
                    ran = range_low_TG_per.tolist() * tot_cv
                else:
                    ran = range_low_TG.tolist() * tot_cv
            result = [j for i in result for j in i]
            [(r.append(ran_i),r.append(alg), data.append(r)) for r, ran_i in zip(result, ran)]
        else:
            result = pickle.load(open(path, "rb" ))
            [(r.append(exp_range),r.append(exp_alg), data.append(r)) for r in result]

    df = pd.DataFrame(data=data, columns=["mean_absolute_error",
                                          "mean_absolute_error",
                                          "r2_score",
                                          "RRMSE",
                                          "RMSE",
                                          "range",
                                          "alg"])
    if save_path != None:
        print(save_path)
        return df.to_csv(save_path)

    return df


def run():

    result_path = "../../result/aggr/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    aggr("../../result/result_low/", "../../result/aggr/result_low.csv")
    aggr("../../result/result_high/", "../../result/aggr/result_high.csv")
    aggr("../../result/result_low_percentil/", "../../result/aggr/result_low_percentil.csv")
    aggr("../../result/result_high_percentil/", "../../result/aggr/result_high_percentil.csv")
    aggr_all("../../result/result_all/", "../../result/aggr/result_all.csv", "../../data/clean/oxides_Tg_train.csv", "Tg")
    aggr_all("../../result/result_all_percentil/", "../../result/aggr/result_all_percentil.csv", "../../data/clean/oxides_Tg_train.csv", "Tg")


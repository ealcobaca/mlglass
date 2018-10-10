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

def run():

    result_path = "../../result/aggr/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    aggr("../../result/result_low/", "../../result/aggr/result_low.csv")
    aggr("../../result/result_high/", "../../result/aggr/result_high.csv")
    aggr("../../result/result_low_percentil/", "../../result/aggr/result_low_percentil.csv")
    aggr("../../result/result_high_percentil/", "../../result/aggr/result_high_percentil.csv")
    aggr("../../result/result_all/result_all_", "../../result/aggr/result_low.csv")


import pandas as pd
import numpy as np
import pickle
from range.regressors import compute_performance


path_result = "../result/result_oracle/default-model/"

paths = [
    "../result/result_oracle/default-model/mean_test_.list",
    "../result/result_oracle/default-model/mode_test_.list"]
    # "../result/result_oracle/default-model/mean_test_all_leaf.list",
    # "../result/result_oracle/default-model/mode_test_all_leaf.list"]

dic_measure = {"MAE":0, "MSE":1, "R2_S":2, "RRMSE":3, "RMSE":4, "MARE":5, "R2":6}
for path in paths:
    y_true, y_pred = pickle.load(open(path, 'rb'))
    name = path
    print(name)
    perf = compute_performance(y_true, y_pred)
    for measure in  dic_measure.keys():
        print("------ "+measure+" ------")
        print(perf[dic_measure[measure]])
    print("-------------------------\n")

paths = [
    "../result/baselines/log/predictions_raw_RF.list",
    "../result/baselines/log/predictions_raw_DT.list",
    "../result/baselines/log/predictions_raw_MLP.list"]
    # "../result/result_oracle/default-model/mean_test_all_leaf.list",
    # "../result/result_oracle/default-model/mode_test_all_leaf.list"]

dic_measure = {"MAE":0, "MSE":1, "R2_S":2, "RRMSE":3, "RMSE":4, "MARE":5, "R2":6}
for path in paths:
    _, y_true, y_pred = pickle.load(open(path, 'rb'))
    name = path
    print(name)
    perf = compute_performance(y_true, y_pred)
    for measure in  dic_measure.keys():
        print("------ "+measure+" ------")
        print(perf[dic_measure[measure]])
    print("-------------------------\n")


import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

files = glob.glob("../../result/logs/performance_best_models_tg_*")
files = files + glob.glob("../../result/logs/performance_standard_models_tg_*")

files_tag = ["{0}".format(f.split("_")[1]) for f in files]
files_fold = ["{0}".format(f.split('_')[4].split('.csv')[0]) for f in files]

data_frames = []
for f in files:
    df = pd.read_csv(f, index_col=0)
    data_frames.append(df)

result = pd.concat(data_frames).loc["RRMSE", :]
result["folds"] = files_fold
result["tag"] = files_tag
result = result.reset_index(drop=True)


# bp_dt = result.boxplot(column=["dt"], by=["tag"])
# bp_knn = result.boxplot(column=["knn"], by=["tag"])
# bp_svr = result.boxplot(column=["svr"], by=["tag"])
# bp_mlp = result.boxplot(column=["mlp"], by=["tag"])
# bp_rf = result.boxplot(column=["rf"], by=["tag"])
#
# plt.show()

best = result.loc[result["tag"] == 'best']
median = np.abs(best.iloc[:, :5] - best.median())
idxmin = median.idxmin()
for alg, idx in zip(idxmin.index, idxmin.values):
    print("{0} -- fold{1}".format(alg, idx))
# standard = result.loc[result["tag"] == 'standard']




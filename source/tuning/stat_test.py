import glob
import pandas as pd
import numpy as np
import bayesiantests as bt


folds = glob.glob("../../result/logs/performance_best_models*.csv")

lines = []
idx = pd.read_csv(folds[0], index_col=0).columns.tolist()

for f in folds:
    data = pd.read_csv(f, index_col=0)
    lines.append(data.loc["RRMSE"].values)

matrix = 1 - np.array(lines)
print(matrix)
print(idx)

###############################################################################

stat_test = []
rope=0.01 # We consider two classifers equivalent when the difference of
          # accuracy is less that 1%
rho=1/10  # We are performing 10 folds, cross-validation

for i in range(4, len(idx)-1):
    print("rf vs ", idx[i])
    names = [idx[i], "rf"]
    diff = np.array([(matrix[:, -1]) - matrix[:, i]])
    print(diff)
    stat_test.append(bt.hierarchical(
        diff=diff,
        rope=rope,
        rho=rho,
        lowerAlpha=0.5,
        upperAlpha=5,
        lowerBeta=0.05,
        upperBeta=.15,
        verbose=True, names=names))

for i in range(0, len(idx)-1):
    pl, pe, pr = stat_test[i]
    names = [idx[i],"rf"]
    print('P({c1} > {c2}) = {pl}, P(rope) = {pe}, P({c2} > {c1}) = {pr}'.
          format(c1=names[0], c2=names[1], pl=pl, pe=pe, pr=pr))

import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
import os

outer_seed = 1

data = pd.read_csv("../../data/clean/data_tg_dupl_rem.csv")
columns = data.columns
X, y = data.iloc[:, :-1].values, data.iloc[:, -1].values
kf = KFold(n_splits=10, random_state=outer_seed, shuffle=True)

directory = "../../data/clean/train_test_split/"
if not os.path.exists(directory):
    os.makedirs(directory)

count = 0
for k, (train_index, test_index) in enumerate(kf.split(X)):
    X_train, y_train = X[train_index], y[train_index]
    X_test, y_test = X[test_index], y[test_index]
    count += 1
    values = []

    aux = "../../data/clean/train_test_split/tg_{0}_fold{1:02d}.csv"
    path_name = aux.format("train", count)
    df_train = pd.DataFrame(X_train)
    df_train['Tg'] = y_train
    df_train.columns = columns
    df_train.to_csv(path_name, index=False)

    idx_max = np.argmax(y_test)
    idx_min = np.argmin(y_test)

    path_name = aux.format("test", count)
    df_test = pd.DataFrame(X_test)
    df_test['Tg'] = y_test
    df_test.columns = columns
    df_test.to_csv(path_name, index=False)

    values.append(np.append(X_test[idx_max], y_test[idx_max]))
    values.append(np.append(X_test[idx_min], y_test[idx_min]))
    df_test_extreme = pd.DataFrame(values, columns=columns)
    df_test_extreme.to_csv(
        "../../data/clean/train_test_split/tg_test_fold{0:02d}_extreme.csv".format(count),
        index=False)


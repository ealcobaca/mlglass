import os
import pandas as pd
from sklearn.model_selection import KFold
from constants import SPLIT_DATA_PATH as split_path
from constants import TARGETS_FORMATTED as targets
from constants import DATA_PATH as data_path
from constants import DATASET_PREFIX as data_prefix
from constants import REMOVE_ID_COLUMN


if not os.path.exists(split_path):
    os.makedirs(split_path)

outer_seed = 1

for target, targetf in targets.items():
    data = pd.read_csv(
        '{0}/{1}{2}.csv'.format(data_path, data_prefix, target)
    )

    # Remove material id
    if REMOVE_ID_COLUMN:
        data = data.iloc[:, 1:]

    columns = data.columns
    X, y = data.iloc[:, :-1].values, data.iloc[:, -1].values
    kf = KFold(n_splits=10, random_state=outer_seed, shuffle=True)

    count = 0
    for k, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        count += 1
        values = []

        path_name = '{0}/{1}_train_fold{2:02d}.csv'.format(
            split_path, target, count
        )
        df_train = pd.DataFrame(X_train)
        df_train[target] = y_train
        df_train.columns = columns
        df_train.to_csv(path_name, index=False)

        # idx_max = np.argmax(y_test)
        # idx_min = np.argmin(y_test)

        path_name = '{0}/{1}_test_fold{2:02d}.csv'.format(
            split_path, target, count
        )
        df_test = pd.DataFrame(X_test)
        df_test[target] = y_test
        df_test.columns = columns
        df_test.to_csv(path_name, index=False)

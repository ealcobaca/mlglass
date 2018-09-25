import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def mult_target_split(data, targets, save_path, test_size=0.2, seed=123):
    np.random.seed(seed)
    seeds = np.random.randint(np.iinfo(np.int32).max, size=len(targets))

    for target, seed in zip(targets, seeds):
        data_cl = data.dropna(subset=[target])
        y = data_cl[[target]].values
        X = data_cl.drop(targets, axis=1).values # drop all targets

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

        data_train = np.concatenate((X_train, y_train), axis=1)
        data_test = np.concatenate((X_test, y_test), axis=1)

        file_name_train = save_path+"/oxides_"+target+"_train.csv"
        file_name_test = save_path+"/oxides_"+target+"_test.csv"
        columns = data.columns[:-3].append(pd.Index([target]))
        pd.DataFrame(data_train, columns=columns).to_csv(file_name_train, header=True, index=False)
        pd.DataFrame(data_test, columns=columns).to_csv(file_name_test, header=True, index=False)

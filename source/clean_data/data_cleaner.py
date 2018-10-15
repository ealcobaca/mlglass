import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def mult_target_split(data, targets, save_path, test_size=0.2, amount=None, seed=123):
    np.random.seed(seed)
    seeds = np.random.randint(np.iinfo(np.int32).max, size=len(targets))
    # X, y, X_rem, y_rem = None

    for target, seed in zip(targets, seeds):
        data_cl = data.dropna(subset=[target])
        y = data_cl[[target]].values
        X = data_cl.drop(targets, axis=1).values # drop all targets

        if amount != None:
            X, y, X_rem, y_rem = remove_data(X, y, amount)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

        data_train = np.concatenate((X_train, y_train), axis=1)
        data_test = np.concatenate((X_test, y_test), axis=1)

        file_name_train = save_path+"/oxides_"+target+"_train.csv"
        file_name_test = save_path+"/oxides_"+target+"_test.csv"
        columns = data.columns[:-3].append(pd.Index([target]))
        pd.DataFrame(data_train, columns=columns).to_csv(file_name_train, header=True, index=False)
        pd.DataFrame(data_test, columns=columns).to_csv(file_name_test, header=True, index=False)

        if amount != None:
            data_test_rem = np.concatenate((X_rem, y_rem), axis=1)
            file_name_test_rem = save_path+"/oxides_"+target+"_test_rem.csv"
            pd.DataFrame(data_test_rem, columns=columns).to_csv(file_name_test_rem, header=True, index=False)


def remove_data(X, y, amount=6):
    y_sort = np.sort([i[0] for i in y])
    min_y = y_sort[amount-1]
    max_y = y_sort[len(y_sort) - amount]

    aux1 = [i[0] < max_y and i[0] > min_y for i in y]
    aux2 = np.logical_not(aux1)

    return X[aux1], y[aux1], X[aux2], y[aux2]

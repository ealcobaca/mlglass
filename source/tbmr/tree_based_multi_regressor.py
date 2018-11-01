import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from range.oracle import summary
from scipy import stats
import os
import pickle


class TBMR:
    def __init__(self, alg=("RF"), range_cut=(), seed=None):
        if seed is not None:
            self.seed = seed
            np.random.seed(self.seed)

        self.range_cut = range_cut

    def next_seed(self):
        return np.random.randint(100000)

    def fill_data(self, X, y, range_type, range_size, y_disc, value):
        if(range_type == "beggining"):
            y_disc[y < range_size] = value
            return X[y < range_size], y[y < range_size], y_disc
        elif(range_type == "middle"):
            low, high = range_size
            aux = [i >= low and i < high for i in y]
            y_disc[aux] = value
            X_middle, y_middle = X[aux], y[aux]
            return X_middle, y_middle, y_disc
        elif(range_type == "end"):
            y_disc[y >= range_size] = value
            return X[y >= range_size], y[y >= range_size], y_disc
        else:
            print("Error")
            return None, None

    def fit(self, X, y):
        data = []
        aux = len(self.range_cut)
        y_disc = np.zeros_like(y)

        previous = -1
        for i in range(0, len(self.range_cut)+1):
            if(i == 0):
                X_cut, y_cut, y_disc = self.fill_data(X, y, "beggining", self.range_cut[i], y_disc, i+1)
            elif(i == aux):
                X_cut, y_cut, y_disc = self.fill_data(X, y, "end", self.range_cut[previous], y_disc, i+1)
            else:
                X_cut, y_cut, y_disc = self.fill_data(X, y, "middle", (self.range_cut[previous], self.range_cut[i]), y_disc, i+1)
            previous = i
            data.append([X_cut, y_cut])

        # root model
        # y_disc = [str(i) for i in y_disc]
        self.classif_root = RandomForestClassifier(n_estimators=100, random_state=self.next_seed())
        self.classif_root.fit(X, y_disc)

        # leaf models
        self.regrs_leaf = []
        for d in data:
            regr = RandomForestRegressor(n_estimators=100, random_state=self.next_seed())
            regr.fit(d[0], d[1])
            self.regrs_leaf.append(regr)

    def predict(self, X):
        pred_root = self.classif_root.predict(X)
        pred_leaf = []
        pred_root = [int(i) for i in pred_root]

        pred_leaf = [self.regrs_leaf[i-1].predict([x])[0] for i, x in zip(pred_root, X)]
        return np.array(pred_leaf)

        # # Speeding up prediction computation
        # pred_leaf = np.zeros((X.shape[0]))
        # for cat_val in np.unique(pred_root):
        #     sel_samples = pred_root == cat_val
        #     pred_leaf[sel_samples] = \
        #         self.regrs_leaf[cat_val-1].predict(X[sel_samples, :])

    def to_class(self, y):
        y_disc = np.zeros_like(y)
        len_range = len(self.range_cut)

        previous = -1
        for i in range(0, len(self.range_cut)+1):
            if(i == 0):
                y_disc[y < self.range_cut[i]] = int(i+1)
            elif(i == len_range):
                y_disc[y >= self.range_cut[previous]] = int(i+1)
            else:
                low = self.range_cut[previous]
                high = self.range_cut[i]
                aux = [i >= low and i < high for i in y]
                y_disc[aux] = int(i+1)
            previous = i
        return y_disc

    def predict_all_leaf(self, X, y):
        pred_leaf = []
        pred_root = self.to_class(y)
        pred_root = [int(i) for i in pred_root]
        pred_leaf = [self.regrs_leaf[i-1].predict([x])[0] for i,x in zip(pred_root,X)]
        return [y, pred_leaf]


    def predict_root(self, X, y):
        pred_root = self.classif_root.predict(X)
        # pred_root = [int(i) for i in pred_root]
        truth = self.to_class(y)
        truth = [int(i) for i in truth]
        return [truth, pred_root]


    def predict_leaf(self, X, y, index):
        pred_leaf = []
        pred_root = self.to_class(y)
        pred_root = np.array([int(i) for i in pred_root])
        x = X[pred_root == (index+1)]
        y = y[pred_root == (index+1)]
        pred_leaf = self.regrs_leaf[index].predict(x)
        return [y, pred_leaf]


def run_default(data_train_path, data_test_path, str_class, result_path, f_name):
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    data_train = pd.read_csv(data_train_path)
    X_train = data_train.drop([str_class], axis=1).values
    y_train = data_train[str_class].values

    data_test = pd.read_csv(data_test_path)
    X_test = data_test.drop([str_class], axis=1).values
    y_test = data_test[str_class].values

    summ = summary()
    aux = stats.mode(summ)[0][0]
    start = float(aux.split("-")[0])
    end = float(aux.split("_")[0].split("-")[1])
    mode = (start, end)

    start = np.array([s.split("_")[0].split('-')[0] for s in summ]).astype(np.float)
    end = np.array([s.split("_")[0].split('-')[1] for s in summ]).astype(np.float)
    mean = (round(np.mean(start),2), round(np.mean(end),2))
    conf = {"mode": mode, "mean":mean}

    for c in conf.keys():
        regr = TBMR(alg=("RF"), range_cut=conf[c], seed=None)
        regr.fit(X_train, y_train)
        pred = regr.predict(X_test)

        result = [y_test, pred]
        file_name = result_path+"{0}_{1}_.list".format(c,f_name)
        pickle.dump(result, open(file_name, "wb" ))

        result = regr.predict_all_leaf(X_test, y_test)
        file_name = result_path+"{0}_{1}_all_leaf.list".format(c,f_name)
        pickle.dump(result, open(file_name, "wb" ))

        result = regr.predict_leaf(X_test, y_test, 0)
        file_name = result_path+"{0}_{1}_leaf-start_.list".format(c,f_name)
        pickle.dump(result, open(file_name, "wb" ))

        result = regr.predict_leaf(X_test, y_test, 1)
        file_name = result_path+"{0}_{1}_leaf-middle_.list".format(c,f_name)
        pickle.dump(result, open(file_name, "wb" ))

        result = regr.predict_leaf(X_test, y_test, 2)
        file_name = result_path+"{0}_{1}_leaf-end_.list".format(c,f_name)
        pickle.dump(result, open(file_name, "wb" ))

        result = regr.predict_root(X_test, y_test)
        file_name = result_path+"{0}_{1}_root_.list".format(c,f_name)
        pickle.dump(result, open(file_name, "wb" ))


run_default(
    "../data/clean/oxides_Tg_train.csv",
    "../data/clean/oxides_Tg_test.csv",
    "Tg",
    "../result/result_oracle/default-model/", "test")
run_default(
    "../data/clean/oxides_Tg_train.csv",
    "../data/clean/oxides_Tg_test_rem.csv",
    "Tg",
    "../result/result_oracle/default-model/", "test_rem")

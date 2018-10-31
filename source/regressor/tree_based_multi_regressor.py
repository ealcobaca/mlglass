import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier


class TBMR:
    def __init__(self, alg=("RF"), range_cut=(), seed=None):
        if seed != None:
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

        for i in range(0,len(self.range_cut)+1):
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
        print(pred_root)

        pred_leaf = [self.regrs_leaf[i-1].predict([x])[0] for i,x in zip(pred_root,X)]
        return pred_leaf

data_path = "../../data/clean/oxides_Tg_train.csv"
str_class = "Tg"
data = pd.read_csv(data_path)
X = data.drop([str_class], axis=1).values
y = data[str_class].values

regr = TBMR(alg=("RF"), range_cut=(658.15, 838.15), seed=None)
regr.fit(X,y)
pred = regr.predict(X)

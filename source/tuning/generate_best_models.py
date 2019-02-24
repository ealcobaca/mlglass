import os
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
# from catboost import CatBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

input_file = '../../data/clean/oxides_Tg_train.csv'
output_path = '../../result'
target = 'tg'
regressors = {
    'dt': (DecisionTreeRegressor, {}),
    'knn': (KNeighborsRegressor, {}),
    'mlp': (MLPRegressor, {}),
    'rf': (RandomForestRegressor, {'n_estimators': 100, 'n_jobs': 8}),
    'svr': (SVR, {'gamma': 'auto'})
}


def train_default_models(train_data, regressors, output_path, target):
    for id_reg, (reg, conf) in regressors.items():
        model = reg(**conf)
        model.fit(train_data[:, :-1], train_data[:, -1])
        print('{} generated.'.format(id_reg))
        with open(os.path.join(output_path, id_reg,
                               'default_{}_{}.model'.
                               format(id_reg, target)), 'wb') as f:
            pickle.dump(file=f, obj=model, protocol=-1)


def train_best_models(train_data, regressors, output_path, target):
    for id_reg, (reg, _) in regressors.items():
        with open(
            os.path.join(output_path, id_reg, 'best_configuration_{}_{}_.rcfg'
                         .format(id_reg, target)),
            'rb'
        ) as f:
            conf = pickle.load(f)
        model = reg(**conf[1])
        model.fit(train_data[:, :-1], train_data[:, -1])
        print('{} generated.'.format(id_reg))
        with open(os.path.join(output_path, id_reg,
                               'best_{}_{}.model'.
                               format(id_reg, target)), 'wb') as f:
            pickle.dump(file=f, obj=model, protocol=-1)


def main():
    train_data = pd.read_csv(input_file)
    train_data = train_data.values
    train_default_models(train_data, regressors, output_path, target)
    train_best_models(train_data, regressors, output_path, target)


if __name__ == '__main__':
    main()

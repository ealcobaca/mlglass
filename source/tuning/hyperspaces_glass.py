import os
import datetime
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from sklearn.neural_network import MLPRegressor
from paje.opt.hp_space import HPSpace
from paje.opt.random_search import RandomSearch
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR


def mlp_archictecture_builder():
    hidd1_min = 10
    hidd1_max = 100
    hidd2_min = 0
    hidd2_max = 100
    hidd3_min = 0
    hidd3_max = 100

    def mlp_tuple():
        l1 = np.random.randint(hidd1_min, hidd1_max)
        l2 = np.random.randint(hidd2_min, hidd2_max)
        l3 = np.random.randint(hidd3_min, hidd3_max)

        if l2 == 0:
            return (l1,)
        if l3 == 0:
            return (l1, l2,)

        return (l1, l2, l3)

    return mlp_tuple


def RRMSE(y, y_pred):
    num = np.sum((y - y_pred) ** 2)
    dem = np.sum((y - np.mean(y)) ** 2)
    return np.sqrt(num/dem)


def get_regressor(algorithm):
    if algorithm == 'rf':
        return RandomForestRegressor
    elif algorithm == 'catboost':
        return CatBoostRegressor
    elif algorithm == 'dt':
        return DecisionTreeRegressor
    elif algorithm == 'mlp':
        return MLPRegressor
    elif algorithm =='knn':
        return KNeighborsRegressor
    elif algorithm == 'svr':
        return SVR
    else:
        print('Invalid regression technique.')
        return None


def catboost_space():
    hp_catboost = HPSpace(name='Catboost')
    hp_catboost.add_axis(hp_catboost, 'one_hot_max_size', 'c', None, None,
                         [1])
    hp_catboost.add_axis(hp_catboost, 'iterations', 'z', 100, 1000,
                         np.random.ranf)
    hp_catboost.add_axis(hp_catboost, 'learning_rate', 'r', 0.01, 0.4,
                         np.random.ranf)
    hp_catboost.add_axis(hp_catboost, 'depth', 'z', 1, 16, np.random.ranf)
    hp_catboost.add_axis(hp_catboost, 'l2_leaf_reg', 'r', 0, 7,
                         np.random.ranf)
    hp_catboost.add_axis(hp_catboost, 'random_strength', 'r', 0.0, 1,
                         np.random.ranf)
    hp_catboost.add_axis(hp_catboost, 'bagging_temperature', 'r', 0.0, 1.5,
                         np.random.ranf)
    hp_catboost.add_axis(hp_catboost, 'border_count', 'z', 128, 254,
                         np.random.ranf)
    hp_catboost.add_axis(hp_catboost, 'verbose', 'c', None, None,
                         [False])
    hp_catboost.add_axis(hp_catboost, 'allow_writing_files', 'c', None, None,
                         [False])
    hp_catboost.print(data=True)

    return hp_catboost


def dt_space():
    hp_dt = HPSpace(name='DT')
    # Verificar a possibilidade de permitir qualquer profundidade: None
    hp_dt.add_axis(hp_dt, 'max_depth', 'z', 1, 65, np.random.ranf)
    hp_dt.add_axis(hp_dt, 'min_samples_split', 'z', 2, 500,
                   np.random.ranf)
    hp_dt.add_axis(hp_dt, 'min_samples_leaf', 'z', 1, 250,
                   np.random.ranf)
    hp_dt.print(data=True)

    return hp_dt


def rf_space():
    hp_rf = HPSpace(name='RF')
    hp_rf.add_axis(hp_rf, 'n_estimators', 'z', 100, 1000, np.random.ranf)
    # Tamanho maximo como o numero de features do problema
    hp_rf.add_axis(hp_rf, 'max_depth', 'c', None, None,
                   [None, 10, 20, 30, 40, 60, 70, 80, 90, 100])
    hp_rf.add_axis(hp_rf, 'min_samples_split', 'z', 2, 200,
                   np.random.ranf)
    hp_rf.add_axis(hp_rf, 'min_samples_leaf', 'z', 1, 100,
                   np.random.ranf)
    hp_rf.add_axis(hp_rf, 'max_features', 'z', 2, 65, np.random.ranf)
    hp_rf.print(data=True)

    return hp_rf


def mlp_space():
    hp_mlp = HPSpace(name='MLP')
    hp_mlp.add_axis(hp_mlp, 'hidden_layer_sizes', 'f', None, None,
                    mlp_archictecture_builder())
    hp_mlp.add_axis(hp_mlp, 'solver', 'c', None, None,
                    ['lbfgs', 'sgd', 'adam'])
    hp_mlp.add_axis(hp_mlp, 'activation', 'c', None, None,
                    ['logistic', 'tanh', 'relu'])
    hp_mlp.add_axis(hp_mlp, 'alpha', 'c', None, None,
                    [10 ** -5, 10 ** -4, 10 ** -3])
    hp_mlp.add_axis(hp_mlp, 'learning_rate', 'c', None, None,
                    ['constant', 'adaptive'])
    hp_mlp.add_axis(hp_mlp, 'learning_rate_init', 'r', 10 ** -3, 10 ** -1,
                    np.random.ranf)
    hp_mlp.add_axis(hp_mlp, 'batch_size', 'c', None, None,
                    [200, 500, 1000])
    hp_mlp.add_axis(hp_mlp, 'max_iter', 'z', 200, 1000,
                    np.random.ranf)
    hp_mlp.add_axis(hp_mlp, 'momentum', 'r', 0, 1,
                    np.random.ranf)
    hp_mlp.print(data=True)

    return hp_mlp


def knn_space():
    hp_knn = HPSpace(name='k-NN')
    hp_knn.add_axis(hp_knn, 'n_neighbors', 'z', 1, 1000, np.random.ranf)
    hp_knn.add_axis(hp_knn, 'weights', 'c', None, None,
                    ['uniform', 'distance'])
    hp_knn.print(data=True)

    return hp_knn


def svr_space():
    hp_svr = HPSpace(name='SVR')
    hp_svr.add_axis(hp_svr, 'kernel', 'c', None, None,
                    ['linear', 'poly', 'rbf'])
    hp_svr.add_axis(hp_svr, 'degree', 'c', None, None,
                    [2, 3, 4])
    hp_svr.add_axis(hp_svr, 'gamma', 'c', None, None, ['auto', 'scale'])
    hp_svr.add_axis(hp_svr, 'coef0', 'r', 0.0, 100.0, np.random.ranf)
    hp_svr.add_axis(hp_svr, 'tol', 'c', None, None, [1e-4])
    hp_svr.add_axis(hp_svr, 'C', 'r', 0.01, 100, np.random.ranf)
    # Verificar
    hp_svr.add_axis(hp_svr, 'epsilon', 'r', 5.0, 50.0, np.random.ranf)
    hp_svr.add_axis(hp_svr, 'max_iter', 'c', None, None, [1e+6])
    hp_svr.print(data=True)

    return hp_svr


def get_search_space(algorithm):
    if algorithm == 'rf':
        return rf_space()
    elif algorithm == 'catboost':
        return catboost_space()
    elif algorithm == 'dt':
        return dt_space()
    elif algorithm == 'mlp':
        return mlp_space()
    elif algorithm == 'knn':
        return knn_space()
    elif algorithm == 'svr':
        return svr_space()
    else:
        print('Invalid regression technique.')
        return None


def objective(**kwargs):
    model = kwargs.pop('predictor')
    X = kwargs.pop('X')
    y = kwargs.pop('y')
    loss_func = kwargs.pop('loss_func_tuning')
    seed = kwargs.pop('seed')
    model_name = kwargs.pop('model_name')
    output_folder = kwargs.pop('output_folder')
    data_tag = kwargs.pop('data_tag')

    kf = KFold(n_splits=10, random_state=seed, shuffle=True)
    errors = []
    for train_index, test_index in kf.split(X):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        regressor = model(**kwargs)
        regressor.fit(X_train, y_train)
        error = loss_func(y_test, regressor.predict(X_test))
        errors.append(error)

    with open('{0}/{1}_{2}_{3}_.rcfg'.format(
            output_folder, model_name,
            str(datetime.datetime.now()), data_tag), 'wb') as \
            file:
        out_data = {
            'reg_conf': kwargs,
            'errors': errors
        }
        pickle.dump(file=file, obj=out_data, protocol=-1)
    return np.median(errors)


def main(parameters):
    if (len(parameters) - 1) != 7:
        print("Missing required parameters: (regressor, input_file, \
              output_folder, max_iter, seed)")
    regressor = parameters[1]
    input_file = parameters[2]
    output_folder = parameters[3]
    max_iter = int(parameters[4])
    seed = int(parameters[5])
    n_jobs = int(parameters[6])
    data_tag = parameters[7]

    output_folder = os.path.join(output_folder, regressor, data_tag, str(seed))
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    data = pd.read_csv(input_file)
    X, y = data.iloc[:, :-1].values, data.iloc[:, -1].values

    rs = RandomSearch(get_search_space(algorithm=regressor),
                      max_iter=max_iter, n_jobs=n_jobs)
    best_conf = rs.fmin(
        objective=objective,
        predictor=get_regressor(algorithm=regressor),
        loss_func_tuning=RRMSE,
        X=X,
        y=y,
        seed=seed,
        # id_gen=id_generator(),
        model_name=regressor,
        output_folder=output_folder,
        data_tag=data_tag
    )

    with open('{0}/best_configuration_{1}_{2}_.rcfg'.format(output_folder,
              regressor, data_tag), 'wb') as file:
        pickle.dump(file=file, obj=best_conf, protocol=-1)

    output_folder = os.path.join(output_folder, 'log')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open('{0}/log_{1}_.txt'.format(output_folder,
                                            data_tag), 'w') as file:
        file.write("regressor = {0}\n".format(regressor))
        file.write("input_file = {0}\n".format(input_file))
        file.write("output_folder = {0}\n".format(output_folder))
        file.write("max_iter = {0}\n".format(max_iter))
        file.write("seed = {0}\n".format(seed))
        file.write("n_jobs = {0}\n".format(n_jobs))
        file.write("data_tag = {0}\n".format(data_tag))


if __name__ == '__main__':
    main(sys.argv)

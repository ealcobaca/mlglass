import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from sklearn.neural_network import MLPRegressor
from paje.opt.hp_space import HPSpace
from sklearn.model_selection import KFold


def mlp_archictecture_generator():
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


def get_loss_function():
    def RRMSE(y, y_pred):
        num = np.sum((y - y_pred) ** 2)
        dem = np.sum((y - np.mean(y)) ** 2)
        return np.sqrt(num/dem)

    return RRMSE


def get_regressor(algorithm):
    if algorithm == 'rf':
        return RandomForestRegressor
    elif algorithm == 'catboost':
        return CatBoostRegressor
    elif algorithm == 'dt':
        return DecisionTreeRegressor
    elif algorithm == 'mlp':
        return MLPRegressor
    else:
        print('Invalid regression technique.')
        return None


def objective(**kwargs):
    model = kwargs.pop('predictor')
    X = kwargs.pop('X')
    y = kwargs.pop('y')
    seed = kwargs.pop('seed')
    kf = KFold(n_splits=10, random_state=seed, shuffle=True)
    loss_func = get_loss_function()
    errors = []
    for train_index, test_index in kf.split(X):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        regressor = model(**kwargs)
        regressor.fit(X_train, y_train)
        error = loss_func(y_test, regressor.predict(X_test))
        errors.append(error)
    # Verificar se vamos salvar mais coisas
    return np.median(errors)


def catboost_space():
    hp_catboost = HPSpace(name='Catboost')
    hp_catboost.add_axis(hp_catboost, 'one_hot_max_size', 'z', 2, 5,
                         np.random.ranf)
    hp_catboost.add_axis(hp_catboost, 'iterations', 'z', 100, 1000,
                         np.random.ranf)
    hp_catboost.add_axis(hp_catboost, 'learning_rate', 'r', 0.01, 0.4,
                         np.random.ranf)
    hp_catboost.add_axis(hp_catboost, 'depth', 'z', 6, 10, np.random.ranf)
    hp_catboost.add_axis(hp_catboost, 'l2_leaf_reg', 'z', 3, 7,
                         np.random.ranf)
    hp_catboost.add_axis(hp_catboost, 'random_strength', 'r', 0.3, 1,
                         np.random.ranf)
    hp_catboost.add_axis(hp_catboost, 'bagging_temperature', 'r', 0.5, 1.5,
                         np.random.ranf)
    hp_catboost.add_axis(hp_catboost, 'border_count', 'z', 128, 254,
                         np.random.ranf)
    hp_catboost.print(data=True)

    return hp_catboost


def dt_space():
    hp_dt = HPSpace(name='DT')
    # Verificar a possibilidade de permitir qualquer profundidade: None
    hp_dt.add_axis(hp_dt, 'max_depth', 'z', 4, 15, np.random.ranf)
    hp_dt.add_axis(hp_dt, 'min_samples_split', 'r', 0.0002, 0.01,
                   np.random.ranf)
    hp_dt.add_axis(hp_dt, 'min_samples_leaf', 'r', 0.0001, 0.01,
                   np.random.ranf)
    hp_dt.print(data=True)

    return hp_dt


def rf_space():
    hp_rf = HPSpace(name='RF')
    hp_rf.add_axis(hp_rf, 'n_estimators', 'z', 100, 1000, np.random.ranf)
    hp_rf.add_axis(hp_rf, 'max_depth', 'z', 4, 15, np.random.ranf)
    hp_rf.add_axis(hp_rf, 'min_samples_split', 'r', 0.0002, 0.01,
                   np.random.ranf)
    hp_rf.add_axis(hp_rf, 'min_samples_leaf', 'r', 0.0001, 0.01,
                   np.random.ranf)
    hp_rf.add_axis(hp_rf, 'max_features', 'z', 3, 15, np.random.ranf)
    hp_rf.print(data=True)

    return hp_rf


def mlp_space():
    hp_mlp = HPSpace(name='MLP')
    hp_mlp.add_axis(hp_mlp, 'hidden_layer_sizes', 'f', None, None,
                    mlp_archictecture_generator)
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


def get_search_space(algorithm):
    if algorithm == 'rf':
        return rf_space()
    elif algorithm == 'catboost':
        return catboost_space()
    elif algorithm == 'dt':
        return dt_space()
    elif algorithm == 'mlp':
        return mlp_space()
    else:
        print('Invalid regression technique.')
        return None

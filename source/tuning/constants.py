from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
# from catboost import CatBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

# Constants sections
N_FOLDS_OUTER = 10
N_FOLDS_INNER = 5
N_ITER_TUNING = 500

# Regressors section
REGRESSORS_LIST = ['dt', 'knn', 'svr', 'mlp', 'rf']
REGRESSORS_DEFAULT = {
    'dt': (DecisionTreeRegressor, {}),
    'knn': (KNeighborsRegressor, {}),
    'mlp': (MLPRegressor, {}),
    # 'catboost': (CatBoostRegressor, {'verbose': False,
    #                                  'allow_writing_files': False}),
    'rf': (RandomForestRegressor, {'n_estimators': 100, 'n_jobs': 8}),
    'svr': (SVR, {'gamma': 'auto'})
}
REGRESSORS_FORMATTED = {
    'dt': 'DT',
    'knn': 'k-NN',
    'mlp': 'MLP',
    # 'catboost': 'CatBoost',
    'rf': 'RF',
    'svr': 'SVR',
}


# Targets section
# TARGETS_LIST = ['tg', 'nd300', 'tl']
TARGETS_LIST = ['tg']
TARGETS_FORMATTED = {
    'tg': '$T_g$',
    # 'nd300': 'ND300',
    # 'tl': 'Tliquidus'
}

METRICS_FORMATTED = {
    'relative_deviation': 'RD',
    'R2': '$R^2$',
    'RMSE': 'RMSE',
    'RRMSE': 'RRMSE'
}


# Paths sections
OUTPUT_PATH = '../../result'
SPLIT_DATA_PATH = '../../data/clean/train_test_split'

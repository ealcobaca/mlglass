from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
# from catboost import CatBoostRegressor
# from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
# from sklearn.svm import SVR

# Constants sections
N_FOLDS_OUTER = 10
N_FOLDS_INNER = 5
N_ITER_TUNING = 500

# Regressors section
REGRESSORS_LIST = ['dt', 'knn', 'rf']
# REGRESSORS_LIST = ['catboost', 'dt', 'knn', 'svr', 'mlp', 'rf']
REGRESSORS_DEFAULT = {
    # 'catboost': (CatBoostRegressor, {'verbose': False,
    #                                  'allow_writing_files': False}),
    'dt': (DecisionTreeRegressor, {}),
    'knn': (KNeighborsRegressor, {}),
    # 'mlp': (MLPRegressor, {}),
    'rf': (RandomForestRegressor, {'n_estimators': 100, 'n_jobs': 8}),
    # 'svr': (SVR, {'gamma': 'auto'})
}
REGRESSORS_FORMATTED = {
    # 'catboost': 'CatBoost',
    'dt': 'Cart',
    'knn': 'k-NN',
    # 'mlp': 'MLP',
    'rf': 'RF',
    # 'svr': 'SVR',
}


# Targets
TARGETS_LIST = ['ANY_TEC', 'MOD_UNG', 'ND300', 'NUD300', 'TG', 'TLiq']
# TARGETS_LIST = ['MOD_UNG']
TARGETS_FORMATTED = {
    'ANY_TEC': 'TEC',
    #'MOD_UNG': 'Young',
    'ND300': 'ND300',
    'NUD300': 'NUD300',
    'TG': 'T$_g$',
    'TLiq': 'TLiq'
}

METRICS_FORMATTED = {
    'relative_deviation': 'RD',
    'R2': '$R^2$',
    'RMSE': 'RMSE',
    'RRMSE': 'RRMSE'
}


# Paths sections
OUTPUT_PATH = '/home/mastelini/glass_ml/six_properties_results'
SPLIT_DATA_PATH = '/home/mastelini/glass_ml/six_properties_results/train_test_split'
DATA_PATH = '/home/mastelini/glass_ml/glass_project/playground/run_pipeline/out_data'
DATASET_PREFIX = 'out_pipeline_six_'
REMOVE_ID_COLUMN = True

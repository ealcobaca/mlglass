from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import numpy as np


def RRMSE(target, pred):
    num = np.sum((target - pred) ** 2)
    dem = np.sum((np.mean(target) - target) ** 2)
    if(dem == 0):
        print(target.shape)
        print(target.pred)
        print()
    return np.sqrt(num/dem)


def MSE(target, pred):
    N = len(target)
    return np.sum((target-pred)**2)/N


def RMSE(target, pred):
    return np.sqrt(MSE(target, pred))


def train_regressors(X_train, y_train, regressor, seed):
    reg = None
    if regressor == "RF":
        reg = RandomForestRegressor(
            n_estimators=100,
            criterion="mse",
            max_depth=None,
            min_samples_split=0.01,
            min_samples_leaf=0.005,
            n_jobs=10,
            random_state=seed).fit(X_train,y_train)
    elif regressor == "DT":
        reg = DecisionTreeRegressor(
            criterion="mse",
            splitter="best",
            max_depth=None,
            min_samples_split=0.01,
            min_samples_leaf=0.005,
            random_state=seed).fit(X_train, y_train)
    elif regressor == "XG":
        # data = xgb.DMatrix(data=X_train,label=y_train)
        reg = xgb.XGBRegressor(
            objective ='reg:linear',
            n_jobs=10,
            n_estimators = 100,
            random_state=seed).fit(X_train, y_train)
    elif regressor == "SVM":
        reg = SVR(
            kernel="rbf",
            gamma="auto",
            C=1.0,
            epsilon=0.1,
            max_iter=10000000).fit(X_train, y_train)
    elif regressor == "MLP":
        reg = MLPRegressor(
            hidden_layer_sizes=50,
            activation="relu",
            solver="adam",
            alpha=0.0001,
            max_iter=800,
            early_stopping=True,
            random_state=seed).fit(X_train, y_train)
    else:
        print("Error")
        reg = None
    return reg


def apply_regressors(reg, X_test):
    return reg.predict(X_test)


def compute_performance(y_pred, y_true):
    result = [mean_absolute_error(y_true, y_pred),
              mean_squared_error(y_true, y_pred),
              r2_score(y_true, y_pred),
              RRMSE(y_true, y_pred),
              RMSE(y_true, y_pred)]
    return result

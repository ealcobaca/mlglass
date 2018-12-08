from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from catboost import CatBoostRegressor
import pandas as pd
import numpy as np


def MSE(target, pred):
    N = len(target)
    return np.sum((target-pred)**2)/N


def RRMSE(target, pred):
    num = np.sum((target - pred) ** 2)
    dem = np.sum((np.mean(target) - target) ** 2)
    if(dem == 0):
        print(target.shape)
        print(pred.shape)
        print()
    return np.sqrt(num/dem)


def MARE(target, pred, alpha=1e-6):
    # print(target.shape)
    # print(pred.shape)
    # return 1
    return np.mean((np.abs(target - pred)+alpha)/(target+alpha))


def RMSE(target, pred):
    return np.sqrt(MSE(target, pred))


train_data = pd.read_csv('../../data/clean/oxides_Tg_train.csv')
test_data = pd.read_csv('../../data/clean/oxides_Tg_test.csv')

# ada = AdaBoostRegressor(
#     base_estimator=DecisionTreeRegressor(max_depth=10),
#     n_estimators=100,
#     learning_rate=1,
#     loss='exponential',
#     random_state=2018
# )
ctboost = CatBoostRegressor(iterations=1000, learning_rate=0.7,
                            depth=7, loss_function='RMSE')
# rf = RandomForestRegressor(n_estimators=300)

# ada.fit(train_data.iloc[:, :-1], train_data.iloc[:, -1])
# rf.fit(train_data.iloc[:, :-1], train_data.iloc[:, -1])
ctboost.fit(train_data.iloc[:, :-1], train_data.iloc[:, -1])
# preds = ada.predict(test_data.iloc[:, :-1])
# preds = rf.predict(test_data.iloc[:, :-1])
preds = ctboost.predict(test_data.iloc[:, :-1])

print('RMSE: {}'.format(RMSE(test_data.iloc[:, -1], preds)))
print('RRMSE: {}'.format(RRMSE(test_data.iloc[:, -1], preds)))
print('MARE: {}'.format(MARE(test_data.iloc[:, -1], preds)))

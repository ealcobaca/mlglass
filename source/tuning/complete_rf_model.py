import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


data = pd.read_csv('../../data/clean/data_tg_dupl_rem.csv')

print('Data loaded')

rf = RandomForestRegressor(
    n_estimators=933,
    max_features='sqrt',
    n_jobs=-1
)

rf.fit(data.iloc[:, :-1], data.iloc[:, -1])

print('Model trained... now saving it')

out_file = '../../result/rf/rf_tg_final.model'

with open(out_file, 'wb') as file:
    pickle.dump(file=file, obj=rf, protocol=-1)

print('Done!')

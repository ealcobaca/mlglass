import os
import sys
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

from constants import OUTPUT_PATH as output_path
from constants import REGRESSORS_DEFAULT as regressors
from constants import TARGETS_LIST as targets
from constants import N_FOLDS_OUTER as n_folds
from constants import SPLIT_DATA_PATH as data_path


def train_default_models(train_data, regressors, output_path, target, fold,
                         must_normalize):
    for id_reg, (reg, conf) in regressors.items():
        model_name = os.path.join(
            output_path, id_reg,
            'default_{0}_{1}_fold{2:02d}.model'.format(id_reg, target, fold)
        )
        if os.path.exists(model_name):
            continue
        model = reg(**conf)

        tscaler = None
        if must_normalize:
            tscaler = StandardScaler()
            scaled_y = tscaler.fit_transform(
                train_data[:, -1].reshape(-1, 1))[:, 0]
            model.fit(train_data[:, :-1], scaled_y)
        else:
            model.fit(train_data[:, :-1], train_data[:, -1])
        print('{} generated.'.format(id_reg))
        to_save = {
            'model': model,
            'scaler': tscaler
        }
        with open(model_name, 'wb') as f:
            pickle.dump(file=f, obj=to_save, protocol=-1)


def train_best_models(train_data, regressors, output_path, target, fold,
                      must_normalize):
    for id_reg, (reg, _) in regressors.items():
        with open(
            os.path.join(
                output_path, id_reg,
                'best_configuration_{0}_{1}_fold{2:02d}_.rcfg'.format(
                    id_reg, target, fold
                )
            ),
            'rb'
        ) as f:
            conf = pickle.load(f)
        model_name = os.path.join(
            output_path, id_reg,
            'best_{0}_{1}_fold{2:02d}.model'.format(id_reg, target, fold)
        )

        if os.path.exists(model_name):
            continue

        if id_reg == 'catboost':
            conf[1]['thread_count'] = None
        model = reg(**conf[1])

        tscaler = None
        if must_normalize:
            tscaler = StandardScaler()
            scaled_y = tscaler.fit_transform(
                train_data[:, -1].reshape(-1, 1))[:, 0]
            model.fit(train_data[:, :-1], scaled_y)
        else:
            model.fit(train_data[:, :-1], train_data[:, -1])

        print('{} generated.'.format(id_reg))
        to_save = {
            'model': model,
            'scaler': tscaler
        }
        with open(model_name, 'wb') as f:
            pickle.dump(file=f, obj=to_save, protocol=-1)


def main(parameters):
    must_normalize = bool(parameters[1])
    for target in targets:
        print('Training models for {}'.format(target))
        for k in range(1, n_folds + 1):
            input_file = '{0}/{1}_train_fold{2:02d}.csv'.format(
                data_path, target, k
            )

            train_data = pd.read_csv(input_file)
            train_data = train_data.values
            print('Default models')
            print('Fold {:02d}'.format(k))
            train_default_models(
                train_data, regressors, output_path, target, k, must_normalize
            )
            print('Tuned models')
            print('Fold {:02d}'.format(k))
            train_best_models(
                train_data, regressors, output_path, target, k, must_normalize
            )


if __name__ == '__main__':
    main(sys.argv)

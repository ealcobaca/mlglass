import os
import pandas as pd
import pickle

from constants import OUTPUT_PATH as output_path
from constants import REGRESSORS_DEFAULT as regressors
from constants import TARGETS_LIST as targets
from constants import N_FOLDS_OUTER as n_folds
from constants import SPLIT_DATA_PATH as data_path


def train_default_models(train_data, regressors, output_path, target, fold):
    for id_reg, (reg, conf) in regressors.items():
        model_name = os.path.join(
            output_path, id_reg,
            'default_{0}_{1}_fold{2:02d}.model'.format(id_reg, target, fold)
        )
        if os.path.exists(model_name):
            continue
        model = reg(**conf)
        model.fit(train_data[:, :-1], train_data[:, -1])
        print('{} generated.'.format(id_reg))
        with open(model_name, 'wb') as f:
            pickle.dump(file=f, obj=model, protocol=-1)


def train_best_models(train_data, regressors, output_path, target, fold):
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
        model.fit(train_data[:, :-1], train_data[:, -1])
        print('{} generated.'.format(id_reg))
        with open(model_name, 'wb') as f:
            pickle.dump(file=f, obj=model, protocol=-1)


def main():
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
                train_data, regressors, output_path, target, k
            )
            print('Tuned models')
            print('Fold {:02d}'.format(k))
            train_best_models(train_data, regressors, output_path, target, k)


if __name__ == '__main__':
    main()

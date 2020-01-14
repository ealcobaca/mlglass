import zipfile
import tempfile
import os
import pickle
import numpy as np
from constants import TARGETS_LIST as targets
from constants import REGRESSORS_LIST as regressors
from constants import OUTPUT_PATH as output_path
from constants import N_FOLDS_OUTER as n_folds
from constants import N_ITER_TUNING as n_iter


def get_best_conf(data_dir, regressor, target):
    best_error = float('Inf')
    best_conf = None

    for iter in range(n_iter):
        iter_name = '{0}_{1}_{2}_.rcfg'.format(regressor, iter, target)
        with open(os.path.join(data_dir, iter_name), 'rb') as f:
            t_data = pickle.load(f)
            error = np.mean(t_data['errors'])
            conf = t_data['reg_conf']
            if error < best_error:
                best_error = error
                best_conf = conf
    return best_error, best_conf


if __name__ == '__main__':
    for target in targets:
        for regressor in regressors:
            # data_path = '{0}/{1}/{1}-{2}.zip'.format(
            #     output_path, regressor, target
            # )
            data_path = '{0}/{1}/{2}'.format(
                output_path, regressor, target
            )
            with tempfile.TemporaryDirectory() as td:
                # with zipfile.ZipFile(data_path, 'r') as zf:
                #     zf.extractall(path=td)
                for k in range(1, n_folds + 1):
                    # dir_prefix = '{0}/outer_fold{1}'.format(
                    #     target, k
                    # )
                    dir_prefix = 'outer_fold{}'.format(k)
                    f_name = 'best_configuration_{0}_{1}_fold{2:02d}_.rcfg'.\
                        format(regressor, target, k)
                    best_error, best_conf = get_best_conf(
                        # '{0}/{1}'.format(td, dir_prefix), regressor, target
                        '{0}/{1}'.format(data_path, dir_prefix),
                        regressor, target
                    )
                    out_file = '{0}/{1}/{2}'.format(
                        output_path, regressor, f_name
                    )
                    with open(out_file, 'wb') as f:
                        pickle.dump(
                            file=f, obj=(best_error, best_conf), protocol=-1
                        )

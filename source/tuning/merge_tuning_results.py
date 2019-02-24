import zipfile
import tempfile
import os
import pickle

regressor = 'mlp'
target = 'tg'
tuning_prefix = [100, 200, 300, 400, 500]


def get_best_conf(data_path, folder_prefix, tuning_prefix, file_name):
    best_error = float('Inf')
    best_conf = None
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(data_path, 'r') as zf:
            zf.extractall(path=td)
        for t_prefix in tuning_prefix:
            with open(os.path.join(td, folder_prefix, str(t_prefix),
                      file_name), 'rb') as f:
                t_data = pickle.load(f)
            error = t_data[0]
            conf = t_data[1]
            if error < best_error:
                best_error = error
                best_conf = conf
    return best_error, best_conf


if __name__ == '__main__':
    data_path = '../../result/{0}/result_{0}.zip'.format(regressor)
    folder_prefix = 'result/{0}/{1}'.format(regressor, target)
    file_name = 'best_configuration_{0}_{1}_.rcfg'.format(regressor, target)
    best_error, best_conf = get_best_conf(data_path, folder_prefix,
                                          tuning_prefix, file_name)
    with open('../../result/{0}/best_configuration_{0}_{1}_.rcfg'.
              format(regressor, target), 'wb') as f:
        pickle.dump(file=f, obj=(best_error, best_conf), protocol=-1)

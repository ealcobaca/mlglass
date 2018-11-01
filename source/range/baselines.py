import numpy as np
import pandas as pd
import pickle
from regressors import train_regressors, apply_regressors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def save_obj(object, file_name):
    pickle.dump(obj=object, file=open(file_name, "wb"), protocol=-1)


def create_baselines(tr_data_path, ts_data_path, ts_ext_data_path,
                     output_path):
    seed = 2018
    regressors = ['MLP', 'DT', 'RF']

    data_tr = pd.read_csv(tr_data_path)
    X_tr, y_tr = data_tr.values[:, :-1], data_tr.values[:, -1:][:, 0]

    data_ts = pd.read_csv(ts_data_path)
    X_ts, y_ts = data_ts.values[:, :-1], data_ts.values[:, -1:][:, 0]

    data_ts_ext = pd.read_csv(ts_ext_data_path)
    X_ts_ext, y_ts_ext = data_ts_ext.values[:, :-1], \
        data_ts_ext.values[:, -1:][:, 0]
    test_index = data_ts.index
    test_ext_index = data_ts_ext.index

    # Baselines
    # Baseline #1: using the whole available data
    for reg_name in regressors:
        reg = train_regressors(X_tr, y_tr, reg_name, seed)
        pred = apply_regressors(reg=reg, X_test=X_ts)
        pred_ext = apply_regressors(reg=reg, X_test=X_ts_ext)

        # # Save regressor
        # save_obj(reg, '{0}models/baseline_raw_{1}.reg'.format(
        #     output_path, reg_name)
        # )
        # Save predictions
        save_obj(
            [test_index, y_ts, pred],
            '{0}log/predictions_raw_{1}.list'.format(output_path, reg_name)
        )
        save_obj(
            [test_ext_index, y_ts_ext, pred_ext],
            '{0}log/predictions_ext_raw_{1}.list'.format(output_path, reg_name)
        )
    # Baseline #2: using pca to reduce dimensions

    # Firstly, apply standardization
    scaler = StandardScaler().fit(X_tr)
    # save_obj(scaler, '{0}models/z-score_scaler.sclr'.format(
    #     output_path)
    # )
    z_X_tr, z_X_ts, z_XT_ts_ext = scaler.transform(X_tr), \
        scaler.transform(X_ts), scaler.transform(X_ts_ext)

    pca = PCA(n_components=X_tr.shape[1])
    pca.fit(X_tr)
    # save_obj(pca, '{0}models/extractor_pca.extr'.format(
    #     output_path)
    # )

    # Get a suitable number of Principal Components
    n_comp = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1
    # transform the data
    X_tr_pca = pca.transform(z_X_tr)[:, :n_comp]
    X_ts_pca = pca.transform(z_X_ts)[:, :n_comp]
    X_ts_ext_pca = pca.transform(z_XT_ts_ext)[:, :n_comp]

    for reg_name in regressors:
        reg = train_regressors(X_tr_pca, y_tr, reg_name, seed)
        pred = apply_regressors(reg=reg, X_test=X_ts_pca)
        pred_ext = apply_regressors(reg=reg, X_test=X_ts_ext_pca)

        # Save regressor
        # save_obj(reg, '{0}models/baseline_pca{1}_{2}.reg'.format(
        #     output_path, n_comp, reg_name)
        # )
        # Save predictions
        save_obj(
            [test_index, y_ts, pred],
            '{0}log/predictions_pca{1}_{2}.list'.format(
                output_path, n_comp, reg_name
            )
        )
        save_obj(
            [test_ext_index, y_ts_ext, pred_ext],
            '{0}log/predictions_ext_pca{1}_{2}.list'.format(
                output_path, n_comp, reg_name
            )
        )

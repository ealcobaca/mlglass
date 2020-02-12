import glob
import pickle
import pandas as pd
import numpy as np
from constants import TARGETS_LIST as targets
# Verificar se o local dos resultados está correto no arquivo constants.py
from constants import OUTPUT_PATH as output_path


for target in targets:
    print('\n{}'.format(target))
    files = glob.glob(
        "{}/logs/performance_best_models_{}_*".format(output_path, target)
    )

    data_frames = []
    for f in files:
        df = pd.read_csv(f, index_col=0)
        data_frames.append(df)

    result = pd.concat(data_frames).loc["RRMSE", :]
    result = result.reset_index(drop=True)

    median = np.abs(result - result.median())
    idxmin = median.idxmin()

    for alg, idx in zip(idxmin.index, idxmin.values):
        # Comentar esse if se quiser saber a configuração para todos
        # os algoritmos
        if alg != "rf":
            continue
        print(
            "Median configuration for {0} was at fold {1:02d}".format(
                alg, idx + 1
            )
        )
        conf_path = '{0}/{1}/best_configuration_{1}_{2}_fold{3:02d}_.rcfg'.\
            format(output_path, alg, target, int(idx + 1))
        print('Configuration:')
        with open(conf_path, 'rb') as f:
            median_conf = pickle.load(f)
            print(median_conf[1])

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from constants import REGRESSORS_FORMATTED as regressors
from constants import TARGETS_LIST as targets
from constants import N_FOLDS_OUTER as n_folds
from constants import OUTPUT_PATH as prefix_out


intervals = [0, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300,
             float('Inf')]
offset = 50
log_path = '{0}/logs'.format(prefix_out)


def relative_deviation(obs, pred):
    obs = np.array(obs)
    pred = np.array(pred)
    # rds = np.abs(obs-pred)/obs * 100
    rds = obs - pred
    return rds.tolist()


for target in targets:
    logs = {}
    for regressor in regressors:
        logs[regressor] = []
        logs['{}_pred'.format(regressor)] = []

    for k in range(1, n_folds + 1):
        log_name = '{0}/predictions_best_models_{1}_fold{2:02d}.csv'.format(
            log_path, target, k
        )
        log = pd.read_csv(log_name)
        for regressor in regressors:
            logs[regressor].extend(
                log.loc[:, regressor].values
            )
            logs['{}_pred'.format(regressor)].extend(
                log.loc[:, '{}_pred'.format(regressor)].values
            )
    for regressor, regressorf in regressors.items():
        bars = []
        for i in range(1, len(intervals)):
            idx = [j for j, v in enumerate(logs[regressor])
                   if v > intervals[i - 1] + offset and
                   v <= intervals[i] + offset]

            bars.append(
                (
                    [logs[regressor][j] for j in idx],
                    [logs['{}_pred'.format(regressor)][j] for j in idx]
                )
            )
        print(len(bars))
        for b in bars:
            print(len(b[0]))
        exit()
        bars_ = {}

        for i in range(1, len(intervals) - 1):
            bars_[str(intervals[i])] = \
                relative_deviation(bars[i - 1][0], bars[i - 1][1])

        data = [bars_[k] for k in bars_]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.axhline(linewidth=1, color='gray', linestyle='--')
        ax.boxplot(
            data,
            labels=[str(intervals[i]) for i in range(1, len(intervals) - 1)]
        )
        ax.set_title(regressorf)
        plt.ylabel('Observed - Predicted')
        plt.ylim([-700, 700])
        plt.tight_layout()
        plt.savefig('{0}/boxplots/bands_{1}.png'.format(prefix_out, regressor))

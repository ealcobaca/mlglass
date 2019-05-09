import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from constants import REGRESSORS_FORMATTED as regressors
from constants import TARGETS_LIST as targets
from constants import N_FOLDS_OUTER as n_folds
from constants import OUTPUT_PATH as prefix_out
from constants import DATA_PATH as input_path
from collections import Counter


intervals = [0, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300,
             float('Inf')]
offset = 50
top_k = 5
log_path = '{0}/logs'.format(prefix_out)


def deviation(obs, pred):
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

    # To relate the sample idx to their fold
    idx_all_fold = {}
    for k in range(1, n_folds + 1):
        log_name = '{0}/predictions_best_models_{1}_fold{2:02d}.csv'.format(
            log_path, target, k
        )
        log = pd.read_csv(log_name)

        # Links the sample id with its fold
        s_f = len(log)
        s_dict = len(idx_all_fold)
        idx_all_fold.update(
            {s_dict + m: (m, k) for m in range(s_f)}
        )
        for regressor in regressors:
            logs[regressor].extend(
                log.loc[:, regressor].values
            )
            logs['{}_pred'.format(regressor)].extend(
                log.loc[:, '{}_pred'.format(regressor)].values
            )

    worse_cases = {}
    who_missed_it = {}
    for regressor, regressorf in regressors.items():
        worse_cases[regressor] = []
        bars = []
        idx_bars = []
        for i in range(1, len(intervals)):
            idx = [j for j, v in enumerate(logs[regressor])
                   if v > intervals[i - 1] + offset and
                   v <= intervals[i] + offset]
            idx_bars.append(idx)
            bars.append(
                (
                    [logs[regressor][j] for j in idx],
                    [logs['{}_pred'.format(regressor)][j] for j in idx]
                )
            )
        bars_ = {}

        for i in range(1, len(intervals) - 1):
            bars_[str(intervals[i])] = \
                deviation(bars[i - 1][0], bars[i - 1][1])
            # Skip the last band (too few elements)
            if intervals[i] == 1300:
                continue
            # Get deviations
            devs = np.array(bars_[str(intervals[i])])
            # Sort them
            sorted_idx = devs.argsort()
            # The greatest negative errors
            worse_cases[regressor].extend(
                [(idx_bars[i - 1][sorted_idx[m]], devs[sorted_idx[m]])
                    for m in range(top_k)]
            )
            # The greatest positive errors
            worse_cases[regressor].extend(
                [(idx_bars[i - 1][sorted_idx[-(m+1)]],
                  devs[sorted_idx[-(m+1)]])
                 for m in range(top_k)]
            )

            # Save the regressors who missed the samples for later conference
            for m in range(top_k):
                ws = idx_bars[i - 1][sorted_idx[m]]
                wg = idx_bars[i - 1][sorted_idx[-(m+1)]]
                if ws in who_missed_it:
                    who_missed_it[ws].append(regressor)
                else:
                    who_missed_it[ws] = [regressor]
                if wg in who_missed_it:
                    who_missed_it[wg].append(regressor)
                else:
                    who_missed_it[wg] = [regressor]

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
    counter_worse = Counter()
    for regressor in regressors:
        counter_worse.update(
            [elem[0] for elem in worse_cases[regressor]]
        )
    err_log = {}
    err_log['id'] = []
    sorted_c = sorted(counter_worse, key=counter_worse.get, reverse=True)

    for j in sorted_c:
        id, fold = idx_all_fold[j]
        l_name = '{0}/train_test_split_named/{1}_test_fold{2:02d}.csv'.format(
            input_path, target, fold
        )
        log = pd.read_csv(l_name)
        err_log['id'].append(log.iloc[id, 0])
    err_log['qtd_worse'] = [counter_worse[k] for k in sorted_c]
    err_log['missed_it'] = ['-'.join(who_missed_it[k]) for k in sorted_c]
    pd.DataFrame.from_dict(err_log).to_csv(
        '{0}/boxplots/worst_cases_{1}_top{2}.csv'.format(
            prefix_out, target, top_k
        ),
        index=False
    )

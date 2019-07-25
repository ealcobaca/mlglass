import os
import pickle
import pandas as pd
from constants import DATA_PATH as input_path
from constants import OUTPUT_PATH as output_path
from multiprocessing import Pool


def path2rule(estimator, features, data, tree_id, out_path):
    for i in range(len(data)):
        sample = data[i, :].reshape(1, -1)

        prediction = estimator.predict(sample[:, :-1])
        feature = estimator.tree_.feature
        threshold = estimator.tree_.threshold

        node_indicator = estimator.decision_path(sample[:, :-1])
        leave_id = estimator.apply(sample[:, :-1])

        node_index = node_indicator.indices[node_indicator.indptr[0]:
                                            node_indicator.indptr[1]]
        decisions = []
        for node_id in node_index:
            if leave_id[0] == node_id:
                continue

            if (sample[0, feature[node_id]] <= threshold[node_id]):
                threshold_sign = '<='
            else:
                threshold_sign = '>'

            decisions.append(
                '{0} {1} {2:.3f}'.format(
                    features[feature[node_id]], threshold_sign,
                    threshold[node_id]
                )
            )
        # decisions = simplify_rules(decisions)
        rule = ' & '.join(decisions)
        line = ','.join([
            rule, str(sample[0, -1]), str(prediction[0]), str(i),
            str(tree_id)
        ])
        file_name = '{0}/rules_t{1:03d}.txt'.format(
            out_path, tree_id
        )
        with open(file_name, 'a') as f:
            f.write('\n' + line)


n_cpus = 40
if __name__ == '__main__':
    model_name = '{0}/rf/rf_tg_final.model'.format(output_path)

    with open(model_name, 'rb') as f:
        rf = pickle.load(f)

    data = pd.read_csv(
        '{0}/data_tg_dupl_rem.csv'.format(input_path)
    )
    features_names = list(data)
    data = data.values

    out_path = '{0}/interpretation/raw_rf_rules'.format(output_path)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for j in range(len(rf.estimators_)):
        folder = '{0}/t{1:03d}'.format(out_path, j)
        if not os.path.exists(folder):
            os.makedirs(folder)
    print('Folders created')
    pool = Pool(processes=n_cpus)
    for j, estimator in enumerate(rf.estimators_):
        pool.apply_async(
            path2rule,
            (estimator, features_names, data, j, out_path)
        )
    pool.close()
    pool.join()

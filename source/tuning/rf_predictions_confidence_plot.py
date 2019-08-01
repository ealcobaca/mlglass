import pickle
import numpy as np
import pandas as pd
from constants import SPLIT_DATA_PATH as input_data_path
from constants import OUTPUT_PATH as output_path


def get_rf_predictions_and_confidence(rf, X_new):
    std_data = np.zeros((len(X_new), len(rf.estimators_)))
    predictions = rf.predict(X_new)

    for t, tree in enumerate(rf.estimators_):
        std_data[:, t] = tree.predict(X_new)
    confidence = np.std(std_data, axis=1)
    return predictions, confidence


if __name__ == '__main__':
    predictions = []
    confidence = []
    observations = []
    for k in range(10):
        test = pd.read_csv(
            '{0}/tg_test_fold{1:02d}.csv'.format(
                input_data_path, k + 1
            )
        ).values

        model_name = '{0}/rf/best_rf_tg_fold{1:02d}.model'.format(
            output_path, k + 1
        )
        with open(model_name, 'rb') as f:
            rf = pickle.load(f)
        aux_p, aux_c = get_rf_predictions_and_confidence(
            rf, test[:, :-1]
        )
        observations.extend(test[:, -1].tolist())
        predictions.extend(aux_p.tolist())
        confidence.extend(aux_c.tolist())

    with open('{0}/rf/confidence_pred_data.data'.format(output_path), 'wb') as f:
        pickle.dump(
            obj=(observations, predictions, confidence),
            file=f,
            protocol=-1
        )

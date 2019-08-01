import pickle
import numpy as np
import pandas as pd
from constants import SPLIT_DATA_PATH as input_data_path
from constants import OUTPUT_PATH as output_path
import matplotlib.pyplot as plt


def R2(target, pred):
    return np.corrcoef(target, pred)[0, 1] ** 2


def sample_relative_deviation(obs, pred):
    return np.abs(obs - pred)/obs * 100


def get_rf_predictions_and_confidence(rf, X_new):
    std_data = np.zeros((len(X_new), len(rf.estimators_)))
    predictions = rf.predict(X_new)

    for t, tree in enumerate(rf.estimators_):
        std_data[:, t] = tree.predict(X_new)
    confidence = np.std(std_data, axis=1)
    return predictions, confidence


def plot_confidence(observations, predictions, confidence):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        range(len(observations)), predictions, 'r-', label='Predictions',
        alpha=.7
    )
    ax.fill_between(
        range(len(observations)),
        predictions - confidence,
        predictions + confidence,
        color='blue', alpha=.5, label='Confidence'
    )
    ax.plot(range(len(observations)), observations, 'k-', label='Observations')
    ax.set_xlabel('Material')
    ax.set_ylabel('$T_g$')
    fig.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # predictions = []
    # confidence = []
    # observations = []
    # for k in range(10):
    #     test = pd.read_csv(
    #         '{0}/tg_test_fold{1:02d}.csv'.format(
    #             input_data_path, k + 1
    #         )
    #     ).values
    #
    #     model_name = '{0}/rf/best_rf_tg_fold{1:02d}.model'.format(
    #         output_path, k + 1
    #     )
    #     with open(model_name, 'rb') as f:
    #         rf = pickle.load(f)
    #     aux_p, aux_c = get_rf_predictions_and_confidence(
    #         rf, test[:, :-1]
    #     )
    #     observations.extend(test[:, -1].tolist())
    #     predictions.extend(aux_p.tolist())
    #     confidence.extend(aux_c.tolist())
    #
    # with open('{0}/rf/confidence_pred_data.data'.format(output_path), 'wb') as f:
    #     pickle.dump(
    #         obj=(observations, predictions, confidence),
    #         file=f,
    #         protocol=-1
    #     )
    with open('{0}/rf/confidence_pred_data.data'.format(output_path), 'rb') as f:
        observations, predictions, confidence = pickle.load(f)

    observations = np.array(observations)
    predictions = np.array(predictions)
    confidence = np.array(confidence)
    order = np.argsort(observations)
    plot_confidence(
        observations[order], predictions[order], confidence[order]
    )

    print(
        'R2 RDxSTD_Trees:',
        R2(sample_relative_deviation(observations, predictions), confidence)
    )

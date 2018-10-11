import numpy as np
from sklearn.model_selection import cross_val_predict
import pandas as pd


class StackedSingleTarget:
    INTERNAL_CV = 'internal_cv'
    PREDICTIONS = 'predictions'
    TARGETS_VALUES = 'targets_values'

    def __init__(self, regressor, regressor_params, method='predictions',
                 n_part=None):
        self._regressor = regressor
        self._regressor_params = regressor_params

        if method == self.INTERNAL_CV:
            if n_part is None or not isinstance(n_part, int) or n_part <= 2:
                raise ValueError(
                    '"n_part" must be a interger greater than 2'
                )
            self._cv = n_part
        self.method = method

        if method not in [self.INTERNAL_CV, self.PREDICTIONS,
                          self.TARGETS_VALUES]:
            raise ValueError('Invalid "method" value (Options are: \
                             "internal_cv", "predictions", "targets_values").')

    def fit(self, X, Y):
        if isinstance(X, pd.core.frame.DataFrame):
            X = X.values
        elif not isinstance(X, np.ndarray):
            raise ValueError('"X" must be a numpy matrix or pandas \
                             DataFrame')
        if isinstance(Y, pd.core.frame.DataFrame):
            Y = Y.values
        elif not isinstance(Y, np.ndarray):
            raise ValueError('"Y" must be a numpy matrix or pandas \
                             DataFrame')

        self.n_targets = Y.shape[1]

        self._base_models = {}
        self._meta_models = {}

        # Training the base models
        for t in range(self.n_targets):
            self._base_models[t] = self._regressor(**self._regressor_params)
            self._base_models[t].fit(X, Y[:, t])

        base_predictions = np.zeros_like(Y)

        # Getting the base predictions
        if self.method == self.INTERNAL_CV:
            for t in range(self.n_targets):
                base_predictions[:, t] = cross_val_predict(
                    self._regressor(**self._regressor_params),
                    X,
                    Y[:, t],
                    cv=self._cv
                )
        elif self.method == self.PREDICTIONS:
            for t in range(self.n_targets):
                base_predictions[:, t] = self._base_models[t].predict(X)
        elif self.method == self.TARGETS_VALUES:
            base_predictions = Y.copy()

        # Augmented training set
        X_aug = np.column_stack((X, base_predictions))

        # Training the meta_models
        for t in range(self.n_targets):
            self._meta_models[t] = self._regressor(**self._regressor_params)
            self._meta_models[t].fit(X_aug, Y[:, t])

    def predict(self, X):
        if isinstance(X, pd.core.frame.DataFrame):
            X = X.values
        elif not isinstance(X, np.ndarray):
            raise ValueError('"X" must be a numpy matrix or pandas \
                             DataFrame')

        n_rows = X.shape[0]
        base_predictions = np.zeros((n_rows, self.n_targets))
        meta_predictions = np.zeros((n_rows, self.n_targets))

        for t in range(self.n_targets):
            base_predictions[:, t] = self._base_models[t].predict(X)

        X_aug = np.column_stack((X, base_predictions))

        for t in range(self.n_targets):
            meta_predictions[:, t] = self._meta_models[t].predict(X_aug)

        return meta_predictions

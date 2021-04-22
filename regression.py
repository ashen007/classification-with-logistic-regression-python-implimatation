import numpy as np


class LinearRegression:
    def __init__(self):
        self.coef_ = 0
        self.intercept_ = 0

    def fit(self, x, y):
        ones = np.ones([x.shape[0], 1])
        x = np.hstack([x, ones])
        self.coef_, self.intercept_ = np.linalg.lstsq(x, y, rcond=None)[0]

        return self.coef_, self.intercept_

    def predict(self, x):
        ones = np.ones([x.shape[0], 1])
        x = np.hstack([x, ones])

        return np.dot(x, [self.coef_, self.intercept_])

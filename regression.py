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


class LogisticRegression:
    def __init__(self):
        self.weights_ = 0
        self.cost = []

    def fit(self, x, y, learning_rate, epochs):
        weights_ = np.zeros(x.shape[1] + 1)
        x = np.hstack([x, np.ones((x.shape[0], 1))])

        for i in range(epochs):
            hypothesis = 1 / (1 + np.exp(-(np.dot(x, weights_))))
            loss = y - hypothesis
            gradient = np.dot(x.T, loss) / len(y)
            weights_ = weights_ - gradient * learning_rate
            cost = (-1 / len(y)) * (np.sum((y * np.log(hypothesis)) + ((1 - y) * (np.log(1 - hypothesis)))))
            self.cost.append(cost)

        self.weights_ = weights_
        return self.weights_

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
        theta = np.zeros(x.shape[1] + 1)
        ones = np.ones(x.shape[0])
        x = np.hstack((np.asarray(x), ones.reshape(-1, 1)))

        for i in range(epochs):
            hypothesis = 1 / (1 + np.exp(-(np.dot(x, theta))))
            loss = hypothesis - y
            gradient = np.dot(x.T, loss) / len(y)
            theta = theta - gradient * learning_rate
            cost = (-1 / len(y)) * (np.sum((y * np.log(hypothesis)) + ((1 - y) * (np.log(1 - hypothesis)))))
            self.cost.append(cost)

        self.weights_ = theta
        return theta

    def predict(self, x):
        ones = np.ones(x.shape[0])
        x = np.hstack((np.asarray(x), ones.reshape(-1, 1)))
        pred = 1 / (1 + np.exp(-(np.dot(x, self.weights_))))

        return pred

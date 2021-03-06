import numpy as np
import scipy.stats as ss
from scipy.special import softmax
from feature_engine.encoding import OneHotEncoder


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


class MultiClassLogisticRegression:
    def __init__(self):
        self.weights_ = []
        self.cost = []

    def encorder(self, y):
        """Y dataframe"""
        encode = OneHotEncoder()
        encode.fit(y)
        return encode.transform(y)

    def fit(self, x, y, learning_rate, epochs):
        """Y dataframe"""
        ones = np.ones(x.shape[0])
        y = self.encorder(y)
        x = np.hstack((np.asarray(x), ones.reshape(-1, 1)))
        theta = np.zeros([x.shape[1], y.shape[1]])

        for i in range(epochs):
            hypothesis = softmax(np.dot(x, theta),axis=1)
            loss = hypothesis - y
            gradient = np.dot(x.T, loss) / len(y)
            theta = theta - gradient * learning_rate
            cost = 1/len(y)*(np.trace(np.dot(np.dot(x,theta),y.T))) + np.sum(np.log(np.sum(np.exp(np.dot(x,theta)),axis=1)))
            self.cost.append(cost)

        self.weights_ = theta
        return theta

    def predict(self,x):
        ones = np.ones(x.shape[0])
        x = np.hstack((np.asarray(x), ones.reshape(-1, 1)))
        return np.argmax(softmax(np.dot(x,self.weights_),axis=1),axis=1)


class KNN:
    def __init__(self,x,y,k=5):
        self.hyper_parameter = x
        self.target = y
        self.neighbors = k
        self.predicted_cluster = []

    def predict(self,x):
        # add rules for inputs: TODO
        for i in range(x.shape[0]):
            distances = np.sqrt(np.sum(np.square(self.hyper_parameter - x.loc[i]),axis=1))
            closest_neighbors = self.target.loc[distances.sort_values().head(self.neighbors).index]
            cluster = ss.mode(closest_neighbors.values).mode[0][0]
            self.predicted_cluster.append(cluster)

        return self.predicted_cluster
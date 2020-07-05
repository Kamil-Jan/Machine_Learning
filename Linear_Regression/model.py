import numpy as np


class Model:
    def __init__(self, data, output):
        """
        :data - Matrix of data: e.g. np.zeros((m, j))
        :output - Actual output for given data: np.zeros((m, 1))
        :theta - Vector of thetas
        
        where:
            j - number of features
            m - number of examples
        """
        self.data = np.c_[np.ones((len(data), 1)), data] # add bias units
        self.output = output
        self.theta = np.zeros((len(self.data[0]), 1))

    def hypothesis(self, X):
        """
        Returns predicted value depending on X.
        
        :X - column vector of features.
        """
        return np.c_[np.ones((len(X), 1)), X].dot(self.theta)

    def cost_function(self, theta):
        """
        Calculates how wrong the model is
        in terms of its prediction.
        """
        m = len(self.output)

        predictions = np.dot(self.data, theta)
        cost = (1 / (2 * m)) * np.sum(np.square(predictions - self.output))
        return cost

    def gradient_descent(self, iterations=100, learning_rate=0.01):
        """
        Gradient descent algorithm.
        """
        m = len(self.output)
        J_history = np.zeros((iterations, 1))
        for i in range(iterations):
            prediction = np.dot(self.data, self.theta)
            self.theta = self.theta - learning_rate * ((1 / m) * self.data.T.dot((prediction - self.output)))
            J_history[i][0] = self.cost_function(self.theta)
        return J_history

    def reg_cost_function(self, theta, lambda_val=0.01):
        m = len(self.output)
        predictions = np.dot(self.data, theta)
        reg_value = lambda_val * np.sum(np.square(theta[1:][0]))
        cost = ((1 / (2 * m)) * np.sum(np.square(predictions - self.output))) + reg_value
        return cost

    def reg_gradient_descent(self, iterations=100, learning_rate=0.01, lambda_val=0.01):
        m = len(self.output)
        J_history = np.zeros((iterations, 1))
        for i in range(iterations):
            prediction = np.dot(self.data, self.theta)

            reg_value = 1 - learning_rate * (lambda_val / m)
            self.theta = self.theta * reg_value - learning_rate * (self.data.T.dot((prediction - self.output)) / m)
            J_history[i][0] = self.reg_cost_function(self.theta, lambda_val)
        return J_history

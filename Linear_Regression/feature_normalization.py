import numpy as np


def feature_normalize(X):
    """
    Normalizes the range of features of data
    for faster convergence with global optima.
    """
    X_norm = 0
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)

    X_norm = X - mu
    X_norm = X_norm / sigma
    return X_norm, mu, sigma

def convert_values(values, mu, sigma):
    """
    Normalizes values using given mu and
    sigma values.
    """
    return (values - mu) / sigma

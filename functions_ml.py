import numpy as np

def compute_loss_mse(y, tx, w):
    """
    Calculate the loss using either MSE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    e = np.squeeze(y - (tx @ w))
    return np.dot(e, e) / (2.0 * e.size)


def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using normal equations.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
    """
    n = y.shape[0]
    lambda_prime_ = lambda_ * 2 * n
    w = np.linalg.solve(tx.T @ tx + lambda_prime_ * np.identity(tx.shape[1]), tx.T @ y)
    return w, compute_loss_mse(y, tx, w)
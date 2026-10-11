"""Logistic regression written by hand with NumPy (Task 3.1-3.3)."""
import numpy as np


def sigmoid(z):
    z = np.clip(z, -500, 500)               # avoid overflow in exp
    return 1.0 / (1.0 + np.exp(-z))


def hypothesis(X, theta):
    """h_theta(x) = g(theta^T x) for every row of X (X already has x0 = 1)."""
    return sigmoid(X @ theta)


def cost(X, y, theta):
    """J(theta) = -(1/n) * sum[ y log h + (1-y) log(1-h) ], h clipped."""
    h = np.clip(hypothesis(X, theta), 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(h) + (1 - y) * np.log(1 - h)))


def fit_logistic(X, y, alpha, n_iters):
    """Batch gradient descent from theta = 0.
    theta_j := theta_j + (alpha/n) * sum (y - h) x_j
    Returns theta and J(theta) after every iteration
    (history[0] is J at theta = 0, so it has n_iters + 1 entries).
    """
    n = X.shape[0]
    theta = np.zeros(X.shape[1])
    history = [cost(X, y, theta)]
    for _ in range(n_iters):
        theta = theta + (alpha / n) * (X.T @ (y - hypothesis(X, theta)))
        history.append(cost(X, y, theta))
    return theta, history


def predict(X, theta):
    return (hypothesis(X, theta) >= 0.5).astype(int)

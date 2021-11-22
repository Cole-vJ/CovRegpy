
import numpy as np
import matplotlib.pyplot as plt

# Hassani, H. (2007). Singular Spectrum Analysis: Methodology and Comparison.
# Cardiff University and Central Bank of the Islamic Republic of Iran.
# https://mpra.ub.uni-muenchen.de/4991/


def ssa(time_series, L, est=3, plot=False):

    # decomposition - embedding
    X = np.zeros((L, len(time_series) - L + 1))
    for col in range(len(time_series) - L + 1):
        X[:, col] = time_series[col:int(L + col)]

    # decomposition - singular value decomposition
    eigen_values, eigen_vectors = np.linalg.eig(np.matmul(X, X.T))
    # eigen-vectors in columns
    V_storage = {}
    X_storage = {}
    for i in range(sum(eigen_values > 0)):
        V_storage[i] = np.matmul(X.T, eigen_vectors[:, i].reshape(-1, 1)) / np.sqrt(eigen_values[i])
        X_storage[i] = np.sqrt(eigen_values[i]) * np.matmul(eigen_vectors[:, i].reshape(-1, 1), V_storage[i].T)

    # reconstruction - grouping
    X_estimate = X_storage[0]
    for j in range(1, est):
        X_estimate += X_storage[j]

    # reconstruction - averaging
    time_series_est = np.zeros_like(time_series)
    averaging_vector = L * np.ones_like(time_series_est)
    for col in range(len(time_series) - L + 1):
        time_series_est[col:int(L + col)] += X_estimate[:, col]
        if col < L:
            averaging_vector[col] -= (L - col - 1)
            averaging_vector[-int(col + 1)] -= (L - col - 1)
    time_series_est /= averaging_vector

    if plot:
        plt.plot(time_series)
        plt.plot(time_series_est, '--')
        plt.show()

    return time_series_est

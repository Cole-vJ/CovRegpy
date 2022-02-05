
# Document Strings Publication

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
import matplotlib.pyplot as plt

# Main reference: Hassani (2007)
# Hassani, H. (2007). Singular Spectrum Analysis: Methodology and Comparison.
# Cardiff University and Central Bank of the Islamic Republic of Iran.
# https://mpra.ub.uni-muenchen.de/4991/


def ssa(time_series, L, est=3, plot=False, KS_test=False, plot_KS_test=False, KS_scale_limit=1):
    """
    Singular Spectrum Analysis (SSA) as in Hassani (2007).

    Parameters
    ----------
    time_series : real ndarray
        Time series on which SSA will be applied.

    L : positive integer
        Embedding dimension as in Hassani (2007).

    est : positive integer
        Number of components to use in trend estimation. Necessarily true that est < L.

    Returns
    -------
    output : real ndarray
        Single basis spline of degree: "degree".

    Notes
    -----
    Continually subsets knot vector by one increment until base case is reached.

    """
    KS_value = 1000

    if KS_test:
        for L_test in np.arange(10, int(len(time_series) / 3), 10):
            for est_test in np.arange(L_test):
                trend = ssa(time_series, L=L_test, est=est_test, KS_test=False)

                errors = time_series - trend
                std_of_errors = np.std(errors)
                x = np.linspace(-5 * std_of_errors, 5 * std_of_errors, len(errors))
                sorted_errors = np.sort(errors)
                ks_vector = np.zeros_like(x)
                for error in sorted_errors:
                    for index in range(len(x)):
                        if x[index] < error:
                            ks_vector[index] += 1
                ks_vector = 1 - ks_vector / 100
                test_value = np.max(np.abs(norm.cdf(x, scale=std_of_errors) - ks_vector))
                x_test_value = x[np.max(np.abs(norm.cdf(x, scale=std_of_errors) - ks_vector)) ==
                                 np.abs(norm.cdf(x, scale=std_of_errors) - ks_vector)]
                if plot_KS_test:
                    plt.title('Kolmogorov-Smirnoff Test')
                    plt.plot(x, norm.cdf(x, scale=std_of_errors), label='Cumulative distribution')
                    plt.plot(x, ks_vector, label='Test distribution')
                    plt.plot(x_test_value * np.ones(100),
                             np.linspace(norm.cdf(x_test_value, scale=std_of_errors),
                                         ks_vector[np.max(np.abs(norm.cdf(x, scale=std_of_errors) - ks_vector)) ==
                                                   np.abs(norm.cdf(x, scale=std_of_errors) - ks_vector)], 100),
                             'k', label='KS distance')
                    plt.legend(loc='best')
                    plt.text(std_of_errors, 0.5, f'KS Test Statistic = {np.round(test_value, 5)}')
                    plt.show()

                if (test_value < KS_value) and (std_of_errors > KS_scale_limit):
                    KS_value = test_value
                    L_opt = L_test
                    est_opt = est_test

        print(L_opt)
        print(est_opt)

        L = L_opt
        est = est_opt

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


if __name__ == "__main__":

    # pull all close data
    tickers_format = ['MSFT', 'AAPL', 'GOOGL', 'AMZN', 'TSLA']
    data = yf.download(tickers_format, start="2018-10-15", end="2021-10-16")
    close_data = data['Close']
    del data, tickers_format

    # create date range and interpolate
    date_index = pd.date_range(start='16/10/2018', end='16/10/2021')
    close_data = close_data.reindex(date_index).interpolate()
    close_data = close_data[::-1].interpolate()
    close_data = close_data[::-1]
    del date_index

    # singular spectrum analysis
    plt.title('Singular Spectrum Analysis Example')
    for i in range(1, 11):
        plt.plot(ssa(np.asarray(close_data['MSFT'][-100:]), L=10, est=i), '--', label=f'Components = {i}')
    plt.legend(loc='best', fontsize=8)
    plt.show()

    opt_trend = ssa(np.asarray(close_data['MSFT'][-100:]), L=10, est=5, KS_test=True)
    plt.plot(np.asarray(close_data['MSFT'][-100:]))
    plt.plot(opt_trend)
    plt.show()

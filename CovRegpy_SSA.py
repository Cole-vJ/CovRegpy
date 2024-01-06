
#     ________
#            /
#      \    /
#       \  /
#        \/

# Main reference: Hassani (2007)
# Hassani, H. (2007). Singular Spectrum Analysis: Methodology and Comparison.
# Cardiff University and Central Bank of the Islamic Republic of Iran.
# https://mpra.ub.uni-muenchen.de/4991/

import numpy as np
import seaborn as sns
from scipy.stats import norm
import matplotlib.pyplot as plt

np.random.seed(0)

sns.set(style='darkgrid')


def CovRegpy_ssa(time_series, L, est=3, plot=False, KS_test=False, plot_KS_test=False, KS_scale_limit=1.0,
                 max_eig_ratio=0.0001, KS_start=1, KS_end=3, KS_interval=1):
    """
    Singular Spectrum Analysis (SSA) as in Hassani (2007).

    Parameters
    ----------
    time_series : real ndarray
        Time series on which SSA will be applied.

    L : positive integer
        Embedding dimension as in Hassani (2007).

    est : positive integer
        Number of components to use in trend estimation. Necessarily true that est <= L.

    plot : boolean
        Whether to plot time series and trend estimation before returning trend and decomposition (one plot at end).

    KS_test : boolean
        Whether to test for optimal L and est under Kolmogorov-Smirnov cumulative distribution conditions.

    plot_KS_test : boolean
        Whether to plot all intermediate Kolmogorov-Smirnov tests.

    KS_scale_limit : positive float
        Kolmogorov-Smirnov conditions optimised subject to a minimum standard deviation.
        Without this limit, 'optimised' trend estimate would be the exact time series.
        Set to zero or small for a test - see example.

    max_eig_ratio : float
        Not in original algorithm, but we conjecture that is essential for time-saving and prevention of inclusion
        of probably inconsequential components i.e. components with minimal contribution to variation.

    KS_start : integer
        Beginning value of grid search for Kolmogorov Smirnov Test.

    KS_end : integer
        Possible end value of grid search for Kolmogorov Smirnov Test.
        Only possible as we take the minimum of KS_end and len(time_series) / 3.

    KS_interval : integer
        Interval of grid search for Kolmogorov Smirnov Test.

    Returns
    -------
    time_series_est : real ndarray
        SSA trend estimation of time series using est number of components to estimate trend.

    time_series_decomp : real ndarray
        SSA decomposition of time series into est number of components.

    Notes
    -----
    Kolmogorov-Smirnov Singular Spectrum Analysis (KS-SSA) is experimental and effective.

    """

    if not isinstance(time_series, (type(np.asarray([[1.0, 2.0], [3.0, 4.0]])))):
        raise TypeError('time_series must be of type np.ndarray.')
    if np.array(time_series).dtype != np.array([[1., 1., 1., 1.]]).dtype:
        raise TypeError('time_series must only contain floats.')
    if (not isinstance(L, int)) or (not L > 0) or (not L < len(time_series)):
        raise ValueError('\'L\' must be a positive integer of appropriate magnitude: L < len(time_series).')
    if (not isinstance(est, int)) or (not est > 0) or (not est <= L):
        raise ValueError('\'est\' must be a positive integer of appropriate magnitude: est <= L.')
    if not isinstance(plot, bool):
        raise TypeError('\'plot\' must be boolean.')
    if not isinstance(KS_test, bool):
        raise TypeError('\'KS_test\' must be boolean.')
    if not isinstance(plot_KS_test, bool):
        raise TypeError('\'plot_KS_test\' must be boolean.')
    if (not isinstance(KS_scale_limit, float)) or KS_scale_limit <= 0:
        raise ValueError('\'KS_scale_limit\' must be a positive float.')
    if (not isinstance(max_eig_ratio, float)) or max_eig_ratio <= 0 or max_eig_ratio >= 1:
        raise ValueError('\'max_eig_ratio\' must be a float percentage between 0 and 1.')
    if (not isinstance(KS_start, int)) or (not KS_start > 0) or (not KS_start <= len(time_series) / 3):
        raise ValueError('\'KS_start\' must be a positive integer of appropriate magnitude: '
                         'KS_start <= len(time_series) / 3.')
    if (not isinstance(KS_end, int)) or (not KS_end > KS_start) or (not KS_end <= len(time_series) / 3):
        raise ValueError('\'KS_end\' must be a positive integer of appropriate magnitude: '
                         'KS_end > KS_start and KS_end <= len(time_series) / 3.')
    if (not isinstance(KS_interval, int)) or (not KS_interval > 0) or (not KS_interval < (KS_end - KS_start)):
        raise ValueError('\'KS_interval\' must be a positive integer of appropriate magnitude: '
                         'KS_interval < (KS_end - KS_start).')

    KS_value = 1000

    fig_plot_count = 0

    if KS_test:
        for L_test in np.arange(KS_start, int(min(len(time_series) / 3, KS_end)), KS_interval):
            for est_test in np.arange(1, int(L_test)):

                prev_test_value = 1.0
                if int(est_test) > 1:
                    prev_test_value = test_value.copy()

                trend = CovRegpy_ssa(time_series, L=int(L_test), est=int(est_test), KS_test=False)[0]

                errors = time_series - trend
                std_of_errors = np.std(errors)
                x = np.linspace(-5 * std_of_errors, 5 * std_of_errors, len(errors))
                sorted_errors = np.sort(errors)
                ks_vector = np.zeros_like(x)
                for error in sorted_errors:
                    for index in range(len(x)):
                        if x[index] < error:
                            ks_vector[index] += 1
                ks_vector = 1 - ks_vector / len(x)
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
                    plt.text(x[0], 0.5, f'KS Test Statistic = {np.round(test_value, 5)}', fontsize=8)
                    plt.xticks(fontsize=8)
                    plt.xlabel('Errors', fontsize=10)
                    plt.yticks(fontsize=8)
                    plt.ylabel('Cumulative Distribution', fontsize=10)
                    plt.show()

                if (test_value < KS_value) and (std_of_errors > KS_scale_limit):
                    KS_value = test_value
                    L_opt = int(L_test)
                    est_opt = int(est_test)
                if test_value == prev_test_value:
                    break
                prev_test_value = test_value.copy()

        print(L_opt)
        print(est_opt)

        L = L_opt
        est = est_opt

    # decomposition - embedding
    X = np.zeros((L, len(time_series) - L + 1))
    for col in range(len(time_series) - L + 1):
        X[:, col] = time_series[col:int(L + col)]

    # decomposition - singular value decomposition
    eigen_values, eigen_vectors = np.linalg.eig(np.matmul(X.T, X))
    eigen_values = np.real(eigen_values)

    eigen_vectors = np.real(eigen_vectors)
    # eigen-vectors in columns
    V_storage = {}
    X_storage = {}
    for i in range(sum(eigen_values > max_eig_ratio * np.max(eigen_values))):
        V_storage[i] = np.matmul(X, eigen_vectors[:, i].reshape(-1, 1)) / np.sqrt(eigen_values[i])
        X_storage[i] = np.sqrt(eigen_values[i]) * np.matmul(eigen_vectors[:, i].reshape(-1, 1), V_storage[i].T)

    # reconstruction - grouping
    X_estimate = np.zeros_like(X)
    for j in range(min(len(X_storage), est)):
        X_estimate += X_storage[j].T

    # reconstruction - averaging
    time_series_est = np.zeros_like(time_series)
    averaging_vector = L * np.ones_like(time_series_est)
    for col in range(len(time_series) - L + 1):
        time_series_est[col:int(L + col)] += X_estimate[:, col]
        if col < L:
            averaging_vector[col] -= (L - col - 1)
            averaging_vector[-int(col + 1)] -= (L - col - 1)
    time_series_est /= averaging_vector

    # Decomposing Singular Spectrum Analysis (D-SSA)
    time_series_decomp = np.zeros((est, len(time_series_est)))

    for comp in range(min(len(X_storage), est)):
        comp_est = np.zeros_like(time_series)
        for col in range(len(time_series) - L + 1):
            comp_est[col:int(L + col)] += X_storage[comp][col, :]
        time_series_decomp[comp, :] = comp_est / averaging_vector

    if plot:
        plt.plot(time_series)
        plt.plot(time_series_est, '--')
        plt.show()

    return time_series_est, time_series_decomp

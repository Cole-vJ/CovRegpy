
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


def CovRegpy_ssa(time_series, L, est=3, plot=False, KS_test=False, plot_KS_test=False, KS_scale_limit=1,
                 figure_plot=False, max_eig_ratio=0.0001, KS_start=10, KS_end=100, KS_interval=10):
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

    KS_scale_limit : float
        Kolmogorov-Smirnov conditions optimised subject to a minimum standard deviation.
        Without this limit, 'optimised' trend estimate would be the exact time series.
        Set to zero or small for a test - see example at end of script.

    figure_plot : boolean
        This is not integral to core code - demonstration purposes only - plots demonstration of KS test.

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
    KS_value = 1000

    fig_plot_count = 0

    if KS_test:
        for L_test in np.arange(KS_start, int(min(len(time_series) / 3, KS_end)), KS_interval):
            for est_test in np.arange(L_test):

                prev_test_value = 1.0
                if est_test > 0:
                    prev_test_value = test_value.copy()

                trend = CovRegpy_ssa(time_series, L=L_test, est=est_test, KS_test=False)[0]

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

                if figure_plot and fig_plot_count < 2:
                    if fig_plot_count == 0:
                        fig, axs = plt.subplots(2, 2)
                        plt.subplots_adjust(hspace=0.3, wspace=0.3)
                        axs[fig_plot_count, 0].set_title('Time Series and Trend', fontsize=10)
                        axs[fig_plot_count, 1].set_title('Kolmogorov-Smirnov Test', fontsize=10)
                        axs[fig_plot_count, 1].set_xticks([-1500, 0, 1500])
                        axs[fig_plot_count, 1].set_xticklabels([-1500, 0, 1500], fontsize=6)
                        axs[fig_plot_count, 0].set_xlabel('Months', fontsize=8)
                        axs[fig_plot_count, 1].set_xlabel('Errors', fontsize=8)
                        axs[fig_plot_count, 1].set_ylabel('Cumulative Distribution', fontsize=8)
                    if fig_plot_count == 1:
                        axs[fig_plot_count, 1].set_xticks([-500, 0, 500])
                        axs[fig_plot_count, 1].set_xticklabels([-500, 0, 500], fontsize=6)
                        axs[fig_plot_count, 0].set_xlabel('Months', fontsize=8)
                        axs[fig_plot_count, 1].set_xlabel('Errors', fontsize=8)
                        axs[fig_plot_count, 1].set_ylabel('Cumulative Distribution', fontsize=8)
                    axs[fig_plot_count, 0].plot(time_series, label='Time series')
                    axs[fig_plot_count, 0].plot(trend, label='Trend estimate')
                    axs[fig_plot_count, 0].set_yticks([0, 500, 1000, 1500])
                    axs[fig_plot_count, 0].set_yticklabels([0, 500, 1000, 1500], fontsize=6)
                    axs[fig_plot_count, 0].set_xticks([0, 60, 120])
                    axs[fig_plot_count, 0].set_xticklabels([0, 60, 120], fontsize=6)
                    axs[fig_plot_count, 1].set_yticks([0, 0.5, 1])
                    axs[fig_plot_count, 1].set_yticklabels([0, 0.5, 1], fontsize=6)
                    axs[fig_plot_count, 1].plot(x, norm.cdf(x, scale=std_of_errors),
                                                    label='Cumulative distribution')
                    axs[fig_plot_count, 1].plot(x, ks_vector, label='Test distribution')
                    axs[fig_plot_count, 1].plot(x_test_value * np.ones(100),
                                                np.linspace(norm.cdf(x_test_value, scale=std_of_errors),
                                                            ks_vector[np.max(np.abs(norm.cdf(x, scale=std_of_errors) -
                                                                                    ks_vector)) ==
                                                                      np.abs(norm.cdf(x, scale=std_of_errors) -
                                                                             ks_vector)], 100),
                                                'k', label='KS distance')
                    if fig_plot_count == 1:
                        axs[fig_plot_count, 0].legend(loc='best', fontsize=6)
                        axs[fig_plot_count, 1].legend(loc='best', fontsize=6)
                        plt.savefig('../aas_figures/Example_ks_test')
                        plt.show()
                    fig_plot_count += 1

                if (test_value < KS_value) and (std_of_errors > KS_scale_limit):
                    KS_value = test_value
                    L_opt = L_test
                    est_opt = est_test
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

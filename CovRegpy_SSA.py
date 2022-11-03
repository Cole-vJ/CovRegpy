
# Document Strings Publication

# Main reference: Hassani (2007)
# Hassani, H. (2007). Singular Spectrum Analysis: Methodology and Comparison.
# Cardiff University and Central Bank of the Islamic Republic of Iran.
# https://mpra.ub.uni-muenchen.de/4991/

import textwrap
import numpy as np
import pandas as pd
import yfinance as yf
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
        Whether to plot time series and trend estimation.

    KS_test : boolean
        Whether to test for optimal L and est under Kolmogorov-Smirnov cumulative distribution conditions.

    plot_KS_test : boolean
        Whether to plot all intermediate Kolmogorov-Smirnov tests.

    KS_scale_limit : float
        Kolmogorov-Smirnov conditions optimised subject to a minimum standard deviation.

    Returns
    -------
    time_series_est : real ndarray
        SSA trend estimation of time series using est number of components to estimate.

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
                        plt.savefig('aas_figures/Example_ks_test')
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
    # fig, axs = plt.subplots(1, 2)
    # plt.subplots_adjust(hspace=0.5)
    # axs[0].set_title('Eigenvalues')
    # axs[1].set_title('Significant eigenvalues')
    # axs[0].plot(eigen_values)
    # axs[0].plot(np.linspace(-5, 130), 82500 * np.ones(50), 'k--')
    # axs[0].plot(np.linspace(-5, 130), -2500 * np.ones(50), 'k--')
    # axs[0].plot(-5 * np.ones(50), np.linspace(-2500, 82500), 'k--')
    # axs[0].plot(130 * np.ones(50), np.linspace(-2500, 82500), 'k--')
    # axs[1].plot(eigen_values)
    # axs[1].text(50, 0.9 * max(eigen_values), r'$\lambda_K$$_S = 0.9$: sig eig = {}'.format(sum(eigen_values > 0.9 * max(eigen_values))), fontsize=8)
    # axs[1].plot(np.linspace(-2.5, 42.5), 0.9 * max(eigen_values) * np.ones(50), '--', c='darkgrey')
    # axs[1].text(50, 0.8 * max(eigen_values), r'$\lambda_K$$_S = 0.8$: sig eig = {}'.format(sum(eigen_values > 0.8 * max(eigen_values))), fontsize=8)
    # axs[1].plot(np.linspace(-2.5, 42.5), 0.8 * max(eigen_values) * np.ones(50), '--', c='darkgrey')
    # axs[1].text(50, 0.7 * max(eigen_values), r'$\lambda_K$$_S = 0.7$: sig eig = {}'.format(sum(eigen_values > 0.7 * max(eigen_values))), fontsize=8)
    # axs[1].plot(np.linspace(-2.5, 42.5), 0.7 * max(eigen_values) * np.ones(50), '--', c='darkgrey')
    # axs[1].text(50, 0.6 * max(eigen_values), r'$\lambda_K$$_S = 0.6$: sig eig = {}'.format(sum(eigen_values > 0.6 * max(eigen_values))), fontsize=8)
    # axs[1].plot(np.linspace(-2.5, 42.5), 0.6 * max(eigen_values) * np.ones(50), '--', c='darkgrey')
    # axs[1].text(50, 0.5 * max(eigen_values), r'$\lambda_K$$_S = 0.5$: sig eig = {}'.format(sum(eigen_values > 0.5 * max(eigen_values))), fontsize=8)
    # axs[1].plot(np.linspace(-2.5, 42.5), 0.5 * max(eigen_values) * np.ones(50), '--', c='darkgrey')
    # axs[1].text(50, 0.4 * max(eigen_values), r'$\lambda_K$$_S = 0.4$: sig eig = {}'.format(sum(eigen_values > 0.4 * max(eigen_values))), fontsize=8)
    # axs[1].plot(np.linspace(-2.5, 42.5), 0.4 * max(eigen_values) * np.ones(50), '--', c='darkgrey')
    # axs[1].text(50, 0.3 * max(eigen_values),
    #             r'$\lambda_K$$_S = 0.3$: sig eig = {}'.format(sum(eigen_values > 0.3 * max(eigen_values))), fontsize=8)
    # axs[1].plot(np.linspace(-2.5, 42.5), 0.3 * max(eigen_values) * np.ones(50), '--', c='darkgrey')
    # axs[0].set_ylim(-5000, 85000)
    # axs[1].set_ylim(-2500, 82500)
    # axs[1].set_xlim(-5.0, 130)
    # x_points = [0, 500, 1000, 1500, 2000, 2500]
    # y_points = [0, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000]
    # x_zoomed_points = [0, 20, 40, 60, 80, 100, 120]
    # x_names = [0, 500, 1000, 1500, 2000, 2500]
    # y_names = [0, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000]
    # y_names_empty = ['', '', '', '', '', '', '', '', '']
    # x_zoomed_names = [0, 20, 40, 60, 80, 100, 120]
    #
    # axis = 0
    # for ax in axs.flat:
    #     if axis == 0:
    #         ax.set_xticks(x_points)
    #         ax.set_yticks(y_points)
    #         ax.set_xticklabels(x_names, fontsize=8)
    #         ax.set_yticklabels(y_names, fontsize=8)
    #         ax.set(ylabel=r'$\lambda_k$')
    #         ax.set(xlabel='k')
    #     else:
    #         ax.set_xticks(x_zoomed_points)
    #         ax.set_yticks(y_points)
    #         ax.set_yticklabels(y_names, fontsize=8)
    #         ax.set_xticklabels(x_zoomed_names, fontsize=8)
    #         ax.set(xlabel='k')
    #     axis += 1
    #
    # box_0 = axs[0].get_position()
    # box_1 = axs[1].get_position()
    # axs[0].set_position([box_0.x0 + 0.01, box_0.y0, box_0.width * 0.95, box_0.height])
    # axs[1].set_position([box_1.x0 + 0.01, box_1.y0, box_1.width * 0.95, box_1.height])
    # # axs[0].legend(loc='center left', bbox_to_anchor=(1, -0.3), fontsize=9)
    # plt.show()

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


if __name__ == "__main__":

    # # pull all close data
    # tickers_format = ['MSFT', 'AAPL', 'GOOGL', 'AMZN', 'TSLA']
    # data = yf.download(tickers_format, start="2018-10-15", end="2021-10-16")
    # close_data = data['Close']
    # del data, tickers_format
    #
    # # create date range and interpolate
    # date_index = pd.date_range(start='16/10/2018', end='16/10/2021')
    # close_data = close_data.reindex(date_index).interpolate()
    # close_data = close_data[::-1].interpolate()
    # close_data = close_data[::-1]
    # del date_index
    #
    # # singular spectrum analysis
    # plt.title('Singular Spectrum Analysis Example')
    # for i in range(1, 11):
    #     plt.plot(CovRegpy_ssa(np.asarray(close_data['MSFT'][-100:]), L=10, est=i), '--', label=f'Components = {i}')
    # plt.legend(loc='best', fontsize=8)
    # plt.show()
    #
    # opt_trend = CovRegpy_ssa(np.asarray(close_data['MSFT'][-100:]), L=10, est=5, KS_test=True, plot_KS_test=True)
    # plt.plot(np.asarray(close_data['MSFT'][-100:]))
    # plt.plot(opt_trend)
    # plt.show()

    # figures for paper

    np.random.seed(0)

    x11_time = np.linspace(0, 120, 121)
    x11_trend_cycle = (1 / 100) * (x11_time - 10) * (x11_time - 60) * (x11_time - 110) + 1000
    x11_seasonal = 100 * np.sin((2 * np.pi / 12) * x11_time)
    x11_noise = 100 * np.random.normal(0, 1, 121)
    x11_time_series = x11_trend_cycle + x11_seasonal + x11_noise

    plt.plot(x11_time, x11_time_series)
    plt.title('Additive X11 Example Time Series')
    plt.xticks([0, 20, 40, 60, 80, 100, 120], fontsize=8)
    plt.yticks([400, 600, 800, 1000, 1200, 1400, 1600], fontsize=8)
    plt.show()

    ssa_decomp = CovRegpy_ssa(x11_time_series, L=10, est=8)

    fig, axs = plt.subplots(3, 1)
    plt.subplots_adjust(hspace=0.3)
    fig.suptitle('Additive Decomposition SSA Demonstration')
    axs[0].plot(x11_time, x11_trend_cycle, label='Component')
    axs[0].plot(x11_time, ssa_decomp[0], 'r--', label='SSA trend')
    axs[0].plot(x11_time, ssa_decomp[1][0, :], 'g--', label=textwrap.fill('D-SSA first component', 12))
    axs[0].set_xticks([0, 20, 40, 60, 80, 100, 120])
    axs[0].set_xticklabels(['', '', '', '', '', '', ''], fontsize=8)
    axs[0].set_yticks([500, 1000, 1500])
    axs[0].set_yticklabels(['500', '1000', '1500'], fontsize=8)
    axs[0].set_title('Trend-Cycle Component')
    box_0 = axs[0].get_position()
    axs[0].set_position([box_0.x0 - 0.05, box_0.y0, box_0.width * 0.84, box_0.height])
    axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    axs[1].plot(x11_time, x11_seasonal, label='Component')
    axs[1].plot(x11_time, ssa_decomp[1][1, :], 'g--', label=textwrap.fill('D-SSA second component', 12))
    axs[1].plot(x11_time, ssa_decomp[1][1, :] + ssa_decomp[1][2, :], 'r--',
                label=textwrap.fill('D-SSA second and third components', 12))
    axs[1].set_xticks([0, 20, 40, 60, 80, 100, 120])
    axs[1].set_xticklabels(['', '', '', '', '', '', ''], fontsize=8)
    axs[1].set_yticks([-100, 0, 100])
    axs[1].set_yticklabels(['-100', '0', '100'], fontsize=8)
    axs[1].set_ylim(-175, 175)
    axs[1].set_title('Seasonal Component')
    box_1 = axs[1].get_position()
    axs[1].set_position([box_1.x0 - 0.05, box_1.y0, box_1.width * 0.84, box_1.height])
    axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    axs[2].plot(x11_time, x11_noise, label='Component')
    axs[2].plot(x11_time, ssa_decomp[1][3, :], 'g--', label=textwrap.fill('D-SSA fourth component', 12))
    axs[2].plot(x11_time, np.sum(ssa_decomp[1][3:, :], axis=0), 'r--',
                label=textwrap.fill('D-SSA fourth and onwards components', 12))
    axs[2].set_xticks([0, 20, 40, 60, 80, 100, 120])
    axs[2].set_xticklabels(['0', '20', '40', '60', '80', '100', '120'], fontsize=8)
    axs[2].set_yticks([-200, 0, 200])
    axs[2].set_yticklabels(['-200', '0', '200'], fontsize=8)
    axs[2].set_ylim(-250, 250)
    axs[2].set_title('Random Error')
    box_2 = axs[2].get_position()
    axs[2].set_position([box_2.x0 - 0.05, box_2.y0, box_2.width * 0.84, box_2.height])
    axs[2].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    plt.savefig('aas_figures/Example_ssa_decomposition')
    plt.show()

    ssa_decomp = CovRegpy_ssa(x11_time_series, L=10, est=8, KS_test=True, plot_KS_test=False, figure_plot=True)

    plt.plot(x11_time, x11_trend_cycle, label='Component')
    plt.plot(x11_time, ssa_decomp[0], 'r--', label='KS-SSA trend')
    plt.xticks([0, 20, 40, 60, 80, 100, 120], ['0', '20', '40', '60', '80', '100', '120'], fontsize=8)
    plt.xlabel('Months', fontsize=10)
    plt.yticks([500, 1000, 1500], ['500', '1000', '1500'], fontsize=8)
    plt.title('Trend-Cycle Component and KS-SSA Trend Estimate')
    plt.legend(loc='best', fontsize=8)
    plt.savefig('aas_figures/Example_ks_ssa')
    plt.show()


# Case Study - MDLP compare

import textwrap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mdlp.discretization import MDLP
from sklearn.linear_model import Ridge

from AdvEMDpy import AdvEMDpy, emd_preprocess, emd_basis

from CovRegpy_utilities import efficient_frontier, global_minimum_forward_applied_information, \
    sharpe_forward_applied_information, pca_forward_applied_information, \
    sharpe_forward_applied_information_summation_restriction

from CovRegpy_measures import cumulative_return, mean_return, variance_return, value_at_risk_return, \
    max_draw_down_return, omega_ratio_return, sortino_ratio_return, sharpe_ratio_return

from CovRegpy_RPP import risk_parity_weights_summation_restriction, global_obj_fun

from CovRegpy_DCC import covregpy_dcc

from CovRegpy_RCR import cov_reg_given_mean, cubic_b_spline

from CovRegpy_SSA import CovRegpy_ssa

np.random.seed(0)

sns.set(style='darkgrid')

# create S&P 500 index
sp500_close = pd.read_csv('S&P500_Data/sp_500_close_5_year.csv', header=0)
sp500_close = sp500_close.set_index(['Unnamed: 0'])
sp500_market_cap = pd.read_csv('S&P500_Data/sp_500_market_cap_5_year.csv', header=0)
sp500_market_cap = sp500_market_cap.set_index(['Unnamed: 0'])

sp500_returns = np.log(np.asarray(sp500_close)[1:, :] / np.asarray(sp500_close)[:-1, :])
weights = np.asarray(sp500_market_cap) / np.tile(np.sum(np.asarray(sp500_market_cap), axis=1).reshape(-1, 1), (1, 505))
sp500_returns = np.sum(sp500_returns * weights[:-1, :], axis=1)[365:]
sp500_proxy = np.append(1, np.exp(np.cumsum(sp500_returns)))

# load 11 sector indices
sector_11_indices = pd.read_csv('S&P500_Data/sp_500_11_sector_indices.csv', header=0)
sector_11_indices = sector_11_indices.set_index(['Unnamed: 0'])

# approximate daily treasury par yield curve rates for 3 year bonds
risk_free = (0.01 / 365)  # daily risk free rate

# sector numpy array
sector_11_indices_array = np.vstack((np.zeros((1, 11)), np.asarray(sector_11_indices)))

# construct portfolios
end_of_month_vector = np.asarray([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31,
                                  31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31,
                                  31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31,
                                  31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31,
                                  31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
end_of_month_vector_cumsum = np.cumsum(end_of_month_vector)
month_vector = np.asarray(['January', 'February', 'March', 'April', 'May', 'June',
                           'July', 'August', 'September', 'October', 'November', 'December'])
year_vector = np.asarray(['2017', '2018', '2019', '2020', '2021'])

# minimum variance portfolio
sector_11_indices_array = sector_11_indices_array[1:, :]

# how many months to consider when calculating realised covariance
months = 12

# store all weights from respective models
weight_matrix_global_minimum = np.zeros_like(sector_11_indices_array)
weight_matrix_maximum_sharpe_ratio = np.zeros_like(sector_11_indices_array)
weight_matrix_pca = np.zeros_like(sector_11_indices_array)
weight_matrix_dcc = np.zeros_like(sector_11_indices_array)
weight_matrix_realised = np.zeros_like(sector_11_indices_array)

# weight storage
weight_matrix_high = np.zeros_like(sector_11_indices_array)
weight_matrix_mid = np.zeros_like(sector_11_indices_array)
weight_matrix_trend = np.zeros_like(sector_11_indices_array)

# weights calculated on and used on different data (one month ahead)
for day in range(len(end_of_month_vector_cumsum[:-int(months + 1)])):

    if day > 0:
        del x, x_if, y, x_high, x_trend, x_mid, x_ssa

    # calculate annual returns and covariance
    annual_covariance = \
        np.cov(sector_11_indices_array[
               end_of_month_vector_cumsum[int(day)]:end_of_month_vector_cumsum[int(day + months)], :].T)
    annual_returns = \
        np.sum(sector_11_indices_array[
               end_of_month_vector_cumsum[int(day)]:end_of_month_vector_cumsum[int(day + months)], :], axis=0)
    price_signal = \
        np.cumprod(np.exp(sector_11_indices_array[
                          end_of_month_vector_cumsum[int(day)]:end_of_month_vector_cumsum[
                              int(day + months)], :]), axis=0)

    # calculate actual covariance and returns looking forward
    monthly_covariance = \
        np.cov(sector_11_indices_array[
               end_of_month_vector_cumsum[
                   int(day + months)]:end_of_month_vector_cumsum[int(day + months + 1)], :].T)
    # monthly returns are annualised
    monthly_returns = \
        np.sum(sector_11_indices_array[
               end_of_month_vector_cumsum[
                   int(day + months)]:end_of_month_vector_cumsum[int(day + months + 1)], :], axis=0) / \
        ((end_of_month_vector_cumsum[int(day + months + 1)] - end_of_month_vector_cumsum[int(day + months)]) /
         (end_of_month_vector_cumsum[int(day + months + 1)] - end_of_month_vector_cumsum[int(day + 1)]))

    # calculate global minimum variance portfolio
    gm_w, gm_sd, gm_r = global_minimum_forward_applied_information(annual_covariance, monthly_covariance,
                                                                   monthly_returns)
    # calculate maximum sharpe ratio portfolio
    ms_w, ms_sd, ms_r = sharpe_forward_applied_information_summation_restriction(annual_covariance, annual_returns,
                                                                                 monthly_covariance, monthly_returns,
                                                                                 risk_free, gm_w, gm_r)
    # calculate pca portfolio
    pc_w, pc_sd, pc_r = pca_forward_applied_information(annual_covariance, monthly_covariance,
                                                        monthly_returns, factors=3)
    # calculate realised portfolio
    weights_realised = risk_parity_weights_summation_restriction(annual_covariance, short_limit=0.3,
                                                                 long_limit=1.3).x
    # # calculate DCC weights - commented out for speed
    # weights_dcc = risk_parity_weights_summation_restriction(covregpy_dcc(sector_11_indices_array[
    #                       end_of_month_vector_cumsum[int(day)]:end_of_month_vector_cumsum[
    #                           int(day + months)], :]), short_limit=0.3,
    #                                                         long_limit=1.3).x

    # filled weight matrices iteratively
    weight_matrix_global_minimum[end_of_month_vector_cumsum[day]:
                                 end_of_month_vector_cumsum[int(day + 1)], :] = gm_w
    weight_matrix_maximum_sharpe_ratio[end_of_month_vector_cumsum[day]:
                                       end_of_month_vector_cumsum[int(day + 1)], :] = ms_w
    weight_matrix_pca[end_of_month_vector_cumsum[day]:
                      end_of_month_vector_cumsum[int(day + 1)], :] = pc_w
    weight_matrix_realised[end_of_month_vector_cumsum[day]:
                           end_of_month_vector_cumsum[int(day + 1)], :] = weights_realised
    # commented out for speed
    # weight_matrix_dcc[end_of_month_vector_cumsum[day]:
    #                   end_of_month_vector_cumsum[int(day + 1)], :] = weights_dcc

    ##################################################
    # direct application Covariance Regression - TOP #
    ##################################################

    # calculate 'spline_basis'
    knots = 96  # arbitray - can adjust
    end_time = int(end_of_month_vector_cumsum[int(day + months)] - 1 - end_of_month_vector_cumsum[int(day)])
    spline_basis = cubic_b_spline(knots=np.linspace(-12, end_time + 12, knots),
                                  time=np.linspace(0, end_time, end_time + 1))
    spline_basis_direct_trunc = \
        spline_basis[:, :int(end_of_month_vector_cumsum[int(day + months - 1)] - end_of_month_vector_cumsum[int(day)])]

    # calculate 'A_est'
    A_est = np.linalg.lstsq(spline_basis.transpose(), sector_11_indices_array[
                                                      end_of_month_vector_cumsum[int(day)]:end_of_month_vector_cumsum[
                                                          int(day + months)], :], rcond=None)[0]

    # calculate 'x'
    # decompose price data
    for signal in range(np.shape(price_signal)[1]):
        emd = AdvEMDpy.EMD(time_series=np.asarray(price_signal[:, signal]),
                           time=np.linspace(0, end_time, end_time + 1))
        imfs, _, ifs, _, _, _, _ = \
            emd.empirical_mode_decomposition(knot_envelope=np.linspace(-12, end_time + 12, knots),
                                             matrix=True)

        # ssa
        ssa_components = CovRegpy_ssa(np.asarray(price_signal[:, signal]), L=80, plot=False)[0]
        try:
            x_ssa = np.vstack((x_ssa, ssa_components))
        except:
            x_ssa = ssa_components.copy()

        # deal with constant last IMF and insert IMFs in dataframe
        # deal with different frequency structures here
        try:
            imfs = imfs[1:, :]
            if np.isclose(imfs[-1, 0], imfs[-1, -1]):
                imfs[-2, :] += imfs[-1, :]
                imfs = imfs[:-1, :]
            ifs = ifs[1:, :]
            if np.isclose(imfs[-1, 0], imfs[-1, -1]):
                ifs[-2, :] += ifs[-1, :]
                ifs = ifs[:-1, :]
        except:
            pass
        try:
            x = np.vstack((x, imfs))
            x_if = np.vstack((x_if, ifs))
            for m in range(1, int(np.shape(imfs)[0] + 1)):
                y = np.append(y, m)
        except:
            x = imfs.copy()
            x_if = ifs.copy()
            y = np.zeros(1)
            for m in range(1, int(np.shape(imfs)[0] + 1)):
                y = np.append(y, m)
        try:
            x_trend = np.vstack((imfs[2, :], x_trend))
        except:
            try:
                x_trend = imfs[2, :].copy()
            except:
                try:
                    x_trend = np.vstack((np.ones_like(imfs[1, :]), x_trend))
                except:
                    x_trend = np.ones_like(imfs[0, :])
        try:
            x_mid = np.vstack((imfs[1, :], x_mid))
        except:
            x_mid = imfs[1, :].copy()
        try:
            x_high = np.vstack((imfs[0, :], x_high))
        except:
            x_high = imfs[0, :].copy()

    y = y[1:]

    x_if = x_if[y != 4, :]
    y = y[y != 4]
    if len(y) < 33:
        i = 0
        for index in np.arange(int(len(y) - 1))[np.diff(y) == -1]:
            x_if = np.vstack(
                (x_if[:int(index + i + 1), :], 0.001 * np.ones(int(np.shape(price_signal)[0] - 1)).reshape(1, -1),
                 x_if[int(index + i + 1):, :]))
            y = np.hstack((y[:int(index + i + 1)].reshape(1, -1),
                           np.array(3).reshape(1, 1), y[int(index + i + 1):].reshape(1, -1)))[0]
            i += 1
    y = y[::-1]

    time = np.asarray([0])
    cuts_0_1 = np.asarray([0])
    cuts_1_2 = np.asarray([0])

    assets = 2

    # for i in np.arange(2, 30, 3):
    #     mdlp = MDLP()
    #     x_if_median_filtered = np.zeros_like(x_if[int(i - assets):int(i + 2 * assets), :])
    #     for j in range(np.shape(x_if_median_filtered)[0]):
    #         x_if_median_filtered[j, :] = emd_preprocess.Preprocess(time=np.arange(np.shape(x_if_median_filtered)[1]),
    #                                                                time_series=x_if[int(i - assets + j), :]).median_filter(window_width=51)[1]
    #     X_mdlp = mdlp.fit_transform(x_if_median_filtered,
    #                                 y[int(i - assets):int(i + 2 * assets)])
    #     X_mdlp_cut_points = mdlp.cut_points_
    #     plt.title('Two Cut-Point Examples')
    #     multiple_scatter = np.arange(364).reshape(1, -1)
    #     for a in range(1, assets):
    #         multiple_scatter = np.vstack((multiple_scatter, np.arange(364).reshape(1, -1)))
    #     plt.scatter(multiple_scatter,
    #                 x_if_median_filtered[y[int(i - assets):int(i + 2 * assets)] == 1],
    #                 c='red', label='IF 3')
    #     plt.scatter(multiple_scatter,
    #                 x_if_median_filtered[y[int(i - assets):int(i + 2 * assets)] == 2],
    #                 c='green', label='IF 2')
    #     plt.scatter(multiple_scatter,
    #                 x_if_median_filtered[y[int(i - assets):int(i + 2 * assets)] == 3],
    #                 c='blue', label='IF 1')
    #
    #     test = 0
    #     for num, cut_point in enumerate(X_mdlp_cut_points):
    #         if len(cut_point) > 1:
    #             # plt.plot(cut_point[0] * np.ones_like(x_if[6, :]))
    #             # plt.plot(cut_point[1] * np.ones_like(x_if[6, :]))
    #             if test == 0:
    #                 plt.scatter(num, cut_point[0], c='black', s=40, label='Cut-point 0-1', zorder=10)
    #                 plt.scatter(num, cut_point[1], c='gold', s=40, label='Cut-point 1-2', zorder=10)
    #                 test += 1
    #             plt.scatter(num, cut_point[0], c='black', s=40, zorder=10)
    #             plt.scatter(num, cut_point[1], c='gold', s=40, zorder=10)
    #
    #             time = np.hstack((time, num))
    #             cuts_0_1 = np.hstack((cuts_0_1, cut_point[0]))
    #             cuts_1_2 = np.hstack((cuts_1_2, cut_point[1]))
    #
    #     # plt.show()
    #
    #     time_full = np.arange(int(np.shape(price_signal)[0] - 1))
    #     time_series_full = time_full.copy()
    #     knots = np.linspace(-20, 383, 100)
    #
    #     basis = emd_basis.Basis(time=time_full, time_series=time_series_full)
    #     basis = basis.cubic_b_spline(knots=knots)
    #     basis_subset = basis[:, time[1:]]
    #
    #     coef_0_1 = np.linalg.lstsq(basis_subset.T, cuts_0_1[1:], rcond=None)[0]
    #     coef_1_2 = np.linalg.lstsq(basis_subset.T, cuts_1_2[1:], rcond=None)[0]
    #
    #     smoothing_penalty = 1
    #     second_order_matrix = np.zeros((96, 94))  # note transpose
    #
    #     for a in range(94):
    #         second_order_matrix[a:(a + 3), a] = [1, -2, 1]  # filling values for second-order difference matrix
    #
    #     coef_penalised_0_1 = np.linalg.lstsq(np.append(basis_subset,
    #                                                    smoothing_penalty * second_order_matrix, axis=1).T,
    #                                          np.append(cuts_0_1[1:], np.zeros(94)), rcond=None)[0]
    #     coef_penalised_1_2 = np.linalg.lstsq(np.append(basis_subset,
    #                                                    smoothing_penalty * second_order_matrix, axis=1).T,
    #                                          np.append(cuts_1_2[1:], np.zeros(94)), rcond=None)[0]
    #
    #     clf = Ridge(alpha=1.0)
    #     clf.fit(basis_subset.T, cuts_0_1[1:])
    #     coef_penalised_positive_0_1 = clf.coef_
    #     clf = Ridge(alpha=1.0)
    #     clf.fit(basis_subset.T, cuts_1_2[1:])
    #     coef_penalised_positive_1_2 = clf.coef_
    #
    #     spline = np.matmul(coef_0_1.reshape(1, -1), basis)
    #     spline_penalised_0_1 = np.matmul(coef_penalised_0_1.reshape(1, -1), basis)
    #     spline_penalised_1_2 = np.matmul(coef_penalised_1_2.reshape(1, -1), basis)
    #
    #     if np.shape(x_if)[1] == 365:
    #         temp = 0
    #     accuracy_1 = np.mean(x_if[y == 1, :][int(i - assets):int(i + 2 * assets)] < spline_penalised_0_1)
    #     accuracy_2 = np.mean(np.r_[spline_penalised_0_1 < x_if[y == 2, :][int(i - assets):int(i + 2 * assets)]] &
    #                          np.r_[x_if[y == 2, :][int(i - assets):int(i + 2 * assets)] < spline_penalised_1_2])
    #     accuracy_3 = np.mean(x_if[y == 3, :][int(i - assets):int(i + 2 * assets)] > spline_penalised_1_2)
    #     print('Probability structure 1 below 0-1 cut: {}'.format(accuracy_1))
    #     print('Probability structure 2 above 0-1 and below 1-2 cut: {}'.format(accuracy_2))
    #     print('Probability structure 3 above 1-2 cut: {}'.format(accuracy_3))
    #
    #     plt.plot(np.arange(364), spline_penalised_0_1[0, :], label='Sub-decision surface 0-1', Linewidth=3, c='k')
    #     plt.plot(np.arange(364), spline_penalised_1_2[0, :], label='Sub-decision surface 1-2', Linewidth=3, c='gold')
    #     plt.text(100, 0.17, 'Accuracy IF 3 {}'.format(np.round(accuracy_1, 2)), fontsize=12)
    #     plt.text(100, 0.20, 'Accuracy IF 2 {}'.format(np.round(accuracy_2, 2)), fontsize=12)
    #     plt.text(100, 0.23, 'Accuracy IF 1 {}'.format(np.round(accuracy_3, 2)), fontsize=12)
    #     plt.ylim(-0.11, 0.31)
    #     if i == assets:
    #         plt.savefig('experimental_figures/cut_point_demonstration.png')
    #     # plt.show()

    mdlp = MDLP()
    X_mdlp = mdlp.fit_transform(x_if[:int(assets * 3), :], y[:int(assets * 3)])
    X_mdlp_cut_points = mdlp.cut_points_

    x_stack = np.zeros((1, 33))
    for i in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        try:
            x_stack = np.vstack((x_stack, np.searchsorted([i], x_if[:, 0])))
        except:
            x_stack = np.searchsorted([i], x_if[:, 0])

    cut_point_storage = np.empty((3, np.shape(x_if[:int(assets * 3), :])[1]))
    cut_point_storage[:] = np.nan

    for time_increment in range(np.shape(x_if[:int(assets * 3), :])[1]):

        if len(X_mdlp_cut_points[time_increment]) == 2 and time_increment == 28:

            # point = 0
            #
            # fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 4))
            # plt.suptitle('Minimum Description Length Binning Example')
            # ax1.spines['bottom'].set_visible(False)
            # ax1.tick_params(axis='x', which='both', bottom=False)
            # ax2.spines['top'].set_visible(False)
            #
            # bs = 0.08
            # ts = 0.1
            #
            # ax2.set_ylim(0, bs)
            # ax1.set_ylim(ts, 0.6)
            #
            # ax1.scatter(np.zeros(33)[y == 1][:assets], x_if[:, time_increment][y == 1][:assets], c='red', label='IF 3')
            # ax1.scatter(np.zeros(33)[y == 2][:assets], x_if[:, time_increment][y == 2][:assets], c='green',
            #             label='IF 2')
            # ax1.scatter(np.zeros(33)[y == 3][:assets], x_if[:, time_increment][y == 3][:assets], c='blue', label='IF 1')
            #
            # ax2.scatter(np.zeros(33)[y == 1][:assets], x_if[:, time_increment][y == 1][:assets], c='red')
            # ax2.scatter(np.zeros(33)[y == 2][:assets], x_if[:, time_increment][y == 2][:assets], c='green')
            # ax2.scatter(np.zeros(33)[y == 3][:assets], x_if[:, time_increment][y == 3][:assets], c='blue')
            #
            # print(x_if[:, time_increment][y == 1][:assets])
            # print(x_if[:, time_increment][y == 2][:assets])
            # print(x_if[:, time_increment][y == 3][:assets])
            #
            # for tick in ax2.get_xticklabels():
            #     tick.set_rotation(0)
            # d = .015
            # kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
            # ax1.plot((-d, +d), (-d, +d), **kwargs)
            # ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
            # kwargs.update(transform=ax2.transAxes)
            # ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
            # ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
            # ax2.set_yticks(np.arange(0.00, 0.09, 0.01))
            # ax2.set_yticklabels([0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08], fontsize=8)
            # ax1.set_title('Cut-Points at Time Increment: {}'.format(time_increment), fontsize=10)
            # ax1.set_yticks(np.arange(0.1, 0.7, 0.1))
            # ax1.set_yticklabels([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], fontsize=8)
            # ax2.set_xticks([0])
            # ax2.set_xticklabels([''])
            # ax1.set_ylabel('Instantanoues frequency', fontsize=8)
            # ax2.set_ylabel('Instantanoues frequency', fontsize=8)
            # for cut_point in X_mdlp_cut_points[time_increment]:
            #     ax2.plot(np.linspace(-0.5, 0.5, 100), cut_point * np.ones(100), '--')
            #     ax1.plot(np.linspace(-0.5, 0.5, 100), cut_point * np.ones(100), '--',
            #              label='Cut-point {}-{}'.format(point, int(point + 1)))
            #     cut_point_storage[point, time_increment] = cut_point
            #     point += 1
            # ax1.legend(loc='upper left', fontsize=8)
            # plt.savefig('experimental_figures/cut_point_time_point.png')
            # # plt.show()

            point = 0

        pass

    # comparing correlation matrices

    x_high_trunc = \
        x_high[:, :int(end_of_month_vector_cumsum[int(day + months - 1)] - end_of_month_vector_cumsum[int(day)])]
    x_mid_trunc = \
        x_mid[:, :int(end_of_month_vector_cumsum[int(day + months - 1)] - end_of_month_vector_cumsum[int(day)])]
    x_trend_trunc = \
        x_trend[:, :int(end_of_month_vector_cumsum[int(day + months - 1)] - end_of_month_vector_cumsum[int(day)])]

    # ssa
    x_ssa_trunc = \
        x_ssa[:, :int(end_of_month_vector_cumsum[int(day + months - 1)] - end_of_month_vector_cumsum[int(day)])]

    x_trunc = \
        x[:, :int(end_of_month_vector_cumsum[int(day + months - 1)] - end_of_month_vector_cumsum[int(day)])]

    # calculate y - same for both imf and ssa
    y = sector_11_indices_array[end_of_month_vector_cumsum[int(day + months - 11)]:end_of_month_vector_cumsum[
        int(day + months)], :]
    y = y.T

    diff = 0
    # make 'x' and 'y' the same size - ssa
    if np.shape(x_ssa_trunc)[1] != np.shape(y)[1]:
        diff = int(np.abs(np.shape(y)[1] - np.shape(x_ssa_trunc)[1]))
        if np.shape(x_ssa_trunc)[1] < np.shape(y)[1]:
            y = y[:, :np.shape(x_ssa_trunc)[1]]
        elif np.shape(y)[1] < np.shape(x_ssa_trunc)[1]:
            x_ssa_trunc = x_ssa_trunc[:, :np.shape(y)[1]]
            spline_basis_direct_trunc = spline_basis_direct_trunc[:, :np.shape(y)[1]]

    diff = 0
    # make extended model 'x' and 'y' same size:
    if np.shape(x_high_trunc)[1] != np.shape(y)[1]:
        diff = int(np.abs(np.shape(y)[1] - np.shape(x_high_trunc)[1]))
        if np.shape(x_high_trunc)[1] < np.shape(y)[1]:
            y = y[:, :np.shape(x_high_trunc)[1]]
        elif np.shape(y)[1] < np.shape(x_high_trunc)[1]:
            x_high_trunc = x_high_trunc[:, :np.shape(y)[1]]
            x_mid_trunc = x_mid_trunc[:, :np.shape(y)[1]]
            x_trend_trunc = x_trend_trunc[:, :np.shape(y)[1]]
            x_trunc = x_trunc[:, :np.shape(y)[1]]
            spline_basis_direct_trunc = spline_basis_direct_trunc[:, :np.shape(y)[1]]

    # calculate 'A_est'
    A_est_ssa = A_est.copy()

    # calculate B_est and Psi_est - ssa
    B_est_direct_ssa, Psi_est_direct_ssa = \
        cov_reg_given_mean(A_est=A_est_ssa, basis=spline_basis_direct_trunc, x=x_ssa_trunc, y=y, iterations=100)

    # calculate B_est and Psi_est - direct application
    B_est_direct, Psi_est_direct = \
        cov_reg_given_mean(A_est=A_est, basis=spline_basis_direct_trunc, x=x_trunc, y=y, iterations=100)

    # extended model values

    B_est_high, Psi_est_high = \
        cov_reg_given_mean(A_est=A_est, basis=spline_basis_direct_trunc, x=x_high_trunc, y=y, iterations=100)
    B_est_mid, Psi_est_mid = \
        cov_reg_given_mean(A_est=A_est, basis=spline_basis_direct_trunc, x=x_mid_trunc, y=y, iterations=100)
    B_est_trend, Psi_est_trend = \
        cov_reg_given_mean(A_est=A_est, basis=spline_basis_direct_trunc, x=x_trend_trunc, y=y, iterations=100)

    # calculate forecasted variance

    # days in the month where forecasting is to be done
    days_in_month_forecast_direct = int(end_of_month_vector_cumsum[int(day + months)] -
                                        end_of_month_vector_cumsum[int(day + months - 1)] - diff)

    # empty forecasted variance storage matrix - ssa
    variance_Model_forecast_ssa = np.zeros(
        (days_in_month_forecast_direct, np.shape(B_est_direct)[1], np.shape(B_est_direct)[1]))

    # dcc mgarch
    variance_Model_forecast_dcc = np.zeros(
        (days_in_month_forecast_direct, np.shape(B_est_direct)[1], np.shape(B_est_direct)[1]))

    # extended model

    variance_forecast_high = np.zeros(
        (days_in_month_forecast_direct, np.shape(B_est_direct)[1], np.shape(B_est_direct)[1]))
    variance_forecast_mid = np.zeros(
        (days_in_month_forecast_direct, np.shape(B_est_direct)[1], np.shape(B_est_direct)[1]))
    variance_forecast_trend = np.zeros(
        (days_in_month_forecast_direct, np.shape(B_est_direct)[1], np.shape(B_est_direct)[1]))

    # imf days that will be used to forecast variance of returns
    forecast_days = np.arange(end_of_month_vector_cumsum[int(day + months - 1)],
                              end_of_month_vector_cumsum[int(day + months)])[:days_in_month_forecast_direct]
    # commented out for speed
    # DCC forecast
    # known_returns = \
    #     sector_11_indices_array[end_of_month_vector_cumsum[int(day)]:end_of_month_vector_cumsum[int(day + months)], :]
    # dcc_forecast = \
    #     covregpy_dcc(known_returns, days=1)

    # realised covariance
    variance_Model_forecast_realised = np.zeros(
        (days_in_month_forecast_direct, np.shape(B_est_direct)[1], np.shape(B_est_direct)[1]))

    # iteratively calculate variance
    for var_day in forecast_days:
        # convert var_day index to [0 -> end of month length] index
        forecasted_variance_index = int(var_day - end_of_month_vector_cumsum[int(day + months - 1)])

        # extract last days of imf
        extract_x_imf_values = int(var_day - end_of_month_vector_cumsum[day])

        variance_Model_forecast_ssa[forecasted_variance_index] = \
            Psi_est_direct_ssa + \
            np.matmul(np.matmul(B_est_direct_ssa.T, x_ssa[:, extract_x_imf_values]).astype(np.float64).reshape(-1, 1),
                      np.matmul(x_ssa[:, extract_x_imf_values].T,
                                B_est_direct_ssa).astype(np.float64).reshape(1, -1)).astype(np.float64)

        # extended model

        variance_forecast_high[forecasted_variance_index] = \
            Psi_est_high + \
            np.matmul(np.matmul(B_est_high.T, x_high[:, extract_x_imf_values]).astype(np.float64).reshape(-1, 1),
                      np.matmul(x_high[:, extract_x_imf_values].T,
                                B_est_high).astype(np.float64).reshape(1, -1)).astype(np.float64)
        variance_forecast_mid[forecasted_variance_index] = \
            Psi_est_mid + \
            np.matmul(np.matmul(B_est_mid.T, x_mid[:, extract_x_imf_values]).astype(np.float64).reshape(-1, 1),
                      np.matmul(x_mid[:, extract_x_imf_values].T,
                                B_est_mid).astype(np.float64).reshape(1, -1)).astype(np.float64)
        variance_forecast_trend[forecasted_variance_index] = \
            Psi_est_trend + \
            np.matmul(np.matmul(B_est_trend.T, x_trend[:, extract_x_imf_values]).astype(np.float64).reshape(-1, 1),
                      np.matmul(x_trend[:, extract_x_imf_values].T,
                                B_est_trend).astype(np.float64).reshape(1, -1)).astype(np.float64)
        # commented out for speed
        # dcc mgarch
        # variance_Model_forecast_dcc[forecasted_variance_index] = dcc_forecast * np.sqrt(forecasted_variance_index + 1)

        # realised covariance
        variance_Model_forecast_realised[forecasted_variance_index] = annual_covariance

    # debugging step
    # plt.plot(np.mean(np.mean(np.abs(variance_Model_forecast_direct), axis=1), axis=1))
    variance_median_direct_ssa = np.median(variance_Model_forecast_ssa, axis=0)

    variance_median_dcc = np.median(variance_Model_forecast_dcc, axis=0)

    variance_median_realised = np.median(variance_Model_forecast_realised, axis=0)

    variance_median_high = np.median(variance_forecast_high, axis=0)
    variance_median_mid = np.median(variance_forecast_mid, axis=0)
    variance_median_trend = np.median(variance_forecast_trend, axis=0)

    #####################################################
    # direct application Covariance Regression - BOTTOM #
    #####################################################

    weights_forecast_high = risk_parity_weights_summation_restriction(variance_median_high,
                                                                      short_limit=0.3, long_limit=1.3).x
    print('High frequency weight = {}'.format(weights_forecast_high))
    variance_forecast_high = global_obj_fun(weights_forecast_high, monthly_covariance)
    returns_forecast_high = sum(weights_forecast_high * monthly_returns)
    weights_forecast_mid = risk_parity_weights_summation_restriction(variance_median_mid,
                                                                     short_limit=0.3, long_limit=1.3).x
    print('Mid frequency weight = {}'.format(weights_forecast_mid))
    variance_forecast_mid = global_obj_fun(weights_forecast_mid, monthly_covariance)
    returns_forecast_mid = sum(weights_forecast_mid * monthly_returns)
    weights_forecast_trend = risk_parity_weights_summation_restriction(variance_median_trend,
                                                                       short_limit=0.3, long_limit=1.3).x
    print('Trend frequency weight = {}'.format(weights_forecast_trend))
    variance_forecast_trend = global_obj_fun(weights_forecast_trend, monthly_covariance)
    returns_forecast_trend = sum(weights_forecast_trend * monthly_returns)

    # calculate efficient frontier
    plt.title(textwrap.fill(f'Realised Portfolio Returns versus Portfolio Variance for period from '
                            f'1 {month_vector[int(day % 12)]} {year_vector[int((day + 12) // 12)]} to '
                            f'{np.str(end_of_month_vector[int(day + 13)])} {month_vector[int(int(day + 12) % 12)]} '
                            f'{year_vector[int(int(day + 12) // 12)]}', 57), fontsize=12)
    ef_sd, ef_r = efficient_frontier(gm_w, gm_r, gm_sd, ms_w, ms_r, ms_sd, monthly_covariance)
    # plt.plot(ef_sd, ef_r, 'k--', label='Efficient frontier')
    ef_sd, ef_r = efficient_frontier(gm_w, gm_r, gm_sd, ms_w, ms_r, ms_sd, variance_median_high)
    # plt.plot(ef_sd[1:-1], ef_r[1:-1], '--', c='cyan', label=textwrap.fill('Efficient frontier high frequencies', 20))
    ef_sd, ef_r = efficient_frontier(gm_w, gm_r, gm_sd, ms_w, ms_r, ms_sd, variance_median_mid)
    # plt.plot(ef_sd[1:-1], ef_r[1:-1], '--', c='magenta', label=textwrap.fill('Efficient frontier mid frequencies', 20))
    ef_sd, ef_r = efficient_frontier(gm_w, gm_r, gm_sd, ms_w, ms_r, ms_sd, variance_median_trend)
    # plt.plot(ef_sd[1:-1], ef_r[1:-1], '--', c='gold', label=textwrap.fill('Efficient frontier low frequencies', 20))
    # plt.xlabel('Portfolio variance')
    # plt.legend(loc='lower right', fontsize=10)
    # plt.savefig('figures/efficient_frontiers/Efficient_frontiers_{}'.format(int(day + 1)))
    # plt.show()

    # calculate weights, variance, and returns - direct application ssa Covariance Regression - long restraint removed
    weights_Model_forecast_direct_ssa_long_short = risk_parity_weights_summation_restriction(variance_median_direct_ssa,
                                                                                             short_limit=0.3).x
    weights_Model_forecast_direct_ssa_summation_restriction = risk_parity_weights_summation_restriction(
        variance_median_direct_ssa).x

    # fill extended model storage matrices
    weight_matrix_high[end_of_month_vector_cumsum[day]:end_of_month_vector_cumsum[int(day + 1)],
    :] = weights_forecast_high
    weight_matrix_mid[end_of_month_vector_cumsum[day]:end_of_month_vector_cumsum[int(day + 1)],
    :] = weights_forecast_mid
    weight_matrix_trend[end_of_month_vector_cumsum[day]:end_of_month_vector_cumsum[int(day + 1)],
    :] = weights_forecast_trend

    print(day)

with open('experimental_figures/extended_model_shorting_final.npy', 'wb') as f:
    np.save(f, weight_matrix_high)
    np.save(f, weight_matrix_mid)
    np.save(f, weight_matrix_trend)

with open('experimental_figures/extended_model_shorting.npy', 'rb') as f:
    weight_matrix_high = np.load(f)
    weight_matrix_mid = np.load(f)
    weight_matrix_trend = np.load(f)

# plot significant weights
ax = plt.subplot(111)
plt.gcf().subplots_adjust(bottom=0.18)
plt.title('High Frequency Weights', fontsize=12)
for i in range(11):
    if i == 10:
        plt.plot(weight_matrix_high[:end_of_month_vector_cumsum[48], i],
                 label=sector_11_indices.columns[i], c='k')
    else:
        plt.plot(weight_matrix_high[:end_of_month_vector_cumsum[48], i],
                 label=sector_11_indices.columns[i])
plt.yticks(fontsize=8)
plt.ylabel('Weights', fontsize=10)
plt.xticks([0, 365, 730, 1096, 1460],
           ['01-01-2018', '01-01-2019', '01-01-2020', '01-01-2021', '31-12-2021'],
           fontsize=8, rotation=-30)
plt.xlabel('Days', fontsize=10)
plt.legend(loc='best', fontsize=6)
# plt.savefig('figures/S&P 500 - 11 Sectors/emd_mdlp_high_weights.png')
plt.show()

# plot significant weights
ax = plt.subplot(111)
plt.gcf().subplots_adjust(bottom=0.18)
plt.title('Mid-Frequency Weights', fontsize=12)
for i in range(11):
    if i == 10:
        plt.plot(weight_matrix_mid[:end_of_month_vector_cumsum[48], i],
                 label=sector_11_indices.columns[i], c='k')
    else:
        plt.plot(weight_matrix_mid[:end_of_month_vector_cumsum[48], i],
                 label=sector_11_indices.columns[i])
plt.yticks(fontsize=8)
plt.ylabel('Weights', fontsize=10)
plt.xticks([0, 365, 730, 1096, 1460],
           ['01-01-2018', '01-01-2019', '01-01-2020', '01-01-2021', '31-12-2021'],
           fontsize=8, rotation=-30)
plt.xlabel('Days', fontsize=10)
plt.legend(loc='best', fontsize=6)
# plt.savefig('figures/S&P 500 - 11 Sectors/emd_mdlp_mid_weights.png')
plt.show()

# plot significant weights
ax = plt.subplot(111)
plt.gcf().subplots_adjust(bottom=0.18)
plt.title('Low Frequency Weights', fontsize=12)
for i in range(11):
    if i == 10:
        plt.plot(weight_matrix_trend[:end_of_month_vector_cumsum[48], i],
                 label=sector_11_indices.columns[i], c='k')
    else:
        plt.plot(weight_matrix_trend[:end_of_month_vector_cumsum[48], i],
                 label=sector_11_indices.columns[i])
plt.yticks(fontsize=8)
plt.ylabel('Weights', fontsize=10)
plt.xticks([0, 365, 730, 1096, 1460],
           ['01-01-2018', '01-01-2019', '01-01-2020', '01-01-2021', '31-12-2021'],
           fontsize=8, rotation=-30)
plt.xlabel('Days', fontsize=10)
plt.legend(loc='best', fontsize=6)
# plt.savefig('figures/S&P 500 - 11 Sectors/emd_mdlp_trend_weights.png')
plt.show()

cumulative_returns_mdlp_high = cumulative_return(weight_matrix_high[:end_of_month_vector_cumsum[48]].T,
                                                 sector_11_indices_array[end_of_month_vector_cumsum[12]:].T)
cumulative_returns_mdlp_mid = cumulative_return(weight_matrix_mid[:end_of_month_vector_cumsum[48]].T,
                                                sector_11_indices_array[end_of_month_vector_cumsum[12]:].T)
cumulative_returns_mdlp_trend = cumulative_return(weight_matrix_trend[:end_of_month_vector_cumsum[48]].T,
                                                  sector_11_indices_array[end_of_month_vector_cumsum[12]:].T)

# measure performances - cumulative returns
cumulative_returns_global_minimum_portfolio = \
    cumulative_return(weight_matrix_global_minimum[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T)
cumulative_returns_maximum_sharpe_ratio_portfolio_restriction = \
    cumulative_return(weight_matrix_maximum_sharpe_ratio[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T)
cumulative_returns_pca_portfolio = \
    cumulative_return(weight_matrix_pca[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T)
cumulative_returns_dcc = \
    cumulative_return(weight_matrix_dcc[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T)
cumulative_returns_realised = \
    cumulative_return(weight_matrix_realised[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T)

ax = plt.subplot(111)
plt.gcf().subplots_adjust(bottom=0.18)
plt.title('Cumulative Returns', fontsize=12)
plt.plot(sp500_proxy, label='S&P 500 Proxy')
plt.plot(cumulative_returns_mdlp_high, label='High frequency MDLP')
plt.plot(cumulative_returns_mdlp_mid, label='Mid-frequency MDLP')
plt.plot(cumulative_returns_mdlp_trend, label='Low frequency MDLP')
plt.plot(cumulative_returns_realised, label='Realised covariance')
plt.plot(cumulative_returns_global_minimum_portfolio, label=textwrap.fill('Global minimum variance', 15))
plt.plot(cumulative_returns_maximum_sharpe_ratio_portfolio_restriction, label=textwrap.fill('Maximum Sharpe ratio', 15))
plt.plot(cumulative_returns_pca_portfolio, label=textwrap.fill('Principle portfolio with 3 components', 20))
plt.plot(cumulative_returns_dcc, label=textwrap.fill('DCC-MGARCH', 20))
plt.yticks(fontsize=8)
plt.ylabel('Cumulative Returns', fontsize=10)
plt.xticks([0, 365, 730, 1096, 1461],
           ['31-12-2017', '31-12-2018', '31-12-2019', '31-12-2020', '31-12-2021'],
           fontsize=8, rotation=-30)
plt.xlabel('Days', fontsize=10)
box_0 = ax.get_position()
ax.set_position([box_0.x0 - 0.04, box_0.y0, box_0.width * 0.84, box_0.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
# plt.savefig('figures/S&P 500 - 11 Sectors/Sector_11_indices_cumulative_returns.png')
plt.show()

# measure performances - mean returns
window = 30

mean_returns_mdlp_high = mean_return(weight_matrix_high[:end_of_month_vector_cumsum[48]].T,
                                     sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
mean_returns_mdlp_mid = mean_return(weight_matrix_mid[:end_of_month_vector_cumsum[48]].T,
                                    sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
mean_returns_mdlp_trend = mean_return(weight_matrix_trend[:end_of_month_vector_cumsum[48]].T,
                                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)

mean_returns_sp500 = \
    mean_return(np.ones_like(sp500_returns.reshape(1, -1)), sp500_returns.reshape(1, -1), window)

mean_returns_global_minimum_portfolio = \
    mean_return(weight_matrix_global_minimum[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
mean_returns_pca_portfolio = \
    mean_return(weight_matrix_pca[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
mean_returns_mgarch = \
    mean_return(weight_matrix_dcc[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
mean_returns_realised_covariance = \
    mean_return(weight_matrix_realised[:end_of_month_vector_cumsum[48]].T,
                sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
mean_returns_maximum_sharpe_ratio_portfolio = \
    mean_return(weight_matrix_maximum_sharpe_ratio[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)

ax = plt.subplot(111)
plt.gcf().subplots_adjust(bottom=0.18)
plt.title('Mean Returns', fontsize=12)
plt.plot(mean_returns_sp500, label='S&P 500 Proxy')
plt.plot(mean_returns_mdlp_high, label='High frequency MDLP')
plt.plot(mean_returns_mdlp_mid, label='Mid-frequency MDLP')
plt.plot(mean_returns_mdlp_trend, label='Low frequency MDLP')
plt.plot(mean_returns_realised_covariance, label='Realised covariance')
plt.plot(mean_returns_global_minimum_portfolio, label=textwrap.fill('Global minimum variance', 15))
plt.plot(mean_returns_maximum_sharpe_ratio_portfolio, label=textwrap.fill('Maximum Sharpe ratio', 15))
plt.plot(mean_returns_pca_portfolio, label=textwrap.fill('Principle portfolio with 3 components', 20))
plt.plot(mean_returns_mgarch, label='DCC-MGARCH')
plt.yticks(fontsize=8)
plt.ylabel('Mean Daily Returns', fontsize=10)
plt.xticks([0, 334, 699, 1065, 1430],
           ['30-01-2018', '31-12-2018', '31-12-2019', '31-12-2020', '31-12-2021'],
           fontsize=8, rotation=-30)
plt.xlabel('Days', fontsize=10)
box_0 = ax.get_position()
ax.set_position([box_0.x0 - 0.01, box_0.y0, box_0.width * 0.80, box_0.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
# plt.savefig('figures/S&P 500 - 11 Sectors/Sector_11_indices_mean_returns.png')
plt.show()

# measure performances - variance returns
window = 30

variance_returns_mdlp_high = variance_return(weight_matrix_high[:end_of_month_vector_cumsum[48]].T,
                                             sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
variance_returns_mdlp_mid = variance_return(weight_matrix_mid[:end_of_month_vector_cumsum[48]].T,
                                            sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
variance_returns_mdlp_trend = variance_return(weight_matrix_trend[:end_of_month_vector_cumsum[48]].T,
                                              sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)

variance_returns_sp500 = \
    variance_return(np.ones_like(sp500_returns.reshape(1, -1)), sp500_returns.reshape(1, -1), window)
variance_returns_global_minimum_portfolio = \
    variance_return(weight_matrix_global_minimum[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
variance_returns_pca_portfolio = \
    variance_return(weight_matrix_pca[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
variance_returns_realised_covariance = \
    variance_return(weight_matrix_realised[:end_of_month_vector_cumsum[48]].T,
                    sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
variance_returns_mgarch = \
    variance_return(weight_matrix_dcc[:end_of_month_vector_cumsum[48]].T,
                    sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
variance_returns_maximum_sharpe_ratio_portfolio = \
    variance_return(weight_matrix_maximum_sharpe_ratio[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)

ax = plt.subplot(111)
plt.gcf().subplots_adjust(bottom=0.18)
plt.title('Variance Returns', fontsize=12)
plt.plot(variance_returns_sp500, label='S&P 500 Proxy')
plt.plot(variance_returns_mdlp_high, label='High frequency MDLP')
plt.plot(variance_returns_mdlp_mid, label='Mid-frequency MDLP')
plt.plot(variance_returns_mdlp_trend, label='Low frequency MDLP')
plt.plot(variance_returns_realised_covariance, label='Realised covariance')
plt.plot(variance_returns_global_minimum_portfolio, label=textwrap.fill('Global minimum variance', 15))
plt.plot(variance_returns_maximum_sharpe_ratio_portfolio,
         label=textwrap.fill('Maximum Sharpe ratio', 15))
plt.plot(variance_returns_pca_portfolio, label=textwrap.fill('Principle portfolio with 3 components', 20))
plt.plot(variance_returns_mgarch, label='DCC-MGARCH')
plt.yticks(fontsize=8)
plt.ylabel('Variance Daily Returns', fontsize=10)
plt.xticks([0, 334, 699, 1065, 1430],
           ['30-01-2018', '31-12-2018', '31-12-2019', '31-12-2020', '31-12-2021'],
           fontsize=8, rotation=-30)
plt.xlabel('Days', fontsize=10)
box_0 = ax.get_position()
ax.set_position([box_0.x0 - 0.01, box_0.y0, box_0.width * 0.8, box_0.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
# plt.savefig('figures/S&P 500 - 11 Sectors/Sector_11_indices_variance_returns.png')
plt.show()

# measure performances - value-at-risk returns
window = 30

value_at_risk_returns_mdlp_high = value_at_risk_return(weight_matrix_high[:end_of_month_vector_cumsum[48]].T,
                                                       sector_11_indices_array[end_of_month_vector_cumsum[12]:].T,
                                                       window)
value_at_risk_returns_mdlp_mid = value_at_risk_return(weight_matrix_mid[:end_of_month_vector_cumsum[48]].T,
                                                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T,
                                                      window)
value_at_risk_returns_mdlp_trend = value_at_risk_return(weight_matrix_trend[:end_of_month_vector_cumsum[48]].T,
                                                        sector_11_indices_array[end_of_month_vector_cumsum[12]:].T,
                                                        window)

value_at_risk_returns_sp500 = \
    value_at_risk_return(np.ones_like(sp500_returns.reshape(1, -1)), sp500_returns.reshape(1, -1), window)

value_at_risk_returns_global_minimum_portfolio = \
    value_at_risk_return(weight_matrix_global_minimum[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
value_at_risk_returns_pca_portfolio = \
    value_at_risk_return(weight_matrix_pca[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
value_at_risk_returns_realised_covariance = \
    value_at_risk_return(weight_matrix_realised[:end_of_month_vector_cumsum[48]].T,
                         sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
value_at_risk_returns_mgarch = \
    value_at_risk_return(weight_matrix_dcc[:end_of_month_vector_cumsum[48]].T,
                         sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
value_at_risk_returns_maximum_sharpe_ratio_portfolio = \
    value_at_risk_return(weight_matrix_maximum_sharpe_ratio[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)

ax = plt.subplot(111)
plt.gcf().subplots_adjust(bottom=0.18)
plt.title('Value-at-Risk Returns', fontsize=12)
plt.plot(value_at_risk_returns_sp500, label='S&P 500 Proxy')
plt.plot(value_at_risk_returns_mdlp_high, label='High frequency MDLP')
plt.plot(value_at_risk_returns_mdlp_mid, label='Mid-frequency MDLP')
plt.plot(value_at_risk_returns_mdlp_trend, label='Low frequency MDLP')
plt.plot(value_at_risk_returns_realised_covariance, label='Realised covariance')
plt.plot(value_at_risk_returns_global_minimum_portfolio, label=textwrap.fill('Global minimum variance', 15))
plt.plot(value_at_risk_returns_maximum_sharpe_ratio_portfolio,
         label=textwrap.fill('Maximum Sharpe ratio', 15))
plt.plot(value_at_risk_returns_pca_portfolio, label=textwrap.fill('Principle portfolio with 3 components', 20))
plt.plot(value_at_risk_returns_mgarch, label='DCC-MGARCH')
plt.yticks(fontsize=8)
plt.ylabel('Value-at-Risk Daily Returns', fontsize=10)
plt.xticks([0, 334, 699, 1065, 1430],
           ['30-01-2018', '31-12-2018', '31-12-2019', '31-12-2020', '31-12-2021'],
           fontsize=8, rotation=-30)
plt.xlabel('Days', fontsize=10)
box_0 = ax.get_position()
ax.set_position([box_0.x0 - 0.01, box_0.y0, box_0.width * 0.8, box_0.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
# plt.savefig('figures/S&P 500 - 11 Sectors/Sector_11_indices_value_at_risk_returns.png')
plt.show()

# measure performances - maximum draw down returns
window = 30

max_draw_down_returns_mdlp_high = max_draw_down_return(weight_matrix_high[:end_of_month_vector_cumsum[48]].T,
                                                       sector_11_indices_array[end_of_month_vector_cumsum[12]:].T,
                                                       window)
max_draw_down_returns_mdlp_mid = max_draw_down_return(weight_matrix_mid[:end_of_month_vector_cumsum[48]].T,
                                                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T,
                                                      window)
max_draw_down_returns_mdlp_trend = max_draw_down_return(weight_matrix_trend[:end_of_month_vector_cumsum[48]].T,
                                                        sector_11_indices_array[end_of_month_vector_cumsum[12]:].T,
                                                        window)

max_draw_down_returns_sp500 = \
    max_draw_down_return(np.ones_like(sp500_returns.reshape(1, -1)), sp500_returns.reshape(1, -1), window)
max_draw_down_returns_global_minimum_portfolio = \
    max_draw_down_return(weight_matrix_global_minimum[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
max_draw_down_returns_pca_portfolio = \
    max_draw_down_return(weight_matrix_pca[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
max_draw_down_returns_realised_covariance = \
    max_draw_down_return(weight_matrix_realised[:end_of_month_vector_cumsum[48]].T,
                         sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
max_draw_down_returns_mgarch = \
    max_draw_down_return(weight_matrix_dcc[:end_of_month_vector_cumsum[48]].T,
                         sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
max_draw_down_returns_maximum_sharpe_ratio_portfolio = \
    max_draw_down_return(weight_matrix_maximum_sharpe_ratio[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)

ax = plt.subplot(111)
plt.gcf().subplots_adjust(bottom=0.18)
plt.title('Maximum Draw Down Returns', fontsize=12)
plt.plot(max_draw_down_returns_sp500, label='S&P 500 Proxy')
plt.plot(max_draw_down_returns_mdlp_high, label='High frequency MDLP')
plt.plot(max_draw_down_returns_mdlp_mid, label='Mid-frequency MDLP')
plt.plot(max_draw_down_returns_mdlp_trend, label='Low frequency MDLP')
plt.plot(max_draw_down_returns_realised_covariance, label='Realised Covariance')
plt.plot(max_draw_down_returns_global_minimum_portfolio, label=textwrap.fill('Global minimum variance', 15))
plt.plot(max_draw_down_returns_maximum_sharpe_ratio_portfolio,
         label=textwrap.fill('Maximum Sharpe ratio', 15))
plt.plot(max_draw_down_returns_pca_portfolio, label=textwrap.fill('Principle portfolio with 3 components', 20))
plt.plot(max_draw_down_returns_mgarch, label='DCC-MGARCH')
plt.yticks(fontsize=8)
plt.ylabel('Maximum Draw Down Daily Returns', fontsize=10)
plt.xticks([0, 334, 699, 1065, 1430],
           ['30-01-2018', '31-12-2018', '31-12-2019', '31-12-2020', '31-12-2021'],
           fontsize=8, rotation=-30)
plt.xlabel('Days', fontsize=10)
box_0 = ax.get_position()
ax.set_position([box_0.x0 - 0.04, box_0.y0, box_0.width * 0.84, box_0.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
# plt.savefig('figures/S&P 500 - 11 Sectors/Sector_11_indices_max_draw_down_returns.png')
plt.show()

# measure performances - omega ratio returns
window = 30

omega_ratio_returns_mdlp_high = omega_ratio_return(weight_matrix_high[:end_of_month_vector_cumsum[48]].T,
                                                   sector_11_indices_array[end_of_month_vector_cumsum[12]:].T,
                                                   window)
omega_ratio_returns_mdlp_mid = omega_ratio_return(weight_matrix_mid[:end_of_month_vector_cumsum[48]].T,
                                                  sector_11_indices_array[end_of_month_vector_cumsum[12]:].T,
                                                  window)
omega_ratio_returns_mdlp_trend = omega_ratio_return(weight_matrix_trend[:end_of_month_vector_cumsum[48]].T,
                                                    sector_11_indices_array[end_of_month_vector_cumsum[12]:].T,
                                                    window)

omega_ratio_returns_sp500 = \
    omega_ratio_return(np.ones_like(sp500_returns.reshape(1, -1)), sp500_returns.reshape(1, -1), window)
omega_ratio_returns_global_minimum_portfolio = \
    omega_ratio_return(weight_matrix_global_minimum[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
omega_ratio_returns_pca_portfolio = \
    omega_ratio_return(weight_matrix_pca[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
omega_ratio_returns_realised_covariance = \
    omega_ratio_return(weight_matrix_realised[:end_of_month_vector_cumsum[48]].T,
                       sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
omega_ratio_returns_mgarch = \
    omega_ratio_return(weight_matrix_dcc[:end_of_month_vector_cumsum[48]].T,
                       sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
omega_ratio_returns_maximum_sharpe_ratio_portfolio = \
    omega_ratio_return(weight_matrix_maximum_sharpe_ratio[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)

ax = plt.subplot(111)
plt.gcf().subplots_adjust(bottom=0.18)
plt.title('Omega Ratio Returns', fontsize=12)
plt.plot(omega_ratio_returns_sp500, label='S&P 500 Proxy')
plt.plot(omega_ratio_returns_mdlp_high, label='High frequency MDLP')
plt.plot(omega_ratio_returns_mdlp_mid, label='Mid-frequency MDLP')
plt.plot(omega_ratio_returns_mdlp_trend, label='Low frequency MDLP')
plt.plot(omega_ratio_returns_realised_covariance, label='Realised covariance')
plt.plot(omega_ratio_returns_global_minimum_portfolio, label=textwrap.fill('Global minimum variance', 15))
plt.plot(omega_ratio_returns_maximum_sharpe_ratio_portfolio,
         label=textwrap.fill('Maximum Sharpe ratio', 15))
plt.plot(omega_ratio_returns_pca_portfolio, label=textwrap.fill('Principle portfolio with 3 components', 20))
plt.plot(omega_ratio_returns_mgarch, label='DCC-MGARCH')
plt.yticks(fontsize=8)
plt.ylabel('Omega Ratio Daily Returns', fontsize=10)
plt.xticks([0, 334, 699, 1065, 1430],
           ['30-01-2018', '31-12-2018', '31-12-2019', '31-12-2020', '31-12-2021'],
           fontsize=8, rotation=-30)
plt.xlabel('Days', fontsize=10)
box_0 = ax.get_position()
ax.set_position([box_0.x0 - 0.04, box_0.y0, box_0.width * 0.84, box_0.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
# plt.savefig('figures/S&P 500 - 11 Sectors/Sector_11_indices_omega_ratio_returns.png')
plt.show()

# measure performances - Sharpe ratio returns
window = 30

sharpe_ratio_returns_mdlp_high = sharpe_ratio_return(weight_matrix_high[:end_of_month_vector_cumsum[48]].T,
                                                     sector_11_indices_array[end_of_month_vector_cumsum[12]:].T,
                                                     window)
sharpe_ratio_returns_mdlp_mid = sharpe_ratio_return(weight_matrix_mid[:end_of_month_vector_cumsum[48]].T,
                                                    sector_11_indices_array[end_of_month_vector_cumsum[12]:].T,
                                                    window)
sharpe_ratio_returns_mdlp_trend = sharpe_ratio_return(weight_matrix_trend[:end_of_month_vector_cumsum[48]].T,
                                                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T,
                                                      window)

sharpe_ratio_returns_sp500 = \
    sharpe_ratio_return(np.ones_like(sp500_returns.reshape(1, -1)), sp500_returns.reshape(1, -1), window)
sharpe_ratio_returns_global_minimum_portfolio = \
    sharpe_ratio_return(weight_matrix_global_minimum[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
sharpe_ratio_returns_pca_portfolio = \
    sharpe_ratio_return(weight_matrix_pca[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
sharpe_ratio_returns_realised_covariance = \
    sharpe_ratio_return(weight_matrix_realised[:end_of_month_vector_cumsum[48]].T,
                        sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
sharpe_ratio_returns_mgarch = \
    sharpe_ratio_return(weight_matrix_dcc[:end_of_month_vector_cumsum[48]].T,
                        sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
sharpe_ratio_returns_maximum_sharpe_ratio_portfolio = \
    sharpe_ratio_return(weight_matrix_maximum_sharpe_ratio[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)

ax = plt.subplot(111)
plt.gcf().subplots_adjust(bottom=0.18)
plt.title('Sharpe Ratio Returns', fontsize=12)
plt.plot(sharpe_ratio_returns_sp500, label='S&P 500 Proxy')
plt.plot(sharpe_ratio_returns_mdlp_high, label='High frequency MDLP')
plt.plot(sharpe_ratio_returns_mdlp_mid, label='Mid-frequency MDLP')
plt.plot(sharpe_ratio_returns_mdlp_trend, label='Low frequency MDLP')
plt.plot(sharpe_ratio_returns_realised_covariance, label='Realised covariance')
plt.plot(sharpe_ratio_returns_global_minimum_portfolio, label=textwrap.fill('Global minimum variance', 15))
plt.plot(sharpe_ratio_returns_maximum_sharpe_ratio_portfolio,
         label=textwrap.fill('Maximum Sharpe ratio', 15))
plt.plot(sharpe_ratio_returns_pca_portfolio, label=textwrap.fill('Principle portfolio with 3 components', 20))
plt.plot(sharpe_ratio_returns_mgarch, label='DCC-MGARCH')
plt.yticks(fontsize=8)
plt.ylabel('Sharpe Ratio Daily Returns', fontsize=10)
plt.xticks([0, 334, 699, 1065, 1430],
           ['30-01-2018', '31-12-2018', '31-12-2019', '31-12-2020', '31-12-2021'],
           fontsize=8, rotation=-30)
plt.xlabel('Days', fontsize=10)
box_0 = ax.get_position()
ax.set_position([box_0.x0 - 0.02, box_0.y0, box_0.width * 0.8, box_0.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
# plt.savefig('figures/S&P 500 - 11 Sectors/Sector_11_indices_sharpe_ratio_returns.png')
plt.show()

sortino_ratio_returns_mdlp_high = sharpe_ratio_return(weight_matrix_high[:end_of_month_vector_cumsum[48]].T,
                                                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T,
                                                      window)
sortino_ratio_returns_mdlp_mid = sharpe_ratio_return(weight_matrix_mid[:end_of_month_vector_cumsum[48]].T,
                                                     sector_11_indices_array[end_of_month_vector_cumsum[12]:].T,
                                                     window)
sortino_ratio_returns_mdlp_trend = sharpe_ratio_return(weight_matrix_trend[:end_of_month_vector_cumsum[48]].T,
                                                       sector_11_indices_array[end_of_month_vector_cumsum[12]:].T,
                                                       window)

sortino_ratio_returns_sp500 = \
    sharpe_ratio_return(np.ones_like(sp500_returns.reshape(1, -1)), sp500_returns.reshape(1, -1), window)
sortino_ratio_returns_global_minimum_portfolio = \
    sharpe_ratio_return(weight_matrix_global_minimum[:end_of_month_vector_cumsum[48]].T,
                        sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
sortino_ratio_returns_pca_portfolio = \
    sharpe_ratio_return(weight_matrix_pca[:end_of_month_vector_cumsum[48]].T,
                        sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
sortino_ratio_returns_realised_covariance = \
    sharpe_ratio_return(weight_matrix_realised[:end_of_month_vector_cumsum[48]].T,
                        sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
sortino_ratio_returns_mgarch = \
    sharpe_ratio_return(weight_matrix_dcc[:end_of_month_vector_cumsum[48]].T,
                        sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
sortino_ratio_returns_maximum_sharpe_ratio_portfolio = \
    sortino_ratio_return(weight_matrix_maximum_sharpe_ratio[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)

ax = plt.subplot(111)
plt.gcf().subplots_adjust(bottom=0.18)
plt.title('Sortino Ratio Returns', fontsize=12)
plt.plot(sortino_ratio_returns_sp500, label='S&P 500 Proxy')
plt.plot(sortino_ratio_returns_mdlp_high, label='High frequency MDLP')
plt.plot(sortino_ratio_returns_mdlp_mid, label='Mid-frequency MDLP')
plt.plot(sortino_ratio_returns_mdlp_trend, label='Low frequency MDLP')
plt.plot(sortino_ratio_returns_realised_covariance, label='Realised covariance')
plt.plot(sortino_ratio_returns_global_minimum_portfolio, label=textwrap.fill('Global minimum variance', 15))
plt.plot(sortino_ratio_returns_maximum_sharpe_ratio_portfolio,
         label=textwrap.fill('Maximum Sharpe ratio', 15))
plt.plot(sortino_ratio_returns_pca_portfolio, label=textwrap.fill('Principle portfolio with 3 components', 20))
plt.plot(sortino_ratio_returns_mgarch, label='DCC-MGARCH')
plt.yticks(fontsize=8)
plt.ylabel('Sortino Ratio Daily Returns', fontsize=10)
plt.xticks([0, 334, 699, 1065, 1430],
           ['30-01-2018', '31-12-2018', '31-12-2019', '31-12-2020', '31-12-2021'],
           fontsize=8, rotation=-30)
plt.xlabel('Days', fontsize=10)
box_0 = ax.get_position()
ax.set_position([box_0.x0 - 0.02, box_0.y0, box_0.width * 0.8, box_0.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
# plt.savefig('figures/S&P 500 - 11 Sectors/Sector_11_indices_sortino_ratio_returns.png')
plt.show()




#     ________
#            /
#      \    /
#       \  /
#        \/

# RPP using EMD, RCR, and MDLP - direct application (no forecasting)

import textwrap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mdlp.discretization import MDLP

from sklearn.linear_model import Ridge

from CovRegpy_utilities import efficient_frontier, global_minimum_forward_applied_information, \
    sharpe_forward_applied_information, pca_forward_applied_information, \
    global_minimum_forward_applied_information_long, \
    sharpe_forward_applied_information_summation_restriction

from CovRegpy_RCR import cov_reg_given_mean, cubic_b_spline

from CovRegpy_RPP import equal_risk_parity_weights_long_restriction, equal_risk_parity_weights_summation_restriction, global_obj_fun

from CovRegpy_measures import cumulative_return, mean_return, variance_return, value_at_risk_return, \
    max_draw_down_return, omega_ratio_return, sortino_ratio_return, sharpe_ratio_return

from CovRegpy_SSA import CovRegpy_ssa
from CovRegpy_DCC import covregpy_dcc
from AdvEMDpy import AdvEMDpy, emd_basis

np.random.seed(0)

sns.set(style='darkgrid')

fig, axs = plt.subplots(1, 2, sharey=True, figsize=(32, 16))
axs[0].set_title('Cumulative Returns', fontsize=36, pad=20)
axs[0].set_ylabel('Cumulative Returns', fontsize=20)
axs[0].set_xticks([0, 365, 730, 1096, 1461])
axs[0].set_xticklabels(['31-12-2017', '31-12-2018', '31-12-2019', '31-12-2020', '31-12-2021'],
                       fontsize=16, rotation=-30)
axs[0].set_xlabel('Days', fontsize=20)
axs[0].set_yticks([0.8, 1.0, 1.2, 1.4, 1.6])
axs[0].set_yticklabels([0.8, 1.0, 1.2, 1.4, 1.6], fontsize=16)
box_0 = axs[0].get_position()
axs[0].set_position([box_0.x0 - 0.075, box_0.y0 + 0.02, box_0.width * 1.24, box_0.height])
axs[0].legend(loc='upper left', fontsize=16)
axs[0].set_ylim(0.6, 1.8)
axs[1].set_title('Cumulative Returns', fontsize=36)
# axs[1].set_yticks([0.8, 1.0, 1.2, 1.4, 1.6])
# axs[1].set_yticklabels(['', '', '', '', ''], fontsize=8)
# axs[1].set_ylabel('Cumulative Returns', fontsize=10)
axs[1].set_xticks([0, 365, 730, 1096, 1461],
                  ['31-12-2017', '31-12-2018', '31-12-2019', '31-12-2020', '31-12-2021'],
                  fontsize=16, rotation=-30)
axs[1].set_xlabel('Days', fontsize=20)
box_0 = axs[1].get_position()
axs[1].set_position([box_0.x0 - 0.02, box_0.y0 + 0.02, box_0.width * 1.24, box_0.height])
axs[1].legend(loc='upper left', fontsize=16)
axs[0].set_ylim(0.6, 1.8)
plt.show()

# create S&P 500 index
sp500_close = pd.read_csv('../S&P500_Data/sp_500_close_5_year.csv', header=0)
sp500_close = sp500_close.set_index(['Unnamed: 0'])
sp500_market_cap = pd.read_csv('../S&P500_Data/sp_500_market_cap_5_year.csv', header=0)
sp500_market_cap = sp500_market_cap.set_index(['Unnamed: 0'])

sp500_returns = np.log(np.asarray(sp500_close)[1:, :] / np.asarray(sp500_close)[:-1, :])
weights = np.asarray(sp500_market_cap) / np.tile(np.sum(np.asarray(sp500_market_cap), axis=1).reshape(-1, 1), (1, 505))
sp500_returns = np.sum(sp500_returns * weights[:-1, :], axis=1)[365:-1]
sp500_proxy = np.append(1, np.exp(np.cumsum(sp500_returns)))

sp500_returns_001 = (np.log(np.asarray(sp500_close)[1:, :] / np.asarray(sp500_close)[:-1, :]) - 0.0001)
sp500_returns_001 = np.sum(sp500_returns_001 * weights[:-1, :], axis=1)[365:-1]
sp500_proxy_001 = np.append(1, np.exp(np.cumsum(sp500_returns_001)))

# load 11 sector indices
sector_11_indices = pd.read_csv('../S&P500_Data/sp_500_11_sector_indices.csv', header=0)
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
weight_matrix_global_minimum_long = np.zeros_like(sector_11_indices_array)
# weight_matrix_maximum_sharpe_ratio = np.zeros_like(sector_11_indices_array)
weight_matrix_maximum_sharpe_ratio_restriction = np.zeros_like(sector_11_indices_array)
weight_matrix_pca = np.zeros_like(sector_11_indices_array)
weight_matrix_direct_imf_covreg = np.zeros_like(sector_11_indices_array)
weight_matrix_direct_imf_covreg_high = np.zeros_like(sector_11_indices_array)
weight_matrix_direct_ssa_covreg = np.zeros_like(sector_11_indices_array)
weight_matrix_direct_imf_covreg_restriction = np.zeros_like(sector_11_indices_array)
weight_matrix_direct_imf_covreg_high_restriction = np.zeros_like(sector_11_indices_array)
weight_matrix_direct_ssa_covreg_restriction = np.zeros_like(sector_11_indices_array)
weight_matrix_dcc = np.zeros_like(sector_11_indices_array)
weight_matrix_realised = np.zeros_like(sector_11_indices_array)

# extended model

weight_matrix_high = np.zeros_like(sector_11_indices_array)
weight_matrix_mid = np.zeros_like(sector_11_indices_array)
weight_matrix_trend = np.zeros_like(sector_11_indices_array)

ssa_components, x, x_if, y, x_high, x_trend, x_mid, x_high_1 = {}, {}, {}, {}, {}, {}, {}, {}

# weights calculated on and used on different data (one month ahead)
for day in range(len(end_of_month_vector_cumsum[:-int(months + 1)])):

    del ssa_components, x, x_if, y, x_high, x_trend, x_mid, x_high_1

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
    # plt.scatter(gm_sd, gm_r, label='Global minimum variance portfolio', zorder=2)

    # calculate global minimum variance portfolio - long only
    gm_w_long, gm_sd_long, gm_r_long = global_minimum_forward_applied_information_long(annual_covariance,
                                                                                       monthly_covariance,
                                                                                       monthly_returns)
    # plt.scatter(gm_sd, gm_r, label='Global minimum variance portfolio', zorder=2)

    # calculate maximum sharpe ratio portfolio
    ms_w, ms_sd, ms_r = sharpe_forward_applied_information(annual_covariance, annual_returns,
                                                           monthly_covariance, monthly_returns,
                                                           gm_w, gm_r, risk_free)
    # plt.scatter(ms_sd, ms_r, label='Maximum Sharpe ratio portfolio', zorder=2)

    # calculate maximum sharpe ratio portfolio
    msr_w, msr_sd, msr_r = sharpe_forward_applied_information_summation_restriction(annual_covariance,
                                                                                    annual_returns,
                                                                                    monthly_covariance,
                                                                                    monthly_returns,
                                                                                    risk_free=risk_free,
                                                                                    short_limit=0.3,
                                                                                    long_limit=1.3)
    # print(np.sum(msr_w[msr_w < 0]))
    # print(np.sum(msr_w[msr_w > 0]))
    # plt.scatter(msr_sd, msr_r, label='Maximum Sharpe ratio portfolio with restriction', zorder=2)

    # calculate efficient frontier
    ef_sd, ef_r = efficient_frontier(gm_w, gm_r, gm_sd, ms_w, ms_r, ms_sd, monthly_covariance)
    # plt.plot(ef_sd, ef_r, 'k--', label='Efficient frontier', zorder=1)
    # plt.show()

    # calculate pca portfolio
    pc_w, pc_sd, pc_r = pca_forward_applied_information(annual_covariance, monthly_covariance,
                                                        monthly_returns, factors=3)
    # plt.scatter(pc_sd, pc_r, label='Principle portfolio (3 components)', zorder=2)

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
               end_of_month_vector_cumsum[int(day)]:end_of_month_vector_cumsum[int(day + months)], :], rcond=None)[0]

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
        # plt.plot(imfs[-1, :])
        # plt.plot(ssa_components)
        # plt.show()
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
            x_high = np.vstack((imfs[:2, :], x_high))
        except:
            x_high = imfs[:2, :].copy()
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
            x_high_1 = np.vstack((imfs[0, :], x_high_1))
        except:
            x_high_1 = imfs[0, :].copy()

    mdlp = MDLP()

    y = y[1:]

    # # debugging step
    # for i in range(len(y[:-1])):
    #     if y[i] < y[int(i + 1)]:
    #         plt.plot(x_if[i, :])
    #         plt.plot(np.median(x_if[i, :]) * np.ones_like(x_if[i, :]), '--')
    #     else:
    #         plt.plot(x_if[i, :])
    #         plt.plot(np.median(x_if[i, :]) * np.ones_like(x_if[i, :]), '--')
    #         plt.show()
    # plt.plot(x_if[int(i + 1), :])
    # plt.plot(np.median(x_if[int(i + 1), :]) * np.ones_like(x_if[int(i + 1), :]), '--')
    # plt.show()

    x_if = x_if[y != 4, :]
    y = y[y != 4]
    if len(y) < 33:
        i = 0
        for index in np.arange(int(len(y) - 1))[np.diff(y) == -1]:
            x_if = np.vstack((x_if[:int(index + i + 1), :], 0.001 * np.ones(int(np.shape(price_signal)[0] - 1)).reshape(1, -1),
                              x_if[int(index + i + 1):, :]))
            y = np.hstack((y[:int(index + i + 1)].reshape(1, -1),
                           np.array(3).reshape(1, 1), y[int(index + i + 1):].reshape(1, -1)))[0]
            i += 1
    y = y[::-1]

    time = np.asarray([0])
    cuts_0_1 = np.asarray([0])
    cuts_1_2 = np.asarray([0])

    assets = 2
    # for i in np.arange(assets, int(33 - 2 * assets), 3):

    for i in np.arange(assets, int(assets + 3), 3):
        # plt.plot(x_if[int(i - 2), :])
        # plt.plot(x_if[int(i - 1), :])
        # plt.plot(x_if[int(i - 0), :])
        # plt.plot(x_if[int(i + 1), :])
        # plt.plot(x_if[int(i + 2), :])
        # plt.plot(x_if[int(i + 3), :])
        # plt.show()
        mdlp = MDLP()
        # X_mdlp = mdlp.fit_transform(x_if[int(i - 2):int(i + 1), :], y[int(i - 2):int(i + 1)])
        X_mdlp = mdlp.fit_transform(x_if[int(i - assets):int(i + 2 * assets), :], y[int(i - assets):int(i + 2 * assets)])
        X_mdlp_cut_points = mdlp.cut_points_
        plt.title('Two Cut-Point Examples')
        multiple_scatter = np.arange(364).reshape(1, -1)
        for a in range(1, assets):
            multiple_scatter = np.vstack((multiple_scatter, np.arange(364).reshape(1, -1)))
        # plt.scatter(multiple_scatter,
        #             x_if[int(i - assets):int(i + 2 * assets), :][y[int(i - assets):int(i + 2 * assets)] == 1],
        #             c='red', label='IF 3')
        # plt.scatter(multiple_scatter,
        #             x_if[int(i - assets):int(i + 2 * assets), :][y[int(i - assets):int(i + 2 * assets)] == 2],
        #             c='green', label='IF 2')
        # plt.scatter(multiple_scatter,
        #             x_if[int(i - assets):int(i + 2 * assets), :][y[int(i - assets):int(i + 2 * assets)] == 3],
        #             c='blue', label='IF 1')

        test = 0
        for num, cut_point in enumerate(X_mdlp_cut_points):
            if len(cut_point) > 1:
                # plt.plot(cut_point[0] * np.ones_like(x_if[6, :]))
                # plt.plot(cut_point[1] * np.ones_like(x_if[6, :]))
                if test == 0:
                    # plt.scatter(num, cut_point[0], c='black', s=40, label='Cut-point 0-1', zorder=10)
                    # plt.scatter(num, cut_point[1], c='gold', s=40, label='Cut-point 1-2', zorder=10)
                    test += 1
                # plt.scatter(num, cut_point[0], c='black', s=40, zorder=10)
                # plt.scatter(num, cut_point[1], c='gold', s=40, zorder=10)

                time = np.hstack((time, num))
                cuts_0_1 = np.hstack((cuts_0_1, cut_point[0]))
                cuts_1_2 = np.hstack((cuts_1_2, cut_point[1]))

                # accuracy_1 = np.mean(x_if[y == 1, :][:assets] < cut_point[0])
                # accuracy_2 = np.mean(np.r_[cut_point[0] < x_if[y == 2, :][:assets]] & np.r_[x_if[y == 2, :][:assets] < cut_point[1]])
                # accuracy_3 = np.mean(x_if[y == 3, :][:assets] > cut_point[1])
                # print('Probability structure 1 below 0-1 cut: {}'.format(accuracy_1))
                # print('Probability structure 2 above 0-1 and below 1-2 cut: {}'.format(accuracy_2))
                # print('Probability structure 3 above 1-2 cut: {}'.format(accuracy_3))
                # accuracy = (np.mean(x_if[y == 1, :][:assets] < cut_point[0]) +
                #             np.mean(np.r_[cut_point[0] < x_if[y == 2, :][:assets]] & np.r_[x_if[y == 2, :][:assets] < cut_point[1]]) + \
                #             np.mean(x_if[y == 3, :][:assets] < cut_point[1])) / 3
                # print(accuracy)

        time_full = np.arange(int(np.shape(price_signal)[0] - 1))
        time_series_full = time_full.copy()
        knots = np.linspace(-20, 383, 100)

        basis = emd_basis.Basis(time=time_full, time_series=time_series_full)
        basis = basis.cubic_b_spline(knots=knots)
        basis_subset = basis[:, time[1:]]

        coef_0_1 = np.linalg.lstsq(basis_subset.T, cuts_0_1[1:], rcond=None)[0]
        coef_1_2 = np.linalg.lstsq(basis_subset.T, cuts_1_2[1:], rcond=None)[0]

        smoothing_penalty = 1
        second_order_matrix = np.zeros((96, 94))  # note transpose

        for a in range(94):
            second_order_matrix[a:(a + 3), a] = [1, -2, 1]  # filling values for second-order difference matrix

        coef_penalised_0_1 = np.linalg.lstsq(np.append(basis_subset,
                                                       smoothing_penalty * second_order_matrix, axis=1).T,
                                             np.append(cuts_0_1[1:], np.zeros(94)), rcond=None)[0]
        coef_penalised_1_2 = np.linalg.lstsq(np.append(basis_subset,
                                                       smoothing_penalty * second_order_matrix, axis=1).T,
                                             np.append(cuts_1_2[1:], np.zeros(94)), rcond=None)[0]

        clf = Ridge(alpha=1.0)
        clf.fit(basis_subset.T, cuts_0_1[1:])
        coef_penalised_positive_0_1 = clf.coef_
        clf = Ridge(alpha=1.0)
        clf.fit(basis_subset.T, cuts_1_2[1:])
        coef_penalised_positive_1_2 = clf.coef_

        spline = np.matmul(coef_0_1.reshape(1, -1), basis)
        # plt.plot(np.arange(364), spline[0, :])
        spline_penalised_0_1 = np.matmul(coef_penalised_0_1.reshape(1, -1), basis)
        spline_penalised_1_2 = np.matmul(coef_penalised_1_2.reshape(1, -1), basis)
        spline_penalised_positive_0_1 = np.matmul(coef_penalised_positive_0_1.reshape(1, -1), basis)
        spline_penalised_positive_1_2 = np.matmul(coef_penalised_positive_1_2.reshape(1, -1), basis)

        if np.shape(x_if)[1] == 365:
            temp = 0
        accuracy_1 = np.mean(x_if[y == 1, :][int(i - assets):int(i + 2 * assets)] < spline_penalised_0_1)
        accuracy_2 = np.mean(np.r_[spline_penalised_0_1 < x_if[y == 2, :][int(i - assets):int(i + 2 * assets)]] &
                             np.r_[x_if[y == 2, :][int(i - assets):int(i + 2 * assets)] < spline_penalised_1_2])
        accuracy_3 = np.mean(x_if[y == 3, :][int(i - assets):int(i + 2 * assets)] > spline_penalised_1_2)
        print('Probability structure 1 below 0-1 cut: {}'.format(accuracy_1))
        print('Probability structure 2 above 0-1 and below 1-2 cut: {}'.format(accuracy_2))
        print('Probability structure 3 above 1-2 cut: {}'.format(accuracy_3))

        for a in range(np.shape(multiple_scatter)[0]):
            for b in range(np.shape(multiple_scatter)[1]):
                if a == 0 and b == 0:
                    # plt.scatter(multiple_scatter[a, b],
                    #             x_if[int(i - assets):int(i + 2 * assets), :][
                    #                 y[int(i - assets):int(i + 2 * assets)] == 1][a, b],
                    #             c='green' if x_if[int(i - assets):int(i + 2 * assets), :][
                    #                              y[int(i - assets):int(i + 2 * assets)] == 1][a, b] <
                    #                          spline_penalised_0_1[0, b] else 'red')
                    # plt.scatter(multiple_scatter[a, b],
                    #             x_if[int(i - assets):int(i + 2 * assets), :][
                    #                 y[int(i - assets):int(i + 2 * assets)] == 2][a, b],
                    #             c='green' if x_if[int(i - assets):int(i + 2 * assets), :][
                    #                              y[int(i - assets):int(i + 2 * assets)] == 2][a, b] >
                    #                          spline_penalised_0_1[0, b] and x_if[int(i - assets):int(i + 2 * assets), :][
                    #                              y[int(i - assets):int(i + 2 * assets)] == 2][a, b] <
                    #                          spline_penalised_1_2[0, b] else 'red', label='Incorrectly classified')
                    # plt.scatter(multiple_scatter[a, b],
                    #             x_if[int(i - assets):int(i + 2 * assets), :][
                    #                 y[int(i - assets):int(i + 2 * assets)] == 3][a, b],
                    #             c='green' if x_if[int(i - assets):int(i + 2 * assets), :][
                    #                              y[int(i - assets):int(i + 2 * assets)] == 3][a, b] >
                    #                          spline_penalised_1_2[0, b] else 'red', label='Correctly classified')
                    pass
                else:
                    # plt.scatter(multiple_scatter[a, b],
                    #             x_if[int(i - assets):int(i + 2 * assets), :][
                    #                 y[int(i - assets):int(i + 2 * assets)] == 1][a, b],
                    #             c='green' if x_if[int(i - assets):int(i + 2 * assets), :][
                    #                              y[int(i - assets):int(i + 2 * assets)] == 1][a, b] <
                    #                          spline_penalised_0_1[0, b] else 'red')
                    # plt.scatter(multiple_scatter[a, b],
                    #             x_if[int(i - assets):int(i + 2 * assets), :][
                    #                 y[int(i - assets):int(i + 2 * assets)] == 2][a, b],
                    #             c='green' if x_if[int(i - assets):int(i + 2 * assets), :][
                    #                              y[int(i - assets):int(i + 2 * assets)] == 2][a, b] >
                    #                          spline_penalised_0_1[0, b] and x_if[int(i - assets):int(i + 2 * assets), :][
                    #                              y[int(i - assets):int(i + 2 * assets)] == 2][a, b] <
                    #                          spline_penalised_1_2[0, b] else 'red')
                    # plt.scatter(multiple_scatter[a, b],
                    #             x_if[int(i - assets):int(i + 2 * assets), :][
                    #                 y[int(i - assets):int(i + 2 * assets)] == 3][a, b],
                    #             c='green' if x_if[int(i - assets):int(i + 2 * assets), :][
                    #                              y[int(i - assets):int(i + 2 * assets)] == 3][a, b] >
                    #                          spline_penalised_1_2[0, b] else 'red')
                    pass

        # plt.plot(np.arange(364), spline_penalised_0_1[0, :], label='Sub-decision surface 0-1', Linewidth=3, c='k')
        # plt.plot(np.arange(364), spline_penalised_1_2[0, :], label='Sub-decision surface 1-2', Linewidth=3, c='gold')
        # plt.plot(np.arange(364), spline_penalised_positive_0_1[0, :], label='Sub-decision positive surface 0-1')
        # plt.plot(np.arange(364), spline_penalised_positive_1_2[0, :], label='Sub-decision positive surface 1-2')
        # plt.legend(loc='upper right')
        # plt.text(0, 0.5, 'Accuracy IF 3 {}'.format(np.round(accuracy_1, 2)), fontsize=8)
        # plt.text(0, 0.55, 'Accuracy IF 2 {}'.format(np.round(accuracy_2, 2)), fontsize=8)
        # plt.text(0, 0.6, 'Accuracy IF 1 {}'.format(np.round(accuracy_3, 2)), fontsize=8)
        # plt.text(0, 0.35, 'Accuracy Decisions 0 {}'.format(np.round(np.mean(np.matmul(coef_0_1.reshape(1, -1), basis_subset) > x_if[y == 1, :][:assets][:, time[1:]]), 2)), fontsize=8)
        # plt.text(0, 0.40, 'Accuracy Decisions 1 {}'.format(np.round(np.mean(np.r_[np.matmul(coef_1_2.reshape(1, -1), basis_subset) > x_if[y == 2, :][:assets][:, time[1:]]] & np.r_[np.matmul(coef_0_1.reshape(1, -1), basis_subset) < x_if[y == 2, :][:assets][:, time[1:]]]), 2)), fontsize=8)
        # plt.text(0, 0.45, 'Accuracy Decisions 2 {}'.format(np.round(np.mean(np.matmul(coef_1_2.reshape(1, -1), basis_subset) < x_if[y == 3, :][:assets][:, time[1:]]), 2)), fontsize=8)
        # plt.ylim(-0.11, 0.75)
        # plt.fill(np.append(np.linspace(-10, 130, 100), np.linspace(-10, 130, 100)[::-1]),
        #          np.append(0.65 * np.ones(100), 0.33 * np.ones(100)), c='lightgrey')
        # if i == assets:
        #     plt.savefig('../experimental_figures/cut_point_demonstration.png')
        # plt.show()

    mdlp = MDLP()
    X_mdlp = mdlp.fit_transform(x_if[:int(assets * 3), :], y[:int(assets * 3)])
    # plt.plot(x_if[:, 0])
    # plt.plot(y)
    X_mdlp_cut_points = mdlp.cut_points_

    x_stack = np.zeros((1, 33))
    for i in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        # plt.plot(np.searchsorted([i], x_if[:, 0]))
        try:
            x_stack = np.vstack((x_stack, np.searchsorted([i], x_if[:, 0])))
        except:
            x_stack = np.searchsorted([i], x_if[:, 0])
    # temp = np.searchsorted([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], x_if[:, 0])

    # X, Y = np.meshgrid(np.arange(33), np.arange(12))
    # plt.pcolormesh(X, Y, x_stack)
    # plt.show()

    cut_point_storage = np.empty((3, np.shape(x_if[:int(assets * 3), :])[1]))
    cut_point_storage[:] = np.nan

    for time_increment in range(np.shape(x_if[:int(assets * 3), :])[1]):

        # if len(X_mdlp_cut_points[time_increment]) == 2 and time_increment == 28:
        #
        #     point = 0
        #
        #     fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 4))
        #     plt.suptitle('Minimum Description Length Binning Example')
        #     ax1.spines['bottom'].set_visible(False)
        #     ax1.tick_params(axis='x', which='both', bottom=False)
        #     ax2.spines['top'].set_visible(False)
        #
        #     bs = 0.08
        #     ts = 0.1
        #
        #     ax2.set_ylim(0, bs)
        #     ax1.set_ylim(ts, 0.6)
        #
        #     # ax1.scatter(np.zeros(33)[y == 1][:assets], x_if[:, time_increment][y == 1][:assets], c='red', label='IF 3')
        #     # ax1.scatter(np.zeros(33)[y == 2][:assets], x_if[:, time_increment][y == 2][:assets], c='green', label='IF 2')
        #     # ax1.scatter(np.zeros(33)[y == 3][:assets], x_if[:, time_increment][y == 3][:assets], c='blue', label='IF 1')
        #
        #     # ax2.scatter(np.zeros(33)[y == 1][:assets], x_if[:, time_increment][y == 1][:assets], c='red')
        #     # ax2.scatter(np.zeros(33)[y == 2][:assets], x_if[:, time_increment][y == 2][:assets], c='green')
        #     # ax2.scatter(np.zeros(33)[y == 3][:assets], x_if[:, time_increment][y == 3][:assets], c='blue')
        #
        #     print(x_if[:, time_increment][y == 1][:assets])
        #     print(x_if[:, time_increment][y == 2][:assets])
        #     print(x_if[:, time_increment][y == 3][:assets])
        #
        #     for tick in ax2.get_xticklabels():
        #         tick.set_rotation(0)
        #     d = .015
        #     kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
        #     ax1.plot((-d, +d), (-d, +d), **kwargs)
        #     ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        #     kwargs.update(transform=ax2.transAxes)
        #     ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
        #     ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
        #
        #     # for b1, b2 in zip(bars1, bars2):
        #     #     posx = b2.get_x() + b2.get_width() / 2.
        #     #     if b2.get_height() > bs:
        #     #         ax2.plot((posx - 3 * d, posx + 3 * d), (1 - d, 1 + d), color='k', clip_on=False,
        #     #                  transform=ax2.get_xaxis_transform())
        #     #     if b1.get_height() > ts:
        #     #         ax1.plot((posx - 3 * d, posx + 3 * d), (- d, + d), color='k', clip_on=False,
        #     #                  transform=ax1.get_xaxis_transform())
        #     ax2.set_yticks(np.arange(0.00, 0.09, 0.01))
        #     ax2.set_yticklabels([0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08], fontsize=8)
        #     ax1.set_title('Cut-Points at Time Increment: {}'.format(time_increment), fontsize=10)
        #     ax1.set_yticks(np.arange(0.1, 0.7, 0.1))
        #     ax1.set_yticklabels([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], fontsize=8)
        #     ax2.set_xticks([0])
        #     ax2.set_xticklabels([''])
        #     ax1.set_ylabel('Instantanoues frequency', fontsize=8)
        #     ax2.set_ylabel('Instantanoues frequency', fontsize=8)
        #     for cut_point in X_mdlp_cut_points[time_increment]:
        #         # plt.title('Cut-points time time point: {}'.format(time_increment))
        #         ax2.plot(np.linspace(-0.5, 0.5, 100), cut_point * np.ones(100), '--')
        #         ax1.plot(np.linspace(-0.5, 0.5, 100), cut_point * np.ones(100), '--', label='Cut-point {}-{}'.format(point, int(point + 1)))
        #         cut_point_storage[point, time_increment] = cut_point
        #         point += 1
        #     ax1.legend(loc='upper left', fontsize=8)
        #     plt.savefig('../experimental_figures/cut_point_time_point.png')
        #     plt.show()
        #
        #     point = 0
        #
        #     fig = plt.figure(1)
        #     fig.set_size_inches(18, 12)
        #     plt.title('Minimum Description Length Binning Example', fontsize=32, pad=10.0)
        #
        #     plt.scatter(np.zeros(33)[y == 1][:assets], x_if[:, time_increment][y == 1][:assets], c='red',
        #                 label='IF 3', s=200)
        #     plt.scatter(np.zeros(33)[y == 2][:assets], x_if[:, time_increment][y == 2][:assets], c='green',
        #                 label='IF 2', s=200)
        #     plt.scatter(np.zeros(33)[y == 3][:assets], x_if[:, time_increment][y == 3][:assets], c='blue',
        #                 label='IF 1', s=200)
        #
        #     plt.scatter(np.zeros(33)[y == 1][:assets], x_if[:, time_increment][y == 1][:assets], c='red', s=200)
        #     plt.scatter(np.zeros(33)[y == 2][:assets], x_if[:, time_increment][y == 2][:assets], c='green', s=200)
        #     plt.scatter(np.zeros(33)[y == 3][:assets], x_if[:, time_increment][y == 3][:assets], c='blue', s=200)
        #
        #     plt.ylabel('Instantanoues frequency', fontsize=20)
        #     plt.xticks([0], [' '])
        #     plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], fontsize=16)
        #     for cut_point in X_mdlp_cut_points[time_increment]:
        #         # plt.title('Cut-points time time point: {}'.format(time_increment))
        #         plt.plot(np.linspace(-0.5, 0.5, 100), cut_point * np.ones(100), '--', linewidth=5)
        #         plt.plot(np.linspace(-0.5, 0.5, 100), cut_point * np.ones(100), '--',
        #                  label='Cut-point {}-{}'.format(point, int(point + 1)), linewidth=5)
        #         cut_point_storage[point, time_increment] = cut_point
        #         point += 1
        #     plt.legend(loc='upper left', fontsize=16)
        #     plt.savefig('../experimental_figures/cut_point_time_point_example.pdf')
        #     plt.show()
        #
        #     # plt.scatter(np.zeros(33)[y == 1][:assets], x_if[:, time_increment][y == 1][:assets], c='red')
        #     # plt.scatter(np.zeros(33)[y == 2][:assets], x_if[:, time_increment][y == 2][:assets], c='green')
        #     # plt.scatter(np.zeros(33)[y == 3][:assets], x_if[:, time_increment][y == 3][:assets], c='blue')
        #
        #     point = 0
        #
        #     # for cut_point in X_mdlp_cut_points[time_increment]:
        #     #     plt.title('Cut-points time time point: {}'.format(time_increment))
        #     #     plt.plot(np.linspace(-0.5, 0.5, 100), cut_point * np.ones(100), '--')
        #     #     cut_point_storage[point, time_increment] = cut_point
        #     #     point += 1
        #     #
        #     # # plt.ylim(0.00, 0.08)
        #     # plt.show()

        pass

    X_mdlp_median_sort = np.median(X_mdlp, axis=1)

    distinct_vector = np.arange(len(np.unique(X_mdlp_median_sort)))
    distinct_vector_count = np.zeros_like(distinct_vector)

    distinct_structures = np.zeros((len(distinct_vector), np.shape(X_mdlp)[1]))

    # comparing correlation matrices

    colors_1 = ['red', 'orange', 'gold', 'lawngreen', 'green', 'cyan', 'magenta', 'blue', 'darkviolet', 'pink',
                'yellow']
    colors_2 = ['red', 'red', 'orange', 'orange', 'gold', 'gold', 'lawngreen', 'lawngreen', 'green', 'green', 'cyan',
                'cyan', 'magenta', 'magenta', 'blue', 'blue', 'darkviolet', 'darkviolet', 'pink', 'pink', 'yellow',
                'yellow']

    # plt.title('High-Frequency Structures used in Covariance Regression')
    # plt.gca().set_prop_cycle(color=colors_2)
    # for col, sector in enumerate(sector_11_indices.columns):
    #     plt.plot(x_high[int(2 * col), :], label=sector)
    #     plt.plot(x_high[int(2 * col + 1), :])
    # plt.plot(333 * np.ones(100), np.linspace(-0.15, 0.15, 100), 'k--', label='Fit & forecast boundary')
    # plt.xticks([0, 333, 364], ['01-01-2017', '30-11-2017', '31-12-2017'], fontsize=8, rotation=-30)
    # plt.legend(loc='best', fontsize=6)
    # plt.show()

    # plt.title('Mid-Frequency Structures used in Covariance Regression')
    # plt.gca().set_prop_cycle(color=colors_2)
    # for col, sector in enumerate(sector_11_indices.columns):
    #     plt.plot(x_high[int(2 * col + 1), :], label=sector)
    # plt.plot(333 * np.ones(100), np.linspace(-0.15, 0.15, 100), 'k--', label='Fit & forecast boundary')
    # plt.xticks([0, 333, 364], ['01-01-2017', '30-11-2017', '31-12-2017'], fontsize=8, rotation=-30)
    # plt.legend(loc='best', fontsize=6)
    # plt.show()

    # plt.title('Low-Frequency Structures used in Covariance Regression')
    # plt.gca().set_prop_cycle(color=colors_1)
    # for col, sector in enumerate(sector_11_indices.columns):
    #     plt.plot(x_ssa[col, :], label=sector)
    # plt.plot(333 * np.ones(100), np.linspace(0.8, 1.4, 100), 'k--', label='Fit & forecast boundary')
    # plt.xticks([0, 333, 364], ['01-01-2017', '30-11-2017', '31-12-2017'], fontsize=8, rotation=-30)
    # plt.legend(loc='best', fontsize=6)
    # plt.show()

    # for col, sector in enumerate(sector_11_indices.columns):
    #     plt.title('{} - Mid-Frequency and Trend'.format(sector))
    #     plt.plot(x_trend[col, :])
    #     plt.plot(x_ssa[col, :] - np.mean(x_ssa[col, :]))
    #     plt.plot(x_high[int(2 * col + 1), :])
    #     plt.show()

    # for col, sector in enumerate(sector_11_indices.columns):
    #     plt.title('{} - IMF1 & IMF2'.format(sector))
    #     plt.plot(x_high[int(2 * col), :])
    #     plt.plot(x_high[int(2 * col + 1), :])
    #     plt.show()

    #############################################
    ### Residual IF Classification Discussion ###
    #############################################

    # classification without trends

    ##############################################
    ### begin truncation of further break-down ###
    ##############################################

    x_high_1_trunc = \
        x_high_1[:, :int(end_of_month_vector_cumsum[int(day + months - 1)] - end_of_month_vector_cumsum[int(day)])]
    x_mid_trunc = \
        x_mid[:, :int(end_of_month_vector_cumsum[int(day + months - 1)] - end_of_month_vector_cumsum[int(day)])]
    x_trend_trunc = \
        x_trend[:, :int(end_of_month_vector_cumsum[int(day + months - 1)] - end_of_month_vector_cumsum[int(day)])]

    # ssa
    x_ssa_trunc = \
        x_ssa[:, :int(end_of_month_vector_cumsum[int(day + months - 1)] - end_of_month_vector_cumsum[int(day)])]

    x_trunc = \
        x[:, :int(end_of_month_vector_cumsum[int(day + months - 1)] - end_of_month_vector_cumsum[int(day)])]
    x_high_trunc = \
        x_high[:, :int(end_of_month_vector_cumsum[int(day + months - 1)] - end_of_month_vector_cumsum[int(day)])]

    # calculate y - same for both imf and ssa
    y = sector_11_indices_array[end_of_month_vector_cumsum[int(day + months - 11)]:end_of_month_vector_cumsum[
        int(day + months)], :]
    y = y.T

    # make 'x' and 'y' the same size (working in terms of months and this can occur)
    diff = 0
    if np.shape(x_trunc)[1] != np.shape(y)[1]:
        diff = int(np.abs(np.shape(y)[1] - np.shape(x_trunc)[1]))
        if np.shape(x_trunc)[1] < np.shape(y)[1]:
            y = y[:, :np.shape(x_trunc)[1]]
        elif np.shape(y)[1] < np.shape(x_trunc)[1]:
            x_trunc = x_trunc[:, :np.shape(y)[1]]
            x_high_trunc = x_high_trunc[:, :np.shape(y)[1]]
            spline_basis_direct_trunc = spline_basis_direct_trunc[:, :np.shape(y)[1]]

    diff = 0
    # make 'x' and 'y' the same size - ssa
    if np.shape(x_ssa_trunc)[1] != np.shape(y)[1]:
        diff = int(np.abs(np.shape(y)[1] - np.shape(x_ssa_trunc)[1]))
        if np.shape(x_ssa_trunc)[1] < np.shape(y)[1]:
            y = y[:, :np.shape(x_ssa_trunc)[1]]
        elif np.shape(y)[1] < np.shape(x_ssa_trunc)[1]:
            x_ssa_trunc = x_ssa_trunc[:, :np.shape(y)[1]]

    diff = 0
    # make extended model 'x' and 'y' same size:
    if np.shape(x_high_1_trunc)[1] != np.shape(y)[1]:
        diff = int(np.abs(np.shape(y)[1] - np.shape(x_high_1_trunc)[1]))
        if np.shape(x_high_1_trunc)[1] < np.shape(y)[1]:
            y = y[:, :np.shape(x_high_1_trunc)[1]]
        elif np.shape(y)[1] < np.shape(x_high_1_trunc)[1]:
            x_high_1_trunc = x_high_1_trunc[:, :np.shape(y)[1]]
            x_mid_trunc = x_mid_trunc[:, :np.shape(y)[1]]
            x_trend_trunc = x_trend_trunc[:, :np.shape(y)[1]]

    # calculate 'A_est'
    A_est_ssa = A_est.copy()

    # calculate B_est and Psi_est - ssa
    B_est_direct_ssa, Psi_est_direct_ssa = \
        cov_reg_given_mean(A_est=A_est_ssa, basis=spline_basis_direct_trunc, x=x_ssa_trunc, y=y, iterations=100)

    # calculate B_est and Psi_est - direct application
    B_est_direct, Psi_est_direct = \
        cov_reg_given_mean(A_est=A_est, basis=spline_basis_direct_trunc, x=x_trunc, y=y, iterations=100)

    B_est_direct_high, Psi_est_direct_high = \
        cov_reg_given_mean(A_est=A_est, basis=spline_basis_direct_trunc, x=x_high_trunc, y=y, iterations=100)

    # extended model values

    B_est_high, Psi_est_high = \
        cov_reg_given_mean(A_est=A_est, basis=spline_basis_direct_trunc, x=x_high_1_trunc, y=y, iterations=100)
    B_est_mid, Psi_est_mid = \
        cov_reg_given_mean(A_est=A_est, basis=spline_basis_direct_trunc, x=x_mid_trunc, y=y, iterations=100)
    B_est_trend, Psi_est_trend = \
        cov_reg_given_mean(A_est=A_est, basis=spline_basis_direct_trunc, x=x_trend_trunc, y=y, iterations=100)

    # calculate forecasted variance

    # days in the month where forecasting is to be done
    days_in_month_forecast_direct = int(end_of_month_vector_cumsum[int(day + months)] -
                                        end_of_month_vector_cumsum[int(day + months - 1)] - diff)

    # empty forecasted variance storage matrix - direct
    variance_Model_forecast_direct = np.zeros(
        (days_in_month_forecast_direct, np.shape(B_est_direct)[1], np.shape(B_est_direct)[1]))

    # empty forecasted variance storage matrix - direct high frequency
    variance_Model_forecast_direct_high = np.zeros(
        (days_in_month_forecast_direct, np.shape(B_est_direct)[1], np.shape(B_est_direct)[1]))

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

        variance_Model_forecast_direct[forecasted_variance_index] = \
            Psi_est_direct + np.matmul(np.matmul(B_est_direct.T,
                                       x[:, extract_x_imf_values]).astype(np.float64).reshape(-1, 1),
                                       np.matmul(x[:, extract_x_imf_values].T,
                                                 B_est_direct).astype(np.float64).reshape(1, -1)).astype(np.float64)

        variance_Model_forecast_direct_high[forecasted_variance_index] = \
            Psi_est_direct_high + np.matmul(np.matmul(B_est_direct_high.T,
                                                 x_high[:, extract_x_imf_values]).astype(np.float64).reshape(-1, 1),
                                       np.matmul(x_high[:, extract_x_imf_values].T,
                                                 B_est_direct_high).astype(np.float64).reshape(1, -1)).astype(np.float64)

        variance_Model_forecast_ssa[forecasted_variance_index] = \
            Psi_est_direct_ssa + \
            np.matmul(np.matmul(B_est_direct_ssa.T, x_ssa[:, extract_x_imf_values]).astype(np.float64).reshape(-1, 1),
                      np.matmul(x_ssa[:, extract_x_imf_values].T,
                                B_est_direct_ssa).astype(np.float64).reshape(1, -1)).astype(np.float64)

        # extended model

        variance_forecast_high[forecasted_variance_index] = \
            Psi_est_high + \
            np.matmul(np.matmul(B_est_high.T, x_high_1[:, extract_x_imf_values]).astype(np.float64).reshape(-1, 1),
                      np.matmul(x_high_1[:, extract_x_imf_values].T,
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

        # dcc mgarch
        # variance_Model_forecast_dcc[forecasted_variance_index] = dcc_forecast * np.sqrt(forecasted_variance_index + 1)

        # realised covariance
        variance_Model_forecast_realised[forecasted_variance_index] = annual_covariance

    # debugging step
    # plt.plot(np.mean(np.mean(np.abs(variance_Model_forecast_direct), axis=1), axis=1))
    variance_median_direct = np.median(variance_Model_forecast_direct, axis=0)

    # debugging step
    # plt.plot(np.mean(np.mean(np.abs(variance_Model_forecast_direct), axis=1), axis=1))
    variance_median_direct_high = np.median(variance_Model_forecast_direct_high, axis=0)

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

    weights_forecast_high = equal_risk_parity_weights_summation_restriction(variance_median_high,
                                                                            short_limit=0.3, long_limit=1.3).x
    print('High frequency weight = {}'.format(weights_forecast_high))
    variance_forecast_high = global_obj_fun(weights_forecast_high, monthly_covariance)
    returns_forecast_high = sum(weights_forecast_high * monthly_returns)
    weights_forecast_mid = equal_risk_parity_weights_summation_restriction(variance_median_mid,
                                                                           short_limit=0.3, long_limit=1.3).x
    print('Mid frequency weight = {}'.format(weights_forecast_mid))
    variance_forecast_mid = global_obj_fun(weights_forecast_mid, monthly_covariance)
    returns_forecast_mid = sum(weights_forecast_mid * monthly_returns)
    weights_forecast_trend = equal_risk_parity_weights_summation_restriction(variance_median_trend,
                                                                             short_limit=0.3, long_limit=1.3).x
    print('Trend frequency weight = {}'.format(weights_forecast_trend))
    variance_forecast_trend = global_obj_fun(weights_forecast_trend, monthly_covariance)
    returns_forecast_trend = sum(weights_forecast_trend * monthly_returns)

    # # calculate efficient frontier
    # plt.title(textwrap.fill(f'Realised Portfolio Returns versus Portfolio Variance for period from '
    #                         f'1 {month_vector[int(day % 12)]} {year_vector[int((day + 12) // 12)]} to '
    #                         f'{str(end_of_month_vector[int(day + 13)])} {month_vector[int(int(day + 12) % 12)]} '
    #                         f'{year_vector[int(int(day + 12) // 12)]}', 57), fontsize=12)
    # ef_sd, ef_r = efficient_frontier(gm_w, gm_r, gm_sd, ms_w, ms_r, ms_sd, monthly_covariance)
    # plt.plot(ef_sd, ef_r, 'k--', label='Efficient frontier')
    # ef_sd, ef_r = efficient_frontier(gm_w, gm_r, gm_sd, ms_w, ms_r, ms_sd, variance_median_high)
    # plt.plot(ef_sd[1:-1], ef_r[1:-1], '--', c='cyan', label=textwrap.fill('Efficient frontier high frequencies', 20))
    # ef_sd, ef_r = efficient_frontier(gm_w, gm_r, gm_sd, ms_w, ms_r, ms_sd, variance_median_mid)
    # plt.plot(ef_sd[1:-1], ef_r[1:-1], '--', c='magenta', label=textwrap.fill('Efficient frontier mid frequencies', 20))
    # ef_sd, ef_r = efficient_frontier(gm_w, gm_r, gm_sd, ms_w, ms_r, ms_sd, variance_median_trend)
    # plt.plot(ef_sd[1:-1], ef_r[1:-1], '--', c='gold', label=textwrap.fill('Efficient frontier low frequencies', 20))
    # plt.xlabel('Portfolio variance')
    # plt.legend(loc='lower right', fontsize=10)
    # plt.savefig('../figures/efficient_frontiers/Efficient_frontiers_{}'.format(int(day + 1)))
    # plt.show()

    # calculate weights, variance, and returns - direct application ssa Covariance Regression - long only
    weights_Model_forecast_direct_ssa = equal_risk_parity_weights_long_restriction(variance_median_direct_ssa).x
    model_variance_forecast_direct_ssa = global_obj_fun(weights_Model_forecast_direct_ssa, monthly_covariance)
    model_returns_forecast_direct_ssa = sum(weights_Model_forecast_direct_ssa * monthly_returns)
    # plt.scatter(np.sqrt(model_variance_forecast_direct_ssa), model_returns_forecast_direct_ssa,
    #             label='CovReg Direct Model SSA')

    # calculate weights, variance, and returns - direct application Covariance Regression - long only
    weights_Model_forecast_direct = equal_risk_parity_weights_long_restriction(variance_median_direct).x
    model_variance_forecast_direct = global_obj_fun(weights_Model_forecast_direct, monthly_covariance)
    model_returns_forecast_direct = sum(weights_Model_forecast_direct * monthly_returns)
    # plt.scatter(np.sqrt(model_variance_forecast_direct), model_returns_forecast_direct, label='CovReg Direct Model')

    # calculate weights, variance, and returns - direct application Covariance Regression - long only
    weights_Model_forecast_direct_high = equal_risk_parity_weights_long_restriction(variance_median_direct_high).x
    model_variance_forecast_direct = global_obj_fun(weights_Model_forecast_direct, monthly_covariance)
    model_returns_forecast_direct = sum(weights_Model_forecast_direct * monthly_returns)
    # plt.scatter(np.sqrt(model_variance_forecast_direct), model_returns_forecast_direct, label='CovReg Direct Model')

    # calculate weights, variance, and returns - direct application ssa Covariance Regression - long restraint removed
    weights_Model_forecast_direct_ssa_long_short = equal_risk_parity_weights_summation_restriction(variance_median_direct_ssa, short_limit=0.3).x
    weights_Model_forecast_direct_ssa_summation_restriction = equal_risk_parity_weights_summation_restriction(variance_median_direct_ssa).x

    model_variance_forecast_direct_ssa_long_short = global_obj_fun(weights_Model_forecast_direct_ssa_summation_restriction,
                                                                   monthly_covariance)
    model_returns_forecast_direct_ssa_long_short = sum(weights_Model_forecast_direct_ssa_summation_restriction * monthly_returns)
    # plt.scatter(np.sqrt(model_variance_forecast_direct_ssa_long_short), model_returns_forecast_direct_ssa_long_short,
    #             label='CovReg Direct Model SSA')

    # calculate weights, variance, and returns - direct application Covariance Regression - long restraint removed
    weights_Model_forecast_direct_long_short = equal_risk_parity_weights_summation_restriction(variance_median_direct, short_limit=0.3).x
    weights_Model_forecast_direct_summation_restriction = equal_risk_parity_weights_summation_restriction(variance_median_direct).x

    model_variance_forecast_direct_long_short = global_obj_fun(weights_Model_forecast_direct_summation_restriction,
                                                               monthly_covariance)
    model_returns_forecast_direct_long_short = sum(weights_Model_forecast_direct_summation_restriction * monthly_returns)
    # plt.scatter(np.sqrt(model_variance_forecast_direct_long_short), model_returns_forecast_direct_long_short,
    #             label='CovReg Direct Model')

    weights_Model_forecast_direct_summation_restriction_high = equal_risk_parity_weights_summation_restriction(variance_median_direct_high).x

    weights_Model_forecast_dcc = equal_risk_parity_weights_long_restriction(variance_median_dcc).x

    weights_Model_forecast_realised = equal_risk_parity_weights_long_restriction(variance_median_realised).x

    # fill extended model storage matrices
    weight_matrix_high[end_of_month_vector_cumsum[day]:end_of_month_vector_cumsum[int(day + 1)], :] = weights_forecast_high
    weight_matrix_mid[end_of_month_vector_cumsum[day]:end_of_month_vector_cumsum[int(day + 1)], :] = weights_forecast_mid
    weight_matrix_trend[end_of_month_vector_cumsum[day]:end_of_month_vector_cumsum[int(day + 1)], :] = weights_forecast_trend

    # filled weight matrices iteratively
    weight_matrix_global_minimum[end_of_month_vector_cumsum[day]:end_of_month_vector_cumsum[int(day + 1)], :] = \
        gm_w
    weight_matrix_global_minimum_long[end_of_month_vector_cumsum[day]:end_of_month_vector_cumsum[int(day + 1)], :] = \
        gm_w_long
    # weight_matrix_maximum_sharpe_ratio[end_of_month_vector_cumsum[day]:end_of_month_vector_cumsum[int(day + 1)], :] = \
    #     ms_w
    weight_matrix_maximum_sharpe_ratio_restriction[end_of_month_vector_cumsum[day]:end_of_month_vector_cumsum[int(day + 1)], :] = \
        msr_w
    weight_matrix_pca[end_of_month_vector_cumsum[day]:end_of_month_vector_cumsum[int(day + 1)], :] = \
        pc_w
    weight_matrix_direct_imf_covreg[end_of_month_vector_cumsum[day]:end_of_month_vector_cumsum[int(day + 1)], :] = \
        weights_Model_forecast_direct
    weight_matrix_direct_ssa_covreg[end_of_month_vector_cumsum[day]:end_of_month_vector_cumsum[int(day + 1)], :] = \
        weights_Model_forecast_direct_ssa
    weight_matrix_direct_imf_covreg_restriction[end_of_month_vector_cumsum[day]:end_of_month_vector_cumsum[int(day + 1)], :] = \
        weights_Model_forecast_direct_summation_restriction
    weight_matrix_direct_ssa_covreg_restriction[end_of_month_vector_cumsum[day]:end_of_month_vector_cumsum[int(day + 1)], :] = \
        weights_Model_forecast_direct_ssa_summation_restriction
    weight_matrix_dcc[end_of_month_vector_cumsum[day]:end_of_month_vector_cumsum[int(day + 1)], :] = \
        weights_Model_forecast_dcc
    weight_matrix_realised[end_of_month_vector_cumsum[day]:end_of_month_vector_cumsum[int(day + 1)], :] = \
        weights_Model_forecast_realised

    weight_matrix_direct_imf_covreg_high[end_of_month_vector_cumsum[day]:end_of_month_vector_cumsum[int(day + 1)], :] = \
        weights_Model_forecast_direct_high
    weight_matrix_direct_imf_covreg_high_restriction[end_of_month_vector_cumsum[day]:end_of_month_vector_cumsum[int(day + 1)], :] = \
        weights_Model_forecast_direct_summation_restriction_high

    # graph options
    plt.title(f'Actual Portfolio Returns versus Portfolio Variance for '
              f'1 {month_vector[int(day % 12)]} {year_vector[int((day + 12) // 12)]} to '
              f'{str(end_of_month_vector[int(day + 13)])} {month_vector[int(int(day + 12) % 12)]} '
              f'{year_vector[int(int(day + 12) // 12)]}', fontsize=10)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.xlabel('Portfolio variance', fontsize=10)
    plt.ylabel('Portfolio returns', fontsize=10)
    plt.legend(loc='best', fontsize=8)
    # plt.show()

    print(day)

# with open('../experimental_figures/extended_model_shorting.npy', 'wb') as f:
#     np.save(f, weight_matrix_high)
#     np.save(f, weight_matrix_mid)
#     np.save(f, weight_matrix_trend)

# axs[1].plot(np.arange(0, 1462, 1), cumulative_returns_covreg_realised, label='Realised covariance', linewidth=3)
# axs[1].plot(np.arange(0, 1462, 1), cumulative_returns_covreg_dcc, label='DCC MGARCH', linewidth=3)
# axs[1].plot(np.arange(0, 1462, 1), cumulative_returns_global_minimum_portfolio, label=textwrap.fill('Global minimum variance', 15),
#             linewidth=3)
# axs[1].plot(np.arange(0, 1462, 1), cumulative_returns_pca_portfolio, label=textwrap.fill('Principle portfolio with 3 components', 20),
#             linewidth=3)

with open('../S&P500_Data/benchmarks.npy', 'wb') as f:
    np.save(f, weight_matrix_realised)
    np.save(f, weight_matrix_dcc)
    np.save(f, weight_matrix_global_minimum)
    np.save(f, weight_matrix_pca)

with open('../experimental_figures/extended_model_shorting.npy', 'rb') as f:
    weight_matrix_high = np.load(f)
    weight_matrix_mid = np.load(f)
    weight_matrix_trend = np.load(f)

# # plot significant weights
# ax = plt.subplot(111)
# plt.gcf().subplots_adjust(bottom=0.18)
# plt.title('IMF CovRegpy Weights', fontsize=12)
# for i in range(11):
#     plt.plot(weight_matrix_direct_imf_covreg_restriction[:end_of_month_vector_cumsum[48], i],
#              label=sector_11_indices.columns[i])
# plt.yticks(fontsize=8)
# plt.ylabel('Weights', fontsize=10)
# plt.xticks([0, 365, 730, 1096, 1460],
#            ['01-01-2018', '01-01-2019', '01-01-2020', '01-01-2021', '31-12-2021'],
#            fontsize=8, rotation=-30)
# plt.xlabel('Days', fontsize=10)
# plt.legend(loc='best', fontsize=6)
# # plt.savefig('../figures/S&P 500 - 11 Sectors/Sector_11_significant_weights.png')
# plt.show()

# plot significant weights
ax = plt.subplot(111)
plt.gcf().subplots_adjust(bottom=0.18)
plt.gcf().set_size_inches(12, 8)
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
plt.savefig('../figures/S&P 500 - 11 Sectors/emd_mdlp_high_weights.png')
plt.show()

# plot significant weights
ax = plt.subplot(111)
plt.gcf().subplots_adjust(bottom=0.18)
plt.gcf().set_size_inches(12, 8)
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
plt.savefig('../figures/S&P 500 - 11 Sectors/emd_mdlp_mid_weights.png')
plt.show()

# plot significant weights
ax = plt.subplot(111)
plt.gcf().subplots_adjust(bottom=0.18)
plt.gcf().set_size_inches(12, 8)
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
plt.savefig('../figures/S&P 500 - 11 Sectors/emd_mdlp_trend_weights.png')
plt.show()

cumulative_returns_mdlp_high = cumulative_return(weight_matrix_high[:end_of_month_vector_cumsum[48]].T,
                                                 sector_11_indices_array[end_of_month_vector_cumsum[12]:].T)
cumulative_returns_mdlp_mid = cumulative_return(weight_matrix_mid[:end_of_month_vector_cumsum[48]].T,
                                                sector_11_indices_array[end_of_month_vector_cumsum[12]:].T)
cumulative_returns_mdlp_trend = cumulative_return(weight_matrix_trend[:end_of_month_vector_cumsum[48]].T,
                                                  sector_11_indices_array[end_of_month_vector_cumsum[12]:].T)

ax = plt.subplot(111)
plt.gcf().subplots_adjust(bottom=0.18)
plt.gcf().set_size_inches(12, 8)
plt.title('Cumulative Returns', fontsize=12)
plt.plot(sp500_proxy, label='S&P 500 Proxy')
plt.plot(sp500_proxy_001, label=textwrap.fill('S&P 500 Proxy with 0.01% charge', 15))
plt.plot(cumulative_returns_mdlp_high, label='High frequency IMFs', color='red')
plt.plot(cumulative_returns_mdlp_mid, label='Mid-frequency IMFs', color='gold')
plt.plot(cumulative_returns_mdlp_trend, label='Low frequency IMFs', color='green')
plt.yticks(fontsize=8)
plt.ylabel('Cumulative Returns', fontsize=10)
plt.xticks([0, 365, 730, 1096, 1461],
           ['31-12-2017', '31-12-2018', '31-12-2019', '31-12-2020', '31-12-2021'],
           fontsize=8, rotation=-30)
plt.xlabel('Days', fontsize=10)
box_0 = ax.get_position()
ax.set_position([box_0.x0 - 0.04, box_0.y0, box_0.width * 1.04, box_0.height])
ax.legend(loc='upper left', fontsize=8)
# plt.savefig('../figures/S&P 500 - 11 Sectors/Sector_11_indices_cumulative_returns.png')
plt.show()

# measure performances - cumulative returns
cumulative_returns_global_minimum_portfolio = \
    cumulative_return(weight_matrix_global_minimum[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T)
cumulative_returns_global_minimum_portfolio_long = \
    cumulative_return(weight_matrix_global_minimum_long[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T)
# cumulative_returns_maximum_sharpe_ratio_portfolio = \
#     cumulative_return(weight_matrix_maximum_sharpe_ratio[:end_of_month_vector_cumsum[48]].T,
#                       sector_11_indices_array[end_of_month_vector_cumsum[12]:].T)
cumulative_returns_maximum_sharpe_ratio_portfolio_restriction = \
    cumulative_return(weight_matrix_maximum_sharpe_ratio_restriction[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T)
cumulative_returns_pca_portfolio = \
    cumulative_return(weight_matrix_pca[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T)
cumulative_returns_covreg_imf_direct_portfolio = \
    cumulative_return(weight_matrix_direct_imf_covreg[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T)
cumulative_returns_covreg_ssa_direct_portfolio = \
    cumulative_return(weight_matrix_direct_ssa_covreg[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T)
cumulative_returns_covreg_imf_direct_portfolio_not_long = \
    cumulative_return(weight_matrix_direct_imf_covreg_restriction[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T)
cumulative_returns_covreg_ssa_direct_portfolio_not_long = \
    cumulative_return(weight_matrix_direct_ssa_covreg_restriction[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T)
cumulative_returns_covreg_dcc = \
    cumulative_return(weight_matrix_dcc[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T)
cumulative_returns_covreg_realised = \
    cumulative_return(weight_matrix_realised[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T)

cumulative_returns_covreg_high = \
    cumulative_return(weight_matrix_direct_imf_covreg_high[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T)
cumulative_returns_covreg_high_restriction = \
    cumulative_return(weight_matrix_direct_imf_covreg_high_restriction[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T)

print(f'Realised variance cumulative returns: {cumulative_returns_covreg_realised[-1]}')
print(f'dcc cumulative returns: {cumulative_returns_covreg_dcc[-1]}')
print(f'S&P 500 Proxy: {sp500_proxy[-1]}')
print(f'global minimum variance {cumulative_returns_global_minimum_portfolio[-1]}')
print(f'PCA {cumulative_returns_pca_portfolio[-1]}')

ax = plt.subplot(111)
plt.gcf().subplots_adjust(bottom=0.18)
plt.gcf().set_size_inches(12, 8)
plt.title('Cumulative Returns', fontsize=12)
plt.plot(sp500_proxy, label='S&P 500 Proxy')
plt.plot(sp500_proxy_001, label=textwrap.fill('S&P 500 Proxy with 0.01% charge', 15))
plt.plot(cumulative_returns_covreg_realised, label='Realised covariance')
plt.plot(cumulative_returns_covreg_dcc, label='DCC MGARCH')
plt.plot(cumulative_returns_global_minimum_portfolio, label=textwrap.fill('Global minimum variance', 15))
plt.plot(cumulative_returns_pca_portfolio, label=textwrap.fill('Principle portfolio with 3 components', 20))
# plt.plot(cumulative_returns_maximum_sharpe_ratio_portfolio_restriction,
#          label=textwrap.fill('Maximum Sharpe ratio', 15))
# plt.plot(cumulative_returns_covreg_high,
#          label=textwrap.fill('High frequency IMFs', 19))
# plt.plot(cumulative_returns_covreg_high_restriction,
#          label=textwrap.fill('High frequency (summation restriction)', 19))
# plt.plot(cumulative_returns_covreg_imf_direct_portfolio,
#          label=textwrap.fill('All frequencies (long restriction)', 19))
# plt.plot(cumulative_returns_covreg_imf_direct_portfolio_not_long,
#          label=textwrap.fill('All frequencies (summation restriction)', 19))
# plt.plot(cumulative_returns_covreg_ssa_direct_portfolio,
#          label=textwrap.fill('Low frequency (long restriction)', 19))
# plt.plot(cumulative_returns_covreg_ssa_direct_portfolio_not_long,
#          label=textwrap.fill('Low frequency (summation restriction)', 19), c='k')
plt.yticks(fontsize=8)
plt.ylabel('Cumulative Returns', fontsize=10)
plt.xticks([0, 365, 730, 1096, 1461],
           ['31-12-2017', '31-12-2018', '31-12-2019', '31-12-2020', '31-12-2021'],
           fontsize=8, rotation=-30)
plt.xlabel('Days', fontsize=10)
box_0 = ax.get_position()
ax.set_position([box_0.x0 - 0.04, box_0.y0, box_0.width * 1.04, box_0.height])
ax.legend(loc='upper left', fontsize=8)
# plt.savefig('../figures/S&P 500 - 11 Sectors/Sector_11_indices_cumulative_returns.png')
plt.show()

fig, axs = plt.subplots(1, 2, sharey=True, figsize=(32, 16))
axs[0].set_title('Cumulative Returns', fontsize=36, pad=20)
axs[0].plot(np.arange(0, 1462, 1), sp500_proxy, label='S&P 500 Proxy', linewidth=3)
axs[0].plot(np.arange(0, 1462, 1), sp500_proxy_001, label=textwrap.fill('S&P 500 Proxy with 0.01% charge', 17), linewidth=3)
axs[0].plot(np.arange(0, 1462, 1), cumulative_returns_mdlp_high, label='High frequency IMFs', color='red', linewidth=3)
axs[0].plot(np.arange(0, 1462, 1), cumulative_returns_mdlp_mid, label='Mid-frequency IMFs', color='gold', linewidth=3)
axs[0].plot(np.arange(0, 1462, 1), cumulative_returns_mdlp_trend, label='Low frequency IMFs', color='green', linewidth=3)
axs[0].set_ylabel('Cumulative Returns', fontsize=30)
axs[0].set_xticks([0, 365, 730, 1096, 1461])
axs[0].set_xticklabels(['31-12-2017', '31-12-2018', '31-12-2019', '31-12-2020', '31-12-2021'],
                       fontsize=20, rotation=-30)
axs[0].set_xlabel('Days', fontsize=30)
axs[0].set_yticks([0.8, 1.0, 1.2, 1.4, 1.6])
axs[0].set_yticklabels([0.8, 1.0, 1.2, 1.4, 1.6], fontsize=20)
box_0 = axs[0].get_position()
axs[0].set_position([box_0.x0 - 0.075, box_0.y0 + 0.02, box_0.width * 1.24, box_0.height])
axs[0].legend(loc='upper left', fontsize=20)
axs[0].set_ylim(0.6, 1.8)
axs[1].set_title('Cumulative Returns', fontsize=36, pad=20)
axs[1].plot(np.arange(0, 1462, 1), sp500_proxy, label='S&P 500 Proxy', linewidth=3)
axs[1].plot(np.arange(0, 1462, 1), sp500_proxy_001, label=textwrap.fill('S&P 500 Proxy with 0.01% charge', 17), linewidth=3)
axs[1].plot(np.arange(0, 1462, 1), cumulative_returns_covreg_realised, label='Realised covariance', linewidth=3)
axs[1].plot(np.arange(0, 1462, 1), cumulative_returns_covreg_dcc, label='DCC MGARCH', linewidth=3)
axs[1].plot(np.arange(0, 1462, 1), cumulative_returns_global_minimum_portfolio, label=textwrap.fill('Global minimum variance', 15),
            linewidth=3)
axs[1].plot(np.arange(0, 1462, 1), cumulative_returns_pca_portfolio, label=textwrap.fill('Principle portfolio with 3 components', 20),
            linewidth=3)
axs[1].set_xticks([0, 365, 730, 1096, 1461],
                  ['31-12-2017', '31-12-2018', '31-12-2019', '31-12-2020', '31-12-2021'],
                  fontsize=20, rotation=-30)
axs[1].set_xlabel('Days', fontsize=30)
box_0 = axs[1].get_position()
axs[1].set_position([box_0.x0 - 0.02, box_0.y0 + 0.02, box_0.width * 1.24, box_0.height])
axs[1].legend(loc='upper left', fontsize=20)
axs[0].set_ylim(0.6, 1.8)
plt.savefig('../figures/S&P 500 - 11 Sectors/Sector_11_indices_cumulative_returns_joint_plot.pdf')
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
mean_returns_sp500_001 = \
    mean_return(np.ones_like(sp500_returns.reshape(1, -1)), sp500_returns_001.reshape(1, -1), window)

ax = plt.subplot(111)
plt.gcf().subplots_adjust(bottom=0.18)
plt.gcf().set_size_inches(12, 8)
plt.title('Mean Returns', fontsize=12)
plt.plot(mean_returns_sp500, label='S&P 500 Proxy')
plt.plot(mean_returns_sp500_001, label=textwrap.fill('S&P 500 Proxy with 0.01% charge', 15))
plt.plot(mean_returns_mdlp_high, label='High frequency IMFs', color='red')
plt.plot(mean_returns_mdlp_mid, label='Mid-frequency IMFs', color='gold')
plt.plot(mean_returns_mdlp_trend, label='Low frequency IMFs', color='green')
plt.yticks(fontsize=8)
plt.ylabel('Mean Daily Returns', fontsize=10)
plt.xticks([0, 334, 699, 1065, 1430],
           ['30-01-2018', '31-12-2018', '31-12-2019', '31-12-2020', '31-12-2021'],
           fontsize=8, rotation=-30)
plt.xlabel('Days', fontsize=10)
box_0 = ax.get_position()
ax.set_position([box_0.x0 - 0.04, box_0.y0, box_0.width * 1.04, box_0.height])
ax.legend(loc='lower left', fontsize=8)
# plt.savefig('../figures/S&P 500 - 11 Sectors/Sector_11_indices_mean_returns.png')
plt.show()

mean_returns_global_minimum_portfolio = \
    mean_return(weight_matrix_global_minimum[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
mean_returns_global_minimum_portfolio_long = \
    mean_return(weight_matrix_global_minimum_long[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
mean_returns_maximum_sharpe_ratio_portfolio = \
    mean_return(weight_matrix_maximum_sharpe_ratio_restriction[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
mean_returns_pca_portfolio = \
    mean_return(weight_matrix_pca[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
mean_returns_covreg_imf_direct_portfolio = \
    mean_return(weight_matrix_direct_imf_covreg[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
mean_returns_covreg_ssa_direct_portfolio = \
    mean_return(weight_matrix_direct_ssa_covreg[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
mean_returns_covreg_imf_direct_portfolio_not_long = \
    mean_return(weight_matrix_direct_imf_covreg_restriction[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
mean_returns_covreg_ssa_direct_portfolio_not_long = \
    mean_return(weight_matrix_direct_ssa_covreg_restriction[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)

mean_returns_realised_covariance = \
    mean_return(weight_matrix_realised[:end_of_month_vector_cumsum[48]].T,
                sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
mean_returns_mgarch = \
    mean_return(weight_matrix_dcc[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)

mean_returns_covreg_imf_direct_high = \
    mean_return(weight_matrix_direct_imf_covreg_high[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
mean_returns_covreg_imf_direct_high_not_long = \
    mean_return(weight_matrix_direct_imf_covreg_high_restriction[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)

ax = plt.subplot(111)
plt.gcf().subplots_adjust(bottom=0.18)
plt.gcf().set_size_inches(12, 8)
plt.title('Mean Returns', fontsize=12)
plt.plot(mean_returns_sp500, label='S&P 500 Proxy')
plt.plot(mean_returns_sp500_001, label=textwrap.fill('S&P 500 Proxy with 0.01% charge', 15))
plt.plot(mean_returns_realised_covariance, label='Realised covariance')
plt.plot(mean_returns_mgarch, label='DCC MGARCH')
plt.plot(mean_returns_global_minimum_portfolio, label=textwrap.fill('Global minimum variance', 15))
plt.plot(mean_returns_pca_portfolio, label=textwrap.fill('Principle portfolio with 3 components', 20))
# plt.plot(mean_returns_maximum_sharpe_ratio_portfolio,  label=textwrap.fill('Maximum Sharpe ratio', 15))
# plt.plot(mean_returns_covreg_imf_direct_high, label=textwrap.fill('High frequency (long restriction)', 19))
# plt.plot(mean_returns_covreg_imf_direct_high_not_long, label=textwrap.fill('High frequency (summation restriction)', 19))
# plt.plot(mean_returns_covreg_imf_direct_portfolio, label=textwrap.fill('All frequencies (long restriction)', 19))
# plt.plot(mean_returns_covreg_imf_direct_portfolio_not_long, label=textwrap.fill('All frequencies (summation restriction)', 19))
# plt.plot(mean_returns_covreg_ssa_direct_portfolio, label=textwrap.fill('Low frequency (long restriction)', 19))
# plt.plot(mean_returns_covreg_ssa_direct_portfolio_not_long, label=textwrap.fill('Low frequency (summation restriction)', 19),
#          c='k')
plt.yticks(fontsize=8)
plt.ylabel('Mean Daily Returns', fontsize=10)
plt.xticks([0, 334, 699, 1065, 1430],
           ['30-01-2018', '31-12-2018', '31-12-2019', '31-12-2020', '31-12-2021'],
           fontsize=8, rotation=-30)
plt.xlabel('Days', fontsize=10)
plt.ylim(-0.018, 0.011)
box_0 = ax.get_position()
ax.set_position([box_0.x0 - 0.04, box_0.y0, box_0.width * 1.04, box_0.height])
ax.legend(loc='lower left', fontsize=8)
# plt.savefig('../figures/S&P 500 - 11 Sectors/Sector_11_indices_mean_returns.png')
plt.show()

fig, axs = plt.subplots(1, 2, sharey=True, figsize=(32, 16))
axs[0].set_title('Mean Returns - 30 Day Rolling Window', fontsize=36, pad=20)
axs[0].plot(np.arange(30, 1462, 1), mean_returns_sp500, label='S&P 500 Proxy', linewidth=3)
axs[0].plot(np.arange(30, 1462, 1), mean_returns_sp500_001, label=textwrap.fill('S&P 500 Proxy with 0.01% charge', 17), linewidth=3)
axs[0].plot(np.arange(30, 1462, 1), mean_returns_mdlp_high, label='High frequency IMFs', color='red', linewidth=3)
axs[0].plot(np.arange(30, 1462, 1), mean_returns_mdlp_mid, label='Mid-frequency IMFs', color='gold', linewidth=3)
axs[0].plot(np.arange(30, 1462, 1), mean_returns_mdlp_trend, label='Low frequency IMFs', color='green', linewidth=3)
axs[0].set_xticks([30, 365, 730, 1096, 1461])
axs[0].set_xticklabels(['30-01-2018', '31-12-2018', '31-12-2019', '31-12-2020', '31-12-2021'],
                       fontsize=20, rotation=-30)
axs[0].set_xlabel('Days', fontsize=30)
axs[0].set_yticks([-0.015, -0.010, -0.005, 0, 0.005, 0.010])
axs[0].set_yticklabels(['-0.015', '-0.010', '-0.005', '0.000', '0.005', '0.010'], fontsize=20)
box_0 = axs[0].get_position()
axs[0].set_position([box_0.x0 - 0.075, box_0.y0 + 0.02, box_0.width * 1.24, box_0.height])
axs[0].legend(loc='upper left', fontsize=20)
axs[1].set_title('Mean Returns - 30 Day Rolling Window', fontsize=36, pad=20)
axs[1].plot(np.arange(30, 1462, 1), mean_returns_sp500, label='S&P 500 Proxy', linewidth=3)
axs[1].plot(np.arange(30, 1462, 1), mean_returns_sp500_001, label=textwrap.fill('S&P 500 Proxy with 0.01% charge', 17), linewidth=3)
axs[1].plot(np.arange(30, 1462, 1), mean_returns_realised_covariance, label='Realised covariance', linewidth=3)
axs[1].plot(np.arange(30, 1462, 1), mean_returns_mgarch, label='DCC MGARCH', linewidth=3)
axs[1].plot(np.arange(30, 1462, 1), mean_returns_global_minimum_portfolio,
            label=textwrap.fill('Global minimum variance', 30), linewidth=3)
axs[1].plot(np.arange(30, 1462, 1), mean_returns_pca_portfolio,
            label=textwrap.fill('Principle portfolio with 3 components', 20),
            linewidth=3)
axs[1].set_xticks([30, 365, 730, 1096, 1461])
axs[1].set_xticklabels(['30-01-2018', '31-12-2018', '31-12-2019', '31-12-2020', '31-12-2021'],
                       fontsize=20, rotation=-30)
axs[1].set_xlabel('Days', fontsize=30)
box_0 = axs[1].get_position()
axs[1].set_position([box_0.x0 - 0.02, box_0.y0 + 0.02, box_0.width * 1.24, box_0.height])
axs[1].legend(loc='lower left', fontsize=20)
axs[0].set_ylim(-0.016, 0.011)
plt.savefig('../figures/S&P 500 - 11 Sectors/Sector_11_indices_mean_returns_joint_plot.pdf')
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
variance_returns_sp500_001 = \
    variance_return(np.ones_like(sp500_returns.reshape(1, -1)), sp500_returns_001.reshape(1, -1), window)

ax = plt.subplot(111)
plt.gcf().subplots_adjust(bottom=0.18)
plt.gcf().set_size_inches(12, 8)
plt.title('Variance Returns', fontsize=12)
plt.plot(variance_returns_sp500, label='S&P 500 Proxy')
plt.plot(variance_returns_sp500_001, label=textwrap.fill('S&P 500 Proxy with 0.01% charge', 15))
plt.plot(variance_returns_mdlp_high, label='High frequency IMFs', color='red')
plt.plot(variance_returns_mdlp_mid, label='Mid-frequency IMFs', color='gold')
plt.plot(variance_returns_mdlp_trend, label='Low frequency IMFs', color='green')
plt.yticks(fontsize=8)
plt.ylabel('Variance Daily Returns', fontsize=10)
plt.xticks([0, 334, 699, 1065, 1430],
           ['30-01-2018', '31-12-2018', '31-12-2019', '31-12-2020', '31-12-2021'],
           fontsize=8, rotation=-30)
plt.xlabel('Days', fontsize=10)
box_0 = ax.get_position()
ax.set_position([box_0.x0 - 0.04, box_0.y0, box_0.width * 1.04, box_0.height])
ax.legend(loc='upper left', fontsize=8)
# plt.savefig('../figures/S&P 500 - 11 Sectors/Sector_11_indices_variance_returns.png')
plt.show()

# measure performances - variance returns
window = 30
variance_returns_global_minimum_portfolio = \
    variance_return(weight_matrix_global_minimum[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
variance_returns_global_minimum_portfolio_long = \
    variance_return(weight_matrix_global_minimum_long[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
variance_returns_maximum_sharpe_ratio_portfolio = \
    variance_return(weight_matrix_maximum_sharpe_ratio_restriction[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
variance_returns_pca_portfolio = \
    variance_return(weight_matrix_pca[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
variance_returns_covreg_imf_direct_portfolio = \
    variance_return(weight_matrix_direct_imf_covreg[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
variance_returns_covreg_ssa_direct_portfolio = \
    variance_return(weight_matrix_direct_ssa_covreg[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
variance_returns_covreg_imf_direct_portfolio_not_long = \
    variance_return(weight_matrix_direct_imf_covreg_restriction[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
variance_returns_covreg_ssa_direct_portfolio_not_long = \
    variance_return(weight_matrix_direct_ssa_covreg_restriction[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)

variance_returns_realised_covariance = \
    variance_return(weight_matrix_realised[:end_of_month_vector_cumsum[48]].T,
                    sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
variance_returns_mgarch = \
    variance_return(weight_matrix_dcc[:end_of_month_vector_cumsum[48]].T,
                    sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
variance_returns_sp500 = \
    variance_return(np.ones_like(sp500_returns.reshape(1, -1)), sp500_returns.reshape(1, -1), window)

variance_returns_covreg_imf_direct_high = \
    variance_return(weight_matrix_direct_imf_covreg_high[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
variance_returns_covreg_imf_direct_high_not_long = \
    variance_return(weight_matrix_direct_imf_covreg_high_restriction[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)

ax = plt.subplot(111)
plt.gcf().subplots_adjust(bottom=0.18)
plt.gcf().set_size_inches(12, 8)
plt.title('Variance of Returns', fontsize=12)
plt.plot(variance_returns_sp500, label='S&P 500 Proxy')
plt.plot(variance_returns_sp500_001, label=textwrap.fill('S&P 500 Proxy with 0.01% charge', 15))
plt.plot(variance_returns_realised_covariance, label='Realised covariance')
plt.plot(variance_returns_mgarch, label='DCC MGARCH')
plt.plot(variance_returns_global_minimum_portfolio, label=textwrap.fill('Global minimum variance', 15))
plt.plot(variance_returns_pca_portfolio, label=textwrap.fill('Principle portfolio with 3 components', 20))
# plt.plot(variance_returns_covreg_imf_direct_high, label=textwrap.fill('High frequency (long restriction)', 19))
# plt.plot(variance_returns_covreg_imf_direct_high_not_long, label=textwrap.fill('High frequency (summation restriction)', 19))
# plt.plot(variance_returns_covreg_imf_direct_portfolio, label=textwrap.fill('All frequencies (long restriction)', 19))
# plt.plot(variance_returns_covreg_imf_direct_portfolio_not_long, label=textwrap.fill('All frequencies (summation restriction)', 19))
# plt.plot(variance_returns_covreg_ssa_direct_portfolio, label=textwrap.fill('Low frequency (long restriction)', 19))
# plt.plot(variance_returns_covreg_ssa_direct_portfolio_not_long, label=textwrap.fill('Low frequency (summation restriction)', 19),
#          c='k')
plt.yticks(fontsize=8)
plt.ylabel('Variance', fontsize=10)
plt.xticks([0, 334, 699, 1065, 1430],
           ['30-01-2018', '31-12-2018', '31-12-2019', '31-12-2020', '31-12-2021'],
           fontsize=8, rotation=-30)
plt.xlabel('Days', fontsize=10)
box_0 = ax.get_position()
ax.set_position([box_0.x0 - 0.04, box_0.y0, box_0.width * 1.04, box_0.height])
ax.legend(loc='upper left', fontsize=8)
# plt.savefig('../figures/S&P 500 - 11 Sectors/Sector_11_indices_variance_returns.png')
plt.show()

fig, axs = plt.subplots(1, 2, sharey=True, figsize=(32, 16))
axs[0].set_title('Variance Returns - 30 Day Rolling Window', fontsize=36, pad=20)
axs[0].plot(np.arange(30, 1462, 1), variance_returns_sp500, label='S&P 500 Proxy', linewidth=3)
axs[0].plot(np.arange(30, 1462, 1), variance_returns_sp500_001, label=textwrap.fill('S&P 500 Proxy with 0.01% charge', 17), linewidth=3)
axs[0].plot(np.arange(30, 1462, 1), variance_returns_mdlp_high, label='High frequency IMFs', color='red', linewidth=3)
axs[0].plot(np.arange(30, 1462, 1), variance_returns_mdlp_mid, label='Mid-frequency IMFs', color='gold', linewidth=3)
axs[0].plot(np.arange(30, 1462, 1), variance_returns_mdlp_trend, label='Low frequency IMFs', color='green', linewidth=3)
axs[0].set_xticks([30, 365, 730, 1096, 1461])
axs[0].set_xticklabels(['30-01-2018', '31-12-2018', '31-12-2019', '31-12-2020', '31-12-2021'],
                       fontsize=20, rotation=-30)
axs[0].set_xlabel('Days', fontsize=30)
axs[0].set_yticks([0, 0.00025, 0.0005, 0.00075, 0.00100,
                   0.00125, 0.0015, 0.00175, 0.00200])
axs[0].set_yticklabels(['0.00000', '0.00025', '0.00050', '0.00075', '0.00100',
                        '0.00125', '0.00150', '0.00175', '0.00200'], fontsize=20)
box_0 = axs[0].get_position()
axs[0].set_position([box_0.x0 - 0.075, box_0.y0 + 0.02, box_0.width * 1.24, box_0.height])
axs[0].legend(loc='upper left', fontsize=20)
axs[0].set_ylim(-0.0002, 0.0022)
axs[1].set_title('Variance Returns - 30 Day Rolling Window', fontsize=36, pad=20)
axs[1].plot(np.arange(30, 1462, 1), variance_returns_sp500, label='S&P 500 Proxy', linewidth=3)
axs[1].plot(np.arange(30, 1462, 1), variance_returns_sp500_001,
            label=textwrap.fill('S&P 500 Proxy with 0.01% charge', 17), linewidth=3)
axs[1].plot(np.arange(30, 1462, 1), variance_returns_realised_covariance, label='Realised covariance', linewidth=3)
axs[1].plot(np.arange(30, 1462, 1), variance_returns_mgarch, label='DCC MGARCH', linewidth=3)
axs[1].plot(np.arange(30, 1462, 1), variance_returns_global_minimum_portfolio,
            label=textwrap.fill('Global minimum variance', 15), linewidth=3)
axs[1].plot(np.arange(30, 1462, 1), variance_returns_pca_portfolio,
            label=textwrap.fill('Principle portfolio with 3 components', 20), linewidth=3)
axs[1].set_xticks([30, 365, 730, 1096, 1461])
axs[1].set_xticklabels(['30-01-2018', '31-12-2018', '31-12-2019', '31-12-2020', '31-12-2021'],
                       fontsize=20, rotation=-30)
axs[1].set_xlabel('Days', fontsize=30)
box_0 = axs[1].get_position()
axs[1].set_position([box_0.x0 - 0.02, box_0.y0 + 0.02, box_0.width * 1.24, box_0.height])
axs[1].legend(loc='upper left', fontsize=20)
plt.savefig('../figures/S&P 500 - 11 Sectors/Sector_11_indices_variance_returns_joint_plot.pdf')
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
value_at_risk_returns_sp500_001 = \
    value_at_risk_return(np.ones_like(sp500_returns.reshape(1, -1)), sp500_returns_001.reshape(1, -1), window)

ax = plt.subplot(111)
plt.gcf().subplots_adjust(bottom=0.18)
plt.gcf().set_size_inches(12, 8)
plt.title('Value-at-Risk Returns', fontsize=12)
plt.plot(value_at_risk_returns_sp500, label='S&P 500 Proxy')
plt.plot(value_at_risk_returns_sp500_001, label=textwrap.fill('S&P 500 Proxy with 0.01% charge', 15))
plt.plot(value_at_risk_returns_mdlp_high, label='High frequency IMFs', color='red')
plt.plot(value_at_risk_returns_mdlp_mid, label='Mid-frequency IMFs', color='gold')
plt.plot(value_at_risk_returns_mdlp_trend, label='Low frequency IMFs', color='green')
plt.yticks(fontsize=8)
plt.ylabel('Value-at-Risk Daily Returns', fontsize=10)
plt.xticks([0, 334, 699, 1065, 1430],
           ['30-01-2018', '31-12-2018', '31-12-2019', '31-12-2020', '31-12-2021'],
           fontsize=8, rotation=-30)
plt.xlabel('Days', fontsize=10)
box_0 = ax.get_position()
ax.set_position([box_0.x0 - 0.04, box_0.y0, box_0.width * 1.04, box_0.height])
ax.legend(loc='lower left', fontsize=8)
# plt.savefig('../figures/S&P 500 - 11 Sectors/Sector_11_indices_value_at_risk_returns.png')
plt.show()

# measure performances - value at risk returns
window = 30
value_at_risk_returns_global_minimum_portfolio = \
    value_at_risk_return(weight_matrix_global_minimum[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
value_at_risk_returns_global_minimum_portfolio_long = \
    value_at_risk_return(weight_matrix_global_minimum_long[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
value_at_risk_returns_maximum_sharpe_ratio_portfolio = \
    value_at_risk_return(weight_matrix_maximum_sharpe_ratio_restriction[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
value_at_risk_returns_pca_portfolio = \
    value_at_risk_return(weight_matrix_pca[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
value_at_risk_returns_covreg_imf_direct_portfolio = \
    value_at_risk_return(weight_matrix_direct_imf_covreg[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
value_at_risk_returns_covreg_ssa_direct_portfolio = \
    value_at_risk_return(weight_matrix_direct_ssa_covreg[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
value_at_risk_returns_covreg_imf_direct_portfolio_not_long = \
    value_at_risk_return(weight_matrix_direct_imf_covreg_restriction[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
value_at_risk_returns_covreg_ssa_direct_portfolio_not_long = \
    value_at_risk_return(weight_matrix_direct_ssa_covreg_restriction[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)

value_at_risk_returns_realised_covariance = \
    value_at_risk_return(weight_matrix_realised[:end_of_month_vector_cumsum[48]].T,
                         sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
value_at_risk_returns_mgarch = \
    value_at_risk_return(weight_matrix_dcc[:end_of_month_vector_cumsum[48]].T,
                         sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
value_at_risk_returns_sp500 = \
    value_at_risk_return(np.ones_like(sp500_returns.reshape(1, -1)), sp500_returns.reshape(1, -1), window)

value_at_risk_returns_covreg_imf_direct_high = \
    value_at_risk_return(weight_matrix_direct_imf_covreg_high[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
value_at_risk_returns_covreg_imf_direct_high_not_long = \
    value_at_risk_return(weight_matrix_direct_imf_covreg_high_restriction[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)

ax = plt.subplot(111)
plt.gcf().subplots_adjust(bottom=0.18)
plt.gcf().set_size_inches(12, 8)
plt.title('Value-at-Risk Returns', fontsize=12)
plt.plot(value_at_risk_returns_sp500, label='S&P 500 Proxy')
plt.plot(value_at_risk_returns_sp500_001, label=textwrap.fill('S&P 500 Proxy with 0.01% charge', 15))
plt.plot(value_at_risk_returns_realised_covariance, label='Realised covariance')
plt.plot(value_at_risk_returns_mgarch, label='DCC MGARCH')
plt.plot(value_at_risk_returns_global_minimum_portfolio, label=textwrap.fill('Global minimum variance', 15))
plt.plot(value_at_risk_returns_pca_portfolio, label=textwrap.fill('Principle portfolio with 3 components', 20))
# plt.plot(value_at_risk_returns_covreg_imf_direct_high, label=textwrap.fill('High frequency (long restriction)', 19))
# plt.plot(value_at_risk_returns_covreg_imf_direct_high_not_long, label=textwrap.fill('High frequency (summation restriction)', 19))
# plt.plot(value_at_risk_returns_covreg_imf_direct_portfolio, label=textwrap.fill('All frequencies (long restriction)', 19))
# plt.plot(value_at_risk_returns_covreg_imf_direct_portfolio_not_long, label=textwrap.fill('All frequencies (summation restriction)', 19))
# plt.plot(value_at_risk_returns_covreg_ssa_direct_portfolio, label=textwrap.fill('Low frequency (long restriction)', 19))
# plt.plot(value_at_risk_returns_covreg_ssa_direct_portfolio_not_long,
#          label=textwrap.fill('Low frequency (summation restriction)', 19), c='k')
plt.yticks(fontsize=8)
plt.ylabel('Mean Value-at-Risk', fontsize=10)
plt.xticks([0, 334, 699, 1065, 1430],
           ['30-01-2018', '31-12-2018', '31-12-2019', '31-12-2020', '31-12-2021'],
           fontsize=8, rotation=-30)
plt.xlabel('Days', fontsize=10)
box_0 = ax.get_position()
ax.set_position([box_0.x0 - 0.04, box_0.y0, box_0.width * 1.04, box_0.height])
ax.legend(loc='lower left', fontsize=8)
# plt.savefig('../figures/S&P 500 - 11 Sectors/Sector_11_indices_value_at_risk_returns.png')
plt.show()

fig, axs = plt.subplots(1, 2, sharey=True, figsize=(32, 16))
axs[0].set_title('Value-at-Risk Returns - 30 Day Rolling Window', fontsize=36, pad=20)
axs[0].plot(np.arange(30, 1462, 1), value_at_risk_returns_sp500, label='S&P 500 Proxy', linewidth=3)
axs[0].plot(np.arange(30, 1462, 1), value_at_risk_returns_sp500_001, label=textwrap.fill('S&P 500 Proxy with 0.01% charge', 17), linewidth=3)
axs[0].plot(np.arange(30, 1462, 1), value_at_risk_returns_mdlp_high, label='High frequency IMFs', color='red', linewidth=3)
axs[0].plot(np.arange(30, 1462, 1), value_at_risk_returns_mdlp_mid, label='Mid-frequency IMFs', color='gold', linewidth=3)
axs[0].plot(np.arange(30, 1462, 1), value_at_risk_returns_mdlp_trend, label='Low frequency IMFs', color='green', linewidth=3)
axs[0].set_xticks([30, 365, 730, 1096, 1461])
axs[0].set_xticklabels(['30-01-2018', '31-12-2018', '31-12-2019', '31-12-2020', '31-12-2021'],
                       fontsize=20, rotation=-30)
axs[0].set_xlabel('Days', fontsize=30)
axs[0].set_yticks([0.00, -0.01, -0.02, -0.03, -0.04, -0.05, -0.06, -0.07][::-1])
axs[0].set_yticklabels(['0.00', '-0.01', '-0.02', '-0.03', '-0.04', '-0.05', '-0.06', '-0.07'][::-1], fontsize=20)
box_0 = axs[0].get_position()
axs[0].set_position([box_0.x0 - 0.075, box_0.y0 + 0.02, box_0.width * 1.24, box_0.height])
axs[0].legend(loc='lower left', fontsize=20)
axs[0].set_ylim(-0.075, 0.005)
axs[1].set_title('Value-at-Risk Returns - 30 Day Rolling Window', fontsize=36, pad=20)
axs[1].plot(np.arange(30, 1462, 1), value_at_risk_returns_sp500, label='S&P 500 Proxy', linewidth=3)
axs[1].plot(np.arange(30, 1462, 1), value_at_risk_returns_sp500_001,
            label=textwrap.fill('S&P 500 Proxy with 0.01% charge', 17), linewidth=3)
axs[1].plot(np.arange(30, 1462, 1), value_at_risk_returns_realised_covariance, label='Realised covariance', linewidth=3)
axs[1].plot(np.arange(30, 1462, 1), value_at_risk_returns_mgarch, label='DCC MGARCH', linewidth=3)
axs[1].plot(np.arange(30, 1462, 1), value_at_risk_returns_global_minimum_portfolio,
            label=textwrap.fill('Global minimum variance', 15), linewidth=3)
axs[1].plot(np.arange(30, 1462, 1), value_at_risk_returns_pca_portfolio,
            label=textwrap.fill('Principle portfolio with 3 components', 20),
            linewidth=3)
axs[1].set_xticks([30, 365, 730, 1096, 1461])
axs[1].set_xticklabels(['30-01-2018', '31-12-2018', '31-12-2019', '31-12-2020', '31-12-2021'],
                       fontsize=20, rotation=-30)
axs[1].set_xlabel('Days', fontsize=30)
box_0 = axs[1].get_position()
axs[1].set_position([box_0.x0 - 0.02, box_0.y0 + 0.02, box_0.width * 1.24, box_0.height])
axs[1].legend(loc='lower left', fontsize=20)
plt.savefig('../figures/S&P 500 - 11 Sectors/Sector_11_indices_value_at_risk_returns_joint_plot.pdf')
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
max_draw_down_returns_sp500_001 = \
    max_draw_down_return(np.ones_like(sp500_returns.reshape(1, -1)), sp500_returns_001.reshape(1, -1), window)

ax = plt.subplot(111)
plt.gcf().subplots_adjust(bottom=0.18)
plt.gcf().set_size_inches(12, 8)
plt.title('Maximum Draw Down Returns', fontsize=12)
plt.plot(max_draw_down_returns_sp500, label='S&P 500 Proxy')
plt.plot(max_draw_down_returns_sp500_001, label=textwrap.fill('S&P 500 Proxy with 0.01% charge', 15))
plt.plot(max_draw_down_returns_mdlp_high, label='High frequency IMFs', color='red')
plt.plot(max_draw_down_returns_mdlp_mid, label='Mid-frequency IMFs', color='gold')
plt.plot(max_draw_down_returns_mdlp_trend, label='Low frequency IMFs', color='green')
plt.yticks(fontsize=8)
plt.ylabel('Maximum Draw Down Daily Returns', fontsize=10)
plt.xticks([0, 334, 699, 1065, 1430],
           ['30-01-2018', '31-12-2018', '31-12-2019', '31-12-2020', '31-12-2021'],
           fontsize=8, rotation=-30)
plt.xlabel('Days', fontsize=10)
box_0 = ax.get_position()
ax.set_position([box_0.x0 - 0.04, box_0.y0, box_0.width * 1.04, box_0.height])
ax.legend(loc='lower left', fontsize=8)
# plt.savefig('../figures/S&P 500 - 11 Sectors/Sector_11_indices_max_draw_down_returns.png')
plt.show()

# measure performances - Max draw down returns
window = 30
max_draw_down_returns_global_minimum_portfolio = \
    max_draw_down_return(weight_matrix_global_minimum[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
max_draw_down_returns_global_minimum_portfolio_long = \
    max_draw_down_return(weight_matrix_global_minimum_long[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
max_draw_down_returns_maximum_sharpe_ratio_portfolio = \
    max_draw_down_return(weight_matrix_maximum_sharpe_ratio_restriction[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
max_draw_down_returns_pca_portfolio = \
    max_draw_down_return(weight_matrix_pca[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
max_draw_down_returns_covreg_imf_direct_portfolio = \
    max_draw_down_return(weight_matrix_direct_imf_covreg[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
max_draw_down_returns_covreg_ssa_direct_portfolio = \
    max_draw_down_return(weight_matrix_direct_ssa_covreg[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
max_draw_down_returns_covreg_imf_direct_portfolio_not_long = \
    max_draw_down_return(weight_matrix_direct_imf_covreg_restriction[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
max_draw_down_returns_covreg_ssa_direct_portfolio_not_long = \
    max_draw_down_return(weight_matrix_direct_ssa_covreg_restriction[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)

max_draw_down_returns_realised_covariance = \
    max_draw_down_return(weight_matrix_realised[:end_of_month_vector_cumsum[48]].T,
                         sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
max_draw_down_returns_mgarch = \
    max_draw_down_return(weight_matrix_dcc[:end_of_month_vector_cumsum[48]].T,
                         sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
max_draw_down_returns_sp500 = \
    max_draw_down_return(np.ones_like(sp500_returns.reshape(1, -1)), sp500_returns.reshape(1, -1), window)

max_draw_down_returns_covreg_imf_direct_high = \
    max_draw_down_return(weight_matrix_direct_imf_covreg_high[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
max_draw_down_returns_covreg_imf_direct_high_not_long = \
    max_draw_down_return(weight_matrix_direct_imf_covreg_high_restriction[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)

ax = plt.subplot(111)
plt.gcf().subplots_adjust(bottom=0.18)
plt.gcf().set_size_inches(12, 8)
plt.title('Maximum Draw Down Returns', fontsize=12)
plt.plot(max_draw_down_returns_sp500, label='S&P 500 Proxy')
plt.plot(max_draw_down_returns_sp500_001, label=textwrap.fill('S&P 500 Proxy with 0.01% charge', 15))
plt.plot(max_draw_down_returns_realised_covariance, label='Realised Covariance')
plt.plot(max_draw_down_returns_mgarch, label='DCC MGARCH')
plt.plot(max_draw_down_returns_global_minimum_portfolio, label=textwrap.fill('Global minimum variance', 15))
plt.plot(max_draw_down_returns_pca_portfolio, label=textwrap.fill('Principle portfolio with 3 components', 20))
# plt.plot(max_draw_down_returns_covreg_imf_direct_high, label=textwrap.fill('High frequency (long restriction)', 19))
# plt.plot(max_draw_down_returns_covreg_imf_direct_high_not_long, label=textwrap.fill('High frequency (summation restriction)', 19))
# plt.plot(max_draw_down_returns_covreg_imf_direct_portfolio, label=textwrap.fill('All frequencies (long restriction)', 19))
# plt.plot(max_draw_down_returns_covreg_imf_direct_portfolio_not_long, label=textwrap.fill('All frequencies (summation restriction)', 19))
# plt.plot(max_draw_down_returns_covreg_ssa_direct_portfolio, label=textwrap.fill('Low frequency (long restriction)', 19))
# plt.plot(max_draw_down_returns_covreg_ssa_direct_portfolio_not_long,
#          label=textwrap.fill('Low frequency (summation restriction)', 19), c='k')
plt.yticks(fontsize=8)
plt.ylabel('Max Draw Down', fontsize=10)
plt.xticks([0, 334, 699, 1065, 1430],
           ['30-01-2018', '31-12-2018', '31-12-2019', '31-12-2020', '31-12-2021'],
           fontsize=8, rotation=-30)
plt.xlabel('Days', fontsize=10)
box_0 = ax.get_position()
ax.set_position([box_0.x0 - 0.04, box_0.y0, box_0.width * 1.04, box_0.height])
ax.legend(loc='lower left', fontsize=8)
# plt.savefig('../figures/S&P 500 - 11 Sectors/Sector_11_indices_max_draw_down_returns.png')
plt.show()

fig, axs = plt.subplots(1, 2, sharey=True, figsize=(32, 16))
axs[0].set_title('Maximum Drawdown Returns - 30 Day Rolling Window', fontsize=36, pad=20)
axs[0].plot(np.arange(30, 1462, 1), max_draw_down_returns_sp500, label='S&P 500 Proxy', linewidth=3)
axs[0].plot(np.arange(30, 1462, 1), max_draw_down_returns_sp500_001, label=textwrap.fill('S&P 500 Proxy with 0.01% charge', 17), linewidth=3)
axs[0].plot(np.arange(30, 1462, 1), max_draw_down_returns_mdlp_high, label='High frequency IMFs', color='red', linewidth=3)
axs[0].plot(np.arange(30, 1462, 1), max_draw_down_returns_mdlp_mid, label='Mid-frequency IMFs', color='gold', linewidth=3)
axs[0].plot(np.arange(30, 1462, 1), max_draw_down_returns_mdlp_trend, label='Low frequency IMFs', color='green', linewidth=3)
axs[0].set_xticks([30, 365, 730, 1096, 1461])
axs[0].set_xticklabels(['30-01-2018', '31-12-2018', '31-12-2019', '31-12-2020', '31-12-2021'],
                       fontsize=20, rotation=-30)
axs[0].set_xlabel('Days', fontsize=30)
axs[0].set_yticks([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0][::-1])
axs[0].set_yticklabels([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0][::-1], fontsize=20)
box_0 = axs[0].get_position()
axs[0].set_position([box_0.x0 - 0.075, box_0.y0 + 0.02, box_0.width * 1.24, box_0.height])
axs[0].legend(loc='lower left', fontsize=20)
axs[0].set_ylim(-8.5, -0.5)
axs[1].set_title('Maximum Drawdown Returns - 30 Day Rolling Window', fontsize=36, pad=20)
axs[1].plot(np.arange(30, 1462, 1), max_draw_down_returns_sp500, label='S&P 500 Proxy', linewidth=3)
axs[1].plot(np.arange(30, 1462, 1), max_draw_down_returns_sp500_001,
            label=textwrap.fill('S&P 500 Proxy with 0.01% charge', 17), linewidth=3)
axs[1].plot(np.arange(30, 1462, 1), max_draw_down_returns_realised_covariance, label='Realised covariance', linewidth=3)
axs[1].plot(np.arange(30, 1462, 1), max_draw_down_returns_mgarch, label='DCC MGARCH', linewidth=3)
axs[1].plot(np.arange(30, 1462, 1), max_draw_down_returns_global_minimum_portfolio,
            label=textwrap.fill('Global minimum variance', 30), linewidth=3)
axs[1].plot(np.arange(30, 1462, 1), max_draw_down_returns_pca_portfolio,
            label=textwrap.fill('Principle portfolio with 3 components', 40),
            linewidth=3)
axs[1].set_xticks([30, 365, 730, 1096, 1461])
axs[1].set_xticklabels(['30-01-2018', '31-12-2018', '31-12-2019', '31-12-2020', '31-12-2021'],
                       fontsize=20, rotation=-30)
axs[1].set_xlabel('Days', fontsize=30)
box_0 = axs[1].get_position()
axs[1].set_position([box_0.x0 - 0.02, box_0.y0 + 0.02, box_0.width * 1.24, box_0.height])
axs[1].legend(loc='lower left', fontsize=20)
plt.savefig('../figures/S&P 500 - 11 Sectors/Sector_11_indices_max_draw_down_returns_joint_plot.pdf')
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
omega_ratio_returns_sp500_001 = \
    omega_ratio_return(np.ones_like(sp500_returns.reshape(1, -1)), sp500_returns_001.reshape(1, -1), window)

ax = plt.subplot(111)
plt.gcf().subplots_adjust(bottom=0.18)
plt.gcf().set_size_inches(12, 8)
plt.title('Omega Ratio Returns', fontsize=12)
plt.plot(omega_ratio_returns_sp500, label='S&P 500 Proxy')
plt.plot(omega_ratio_returns_sp500_001, label=textwrap.fill('S&P 500 Proxy with 0.01% charge', 15))
plt.plot(omega_ratio_returns_mdlp_high, label='High frequency IMFs', color='red')
plt.plot(omega_ratio_returns_mdlp_mid, label='Mid-frequency IMFs', color='gold')
plt.plot(omega_ratio_returns_mdlp_trend, label='Low frequency IMFs', color='green')
plt.yticks(fontsize=8)
plt.ylabel('Omega Ratio Daily Returns', fontsize=10)
plt.xticks([0, 334, 699, 1065, 1430],
           ['30-01-2018', '31-12-2018', '31-12-2019', '31-12-2020', '31-12-2021'],
           fontsize=8, rotation=-30)
plt.xlabel('Days', fontsize=10)
box_0 = ax.get_position()
ax.set_position([box_0.x0 - 0.04, box_0.y0, box_0.width * 1.04, box_0.height])
ax.legend(loc='upper left', fontsize=8)
# plt.savefig('../figures/S&P 500 - 11 Sectors/Sector_11_indices_omega_ratio_returns.png')
plt.show()

# measure performances - Omega ratio returns
window = 30
omega_ratio_returns_global_minimum_portfolio = \
    omega_ratio_return(weight_matrix_global_minimum[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
omega_ratio_returns_global_minimum_portfolio_long = \
    omega_ratio_return(weight_matrix_global_minimum_long[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
omega_ratio_returns_maximum_sharpe_ratio_portfolio = \
    omega_ratio_return(weight_matrix_maximum_sharpe_ratio_restriction[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
omega_ratio_returns_pca_portfolio = \
    omega_ratio_return(weight_matrix_pca[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
omega_ratio_returns_covreg_imf_direct_portfolio = \
    omega_ratio_return(weight_matrix_direct_imf_covreg[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
omega_ratio_returns_covreg_ssa_direct_portfolio = \
    omega_ratio_return(weight_matrix_direct_ssa_covreg[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
omega_ratio_returns_covreg_imf_direct_portfolio_not_long = \
    omega_ratio_return(weight_matrix_direct_imf_covreg_restriction[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
omega_ratio_returns_covreg_ssa_direct_portfolio_not_long = \
    omega_ratio_return(weight_matrix_direct_ssa_covreg_restriction[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)

omega_ratio_returns_realised_covariance = \
    omega_ratio_return(weight_matrix_realised[:end_of_month_vector_cumsum[48]].T,
                       sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
omega_ratio_returns_mgarch = \
    omega_ratio_return(weight_matrix_dcc[:end_of_month_vector_cumsum[48]].T,
                       sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
omega_ratio_returns_sp500 = \
    omega_ratio_return(np.ones_like(sp500_returns.reshape(1, -1)), sp500_returns.reshape(1, -1), window)

omega_ratio_returns_covreg_imf_direct_high = \
    omega_ratio_return(weight_matrix_direct_imf_covreg_high[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
omega_ratio_returns_covreg_imf_direct_high_not_long = \
    omega_ratio_return(weight_matrix_direct_imf_covreg_high_restriction[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)

ax = plt.subplot(111)
plt.gcf().subplots_adjust(bottom=0.18)
plt.title('Omega Ratio Returns', fontsize=12)
plt.gcf().set_size_inches(12, 8)
plt.plot(omega_ratio_returns_sp500, label='S&P 500 Proxy')
plt.plot(omega_ratio_returns_sp500_001, label=textwrap.fill('S&P 500 Proxy with 0.01% charge', 15))
plt.plot(omega_ratio_returns_realised_covariance, label='Realised covariance')
plt.plot(omega_ratio_returns_mgarch, label='DCC MGARCH')
plt.plot(omega_ratio_returns_global_minimum_portfolio, label=textwrap.fill('Global minimum variance', 15))
plt.plot(omega_ratio_returns_pca_portfolio, label=textwrap.fill('Principle portfolio with 3 components', 20))
# plt.plot(omega_ratio_returns_covreg_imf_direct_high, label=textwrap.fill('High frequency (long restriction)', 19))
# plt.plot(omega_ratio_returns_covreg_imf_direct_high_not_long, label=textwrap.fill('High frequency (summation restriction)', 19))
# plt.plot(omega_ratio_returns_covreg_imf_direct_portfolio, label=textwrap.fill('All frequencies (long restriction)', 19))
# plt.plot(omega_ratio_returns_covreg_imf_direct_portfolio_not_long, label=textwrap.fill('All frequencies (summation restriction)', 19))
# plt.plot(omega_ratio_returns_covreg_ssa_direct_portfolio, label=textwrap.fill('Low frequency (long restriction)', 19))
# plt.plot(omega_ratio_returns_covreg_ssa_direct_portfolio_not_long,
#          label=textwrap.fill('Low frequency (summation restriction)', 19), c='k')
plt.yticks(fontsize=8)
plt.ylabel('Omega Ratio', fontsize=10)
plt.xticks([0, 334, 699, 1065, 1430],
           ['30-01-2018', '31-12-2018', '31-12-2019', '31-12-2020', '31-12-2021'],
           fontsize=8, rotation=-30)
plt.xlabel('Days', fontsize=10)
box_0 = ax.get_position()
ax.set_position([box_0.x0 - 0.04, box_0.y0, box_0.width * 1.04, box_0.height])
ax.legend(loc='upper left', fontsize=8)
# plt.savefig('../figures/S&P 500 - 11 Sectors/Sector_11_indices_omega_ratio_returns.png')
plt.show()

fig, axs = plt.subplots(1, 2, sharey=True, figsize=(32, 16))
axs[0].set_title('Omega Ratio Returns - 30 Day Rolling Window', fontsize=36, pad=20)
axs[0].plot(np.arange(30, 1462, 1), omega_ratio_returns_sp500, label='S&P 500 Proxy', linewidth=3)
axs[0].plot(np.arange(30, 1462, 1), omega_ratio_returns_sp500_001, label=textwrap.fill('S&P 500 Proxy with 0.01% charge', 17), linewidth=3)
axs[0].plot(np.arange(30, 1462, 1), omega_ratio_returns_mdlp_high, label='High frequency IMFs', color='red', linewidth=3)
axs[0].plot(np.arange(30, 1462, 1), omega_ratio_returns_mdlp_mid, label='Mid-frequency IMFs', color='gold', linewidth=3)
axs[0].plot(np.arange(30, 1462, 1), omega_ratio_returns_mdlp_trend, label='Low frequency IMFs', color='green', linewidth=3)
axs[0].set_xticks([30, 365, 730, 1096, 1461])
axs[0].set_xticklabels(['30-01-2018', '31-12-2018', '31-12-2019', '31-12-2020', '31-12-2021'],
                       fontsize=20, rotation=-30)
axs[0].set_xlabel('Days', fontsize=30)
axs[0].set_yticks([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
axs[0].set_yticklabels([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], fontsize=20)
box_0 = axs[0].get_position()
axs[0].set_position([box_0.x0 - 0.075, box_0.y0 + 0.02, box_0.width * 1.24, box_0.height])
axs[0].legend(loc='upper left', fontsize=20)
axs[0].set_ylim(-0.25, 10.25)
axs[1].set_title('Omega Ratio Returns - 30 Day Rolling Window', fontsize=36, pad=20)
axs[1].plot(np.arange(30, 1462, 1), omega_ratio_returns_sp500, label='S&P 500 Proxy', linewidth=3)
axs[1].plot(np.arange(30, 1462, 1), omega_ratio_returns_sp500_001,
            label=textwrap.fill('S&P 500 Proxy with 0.01% charge', 17), linewidth=3)
axs[1].plot(np.arange(30, 1462, 1), omega_ratio_returns_realised_covariance, label='Realised covariance', linewidth=3)
axs[1].plot(np.arange(30, 1462, 1), omega_ratio_returns_mgarch, label='DCC MGARCH', linewidth=3)
axs[1].plot(np.arange(30, 1462, 1), omega_ratio_returns_global_minimum_portfolio,
            label=textwrap.fill('Global minimum variance', 15), linewidth=3)
axs[1].plot(np.arange(30, 1462, 1), omega_ratio_returns_pca_portfolio,
            label=textwrap.fill('Principle portfolio with 3 components', 20),
            linewidth=3)
axs[1].set_xticks([30, 365, 730, 1096, 1461])
axs[1].set_xticklabels(['30-01-2018', '31-12-2018', '31-12-2019', '31-12-2020', '31-12-2021'],
                       fontsize=20, rotation=-30)
axs[1].set_xlabel('Days', fontsize=30)
box_0 = axs[1].get_position()
axs[1].set_position([box_0.x0 - 0.02, box_0.y0 + 0.02, box_0.width * 1.24, box_0.height])
axs[1].legend(loc='upper right', fontsize=20)
plt.savefig('../figures/S&P 500 - 11 Sectors/Sector_11_indices_omega_ratio_returns_joint_plot.pdf')
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
sharpe_ratio_returns_sp500_001 = \
    sharpe_ratio_return(np.ones_like(sp500_returns.reshape(1, -1)), sp500_returns_001.reshape(1, -1), window)

ax = plt.subplot(111)
plt.gcf().subplots_adjust(bottom=0.18)
plt.gcf().set_size_inches(12, 8)
plt.title('Sharpe Ratio Returns', fontsize=12)
plt.plot(sharpe_ratio_returns_sp500, label='S&P 500 Proxy')
plt.plot(sharpe_ratio_returns_sp500_001, label=textwrap.fill('S&P 500 Proxy with 0.01% charge', 15))
plt.plot(sharpe_ratio_returns_mdlp_high, label='High frequency IMFs', color='red')
plt.plot(sharpe_ratio_returns_mdlp_mid, label='Mid-frequency IMFs', color='gold')
plt.plot(sharpe_ratio_returns_mdlp_trend, label='Low frequency IMFs', color='green')
plt.yticks(fontsize=8)
plt.ylabel('Sharpe Ratio Daily Returns', fontsize=10)
plt.xticks([0, 334, 699, 1065, 1430],
           ['30-01-2018', '31-12-2018', '31-12-2019', '31-12-2020', '31-12-2021'],
           fontsize=8, rotation=-30)
plt.xlabel('Days', fontsize=10)
box_0 = ax.get_position()
ax.set_position([box_0.x0 - 0.04, box_0.y0, box_0.width * 1.04, box_0.height])
ax.legend(loc='upper left', fontsize=8)
# plt.savefig('../figures/S&P 500 - 11 Sectors/Sector_11_indices_sharpe_ratio_returns.png')
plt.show()

# measure performances - Sharpe ratio returns
window = 30
sharpe_ratio_returns_global_minimum_portfolio = \
    sharpe_ratio_return(weight_matrix_global_minimum[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
sharpe_ratio_returns_global_minimum_portfolio_long = \
    sharpe_ratio_return(weight_matrix_global_minimum_long[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
sharpe_ratio_returns_maximum_sharpe_ratio_portfolio = \
    sharpe_ratio_return(weight_matrix_maximum_sharpe_ratio_restriction[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
sharpe_ratio_returns_pca_portfolio = \
    sharpe_ratio_return(weight_matrix_pca[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
sharpe_ratio_returns_covreg_imf_direct_portfolio = \
    sharpe_ratio_return(weight_matrix_direct_imf_covreg[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
sharpe_ratio_returns_covreg_ssa_direct_portfolio = \
    sharpe_ratio_return(weight_matrix_direct_ssa_covreg[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
sharpe_ratio_returns_covreg_imf_direct_portfolio_not_long = \
    sharpe_ratio_return(weight_matrix_direct_imf_covreg_restriction[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
sharpe_ratio_returns_covreg_ssa_direct_portfolio_not_long = \
    sharpe_ratio_return(weight_matrix_direct_ssa_covreg_restriction[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)

sharpe_ratio_returns_realised_covariance = \
    sharpe_ratio_return(weight_matrix_realised[:end_of_month_vector_cumsum[48]].T,
                        sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
sharpe_ratio_returns_mgarch = \
    sharpe_ratio_return(weight_matrix_dcc[:end_of_month_vector_cumsum[48]].T,
                        sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
sharpe_ratio_returns_sp500 = \
    sharpe_ratio_return(np.ones_like(sp500_returns.reshape(1, -1)), sp500_returns.reshape(1, -1), window)

sharpe_ratio_returns_covreg_imf_direct_high = \
    sharpe_ratio_return(weight_matrix_direct_imf_covreg_high[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
sharpe_ratio_returns_covreg_imf_direct_high_not_long = \
    sharpe_ratio_return(weight_matrix_direct_imf_covreg_high_restriction[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)

ax = plt.subplot(111)
plt.gcf().subplots_adjust(bottom=0.18)
plt.gcf().set_size_inches(12, 8)
plt.title('Sharpe Ratio Returns', fontsize=12)
plt.plot(sharpe_ratio_returns_sp500, label='S&P 500 Proxy')
plt.plot(sharpe_ratio_returns_sp500_001, label=textwrap.fill('S&P 500 Proxy with 0.01% charge', 15))
plt.plot(sharpe_ratio_returns_realised_covariance, label='Realised covariance')
plt.plot(sharpe_ratio_returns_mgarch, label='DCC MGARCH')
plt.plot(sharpe_ratio_returns_global_minimum_portfolio, label=textwrap.fill('Global minimum variance', 15))
plt.plot(sharpe_ratio_returns_pca_portfolio, label=textwrap.fill('Principle portfolio with 3 components', 20))
# plt.plot(sharpe_ratio_returns_covreg_imf_direct_high, label=textwrap.fill('High frequency (long restriction)', 19))
# plt.plot(sharpe_ratio_returns_covreg_imf_direct_high_not_long, label=textwrap.fill('High frequency (summation restriction)', 19))
# plt.plot(sharpe_ratio_returns_covreg_imf_direct_portfolio, label=textwrap.fill('All frequencies (long restriction)', 19))
# plt.plot(sharpe_ratio_returns_covreg_imf_direct_portfolio_not_long, label=textwrap.fill('All frequencies (summation restriction)', 19))
# plt.plot(sharpe_ratio_returns_covreg_ssa_direct_portfolio, label=textwrap.fill('Low frequency (long restriction)', 19))
# plt.plot(sharpe_ratio_returns_covreg_ssa_direct_portfolio_not_long,
#          label=textwrap.fill('Low frequency (summation restriction)', 19), c='k')
plt.yticks(fontsize=8)
plt.ylabel('Sharpe Ratio', fontsize=10)
plt.xticks([0, 334, 699, 1065, 1430],
           ['30-01-2018', '31-12-2018', '31-12-2019', '31-12-2020', '31-12-2021'],
           fontsize=8, rotation=-30)
plt.xlabel('Days', fontsize=10)
box_0 = ax.get_position()
ax.set_position([box_0.x0 - 0.04, box_0.y0, box_0.width * 1.04, box_0.height])
ax.legend(loc='upper left', fontsize=8)
# plt.savefig('../figures/S&P 500 - 11 Sectors/Sector_11_indices_sharpe_ratio_returns.png')
plt.show()

fig, axs = plt.subplots(1, 2, sharey=True, figsize=(32, 16))
axs[0].set_title('Sharpe Ratio Returns - 30 Day Rolling Window', fontsize=36, pad=20)
axs[0].plot(np.arange(30, 1462, 1), sharpe_ratio_returns_sp500, label='S&P 500 Proxy', linewidth=3)
axs[0].plot(np.arange(30, 1462, 1), sharpe_ratio_returns_sp500_001, label=textwrap.fill('S&P 500 Proxy with 0.01% charge', 17), linewidth=3)
axs[0].plot(np.arange(30, 1462, 1), sharpe_ratio_returns_mdlp_high, label='High frequency IMFs', color='red', linewidth=3)
axs[0].plot(np.arange(30, 1462, 1), sharpe_ratio_returns_mdlp_mid, label='Mid-frequency IMFs', color='gold', linewidth=3)
axs[0].plot(np.arange(30, 1462, 1), sharpe_ratio_returns_mdlp_trend, label='Low frequency IMFs', color='green', linewidth=3)
axs[0].set_xticks([30, 365, 730, 1096, 1461])
axs[0].set_xticklabels(['30-01-2018', '31-12-2018', '31-12-2019', '31-12-2020', '31-12-2021'],
                       fontsize=20, rotation=-30)
axs[0].set_xlabel('Days', fontsize=30)
axs[0].set_yticks([-0.4, -0.2, 0.0, 0.2, 0.4, 0.6])
axs[0].set_yticklabels(['-0.4', '-0.2', '0.0', '0.2', '0.4', '0.6'], fontsize=20)
box_0 = axs[0].get_position()
axs[0].set_position([box_0.x0 - 0.075, box_0.y0 + 0.02, box_0.width * 1.24, box_0.height])
axs[0].legend(loc='lower left', fontsize=20)
axs[0].set_ylim(-0.55, 0.75)
axs[1].set_title('Sharpe Ratio Returns - 30 Day Rolling Window', fontsize=36, pad=20)
axs[1].plot(np.arange(30, 1462, 1), sharpe_ratio_returns_sp500, label='S&P 500 Proxy', linewidth=3)
axs[1].plot(np.arange(30, 1462, 1), sharpe_ratio_returns_sp500_001,
            label=textwrap.fill('S&P 500 Proxy with 0.01% charge', 17), linewidth=3)
axs[1].plot(np.arange(30, 1462, 1), sharpe_ratio_returns_realised_covariance, label='Realised covariance', linewidth=3)
axs[1].plot(np.arange(30, 1462, 1), sharpe_ratio_returns_mgarch, label='DCC MGARCH', linewidth=3)
axs[1].plot(np.arange(30, 1462, 1), sharpe_ratio_returns_global_minimum_portfolio,
            label=textwrap.fill('Global minimum variance', 30), linewidth=3)
axs[1].plot(np.arange(30, 1462, 1), sharpe_ratio_returns_pca_portfolio,
            label=textwrap.fill('Principle portfolio with 3 components', 40),
            linewidth=3)
axs[1].set_xticks([30, 365, 730, 1096, 1461])
axs[1].set_xticklabels(['30-01-2018', '31-12-2018', '31-12-2019', '31-12-2020', '31-12-2021'],
                       fontsize=20, rotation=-30)
axs[1].set_xlabel('Days', fontsize=30)
box_0 = axs[1].get_position()
axs[1].set_position([box_0.x0 - 0.02, box_0.y0 + 0.02, box_0.width * 1.24, box_0.height])
axs[1].legend(loc='best', fontsize=20)
plt.savefig('../figures/S&P 500 - 11 Sectors/Sector_11_indices_sharpe_ratio_returns_joint_plot.pdf')
plt.show()

# measure performances - Sharpe ratio returns
window = 30

sortino_ratio_returns_mdlp_high = sortino_ratio_return(weight_matrix_high[:end_of_month_vector_cumsum[48]].T,
                                                       sector_11_indices_array[end_of_month_vector_cumsum[12]:].T,
                                                       window)
sortino_ratio_returns_mdlp_mid = sortino_ratio_return(weight_matrix_mid[:end_of_month_vector_cumsum[48]].T,
                                                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T,
                                                      window)
sortino_ratio_returns_mdlp_trend = sortino_ratio_return(weight_matrix_trend[:end_of_month_vector_cumsum[48]].T,
                                                        sector_11_indices_array[end_of_month_vector_cumsum[12]:].T,
                                                        window)

sortino_ratio_returns_sp500 = \
    sortino_ratio_return(np.ones_like(sp500_returns.reshape(1, -1)), sp500_returns.reshape(1, -1), window)
sortino_ratio_returns_sp500_001 = \
    sortino_ratio_return(np.ones_like(sp500_returns.reshape(1, -1)), sp500_returns_001.reshape(1, -1), window)

ax = plt.subplot(111)
plt.gcf().subplots_adjust(bottom=0.18)
plt.gcf().set_size_inches(12, 8)
plt.title('Sortino Ratio Returns', fontsize=12)
plt.plot(sortino_ratio_returns_sp500, label='S&P 500 Proxy')
plt.plot(sortino_ratio_returns_sp500_001, label=textwrap.fill('S&P 500 Proxy with 0.01% charge', 15))
plt.plot(sortino_ratio_returns_mdlp_high, label='High frequency IMFs', color='red')
plt.plot(sortino_ratio_returns_mdlp_mid, label='Mid-frequency IMFs', color='gold')
plt.plot(sortino_ratio_returns_mdlp_trend, label='Low frequency IMFs', color='green')
plt.yticks(fontsize=8)
plt.ylabel('Sharpe Ratio Daily Returns', fontsize=10)
plt.xticks([0, 334, 699, 1065, 1430],
           ['30-01-2018', '31-12-2018', '31-12-2019', '31-12-2020', '31-12-2021'],
           fontsize=8, rotation=-30)
plt.xlabel('Days', fontsize=10)
box_0 = ax.get_position()
ax.set_position([box_0.x0 - 0.04, box_0.y0, box_0.width * 1.04, box_0.height])
ax.legend(loc='upper left', fontsize=8)
# plt.savefig('../figures/S&P 500 - 11 Sectors/Sector_11_indices_sharpe_ratio_returns.png')
plt.show()

# measure performances - Sharpe ratio returns
window = 30
sortino_ratio_returns_global_minimum_portfolio = \
    sortino_ratio_return(weight_matrix_global_minimum[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
sortino_ratio_returns_global_minimum_portfolio_long = \
    sortino_ratio_return(weight_matrix_global_minimum_long[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
sortino_ratio_returns_maximum_sharpe_ratio_portfolio = \
    sortino_ratio_return(weight_matrix_maximum_sharpe_ratio_restriction[:end_of_month_vector_cumsum[48]].T,
                         sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
sortino_ratio_returns_pca_portfolio = \
    sortino_ratio_return(weight_matrix_pca[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
sortino_ratio_returns_covreg_imf_direct_portfolio = \
    sortino_ratio_return(weight_matrix_direct_imf_covreg[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
sortino_ratio_returns_covreg_ssa_direct_portfolio = \
    sortino_ratio_return(weight_matrix_direct_ssa_covreg[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
sortino_ratio_returns_covreg_imf_direct_portfolio_not_long = \
    sortino_ratio_return(weight_matrix_direct_imf_covreg_restriction[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
sortino_ratio_returns_covreg_ssa_direct_portfolio_not_long = \
    sortino_ratio_return(weight_matrix_direct_ssa_covreg_restriction[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)

sortino_ratio_returns_realised_covariance = \
    sortino_ratio_return(weight_matrix_realised[:end_of_month_vector_cumsum[48]].T,
                         sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
sortino_ratio_returns_mgarch = \
    sortino_ratio_return(weight_matrix_dcc[:end_of_month_vector_cumsum[48]].T,
                         sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
sortino_ratio_returns_sp500 = \
    sortino_ratio_return(np.ones_like(sp500_returns.reshape(1, -1)), sp500_returns.reshape(1, -1), window)

sortino_ratio_returns_covreg_imf_direct_high = \
    sortino_ratio_return(weight_matrix_direct_imf_covreg_high[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
sortino_ratio_returns_covreg_imf_direct_high_not_long = \
    sortino_ratio_return(weight_matrix_direct_imf_covreg_high_restriction[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)

ax = plt.subplot(111)
plt.gcf().subplots_adjust(bottom=0.18)
plt.gcf().set_size_inches(12, 8)
plt.title('Sortino Ratio Returns', fontsize=12)
plt.plot(sortino_ratio_returns_sp500, label='S&P 500 Proxy')
plt.plot(sortino_ratio_returns_sp500, label='S&P 500 Proxy')
plt.plot(sortino_ratio_returns_realised_covariance, label='Realised covariance')
plt.plot(sortino_ratio_returns_mgarch, label='DCC MGARCH')
plt.plot(sortino_ratio_returns_global_minimum_portfolio, label=textwrap.fill('Global minimum variance', 15))
plt.plot(sortino_ratio_returns_pca_portfolio, label=textwrap.fill('Principle portfolio with 3 components', 20))
# plt.plot(sortino_ratio_returns_covreg_imf_direct_high, label=textwrap.fill('High frequency (long restriction)', 19))
# plt.plot(sortino_ratio_returns_covreg_imf_direct_high_not_long, label=textwrap.fill('High frequency (summation restriction)', 19))
# plt.plot(sortino_ratio_returns_covreg_imf_direct_portfolio, label=textwrap.fill('All frequencies (long restriction)', 19))
# plt.plot(sortino_ratio_returns_covreg_imf_direct_portfolio_not_long, label=textwrap.fill('All frequencies (summation restriction)', 19))
# plt.plot(sortino_ratio_returns_covreg_ssa_direct_portfolio, label=textwrap.fill('Low frequency (long restriction)', 19))
# plt.plot(sortino_ratio_returns_covreg_ssa_direct_portfolio_not_long,
#          label=textwrap.fill('Low frequency (summation restriction)', 19), c='k')
plt.yticks(fontsize=8)
plt.ylabel('Sortino Ratio', fontsize=10)
plt.xticks([0, 334, 699, 1065, 1430],
           ['30-01-2018', '31-12-2018', '31-12-2019', '31-12-2020', '31-12-2021'],
           fontsize=8, rotation=-30)
plt.xlabel('Days', fontsize=10)
box_0 = ax.get_position()
ax.set_position([box_0.x0 - 0.04, box_0.y0, box_0.width * 1.04, box_0.height])
ax.legend(loc='upper left', fontsize=8)
# plt.savefig('../figures/S&P 500 - 11 Sectors/Sector_11_indices_sortino_ratio_returns.png')
plt.show()

fig, axs = plt.subplots(1, 2, sharey=True, figsize=(32, 16))
axs[0].set_title('Sortino Ratio Returns - 30 Day Rolling Window', fontsize=36, pad=20)
axs[0].plot(np.arange(30, 1462, 1), sortino_ratio_returns_sp500, label='S&P 500 Proxy', linewidth=3)
axs[0].plot(np.arange(30, 1462, 1), sortino_ratio_returns_sp500_001, label=textwrap.fill('S&P 500 Proxy with 0.01% charge', 17), linewidth=3)
axs[0].plot(np.arange(30, 1462, 1), sortino_ratio_returns_mdlp_high, label='High frequency IMFs', color='red', linewidth=3)
axs[0].plot(np.arange(30, 1462, 1), sortino_ratio_returns_mdlp_mid, label='Mid-frequency IMFs', color='gold', linewidth=3)
axs[0].plot(np.arange(30, 1462, 1), sortino_ratio_returns_mdlp_trend, label='Low frequency IMFs', color='green', linewidth=3)
axs[0].set_xticks([30, 365, 730, 1096, 1461])
axs[0].set_xticklabels(['30-01-2018', '31-12-2018', '31-12-2019', '31-12-2020', '31-12-2021'],
                       fontsize=20, rotation=-30)
axs[0].set_xlabel('Days', fontsize=30)
axs[0].set_yticks([0.0, 1.0, 2.0, 3.0])
axs[0].set_yticklabels([0.0, 1.0, 2.0, 3.0], fontsize=20)
box_0 = axs[0].get_position()
axs[0].set_position([box_0.x0 - 0.075, box_0.y0 + 0.02, box_0.width * 1.24, box_0.height])
axs[0].legend(loc='upper left', fontsize=20)
axs[0].set_ylim(-0.9, 3.9)
axs[1].set_title('Sortino Ratio Returns - 30 Day Rolling Window', fontsize=36, pad=20)
axs[1].plot(np.arange(30, 1462, 1), sortino_ratio_returns_sp500, label='S&P 500 Proxy', linewidth=3)
axs[1].plot(np.arange(30, 1462, 1), sortino_ratio_returns_sp500_001,
            label=textwrap.fill('S&P 500 Proxy with 0.01% charge', 17), linewidth=3)
axs[1].plot(np.arange(30, 1462, 1), sortino_ratio_returns_realised_covariance, label='Realised covariance', linewidth=3)
axs[1].plot(np.arange(30, 1462, 1), sortino_ratio_returns_mgarch, label='DCC MGARCH', linewidth=3)
axs[1].plot(np.arange(30, 1462, 1), sortino_ratio_returns_global_minimum_portfolio,
            label=textwrap.fill('Global minimum variance', 15), linewidth=3)
axs[1].plot(np.arange(30, 1462, 1), sortino_ratio_returns_pca_portfolio,
            label=textwrap.fill('Principle portfolio with 3 components', 20),
            linewidth=3)
axs[1].set_xticks([30, 365, 730, 1096, 1461])
axs[1].set_xticklabels(['30-01-2018', '31-12-2018', '31-12-2019', '31-12-2020', '31-12-2021'],
                       fontsize=20, rotation=-30)
axs[1].set_xlabel('Days', fontsize=30)
box_0 = axs[1].get_position()
axs[1].set_position([box_0.x0 - 0.02, box_0.y0 + 0.02, box_0.width * 1.24, box_0.height])
axs[1].legend(loc='upper right', fontsize=20)
plt.savefig('../figures/S&P 500 - 11 Sectors/Sector_11_indices_sortino_ratio_returns_joint_plot.pdf')
plt.show()

# relationship with short limit

short_vector = np.arange(0, -3.0, -0.1)
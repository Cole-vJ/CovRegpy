
import textwrap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from CovRegpy_finance_utils import efficient_frontier, global_minimum_forward_applied_information, \
    sharpe_forward_applied_information, pca_forward_applied_information, \
    global_minimum_forward_applied_information_long, sharpe_forward_applied_information_restriction

from CovRegpy_covariance_regression_functions import cov_reg_given_mean, cubic_b_spline

from CovRegpy_portfolio_weighting_functions import rb_p_weights, rb_p_weights_not_long, global_obj_fun

from CovRegpy_measures import cumulative_return, mean_return, variance_return, value_at_risk_return, \
    max_draw_down_return, omega_ratio_return, sortino_ratio_return, sharpe_ratio_return

from CovRegpy_singular_spectrum_analysis import ssa

from AdvEMDpy import AdvEMDpy

np.random.seed(2)

sns.set(style='darkgrid')

# load 11 sector indices
sector_11_indices = pd.read_csv('S&P500_Data/sp_500_11_sector_indices.csv', header=0)
sector_11_indices = sector_11_indices.set_index(['Unnamed: 0'])

# approximate daily treasury par yield curve rates for 3 year bonds
risk_free = (0.01 / 365)  # daily risk free rate

# sector numpy array
sector_11_indices_array = np.vstack((np.zeros((1, 11)), np.asarray(sector_11_indices)))

for col, sector in enumerate(sector_11_indices.columns):
    plt.plot(np.asarray(np.cumprod(np.exp(sector_11_indices_array[:, col]))), label=sector)
plt.title(textwrap.fill('Cumulative Returns of Eleven Market Cap Weighted Sector Indices of S&P 500 from '
                        '1 January 2017 to 31 December 2021', 60),
          fontsize=10)
plt.legend(loc='upper left', fontsize=8)
plt.xticks([0, 365, 730, 1095, 1461, 1826],
           ['31-12-2016', '31-12-2017', '31-12-2018', '31-12-2019', '31-12-2020', '31-12-2021'],
           fontsize=8, rotation=-30)
plt.yticks([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5], ['0.5', '1.0', '1.5', '2.0', '2.5', '3.0', '3.5'], fontsize=8)
del sector, col
plt.show()

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
weight_matrix_maximum_sharpe_ratio = np.zeros_like(sector_11_indices_array)
weight_matrix_maximum_sharpe_ratio_restriction = np.zeros_like(sector_11_indices_array)
weight_matrix_pca = np.zeros_like(sector_11_indices_array)
weight_matrix_direct_imf_covreg = np.zeros_like(sector_11_indices_array)
weight_matrix_direct_ssa_covreg = np.zeros_like(sector_11_indices_array)
weight_matrix_direct_imf_covreg_not_long = np.zeros_like(sector_11_indices_array)
weight_matrix_direct_ssa_covreg_not_long = np.zeros_like(sector_11_indices_array)

# weights calculated on and used on different data (one month ahead)
for day in range(len(end_of_month_vector_cumsum[:-int(months + 1)])):

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
    ms_w, ms_sd, ms_r = sharpe_forward_applied_information(annual_covariance, monthly_covariance, monthly_returns,
                                                           risk_free, gm_w, gm_r)
    # plt.scatter(ms_sd, ms_r, label='Maximum Sharpe ratio portfolio', zorder=2)

    # calculate maximum sharpe ratio portfolio
    msr_w, msr_sd, msr_r = sharpe_forward_applied_information_restriction(annual_covariance, monthly_covariance,
                                                                          monthly_returns, risk_free, gm_w, gm_r,
                                                                          short_limit=0.1)
    # plt.scatter(msr_sd, msr_r, label='Maximum Sharpe ratio portfolio with restriction', zorder=2)

    # calculate efficient frontier
    ef_sd, ef_r = efficient_frontier(gm_w, gm_r, gm_sd, ms_w, ms_r, ms_sd, monthly_covariance)
    # plt.plot(ef_sd, ef_r, 'k--', label='Efficient frontier', zorder=1)

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
        imfs, _, _, _, _, _, _ = \
            emd.empirical_mode_decomposition(knot_envelope=np.linspace(-12, end_time + 12, knots),
                                             matrix=True)

        # ssa
        ssa_components = ssa(np.asarray(price_signal[:, signal]), L=80, plot=False)
        try:
            x_ssa = np.vstack((ssa_components, x_ssa))
        except:
            x_ssa = ssa_components.copy()

        # deal with constant last IMF and insert IMFs in dataframe
        # deal with different frequency structures here
        try:
            imfs = imfs[1:, :]
            if np.isclose(imfs[-1, 0], imfs[-1, -1]):
                imfs[-2, :] += imfs[-1, :]
                imfs = imfs[:-1, :]
        except:
            pass
        try:
            x = np.vstack((imfs, x))
        except:
            x = imfs.copy()

    # ssa
    x_ssa_trunc = \
        x_ssa[:, :int(end_of_month_vector_cumsum[int(day + months - 1)] - end_of_month_vector_cumsum[int(day)])]

    x_trunc = \
        x[:, :int(end_of_month_vector_cumsum[int(day + months - 1)] - end_of_month_vector_cumsum[int(day)])]

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
            spline_basis_direct_trunc = spline_basis_direct_trunc[:, :np.shape(y)[1]]

    # make 'x' and 'y' the same size - ssa
    if np.shape(x_ssa_trunc)[1] != np.shape(y)[1]:
        diff = int(np.abs(np.shape(y)[1] - np.shape(x_ssa_trunc)[1]))
        if np.shape(x_ssa_trunc)[1] < np.shape(y)[1]:
            y = y[:, :np.shape(x_ssa_trunc)[1]]
        elif np.shape(y)[1] < np.shape(x_ssa_trunc)[1]:
            x_ssa_trunc = x_ssa_trunc[:, :np.shape(y)[1]]

    # calculate 'A_est'
    A_est_ssa = A_est.copy()

    # calculate B_est and Psi_est - ssa
    B_est_direct_ssa, Psi_est_direct_ssa = \
        cov_reg_given_mean(A_est=A_est_ssa, basis=spline_basis_direct_trunc, x=x_ssa_trunc, y=y, iterations=100)

    # calculate B_est and Psi_est - direct application
    B_est_direct, Psi_est_direct = \
        cov_reg_given_mean(A_est=A_est, basis=spline_basis_direct_trunc, x=x_trunc, y=y, iterations=100)

    # calculate forecasted variance

    # days in the month where forecasting is to be done
    days_in_month_forecast_direct = int(end_of_month_vector_cumsum[int(day + months)] -
                                        end_of_month_vector_cumsum[int(day + months - 1)] - diff)

    # empty forecasted variance storage matrix - direct
    variance_Model_forecast_direct = np.zeros((days_in_month_forecast_direct, np.shape(B_est_direct)[1],
                                               np.shape(B_est_direct)[1]))

    # empty forecasted variance storage matrix - ssa
    variance_Model_forecast_ssa = np.zeros(
        (days_in_month_forecast_direct, np.shape(B_est_direct)[1], np.shape(B_est_direct)[1]))

    # imf days that will be used to forecast variance of returns
    forecast_days = np.arange(end_of_month_vector_cumsum[int(day + months - 1)],
                              end_of_month_vector_cumsum[int(day + months)])[:days_in_month_forecast_direct]

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

        variance_Model_forecast_ssa[forecasted_variance_index] = \
            Psi_est_direct_ssa + \
            np.matmul(np.matmul(B_est_direct_ssa.T, x_ssa[:, extract_x_imf_values]).astype(np.float64).reshape(-1, 1),
                      np.matmul(x_ssa[:, extract_x_imf_values].T,
                                B_est_direct_ssa).astype(np.float64).reshape(1, -1)).astype(np.float64)

    # debugging step
    # plt.plot(np.mean(np.mean(np.abs(variance_Model_forecast_direct), axis=1), axis=1))
    variance_median_direct = np.median(variance_Model_forecast_direct, axis=0)

    # debugging step
    # plt.plot(np.mean(np.mean(np.abs(variance_Model_forecast_direct), axis=1), axis=1))
    variance_median_direct_ssa = np.median(variance_Model_forecast_ssa, axis=0)

    #####################################################
    # direct application Covariance Regression - BOTTOM #
    #####################################################

    # calculate weights, variance, and returns - direct application ssa Covariance Regression - long only
    weights_Model_forecast_direct_ssa = rb_p_weights(variance_median_direct_ssa).x
    model_variance_forecast_direct_ssa = global_obj_fun(weights_Model_forecast_direct_ssa, monthly_covariance)
    model_returns_forecast_direct_ssa = sum(weights_Model_forecast_direct_ssa * monthly_returns)
    # plt.scatter(np.sqrt(model_variance_forecast_direct_ssa), model_returns_forecast_direct_ssa,
    #             label='CovReg Direct Model SSA')

    # calculate weights, variance, and returns - direct application Covariance Regression - long only
    weights_Model_forecast_direct = rb_p_weights(variance_median_direct).x
    model_variance_forecast_direct = global_obj_fun(weights_Model_forecast_direct, monthly_covariance)
    model_returns_forecast_direct = sum(weights_Model_forecast_direct * monthly_returns)
    # plt.scatter(np.sqrt(model_variance_forecast_direct), model_returns_forecast_direct, label='CovReg Direct Model')

    # calculate weights, variance, and returns - direct application ssa Covariance Regression - long restraint removed
    weights_Model_forecast_direct_ssa_long_short = rb_p_weights_not_long(variance_median_direct_ssa, short_limit=1).x
    model_variance_forecast_direct_ssa_long_short = global_obj_fun(weights_Model_forecast_direct_ssa_long_short,
                                                                   monthly_covariance)
    model_returns_forecast_direct_ssa_long_short = sum(weights_Model_forecast_direct_ssa_long_short * monthly_returns)
    # plt.scatter(np.sqrt(model_variance_forecast_direct_ssa_long_short), model_returns_forecast_direct_ssa_long_short,
    #             label='CovReg Direct Model SSA')

    # calculate weights, variance, and returns - direct application Covariance Regression - long restraint removed
    weights_Model_forecast_direct_long_short = rb_p_weights_not_long(variance_median_direct, short_limit=1).x
    model_variance_forecast_direct_long_short = global_obj_fun(weights_Model_forecast_direct_long_short,
                                                               monthly_covariance)
    model_returns_forecast_direct_long_short = sum(weights_Model_forecast_direct_long_short * monthly_returns)
    # plt.scatter(np.sqrt(model_variance_forecast_direct_long_short), model_returns_forecast_direct_long_short,
    #             label='CovReg Direct Model')

    # filled weight matrices iteratively
    weight_matrix_global_minimum[end_of_month_vector_cumsum[day]:end_of_month_vector_cumsum[int(day + 1)], :] = \
        gm_w
    weight_matrix_global_minimum_long[end_of_month_vector_cumsum[day]:end_of_month_vector_cumsum[int(day + 1)], :] = \
        gm_w_long
    weight_matrix_maximum_sharpe_ratio[end_of_month_vector_cumsum[day]:end_of_month_vector_cumsum[int(day + 1)], :] = \
        ms_w
    weight_matrix_maximum_sharpe_ratio_restriction[end_of_month_vector_cumsum[day]:end_of_month_vector_cumsum[int(day + 1)], :] = \
        msr_w
    weight_matrix_pca[end_of_month_vector_cumsum[day]:end_of_month_vector_cumsum[int(day + 1)], :] = \
        pc_w
    weight_matrix_direct_imf_covreg[end_of_month_vector_cumsum[day]:end_of_month_vector_cumsum[int(day + 1)], :] = \
        weights_Model_forecast_direct
    weight_matrix_direct_ssa_covreg[end_of_month_vector_cumsum[day]:end_of_month_vector_cumsum[int(day + 1)], :] = \
        weights_Model_forecast_direct_ssa
    weight_matrix_direct_imf_covreg_not_long[end_of_month_vector_cumsum[day]:end_of_month_vector_cumsum[int(day + 1)], :] = \
        weights_Model_forecast_direct_long_short
    weight_matrix_direct_ssa_covreg_not_long[end_of_month_vector_cumsum[day]:end_of_month_vector_cumsum[int(day + 1)], :] = \
        weights_Model_forecast_direct_ssa_long_short

    # graph options
    plt.title(f'Actual Portfolio Returns versus Portfolio Variance for '
              f'1 {month_vector[int(day % 12)]} {year_vector[int((day + 12) // 12)]} to '
              f'{np.str(end_of_month_vector[int(day + 13)])} {month_vector[int(int(day + 12) % 12)]} '
              f'{year_vector[int(int(day + 12) // 12)]}', fontsize=10)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.xlabel('Portfolio variance', fontsize=10)
    plt.ylabel('Portfolio returns', fontsize=10)
    plt.legend(loc='upper left', fontsize=8)
    # plt.show()

    print(day)

# measure performances - cumulative returns
cumulative_returns_global_minimum_portfolio = \
    cumulative_return(weight_matrix_global_minimum[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T)
cumulative_returns_global_minimum_portfolio_long = \
    cumulative_return(weight_matrix_global_minimum_long[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T)
cumulative_returns_maximum_sharpe_ratio_portfolio = \
    cumulative_return(weight_matrix_maximum_sharpe_ratio[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T)
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
    cumulative_return(weight_matrix_direct_imf_covreg_not_long[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T)
cumulative_returns_covreg_ssa_direct_portfolio_not_long = \
    cumulative_return(weight_matrix_direct_ssa_covreg_not_long[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T)

plt.title('Cumulative Returns')
plt.plot(cumulative_returns_global_minimum_portfolio, label='Global minimum variance')
plt.plot(cumulative_returns_global_minimum_portfolio_long, label='Global minimum variance long')
# plt.plot(cumulative_returns_maximum_sharpe_ratio_portfolio, label='Maximum Sharpe ratio portfolio')
plt.plot(cumulative_returns_maximum_sharpe_ratio_portfolio_restriction,
         label='Maximum Sharpe ratio portfolio (with short restriction)')
plt.plot(cumulative_returns_pca_portfolio, label='PCA portfolio (3 components)')
plt.plot(cumulative_returns_covreg_imf_direct_portfolio, label='IMF CovRegpy direct (long only)')
plt.plot(cumulative_returns_covreg_ssa_direct_portfolio, label='SSA CovRegpy direct (long only)')
plt.plot(cumulative_returns_covreg_imf_direct_portfolio_not_long, label='IMF CovRegpy direct')
plt.plot(cumulative_returns_covreg_ssa_direct_portfolio_not_long, label='SSA CovRegpy direct')
plt.legend(loc='upper left', fontsize=8)
plt.show()

# measure performances - mean returns
window = 30
mean_returns_global_minimum_portfolio = \
    mean_return(weight_matrix_global_minimum[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
mean_returns_global_minimum_portfolio_long = \
    mean_return(weight_matrix_global_minimum_long[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
mean_returns_maximum_sharpe_ratio_portfolio = \
    mean_return(weight_matrix_maximum_sharpe_ratio[:end_of_month_vector_cumsum[48]].T,
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

plt.title('Mean Returns')
plt.plot(mean_returns_global_minimum_portfolio, label='Global minimum variance')
plt.plot(mean_returns_global_minimum_portfolio_long, label='Global minimum variance long')
# plt.plot(cumulative_returns_maximum_sharpe_ratio_portfolio, label='Maximum Sharpe ratio portfolio')
plt.plot(mean_returns_pca_portfolio, label='PCA portfolio (3 components)')
plt.plot(mean_returns_covreg_imf_direct_portfolio, label='IMF CovRegpy direct (long only)')
plt.plot(mean_returns_covreg_ssa_direct_portfolio, label='SSA CovRegpy direct (long only)')
plt.legend(loc='upper left', fontsize=8)
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
    variance_return(weight_matrix_maximum_sharpe_ratio[:end_of_month_vector_cumsum[48]].T,
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

plt.title('Variance Returns')
plt.plot(variance_returns_global_minimum_portfolio, label='Global minimum variance')
plt.plot(variance_returns_global_minimum_portfolio_long, label='Global minimum variance long')
# plt.plot(cumulative_returns_maximum_sharpe_ratio_portfolio, label='Maximum Sharpe ratio portfolio')
plt.plot(variance_returns_pca_portfolio, label='PCA portfolio (3 components)')
plt.plot(variance_returns_covreg_imf_direct_portfolio, label='IMF CovRegpy direct (long only)')
plt.plot(variance_returns_covreg_ssa_direct_portfolio, label='SSA CovRegpy direct (long only)')
plt.legend(loc='upper left', fontsize=8)
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
    value_at_risk_return(weight_matrix_maximum_sharpe_ratio[:end_of_month_vector_cumsum[48]].T,
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

plt.title('Value-at-Risk Returns')
plt.plot(value_at_risk_returns_global_minimum_portfolio, label='Global minimum variance')
plt.plot(value_at_risk_returns_global_minimum_portfolio_long, label='Global minimum variance long')
# plt.plot(cumulative_returns_maximum_sharpe_ratio_portfolio, label='Maximum Sharpe ratio portfolio')
plt.plot(value_at_risk_returns_pca_portfolio, label='PCA portfolio (3 components)')
plt.plot(value_at_risk_returns_covreg_imf_direct_portfolio, label='IMF CovRegpy direct (long only)')
plt.plot(value_at_risk_returns_covreg_ssa_direct_portfolio, label='SSA CovRegpy direct (long only)')
plt.legend(loc='upper left', fontsize=8)
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
    max_draw_down_return(weight_matrix_maximum_sharpe_ratio[:end_of_month_vector_cumsum[48]].T,
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

plt.title('Maximum Draw Down Returns')
plt.plot(max_draw_down_returns_global_minimum_portfolio, label='Global minimum variance')
plt.plot(max_draw_down_returns_global_minimum_portfolio_long, label='Global minimum variance long')
# plt.plot(cumulative_returns_maximum_sharpe_ratio_portfolio, label='Maximum Sharpe ratio portfolio')
plt.plot(max_draw_down_returns_pca_portfolio, label='PCA portfolio (3 components)')
plt.plot(max_draw_down_returns_covreg_imf_direct_portfolio, label='IMF CovRegpy direct (long only)')
plt.plot(max_draw_down_returns_covreg_ssa_direct_portfolio, label='SSA CovRegpy direct (long only)')
plt.legend(loc='upper left', fontsize=8)
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
    omega_ratio_return(weight_matrix_maximum_sharpe_ratio[:end_of_month_vector_cumsum[48]].T,
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

plt.title('Omega Ratio Returns')
plt.plot(omega_ratio_returns_global_minimum_portfolio, label='Global minimum variance')
plt.plot(omega_ratio_returns_global_minimum_portfolio_long, label='Global minimum variance long')
# plt.plot(cumulative_returns_maximum_sharpe_ratio_portfolio, label='Maximum Sharpe ratio portfolio')
plt.plot(omega_ratio_returns_pca_portfolio, label='PCA portfolio (3 components)')
plt.plot(omega_ratio_returns_covreg_imf_direct_portfolio, label='IMF CovRegpy direct (long only)')
plt.plot(omega_ratio_returns_covreg_ssa_direct_portfolio, label='SSA CovRegpy direct (long only)')
plt.legend(loc='upper left', fontsize=8)
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
    sharpe_ratio_return(weight_matrix_maximum_sharpe_ratio[:end_of_month_vector_cumsum[48]].T,
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

plt.title('Sharpe Ratio Returns')
plt.plot(sharpe_ratio_returns_global_minimum_portfolio, label='Global minimum variance')
plt.plot(sharpe_ratio_returns_global_minimum_portfolio_long, label='Global minimum variance long')
# plt.plot(cumulative_returns_maximum_sharpe_ratio_portfolio, label='Maximum Sharpe ratio portfolio')
plt.plot(sharpe_ratio_returns_pca_portfolio, label='PCA portfolio (3 components)')
plt.plot(sharpe_ratio_returns_covreg_imf_direct_portfolio, label='IMF CovRegpy direct (long only)')
plt.plot(sharpe_ratio_returns_covreg_ssa_direct_portfolio, label='SSA CovRegpy direct (long only)')
plt.legend(loc='upper left', fontsize=8)
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
    sortino_ratio_return(weight_matrix_maximum_sharpe_ratio[:end_of_month_vector_cumsum[48]].T,
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

plt.title('Sortino Ratio Returns')
plt.plot(sortino_ratio_returns_global_minimum_portfolio, label='Global minimum variance')
plt.plot(sortino_ratio_returns_global_minimum_portfolio_long, label='Global minimum variance long')
# plt.plot(cumulative_returns_maximum_sharpe_ratio_portfolio, label='Maximum Sharpe ratio portfolio')
plt.plot(sortino_ratio_returns_pca_portfolio, label='PCA portfolio (3 components)')
plt.plot(sortino_ratio_returns_covreg_imf_direct_portfolio, label='IMF CovRegpy direct (long only)')
plt.plot(sortino_ratio_returns_covreg_ssa_direct_portfolio, label='SSA CovRegpy direct (long only)')
plt.legend(loc='upper left', fontsize=8)
plt.show()

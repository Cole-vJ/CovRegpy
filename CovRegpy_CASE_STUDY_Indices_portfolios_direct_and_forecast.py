
import textwrap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from CovRegpy_forecasting import gp_forecast
from sklearn.gaussian_process.kernels import ExpSineSquared, WhiteKernel, RBF, RationalQuadratic

from CovRegpy_finance_utils import efficient_frontier, global_minimum_forward_applied_information, \
    sharpe_forward_applied_information, pca_forward_applied_information, \
    global_minimum_forward_applied_information_long, sharpe_forward_applied_information_individual_restriction, \
    sharpe_forward_applied_information_summation_restriction

from CovRegpy_covariance_regression_functions import cov_reg_given_mean, cubic_b_spline

from CovRegpy_portfolio_weighting_functions import risk_parity_weights_long_restrict, risk_parity_weights_short_restriction, global_obj_fun

from CovRegpy_measures import cumulative_return, mean_return, variance_return, value_at_risk_return, \
    max_draw_down_return, omega_ratio_return, sortino_ratio_return, sharpe_ratio_return

from CovRegpy_singular_spectrum_analysis import CovRegpy_ssa

from CovRegpy_GARCH_model import covregpy_dcc_mgarch

from AdvEMDpy import AdvEMDpy

np.random.seed(3)

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

ax = plt.subplot(111)
plt.gcf().subplots_adjust(bottom=0.18)
for col, sector in enumerate(sector_11_indices.columns):
    plt.plot(np.asarray(np.cumprod(np.exp(sector_11_indices_array[:, col]))), label=sector)
plt.title(textwrap.fill('Cumulative Returns of Eleven Market Cap Weighted Sector Indices of S&P 500 from '
                        '1 January 2017 to 31 December 2021', 60),
          fontsize=12)
plt.legend(loc='best', fontsize=8)
plt.xticks([0, 365, 730, 1095, 1461, 1826],
           ['31-12-2016', '31-12-2017', '31-12-2018', '31-12-2019', '31-12-2020', '31-12-2021'],
           fontsize=8, rotation=-30)
plt.xlabel('Days', fontsize=10)
plt.ylabel('Cumulative Returns', fontsize=10)
plt.yticks([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5], ['0.5', '1.0', '1.5', '2.0', '2.5', '3.0', '3.5'], fontsize=8)
del sector, col
plt.savefig('figures/S&P 500 - 11 Sectors/Sector_11_indices.png')
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
weight_matrix_imf_covreg_forecast = np.zeros_like(sector_11_indices_array)
weight_matrix_ssa_covreg_forecast = np.zeros_like(sector_11_indices_array)
weight_matrix_imf_covreg_not_long_forecast = np.zeros_like(sector_11_indices_array)
weight_matrix_ssa_covreg_not_long_forecast = np.zeros_like(sector_11_indices_array)
weight_matrix_dcc = np.zeros_like(sector_11_indices_array)
weight_matrix_realised = np.zeros_like(sector_11_indices_array)

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
    msr_w, msr_sd, msr_r = sharpe_forward_applied_information_summation_restriction(annual_covariance,
                                                                                    monthly_covariance,
                                                                                    monthly_returns, risk_free,
                                                                                    gm_w, gm_r, short_limit=0.3,
                                                                                    long_limit=1.3)
    print(np.sum(msr_w[msr_w < 0]))
    print(np.sum(msr_w[msr_w > 0]))
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
        ssa_components = CovRegpy_ssa(np.asarray(price_signal[:, signal]), L=80, plot=False)
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
    variance_Model_forecast_direct = np.zeros(
        (days_in_month_forecast_direct, np.shape(B_est_direct)[1], np.shape(B_est_direct)[1]))

    # empty forecasted variance storage matrix - ssa
    variance_Model_forecast_ssa = np.zeros(
        (days_in_month_forecast_direct, np.shape(B_est_direct)[1], np.shape(B_est_direct)[1]))

    # dcc mgarch
    variance_Model_forecast_dcc = np.zeros(
        (days_in_month_forecast_direct, np.shape(B_est_direct)[1], np.shape(B_est_direct)[1]))

    # imf days that will be used to forecast variance of returns
    forecast_days = np.arange(end_of_month_vector_cumsum[int(day + months - 1)],
                              end_of_month_vector_cumsum[int(day + months)])[:days_in_month_forecast_direct]

    # DCC forecast
    known_returns = \
        sector_11_indices_array[end_of_month_vector_cumsum[int(day)]:end_of_month_vector_cumsum[int(day + months)], :]
    # dcc_forecast = \
    #     covregpy_dcc_mgarch(known_returns, days=1)

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

        variance_Model_forecast_ssa[forecasted_variance_index] = \
            Psi_est_direct_ssa + \
            np.matmul(np.matmul(B_est_direct_ssa.T, x_ssa[:, extract_x_imf_values]).astype(np.float64).reshape(-1, 1),
                      np.matmul(x_ssa[:, extract_x_imf_values].T,
                                B_est_direct_ssa).astype(np.float64).reshape(1, -1)).astype(np.float64)

        # dcc mgarch
        # variance_Model_forecast_dcc[forecasted_variance_index] = dcc_forecast * np.sqrt(forecasted_variance_index + 1)

        # realised covariance
        variance_Model_forecast_realised[forecasted_variance_index] = annual_covariance

    # debugging step
    # plt.plot(np.mean(np.mean(np.abs(variance_Model_forecast_direct), axis=1), axis=1))
    variance_median_direct = np.median(variance_Model_forecast_direct, axis=0)

    # debugging step
    # plt.plot(np.mean(np.mean(np.abs(variance_Model_forecast_direct), axis=1), axis=1))
    variance_median_direct_ssa = np.median(variance_Model_forecast_ssa, axis=0)

    variance_median_dcc = np.median(variance_Model_forecast_dcc, axis=0)

    variance_median_realised = np.median(variance_Model_forecast_realised, axis=0)

    #####################################################
    # direct application Covariance Regression - BOTTOM #
    #####################################################

    #########################################################
    # Gaussian Process forecast Covariance Regression - TOP #
    #########################################################

    # matrix to store forecasted IMFs
    forecasted_imfs = np.zeros((np.shape(x)[0], end_of_month_vector[int(day + months + 1)]))

    # https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_co2.html#sphx-glr-auto-examples-gaussian-process-plot-gpr-co2-py
    # long term smooth rising trend
    k1 = 1.0 ** 2 * RBF(length_scale=10.0,
                        length_scale_bounds=(1e-00, 1e+02))
    # seasonal component
    k2 = (2.4 ** 2 * RBF(length_scale=90.0,
                         length_scale_bounds=(1e-00, 1e+02)) *
          ExpSineSquared(length_scale=1.3, periodicity=1.0,
                         length_scale_bounds=(1e-01, 1e+01),
                         periodicity_bounds=(1e-02, 1e+02)))
    # medium term irregularity
    k3 = 0.66 ** 2 * RationalQuadratic(length_scale=1.2, alpha=0.78,
                                       length_scale_bounds=(1e-01, 1e+01),
                                       alpha_bounds=(1e-01, 1e+01))
    # noise terms
    k4 = 0.18 ** 2 * RBF(length_scale=0.134, length_scale_bounds=(1e-01, 1e+01)) + WhiteKernel(noise_level=0.19 ** 2)

    k5 = RationalQuadratic(length_scale=1.2, alpha=0.78, length_scale_bounds=(1e-01, 1e+01),
                           alpha_bounds=(1e-01, 1e+01)) * ExpSineSquared(length_scale=1.3, periodicity=1.0,
                                                                         length_scale_bounds=(1e-01, 1e+01),
                                                                         periodicity_bounds=(1e-01, 1e+01))

    k6 = RationalQuadratic(length_scale=1.2, alpha=0.78, length_scale_bounds=(1e-01, 1e+01),
                           alpha_bounds=(1e-01, 1e+01)) * RBF(length_scale=1.3, length_scale_bounds=(1e-01, 1e+01))

    k7 = RBF(length_scale=1.3,
             length_scale_bounds=(1e-01, 1e+01)) * ExpSineSquared(length_scale=1.3, periodicity=1.0,
                                                                  length_scale_bounds=(1e-01, 1e+01),
                                                                  periodicity_bounds=(1e-01, 1e+01))

    k8 = WhiteKernel(noise_level=0.19 ** 2, noise_level_bounds=(1e-01, 1e+00))

    # kernel = k1 + k2 + k3 + k4
    kernel = k5 + k6 + k7 + k8

    x_fit = np.arange(end_of_month_vector_cumsum[day], end_of_month_vector_cumsum[int(day + months)])
    gp_forecast_days = np.arange(end_of_month_vector_cumsum[int(day + months)],
                                 end_of_month_vector_cumsum[int(day + months + 1)])

    forecast_range_imf = 100
    for forecast_day in range(np.shape(forecasted_imfs)[0]):
        full_estimate = gp_forecast(x_fit[-forecast_range_imf:],
                                    x[forecast_day, :][-forecast_range_imf:] -
                                    np.mean(x[forecast_day, :][-forecast_range_imf:]) +
                                    np.random.normal(0, 0.1 *
                                                     (np.max(x[forecast_day, :][-forecast_range_imf:] -
                                                             np.min(x[forecast_day, :][-forecast_range_imf:]))),
                                                     len(x[forecast_day, :][-forecast_range_imf:])),
                                    x_forecast=np.hstack((x_fit[-forecast_range_imf:], gp_forecast_days)),
                                    kernel=kernel, confidence_level=0.95)
        forecasted_imfs[forecast_day, :] = full_estimate[0][-np.shape(forecasted_imfs)[1]:] + \
                                           np.mean(x[forecast_day, :][-forecast_range_imf:])

        # debugging
        # plt.title('IMF Forecast')
        # plt.plot(x_fit[-forecast_range_imf:],
        #          x[forecast_day, :][-forecast_range_imf:])
        # plt.plot(x_fit[-forecast_range_imf:],
        #          x[forecast_day, :][-forecast_range_imf:] -
        #          np.random.normal(0, 0.1 *
        #                           (np.max(x[forecast_day, :][-forecast_range_imf:] -
        #                                   np.min(x[forecast_day, :][-forecast_range_imf:]))),
        #                           len(x[forecast_day, :][-forecast_range_imf:]))
        #          )
        # plt.plot(np.hstack((x_fit[-forecast_range_imf:], gp_forecast_days)),
        #          full_estimate[0] + np.mean(x[forecast_day, :][-forecast_range_imf:]))
        # plt.plot(np.hstack((x_fit[-forecast_range_imf:], gp_forecast_days)),
        #          full_estimate[2] + np.mean(x[forecast_day, :][-forecast_range_imf:]))
        # plt.plot(np.hstack((x_fit[-forecast_range_imf:], gp_forecast_days)),
        #          full_estimate[3] + np.mean(x[forecast_day, :][-forecast_range_imf:]))
        # plt.show()

    # matrix to store forecasted IMFs
    forecasted_ssa = np.zeros((np.shape(x_ssa)[0], end_of_month_vector[int(day + months + 1)]))

    forecast_range_ssa = 100
    for forecast_day in range(np.shape(x_ssa_trunc)[0]):
        full_estimate_ssa = gp_forecast(x_fit[-forecast_range_ssa:],
                                        x_ssa[forecast_day, :][-forecast_range_ssa:] -
                                        np.mean(x_ssa[forecast_day, :][-forecast_range_ssa:]) +
                                        np.random.normal(0, 0.1 *
                                                         (np.max(x[forecast_day, :][-forecast_range_imf:] -
                                                                 np.min(x[forecast_day, :][-forecast_range_imf:]))),
                                                         len(x[forecast_day, :][-forecast_range_imf:])),
                                        x_forecast=np.hstack((x_fit[-forecast_range_ssa:], gp_forecast_days)),
                                        kernel=kernel, confidence_level=0.95)
        forecasted_ssa[forecast_day, :] = full_estimate_ssa[0][-np.shape(forecasted_ssa)[1]:] + \
                                          np.mean(x_ssa[forecast_day, :][-forecast_range_ssa:])

        # debugging
        # plt.title('SSA Forecast')
        # plt.plot(x_fit[-forecast_range_imf:],
        #          x[forecast_day, :][-forecast_range_imf:])
        # plt.plot(x_fit[-forecast_range_imf:],
        #          x[forecast_day, :][-forecast_range_imf:] -
        #          np.random.normal(0, 0.1 *
        #                           (np.max(x[forecast_day, :][-forecast_range_imf:] -
        #                                   np.min(x[forecast_day, :][-forecast_range_imf:]))),
        #                           len(x[forecast_day, :][-forecast_range_imf:]))
        #          )
        # plt.plot(np.hstack((x_fit[-forecast_range_imf:], gp_forecast_days)),
        #          full_estimate_ssa[0] + np.mean(x[forecast_day, :][-forecast_range_imf:]))
        # plt.plot(np.hstack((x_fit[-forecast_range_imf:], gp_forecast_days)),
        #          full_estimate_ssa[2] + np.mean(x[forecast_day, :][-forecast_range_imf:]))
        # plt.plot(np.hstack((x_fit[-forecast_range_imf:], gp_forecast_days)),
        #          full_estimate_ssa[3] + np.mean(x[forecast_day, :][-forecast_range_imf:]))
        # plt.show()

    # Gaussian Process truncation of matrix
    spline_basis_gp_trunc = spline_basis[:, :-1]

    # # calculate truncated x - Gaussian Processes
    x_gp_trunc = x[:, :int(end_of_month_vector_cumsum[int(day + months)] - end_of_month_vector_cumsum[day] - 1)]

    # calculate y - Gaussian Processes
    y_gp = sector_11_indices_array[
           end_of_month_vector_cumsum[day]:int(end_of_month_vector_cumsum[int(day + months)] - 1), :]
    y_gp = y_gp.T

    # ssa gp

    x_gp_trunc_ssa = x_ssa[:, :int(end_of_month_vector_cumsum[int(day + months)] - end_of_month_vector_cumsum[day] - 1)]
    B_est_gp_ssa, Psi_est_gp_ssa = \
        cov_reg_given_mean(A_est=A_est_ssa, basis=spline_basis_gp_trunc, x=x_gp_trunc_ssa, y=y_gp, iterations=100)

    # ssa gp

    # calculate B_est and Psi_est - Gaussian Processes
    B_est_gp, Psi_est_gp = \
        cov_reg_given_mean(A_est=A_est, basis=spline_basis_gp_trunc, x=x_gp_trunc, y=y_gp, iterations=100)

    # imf days that will be used to forecast variance of returns - Gaussain Process
    forecast_days_gp = np.arange(int(end_of_month_vector_cumsum[int(day + months)] - 1),
                                 int(end_of_month_vector_cumsum[int(day + months + 1)] - 1))

    days_in_month_forecast_gp = end_of_month_vector[int(day + months + 1)]

    # empty forecasted variance storage matrix - Gaussian Process
    variance_Model_forecast_gp = np.zeros((days_in_month_forecast_gp, np.shape(B_est_gp)[1], np.shape(B_est_gp)[1]))

    # empty forecasted variance storage matrix - Gaussian Process ssa
    variance_Model_forecast_gp_ssa = np.zeros((days_in_month_forecast_gp, np.shape(B_est_gp_ssa)[1], np.shape(B_est_gp_ssa)[1]))

    # iteratively calculate variance
    for forecasted_variance_index in range(len(forecast_days_gp)):

        variance_Model_forecast_gp[forecasted_variance_index] = \
            Psi_est_gp + np.matmul(np.matmul(B_est_gp.T,
                                             forecasted_imfs[:, forecasted_variance_index]).astype(np.float64).reshape(-1, 1),
                                   np.matmul(forecasted_imfs[:, forecasted_variance_index].T,
                                             B_est_gp).astype(np.float64).reshape(1, -1)).astype(np.float64)

        # ssa gp

        variance_Model_forecast_gp_ssa[forecasted_variance_index] = \
            Psi_est_gp_ssa + \
            np.matmul(np.matmul(B_est_gp_ssa.T,
                                forecasted_ssa[:, forecasted_variance_index]).astype(np.float64).reshape(-1, 1),
                      np.matmul(forecasted_ssa[:, forecasted_variance_index].T,
                                B_est_gp_ssa).astype(np.float64).reshape(1, -1)).astype(np.float64)

        # ssa gp

    # debugging
    # plt.plot(np.mean(np.mean(np.abs(variance_Model_forecast_gp), axis=1), axis=1))
    # plt.plot(np.mean(np.mean(np.abs(variance_Model_forecast_gp_ssa), axis=1), axis=1))
    # plt.show()
    variance_median_gp = np.median(variance_Model_forecast_gp, axis=0)
    variance_median_gp_ssa = np.median(variance_Model_forecast_gp_ssa, axis=0)

    ############################################################
    # Gaussian Process forecast Covariance Regression - BOTTOM #
    ############################################################

    # calculate weights, variance, and returns - direct application ssa Covariance Regression - long only
    weights_Model_forecast_direct_ssa = risk_parity_weights_long_restrict(variance_median_direct_ssa).x
    model_variance_forecast_direct_ssa = global_obj_fun(weights_Model_forecast_direct_ssa, monthly_covariance)
    model_returns_forecast_direct_ssa = sum(weights_Model_forecast_direct_ssa * monthly_returns)
    # plt.scatter(np.sqrt(model_variance_forecast_direct_ssa), model_returns_forecast_direct_ssa,
    #             label='CovReg Direct Model SSA')

    # calculate weights, variance, and returns - direct application Covariance Regression - long only
    weights_Model_forecast_direct = risk_parity_weights_long_restrict(variance_median_direct).x
    model_variance_forecast_direct = global_obj_fun(weights_Model_forecast_direct, monthly_covariance)
    model_returns_forecast_direct = sum(weights_Model_forecast_direct * monthly_returns)
    # plt.scatter(np.sqrt(model_variance_forecast_direct), model_returns_forecast_direct, label='CovReg Direct Model')

    # calculate weights, variance, and returns - direct application ssa Covariance Regression - long restraint removed
    weights_Model_forecast_direct_ssa_long_short = risk_parity_weights_short_restriction(variance_median_direct_ssa, short_limit=1).x
    model_variance_forecast_direct_ssa_long_short = global_obj_fun(weights_Model_forecast_direct_ssa_long_short,
                                                                   monthly_covariance)
    model_returns_forecast_direct_ssa_long_short = sum(weights_Model_forecast_direct_ssa_long_short * monthly_returns)
    # plt.scatter(np.sqrt(model_variance_forecast_direct_ssa_long_short), model_returns_forecast_direct_ssa_long_short,
    #             label='CovReg Direct Model SSA')

    # calculate weights, variance, and returns - direct application Covariance Regression - long restraint removed
    weights_Model_forecast_direct_long_short = risk_parity_weights_short_restriction(variance_median_direct, short_limit=1).x
    model_variance_forecast_direct_long_short = global_obj_fun(weights_Model_forecast_direct_long_short,
                                                               monthly_covariance)
    model_returns_forecast_direct_long_short = sum(weights_Model_forecast_direct_long_short * monthly_returns)
    # plt.scatter(np.sqrt(model_variance_forecast_direct_long_short), model_returns_forecast_direct_long_short,
    #             label='CovReg Direct Model')

    weights_Model_forecast_dcc = risk_parity_weights_long_restrict(variance_median_dcc).x

    weights_Model_forecast_realised = risk_parity_weights_long_restrict(variance_median_realised).x

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
    weight_matrix_dcc[end_of_month_vector_cumsum[day]:end_of_month_vector_cumsum[int(day + 1)], :] = \
        weights_Model_forecast_dcc
    weight_matrix_realised[end_of_month_vector_cumsum[day]:end_of_month_vector_cumsum[int(day + 1)], :] = \
        weights_Model_forecast_realised

    weight_matrix_imf_covreg_forecast[end_of_month_vector_cumsum[day]:end_of_month_vector_cumsum[int(day + 1)],
    :] = risk_parity_weights_long_restrict(variance_median_gp).x
    weight_matrix_ssa_covreg_forecast[end_of_month_vector_cumsum[day]:end_of_month_vector_cumsum[int(day + 1)],
    :] = risk_parity_weights_long_restrict(variance_median_gp_ssa).x

    weight_matrix_imf_covreg_not_long_forecast[end_of_month_vector_cumsum[day]:end_of_month_vector_cumsum[int(day + 1)],
    :] = risk_parity_weights_short_restriction(variance_median_gp, 1).x
    weight_matrix_ssa_covreg_not_long_forecast[end_of_month_vector_cumsum[day]:end_of_month_vector_cumsum[int(day + 1)],
    :] = risk_parity_weights_short_restriction(variance_median_gp_ssa, 1).x

    # graph options
    plt.title(f'Actual Portfolio Returns versus Portfolio Variance for '
              f'1 {month_vector[int(day % 12)]} {year_vector[int((day + 12) // 12)]} to '
              f'{np.str(end_of_month_vector[int(day + 13)])} {month_vector[int(int(day + 12) % 12)]} '
              f'{year_vector[int(int(day + 12) // 12)]}', fontsize=10)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.xlabel('Portfolio variance', fontsize=10)
    plt.ylabel('Portfolio returns', fontsize=10)
    plt.legend(loc='best', fontsize=8)
    # plt.show()

    print(day)

    if day == 6:
        temp = 0

# plot significant weights
ax = plt.subplot(111)
plt.gcf().subplots_adjust(bottom=0.18)
plt.title('IMF CovRegpy Weights', fontsize=12)
for i in range(11):
    plt.plot(weight_matrix_direct_imf_covreg_not_long[:end_of_month_vector_cumsum[48], i],
             label=sector_11_indices.columns[i])
plt.yticks(fontsize=8)
plt.ylabel('Weights', fontsize=10)
plt.xticks([0, 365, 730, 1096, 1460],
           ['01-01-2018', '01-01-2019', '01-01-2020', '01-01-2021', '31-12-2021'],
           fontsize=8, rotation=-30)
plt.xlabel('Days', fontsize=10)
plt.legend(loc='best', fontsize=6)
plt.savefig('figures/S&P 500 - 11 Sectors/Sector_11_significant_weights.png')
plt.show()

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
cumulative_returns_covreg_dcc = \
    cumulative_return(weight_matrix_dcc[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T)
cumulative_returns_covreg_realised = \
    cumulative_return(weight_matrix_realised[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T)

cumulative_returns_covreg_imf_forecast = \
    cumulative_return(weight_matrix_imf_covreg_forecast[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T)
cumulative_returns_covreg_ssa_forecast = \
    cumulative_return(weight_matrix_ssa_covreg_forecast[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T)
cumulative_returns_covreg_imf_not_long_forecast = \
    cumulative_return(weight_matrix_imf_covreg_not_long_forecast[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T)
cumulative_returns_covreg_ssa_not_long_forecast = \
    cumulative_return(weight_matrix_ssa_covreg_not_long_forecast[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T)

ax = plt.subplot(111)
plt.gcf().subplots_adjust(bottom=0.18)
plt.title('Cumulative Returns', fontsize=12)
plt.plot(cumulative_returns_covreg_realised, label='Realised covariance')
plt.plot(cumulative_returns_covreg_dcc, label='DCC MGARCH')
plt.plot(sp500_proxy, label='S&P 500 Proxy')
plt.plot(cumulative_returns_global_minimum_portfolio, label='Global minimum variance')
plt.plot(cumulative_returns_global_minimum_portfolio_long, label='Global minimum variance long')
plt.plot(cumulative_returns_pca_portfolio, label='PCA portfolio with 3 components')
# plt.plot(cumulative_returns_maximum_sharpe_ratio_portfolio, label='Maximum Sharpe ratio portfolio')
# plt.plot(cumulative_returns_maximum_sharpe_ratio_portfolio_restriction,
#          label='Maximum Sharpe ratio portfolio (with short restriction)')
plt.plot(cumulative_returns_covreg_imf_direct_portfolio_not_long, label='IMF CovRegpy')
plt.plot(cumulative_returns_covreg_imf_direct_portfolio, label='IMF CovRegpy long')
plt.plot(cumulative_returns_covreg_ssa_direct_portfolio_not_long, label='SSA CovRegpy')
plt.plot(cumulative_returns_covreg_ssa_direct_portfolio, label='SSA CovRegpy long')

plt.plot(cumulative_returns_covreg_imf_forecast, label='IMF forecast')
plt.plot(cumulative_returns_covreg_imf_not_long_forecast, label='IMF forecast long')
plt.plot(cumulative_returns_covreg_ssa_forecast, label='SSA forecast')
plt.plot(cumulative_returns_covreg_ssa_not_long_forecast, label='SSA forecast long')
plt.yticks(fontsize=8)
plt.ylabel('Cumulative Returns', fontsize=10)
plt.xticks([0, 365, 730, 1096, 1461],
           ['31-12-2017', '31-12-2018', '31-12-2019', '31-12-2020', '31-12-2021'],
           fontsize=8, rotation=-30)
plt.xlabel('Days', fontsize=10)
plt.legend(loc='best', fontsize=8)
plt.savefig('figures/S&P 500 - 11 Sectors/Sector_11_indices_cumulative_returns.png')
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
mean_returns_covreg_imf_direct_portfolio_not_long = \
    mean_return(weight_matrix_direct_imf_covreg_not_long[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
mean_returns_covreg_ssa_direct_portfolio_not_long = \
    mean_return(weight_matrix_direct_ssa_covreg_not_long[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)

mean_returns_realised_covariance = \
    mean_return(weight_matrix_realised[:end_of_month_vector_cumsum[48]].T,
                sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
mean_returns_mgarch = \
    mean_return(weight_matrix_dcc[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
mean_returns_sp500 = \
    mean_return(np.ones_like(sp500_returns.reshape(1, -1)), sp500_returns.reshape(1, -1), window)

ax = plt.subplot(111)
plt.gcf().subplots_adjust(bottom=0.18)
plt.title('Mean Returns', fontsize=12)
plt.plot(mean_returns_realised_covariance, label='Realised covariance')
plt.plot(mean_returns_mgarch, label='DCC MGARCH')
plt.plot(mean_returns_sp500, label='S&P 500 Proxy')
plt.plot(mean_returns_global_minimum_portfolio, label='Global minimum variance')
plt.plot(mean_returns_global_minimum_portfolio_long, label='Global minimum variance long')
# plt.plot(cumulative_returns_maximum_sharpe_ratio_portfolio, label='Maximum Sharpe ratio portfolio')
plt.plot(mean_returns_pca_portfolio, label='PCA portfolio with 3 components')
plt.plot(mean_returns_covreg_imf_direct_portfolio_not_long, label='IMF CovRegpy')
plt.plot(mean_returns_covreg_imf_direct_portfolio, label='IMF CovRegpy only')
plt.plot(mean_returns_covreg_ssa_direct_portfolio_not_long, label='SSA CovRegpy')
plt.plot(mean_returns_covreg_ssa_direct_portfolio, label='SSA CovRegpy only')
plt.yticks(fontsize=8)
plt.ylabel('Mean Daily Returns', fontsize=10)
plt.xticks([0, 334, 699, 1065, 1430],
           ['30-01-2018', '31-12-2018', '31-12-2019', '31-12-2020', '31-12-2021'],
           fontsize=8, rotation=-30)
plt.xlabel('Days', fontsize=10)
plt.legend(loc='best', fontsize=8)
plt.savefig('figures/S&P 500 - 11 Sectors/Sector_11_indices_mean_returns.png')
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
variance_returns_covreg_imf_direct_portfolio_not_long = \
    variance_return(weight_matrix_direct_imf_covreg_not_long[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
variance_returns_covreg_ssa_direct_portfolio_not_long = \
    variance_return(weight_matrix_direct_ssa_covreg_not_long[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)

variance_returns_realised_covariance = \
    variance_return(weight_matrix_realised[:end_of_month_vector_cumsum[48]].T,
                    sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
variance_returns_mgarch = \
    variance_return(weight_matrix_dcc[:end_of_month_vector_cumsum[48]].T,
                    sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
variance_returns_sp500 = \
    variance_return(np.ones_like(sp500_returns.reshape(1, -1)), sp500_returns.reshape(1, -1), window)

ax = plt.subplot(111)
plt.gcf().subplots_adjust(bottom=0.18)
plt.title('Variance of Returns', fontsize=12)
plt.plot(variance_returns_realised_covariance, label='Realised covariance')
plt.plot(variance_returns_mgarch, label='DCC MGARCH')
plt.plot(variance_returns_sp500, label='S&P 500 Proxy')
plt.plot(variance_returns_global_minimum_portfolio, label='Global minimum variance')
plt.plot(variance_returns_global_minimum_portfolio_long, label='Global minimum variance long')
# plt.plot(cumulative_returns_maximum_sharpe_ratio_portfolio, label='Maximum Sharpe ratio portfolio')
plt.plot(variance_returns_pca_portfolio, label='PCA portfolio with 3 components')
plt.plot(variance_returns_covreg_imf_direct_portfolio_not_long, label='IMF CovRegpy')
plt.plot(variance_returns_covreg_imf_direct_portfolio, label='IMF CovRegpy long')
plt.plot(variance_returns_covreg_ssa_direct_portfolio_not_long, label='SSA CovRegpy')
plt.plot(variance_returns_covreg_ssa_direct_portfolio, label='SSA CovRegpy long')
plt.yticks(fontsize=8)
plt.ylabel('Variance', fontsize=10)
plt.xticks([0, 334, 699, 1065, 1430],
           ['30-01-2018', '31-12-2018', '31-12-2019', '31-12-2020', '31-12-2021'],
           fontsize=8, rotation=-30)
plt.xlabel('Days', fontsize=10)
plt.legend(loc='best', fontsize=8)
plt.savefig('figures/S&P 500 - 11 Sectors/Sector_11_indices_variance_returns.png')
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
value_at_risk_returns_covreg_imf_direct_portfolio_not_long = \
    value_at_risk_return(weight_matrix_direct_imf_covreg_not_long[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
value_at_risk_returns_covreg_ssa_direct_portfolio_not_long = \
    value_at_risk_return(weight_matrix_direct_ssa_covreg_not_long[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)

value_at_risk_returns_realised_covariance = \
    value_at_risk_return(weight_matrix_realised[:end_of_month_vector_cumsum[48]].T,
                         sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
value_at_risk_returns_mgarch = \
    value_at_risk_return(weight_matrix_dcc[:end_of_month_vector_cumsum[48]].T,
                         sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
value_at_risk_returns_sp500 = \
    value_at_risk_return(np.ones_like(sp500_returns.reshape(1, -1)), sp500_returns.reshape(1, -1), window)

ax = plt.subplot(111)
plt.gcf().subplots_adjust(bottom=0.18)
plt.title('Value-at-Risk Returns', fontsize=12)
plt.plot(value_at_risk_returns_realised_covariance, label='Realised covariance')
plt.plot(value_at_risk_returns_mgarch, label='DCC MGARCH')
plt.plot(value_at_risk_returns_sp500, label='S&P 500 Proxy')
plt.plot(value_at_risk_returns_global_minimum_portfolio, label='Global minimum variance')
plt.plot(value_at_risk_returns_global_minimum_portfolio_long, label='Global minimum variance long')
# plt.plot(cumulative_returns_maximum_sharpe_ratio_portfolio, label='Maximum Sharpe ratio portfolio')
plt.plot(value_at_risk_returns_pca_portfolio, label='PCA portfolio with 3 components')
plt.plot(value_at_risk_returns_covreg_imf_direct_portfolio_not_long, label='IMF CovRegpy')
plt.plot(value_at_risk_returns_covreg_imf_direct_portfolio, label='IMF CovRegpy long')
plt.plot(value_at_risk_returns_covreg_ssa_direct_portfolio_not_long, label='SSA CovRegpy')
plt.plot(value_at_risk_returns_covreg_ssa_direct_portfolio, label='SSA CovRegpy long')
plt.yticks(fontsize=8)
plt.ylabel('Mean Value-at-Risk', fontsize=10)
plt.xticks([0, 334, 699, 1065, 1430],
           ['30-01-2018', '31-12-2018', '31-12-2019', '31-12-2020', '31-12-2021'],
           fontsize=8, rotation=-30)
plt.xlabel('Days', fontsize=10)
plt.legend(loc='best', fontsize=8)
plt.savefig('figures/S&P 500 - 11 Sectors/Sector_11_indices_value_at_risk_returns.png')
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
max_draw_down_returns_covreg_imf_direct_portfolio_not_long = \
    max_draw_down_return(weight_matrix_direct_imf_covreg_not_long[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
max_draw_down_returns_covreg_ssa_direct_portfolio_not_long = \
    max_draw_down_return(weight_matrix_direct_ssa_covreg_not_long[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)

max_draw_down_returns_realised_covariance = \
    max_draw_down_return(weight_matrix_realised[:end_of_month_vector_cumsum[48]].T,
                         sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
max_draw_down_returns_mgarch = \
    max_draw_down_return(weight_matrix_dcc[:end_of_month_vector_cumsum[48]].T,
                         sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
max_draw_down_returns_sp500 = \
    max_draw_down_return(np.ones_like(sp500_returns.reshape(1, -1)), sp500_returns.reshape(1, -1), window)

ax = plt.subplot(111)
plt.gcf().subplots_adjust(bottom=0.18)
plt.title('Maximum Draw Down Returns', fontsize=12)
plt.plot(max_draw_down_returns_realised_covariance, label='Realised Covariance')
plt.plot(max_draw_down_returns_mgarch, label='DCC MGARCH')
plt.plot(max_draw_down_returns_sp500, label='S&P 500 Proxy')
plt.plot(max_draw_down_returns_global_minimum_portfolio, label='Global minimum variance')
plt.plot(max_draw_down_returns_global_minimum_portfolio_long, label='Global minimum variance long')
# plt.plot(cumulative_returns_maximum_sharpe_ratio_portfolio, label='Maximum Sharpe ratio portfolio')
plt.plot(max_draw_down_returns_pca_portfolio, label='PCA portfolio with 3 components')
plt.plot(max_draw_down_returns_covreg_imf_direct_portfolio_not_long, label='IMF CovRegpy')
plt.plot(max_draw_down_returns_covreg_imf_direct_portfolio, label='IMF CovRegpy long')
plt.plot(max_draw_down_returns_covreg_ssa_direct_portfolio_not_long, label='SSA CovRegpy')
plt.plot(max_draw_down_returns_covreg_ssa_direct_portfolio, label='SSA CovRegpy long')
plt.yticks(fontsize=8)
plt.ylabel('Max Draw Down', fontsize=10)
plt.xticks([0, 334, 699, 1065, 1430],
           ['30-01-2018', '31-12-2018', '31-12-2019', '31-12-2020', '31-12-2021'],
           fontsize=8, rotation=-30)
plt.xlabel('Days', fontsize=10)
plt.legend(loc='best', fontsize=8)
plt.savefig('figures/S&P 500 - 11 Sectors/Sector_11_indices_max_draw_down_returns.png')
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
omega_ratio_returns_covreg_imf_direct_portfolio_not_long = \
    omega_ratio_return(weight_matrix_direct_imf_covreg_not_long[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
omega_ratio_returns_covreg_ssa_direct_portfolio_not_long = \
    omega_ratio_return(weight_matrix_direct_ssa_covreg_not_long[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)

omega_ratio_returns_realised_covariance = \
    omega_ratio_return(weight_matrix_realised[:end_of_month_vector_cumsum[48]].T,
                       sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
omega_ratio_returns_mgarch = \
    omega_ratio_return(weight_matrix_dcc[:end_of_month_vector_cumsum[48]].T,
                       sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
omega_ratio_returns_sp500 = \
    omega_ratio_return(np.ones_like(sp500_returns.reshape(1, -1)), sp500_returns.reshape(1, -1), window)

ax = plt.subplot(111)
plt.gcf().subplots_adjust(bottom=0.18)
plt.title('Omega Ratio Returns', fontsize=12)
plt.plot(omega_ratio_returns_realised_covariance, label='Realised covariance')
plt.plot(omega_ratio_returns_mgarch, label='DCC MGARCH')
plt.plot(omega_ratio_returns_sp500, label='S&P 500 Proxy')
plt.plot(omega_ratio_returns_global_minimum_portfolio, label='Global minimum variance')
plt.plot(omega_ratio_returns_global_minimum_portfolio_long, label='Global minimum variance long')
# plt.plot(cumulative_returns_maximum_sharpe_ratio_portfolio, label='Maximum Sharpe ratio portfolio')
plt.plot(omega_ratio_returns_pca_portfolio, label='PCA portfolio with 3 components')
plt.plot(omega_ratio_returns_covreg_imf_direct_portfolio_not_long, label='IMF CovRegpy')
plt.plot(omega_ratio_returns_covreg_imf_direct_portfolio, label='IMF CovRegpy only')
plt.plot(omega_ratio_returns_covreg_ssa_direct_portfolio_not_long, label='SSA CovRegpy')
plt.plot(omega_ratio_returns_covreg_ssa_direct_portfolio, label='SSA CovRegpy only')
plt.yticks(fontsize=8)
plt.ylabel('Omega Ratio', fontsize=10)
plt.xticks([0, 334, 699, 1065, 1430],
           ['30-01-2018', '31-12-2018', '31-12-2019', '31-12-2020', '31-12-2021'],
           fontsize=8, rotation=-30)
plt.xlabel('Days', fontsize=10)
plt.legend(loc='best', fontsize=8)
plt.savefig('figures/S&P 500 - 11 Sectors/Sector_11_indices_omega_ratio_returns.png')
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
sharpe_ratio_returns_covreg_imf_direct_portfolio_not_long = \
    sharpe_ratio_return(weight_matrix_direct_imf_covreg_not_long[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
sharpe_ratio_returns_covreg_ssa_direct_portfolio_not_long = \
    sharpe_ratio_return(weight_matrix_direct_ssa_covreg_not_long[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)

sharpe_ratio_returns_realised_covariance = \
    sharpe_ratio_return(weight_matrix_realised[:end_of_month_vector_cumsum[48]].T,
                        sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
sharpe_ratio_returns_mgarch = \
    sharpe_ratio_return(weight_matrix_dcc[:end_of_month_vector_cumsum[48]].T,
                        sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
sharpe_ratio_returns_sp500 = \
    sharpe_ratio_return(np.ones_like(sp500_returns.reshape(1, -1)), sp500_returns.reshape(1, -1), window)

ax = plt.subplot(111)
plt.gcf().subplots_adjust(bottom=0.18)
plt.title('Sharpe Ratio Returns', fontsize=12)
plt.plot(sharpe_ratio_returns_realised_covariance, label='Realised covariance')
plt.plot(sharpe_ratio_returns_mgarch, label='DCC MGARCH')
plt.plot(sharpe_ratio_returns_sp500, label='S&P 500 Proxy')
plt.plot(sharpe_ratio_returns_global_minimum_portfolio, label='Global minimum variance')
plt.plot(sharpe_ratio_returns_global_minimum_portfolio_long, label='Global minimum variance long')
# plt.plot(cumulative_returns_maximum_sharpe_ratio_portfolio, label='Maximum Sharpe ratio portfolio')
plt.plot(sharpe_ratio_returns_pca_portfolio, label='PCA portfolio with 3 components')
plt.plot(sharpe_ratio_returns_covreg_imf_direct_portfolio_not_long, label='IMF CovRegpy')
plt.plot(sharpe_ratio_returns_covreg_imf_direct_portfolio, label='IMF CovRegpy long')
plt.plot(sharpe_ratio_returns_covreg_ssa_direct_portfolio_not_long, label='SSA CovRegpy')
plt.plot(sharpe_ratio_returns_covreg_ssa_direct_portfolio, label='SSA CovRegpy only')
plt.yticks(fontsize=8)
plt.ylabel('Sharpe Ratio', fontsize=10)
plt.xticks([0, 334, 699, 1065, 1430],
           ['30-01-2018', '31-12-2018', '31-12-2019', '31-12-2020', '31-12-2021'],
           fontsize=8, rotation=-30)
plt.xlabel('Days', fontsize=10)
plt.legend(loc='best', fontsize=8)
plt.savefig('figures/S&P 500 - 11 Sectors/Sector_11_indices_sharpe_ratio_returns.png')
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
sortino_ratio_returns_covreg_imf_direct_portfolio_not_long = \
    sortino_ratio_return(weight_matrix_direct_imf_covreg_not_long[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
sortino_ratio_returns_covreg_ssa_direct_portfolio_not_long = \
    sortino_ratio_return(weight_matrix_direct_ssa_covreg_not_long[:end_of_month_vector_cumsum[48]].T,
                      sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)

sortino_ratio_returns_realised_covariance = \
    sortino_ratio_return(weight_matrix_realised[:end_of_month_vector_cumsum[48]].T,
                         sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
sortino_ratio_returns_mgarch = \
    sortino_ratio_return(weight_matrix_dcc[:end_of_month_vector_cumsum[48]].T,
                         sector_11_indices_array[end_of_month_vector_cumsum[12]:].T, window)
sortino_ratio_returns_sp500 = \
    sortino_ratio_return(np.ones_like(sp500_returns.reshape(1, -1)), sp500_returns.reshape(1, -1), window)

ax = plt.subplot(111)
plt.gcf().subplots_adjust(bottom=0.18)
plt.title('Sortino Ratio Returns', fontsize=12)
plt.plot(sortino_ratio_returns_realised_covariance, label='Realised covariance')
plt.plot(sortino_ratio_returns_mgarch, label='DCC MGARCH')
plt.plot(sortino_ratio_returns_sp500, label='S&P 500 Proxy')
plt.plot(sortino_ratio_returns_global_minimum_portfolio, label='Global minimum variance')
plt.plot(sortino_ratio_returns_global_minimum_portfolio_long, label='Global minimum variance long')
# plt.plot(cumulative_returns_maximum_sharpe_ratio_portfolio, label='Maximum Sharpe ratio portfolio')
plt.plot(sortino_ratio_returns_pca_portfolio, label='PCA portfolio with 3 components')
plt.plot(sortino_ratio_returns_covreg_imf_direct_portfolio_not_long, label='IMF CovRegpy')
plt.plot(sortino_ratio_returns_covreg_imf_direct_portfolio, label='IMF CovRegpy long')
plt.plot(sortino_ratio_returns_covreg_ssa_direct_portfolio_not_long, label='SSA CovRegpy')
plt.plot(sortino_ratio_returns_covreg_ssa_direct_portfolio, label='SSA CovRegpy long')
plt.yticks(fontsize=8)
plt.ylabel('Sortino Ratio', fontsize=10)
plt.xticks([0, 334, 699, 1065, 1430],
           ['30-01-2018', '31-12-2018', '31-12-2019', '31-12-2020', '31-12-2021'],
           fontsize=8, rotation=-30)
plt.xlabel('Days', fontsize=10)
plt.legend(loc='best', fontsize=8)
plt.savefig('figures/S&P 500 - 11 Sectors/Sector_11_indices_sortino_ratio_returns.png')
plt.show()

# relationship with short limit

short_vector = np.arange(0, -3.0, -0.1)

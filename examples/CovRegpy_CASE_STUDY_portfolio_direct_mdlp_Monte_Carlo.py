
#     ________
#            /
#      \    /
#       \  /
#        \/

# RPP using EMD and RCR - direct application (no forecasting) - performed 250 times for distribution

import numpy as np
import pandas as pd
import seaborn as sns

from CovRegpy import cov_reg_given_mean, cubic_b_spline
from CovRegpy_RPP import equal_risk_parity_weights_summation_restriction
from AdvEMDpy import AdvEMDpy

for seed in np.arange(250):
    np.random.seed(seed)

    sns.set(style='darkgrid')

    # create S&P 500 index
    sp500_close = pd.read_csv('../S&P500_Data/sp_500_close_5_year.csv', header=0)
    sp500_close = sp500_close.set_index(['Unnamed: 0'])
    sp500_market_cap = pd.read_csv('../S&P500_Data/sp_500_market_cap_5_year.csv', header=0)
    sp500_market_cap = sp500_market_cap.set_index(['Unnamed: 0'])

    sp500_returns = np.log(np.asarray(sp500_close)[1:, :] / np.asarray(sp500_close)[:-1, :])
    weights = np.asarray(sp500_market_cap) / np.tile(np.sum(np.asarray(sp500_market_cap), axis=1).reshape(-1, 1), (1, 505))
    sp500_returns = np.sum(sp500_returns * weights[:-1, :], axis=1)[365:]
    sp500_proxy = np.append(1, np.exp(np.cumsum(sp500_returns)))

    # load 11 sector indices
    sector_11_indices = pd.read_csv('../S&P500_Data/sp_500_11_sector_indices.csv', header=0)
    sector_11_indices = sector_11_indices.set_index(['Unnamed: 0'])

    # approximate daily treasury par yield curve rates for 3 year bonds
    risk_free = (0.01 / 365)  # daily risk free rate

    # sector numpy array
    sector_11_indices_array = np.vstack((np.zeros((1, 11)), np.asarray(sector_11_indices)))

    weight_matrix_high = np.zeros_like(sector_11_indices_array)
    weight_matrix_mid = np.zeros_like(sector_11_indices_array)
    weight_matrix_trend = np.zeros_like(sector_11_indices_array)

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
                                                 matrix=True, verbose=False)
            try:
                x_high = np.vstack((x_high, imfs[1, :]))
            except:
                x_high = imfs[1, :].copy()
            try:
                x_mid = np.vstack((x_mid, imfs[2, :]))
            except:
                x_mid = imfs[2, :].copy()
            try:
                x_low = np.vstack((x_low, imfs[3, :]))
            except:
                try:
                    x_low = imfs[3, :].copy()
                except:
                    x_low = np.vstack((x_low, 0.01 * np.random.normal(0, 1, len(imfs[2, :]))))

        y = sector_11_indices_array[end_of_month_vector_cumsum[int(day)]:
                                    end_of_month_vector_cumsum[int(day + months - 1)], :]

        # make 'x' and 'y' the same size (working in terms of months and this can occur)
        diff = 0
        if np.shape(x_high)[1] != np.shape(y)[1]:
            diff = int(np.abs(np.shape(y)[0] - np.shape(x_high)[1]))
            if np.shape(x_high)[1] < np.shape(y)[1]:
                y = y[:, :np.shape(x_high)[1]]
            elif np.shape(y)[1] < np.shape(x_high)[1]:
                x_high_trunc = x_high[:, :np.shape(y)[0]]
                x_mid_trunc = x_mid[:, :np.shape(y)[0]]
                x_low_trunc = x_low[:, :np.shape(y)[0]]
                spline_basis_direct_trunc = spline_basis_direct_trunc[:, :np.shape(y)[0]]

        # calculate 'A_est'
        A_est_ssa = A_est.copy()

        B_est_direct_high, Psi_est_direct_high = \
            cov_reg_given_mean(A_est=A_est, basis=spline_basis_direct_trunc, x=x_high_trunc, y=y.T, iterations=100)
        B_est_direct_mid, Psi_est_direct_mid = \
            cov_reg_given_mean(A_est=A_est, basis=spline_basis_direct_trunc, x=x_mid_trunc, y=y.T, iterations=100)
        B_est_direct_low, Psi_est_direct_low = \
            cov_reg_given_mean(A_est=A_est, basis=spline_basis_direct_trunc, x=x_low_trunc, y=y.T, iterations=100)

        # calculate forecasted variance

        # days in the month where forecasting is to be done
        days_in_month_forecast_direct = diff

        # empty forecasted variance storage matrix - direct high frequency
        variance_Model_forecast_direct_high = np.zeros(
            (days_in_month_forecast_direct, np.shape(B_est_direct_high)[1], np.shape(B_est_direct_high)[1]))
        variance_Model_forecast_direct_mid = np.zeros(
            (days_in_month_forecast_direct, np.shape(B_est_direct_mid)[1], np.shape(B_est_direct_mid)[1]))
        variance_Model_forecast_direct_low = np.zeros(
            (days_in_month_forecast_direct, np.shape(B_est_direct_low)[1], np.shape(B_est_direct_low)[1]))


        # imf days that will be used to forecast variance of returns
        forecast_days = np.arange(end_of_month_vector_cumsum[int(day + months - 1)],
                                  end_of_month_vector_cumsum[int(day + months)])[:days_in_month_forecast_direct]

        # iteratively calculate variance
        for var_day in forecast_days:

            # convert var_day index to [0 -> end of month length] index
            forecasted_variance_index = int(var_day - end_of_month_vector_cumsum[int(day + months - 1)])

            # extract last days of imf
            extract_x_imf_values = int(var_day - end_of_month_vector_cumsum[day])

            variance_Model_forecast_direct_high[forecasted_variance_index] = \
                Psi_est_direct_high + np.matmul(np.matmul(B_est_direct_high.T,
                                           x_high[:, extract_x_imf_values]).astype(np.float64).reshape(-1, 1),
                                           np.matmul(x_high[:, extract_x_imf_values].T,
                                                     B_est_direct_high).astype(np.float64).reshape(1, -1)).astype(np.float64)
            variance_Model_forecast_direct_mid[forecasted_variance_index] = \
                Psi_est_direct_mid + np.matmul(np.matmul(B_est_direct_mid.T,
                                                          x_mid[:, extract_x_imf_values]).astype(
                    np.float64).reshape(-1, 1),
                                                np.matmul(x_mid[:, extract_x_imf_values].T,
                                                          B_est_direct_mid).astype(np.float64).reshape(1, -1)).astype(
                    np.float64)
            variance_Model_forecast_direct_low[forecasted_variance_index] = \
                Psi_est_direct_low + np.matmul(np.matmul(B_est_direct_low.T,
                                                         x_low[:, extract_x_imf_values]).astype(
                    np.float64).reshape(-1, 1),
                                               np.matmul(x_low[:, extract_x_imf_values].T,
                                                         B_est_direct_low).astype(np.float64).reshape(1, -1)).astype(
                    np.float64)

        # debugging step
        variance_median_direct_high = np.median(variance_Model_forecast_direct_high, axis=0)
        variance_median_direct_mid = np.median(variance_Model_forecast_direct_mid, axis=0)
        variance_median_direct_low = np.median(variance_Model_forecast_direct_low, axis=0)

        # calculate weights, variance, and returns - direct application Covariance Regression - long only
        weights_Model_forecast_direct_high = equal_risk_parity_weights_summation_restriction(variance_median_direct_high).x
        weights_Model_forecast_direct_mid = equal_risk_parity_weights_summation_restriction(variance_median_direct_mid).x
        weights_Model_forecast_direct_low = equal_risk_parity_weights_summation_restriction(variance_median_direct_low).x
        print(day)

        weight_matrix_high[end_of_month_vector_cumsum[day]:end_of_month_vector_cumsum[int(day + 1)], :] = \
            weights_Model_forecast_direct_high
        weight_matrix_mid[end_of_month_vector_cumsum[day]:end_of_month_vector_cumsum[int(day + 1)], :] = \
            weights_Model_forecast_direct_mid
        weight_matrix_trend[end_of_month_vector_cumsum[day]:end_of_month_vector_cumsum[int(day + 1)], :] = \
            weights_Model_forecast_direct_low

    pd.DataFrame(weight_matrix_high).to_csv('weights/direct_high_weights_{}.csv'.format(seed))
    pd.DataFrame(weight_matrix_mid).to_csv('weights/direct_mid_weights_{}.csv'.format(seed))
    pd.DataFrame(weight_matrix_trend).to_csv('weights/direct_low_weights_{}.csv'.format(seed))

    print(seed)

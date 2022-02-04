
import numpy as np
import pandas as pd
import seaborn as sns

from CovRegpy_finance_utils import efficient_frontier, global_minimum_forward_applied_information, \
    sharpe_forward_applied_information, pca_forward_applied_information, \
    global_minimum_forward_applied_information_long, \
    sharpe_forward_applied_information_summation_restriction

from CovRegpy_covariance_regression_functions import cov_reg_given_mean, cubic_b_spline

from CovRegpy_portfolio_weighting_functions import rb_p_weights, rb_p_weights_not_long, \
    rb_p_weights_summation_restriction, global_obj_fun

from CovRegpy_measures import cumulative_return, mean_return, variance_return, value_at_risk_return, \
    max_draw_down_return, omega_ratio_return, sortino_ratio_return, sharpe_ratio_return

from CovRegpy_singular_spectrum_analysis import ssa

from CovRegpy_GARCH_model import covregpy_dcc_mgarch

from AdvEMDpy import AdvEMDpy

high_long_freq = np.zeros(250)
high_res_freq = np.zeros(250)
all_long_freq = np.zeros(250)
all_res_freq = np.zeros(250)
low_long_freq = np.zeros(250)
low_res_freq = np.zeros(250)

for seed in np.arange(250):
    np.random.seed(seed)
    # np.random.seed(2)

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
    weight_matrix_global_minimum_long = np.zeros_like(sector_11_indices_array)
    weight_matrix_maximum_sharpe_ratio = np.zeros_like(sector_11_indices_array)
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

        # calculate global minimum variance portfolio - long only
        gm_w_long, gm_sd_long, gm_r_long = global_minimum_forward_applied_information_long(annual_covariance,
                                                                                           monthly_covariance,
                                                                                           monthly_returns)

        # calculate maximum sharpe ratio portfolio
        ms_w, ms_sd, ms_r = sharpe_forward_applied_information(annual_covariance, annual_returns,
                                                               monthly_covariance, monthly_returns,
                                                               risk_free, gm_w, gm_r)

        # calculate maximum sharpe ratio portfolio
        msr_w, msr_sd, msr_r = sharpe_forward_applied_information_summation_restriction(annual_covariance,
                                                                                        annual_returns,
                                                                                        monthly_covariance,
                                                                                        monthly_returns, risk_free,
                                                                                        gm_w, gm_r, short_limit=0.3,
                                                                                        long_limit=1.3)
        print(np.sum(msr_w[msr_w < 0]))
        print(np.sum(msr_w[msr_w > 0]))

        # calculate efficient frontier
        ef_sd, ef_r = efficient_frontier(gm_w, gm_r, gm_sd, ms_w, ms_r, ms_sd, monthly_covariance)

        # calculate pca portfolio
        pc_w, pc_sd, pc_r = pca_forward_applied_information(annual_covariance, monthly_covariance,
                                                            monthly_returns, factors=3)

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
            try:
                x_high = np.vstack((imfs[:2, :], x_high))
            except:
                x_high = imfs[:2, :].copy()

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

        B_est_direct_high, Psi_est_direct_high = \
            cov_reg_given_mean(A_est=A_est, basis=spline_basis_direct_trunc, x=x_high_trunc, y=y, iterations=100)

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

            # dcc mgarch
            # variance_Model_forecast_dcc[forecasted_variance_index] = dcc_forecast * np.sqrt(forecasted_variance_index + 1)

            # realised covariance
            variance_Model_forecast_realised[forecasted_variance_index] = annual_covariance

        # debugging step
        variance_median_direct = np.median(variance_Model_forecast_direct, axis=0)

        # debugging step
        variance_median_direct_high = np.median(variance_Model_forecast_direct_high, axis=0)

        # debugging step
        variance_median_direct_ssa = np.median(variance_Model_forecast_ssa, axis=0)

        variance_median_dcc = np.median(variance_Model_forecast_dcc, axis=0)

        variance_median_realised = np.median(variance_Model_forecast_realised, axis=0)

        #####################################################
        # direct application Covariance Regression - BOTTOM #
        #####################################################

        # calculate weights, variance, and returns - direct application ssa Covariance Regression - long only
        weights_Model_forecast_direct_ssa = rb_p_weights(variance_median_direct_ssa).x
        model_variance_forecast_direct_ssa = global_obj_fun(weights_Model_forecast_direct_ssa, monthly_covariance)
        model_returns_forecast_direct_ssa = sum(weights_Model_forecast_direct_ssa * monthly_returns)

        # calculate weights, variance, and returns - direct application Covariance Regression - long only
        weights_Model_forecast_direct = rb_p_weights(variance_median_direct).x
        model_variance_forecast_direct = global_obj_fun(weights_Model_forecast_direct, monthly_covariance)
        model_returns_forecast_direct = sum(weights_Model_forecast_direct * monthly_returns)

        # calculate weights, variance, and returns - direct application Covariance Regression - long only
        weights_Model_forecast_direct_high = rb_p_weights(variance_median_direct_high).x
        # model_variance_forecast_direct = global_obj_fun(weights_Model_forecast_direct, monthly_covariance)
        # model_returns_forecast_direct = sum(weights_Model_forecast_direct * monthly_returns)

        # calculate weights, variance, and returns - direct application ssa Covariance Regression - long restraint removed
        # weights_Model_forecast_direct_ssa_long_short = rb_p_weights_not_long(variance_median_direct_ssa, short_limit=1).x
        weights_Model_forecast_direct_ssa_summation_restriction = rb_p_weights_summation_restriction(variance_median_direct_ssa).x

        model_variance_forecast_direct_ssa_long_short = global_obj_fun(weights_Model_forecast_direct_ssa_summation_restriction,
                                                                       monthly_covariance)
        model_returns_forecast_direct_ssa_long_short = sum(weights_Model_forecast_direct_ssa_summation_restriction * monthly_returns)

        # calculate weights, variance, and returns - direct application Covariance Regression - long restraint removed
        # weights_Model_forecast_direct_long_short = rb_p_weights_not_long(variance_median_direct, short_limit=1).x
        weights_Model_forecast_direct_summation_restriction = rb_p_weights_summation_restriction(variance_median_direct).x

        model_variance_forecast_direct_long_short = global_obj_fun(weights_Model_forecast_direct_summation_restriction,
                                                                   monthly_covariance)
        model_returns_forecast_direct_long_short = sum(weights_Model_forecast_direct_summation_restriction * monthly_returns)

        weights_Model_forecast_direct_summation_restriction_high = rb_p_weights_summation_restriction(variance_median_direct_high).x

        weights_Model_forecast_dcc = rb_p_weights(variance_median_dcc).x

        weights_Model_forecast_realised = rb_p_weights(variance_median_realised).x

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

        print(day)

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

    high_long_freq[seed] = cumulative_returns_covreg_high[-1]
    high_res_freq[seed] = cumulative_returns_covreg_high_restriction[-1]
    all_long_freq[seed] = cumulative_returns_covreg_imf_direct_portfolio[-1]
    all_res_freq[seed] = cumulative_returns_covreg_imf_direct_portfolio_not_long[-1]
    low_long_freq[seed] = cumulative_returns_covreg_ssa_direct_portfolio[-1]
    low_res_freq[seed] = cumulative_returns_covreg_ssa_direct_portfolio_not_long[-1]

high_long_freq = pd.DataFrame(high_long_freq)
high_long_freq.to_csv('Cumulative Returns/high_long_freq.csv')
high_res_freq = pd.DataFrame(high_res_freq)
high_res_freq.to_csv('Cumulative Returns/high_res_freq.csv')
all_long_freq = pd.DataFrame(all_long_freq)
all_long_freq.to_csv('Cumulative Returns/all_long_freq.csv')
all_res_freq = pd.DataFrame(all_res_freq)
all_res_freq.to_csv('Cumulative Returns/all_res_freq.csv')
low_long_freq = pd.DataFrame(low_long_freq)
low_long_freq.to_csv('Cumulative Returns/low_long_freq.csv')
low_res_freq = pd.DataFrame(low_res_freq)
low_res_freq.to_csv('Cumulative Returns/low_res_freq.csv')

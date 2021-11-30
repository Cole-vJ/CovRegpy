
import textwrap
import numpy as np
import pandas as pd
import yfinance as yf
from AdvEMDpy import AdvEMDpy, emd_basis
from CovRegpy_covariance_regression_functions import cov_reg_given_mean
from CovRegpy_portfolio_weighting_functions import rb_p_weights, global_obj_fun, global_weights, global_weights_long
from CovRegpy_portfolio_sharpe_ratio import sharpe_weights, sharpe_weights_long
import matplotlib.pyplot as plt

np.random.seed(1)

# pull all close data
tickers_format = ['MSFT', 'AAPL', 'GOOGL', 'AMZN', 'TSLA']
data = yf.download(tickers_format, start="2018-10-15", end="2021-10-16")
close_data = data['Close']
del data, tickers_format

# create date range and interpolate
date_index = pd.date_range(start='16/10/2018', end='16/10/2021')
close_data = close_data.reindex(date_index).interpolate()
close_data = close_data = close_data[::-1].interpolate()
close_data = close_data = close_data[::-1]
del date_index

# daily risk free rate
risk_free = (0.02 / 365)

# setup time and knots for EMD
time = np.arange(np.shape(close_data)[0])
knots = 70

# calculate returns for CRC
returns = (np.log(np.asarray(close_data)[1:, :]) -
           np.log(np.asarray(close_data)[:-1, :]))

# store tickers and partition model days and forecast days
tickers = close_data.columns.values.tolist()
model_days = 701  # 2 years - less a month
forecast_days = np.shape(close_data)[0] - model_days - 30

# set up basis for mean extraction in CRC
spline_basis_transform = emd_basis.Basis(time_series=np.arange(model_days), time=np.arange(model_days))
spline_basis_transform = spline_basis_transform.cubic_b_spline(knots=np.linspace(0, model_days - 1, knots))

# store weights calculated throughout model
weights = np.zeros((forecast_days, np.shape(close_data)[1]))

# create list to which accumulated returns will be stored
risk_return_Model = [1]
risk_return_Covariance = [1]

# having created storage vectors, etc - proceed with algorithm
for lag in range(forecast_days):
    print(lag)  # gauge progress of algorithm

    all_data = close_data.iloc[lag:int(model_days + lag + 30)]  # data window used to forecast
    all_data_low_freq = all_data.copy()
    all_data_high_freq = all_data.copy()

    if lag in [0, 30, 61, 91, 122, 153, 181, 212, 242, 273, 303, 334]:  # 0 : 16-09-2021

        for j in range(np.shape(all_data)[1]):  # for individual stock in matrix

            # decompose price data
            emd = AdvEMDpy.EMD(time_series=np.asarray(all_data.iloc[:, j]), time=time[lag:int(model_days + lag + 30)])
            imfs, _, _, _, _, _, _ = \
                emd.empirical_mode_decomposition(knot_envelope=np.linspace(time[lag:int(model_days + lag + 30)][0],
                                                                           time[lag:int(model_days + lag + 30)][-1],
                                                                           knots),
                                                 matrix=True)

            # deal with constant last IMF and insert IMFs in dataframe
            # deal with different frequency structures here
            try:
                imfs = imfs[1:, :]
                if np.isclose(imfs[-1, 0], imfs[-1, -1]):
                    imfs[-2, :] += imfs[-1, :]
                    imfs = imfs[:-1, :]
                for imf in range(np.shape(imfs)[0]):
                    all_data[f'{tickers[j]}_close_imf_{int(imf + 1)}'] = imfs[imf, :]
                all_data_high_freq[f'{tickers[j]}_close_imf_{1}'] = imfs[0, :]
                all_data_low_freq[f'{tickers[j]}_close_imf_last'] = imfs[-1, :]

            except:
                all_data[f'{tickers[j]}_close_imf_{1}'] = imfs

        del _, imf, imfs, j
        # drop original price data
        for i in tickers:
            all_data = all_data.drop(f'{i}', axis=1)
            all_data_low_freq = all_data_low_freq.drop(f'{i}', axis=1)
            all_data_high_freq = all_data_high_freq.drop(f'{i}', axis=1)
        del i

        j = -1
        low_high = ['low', 'high']
        ax = plt.subplot(111)
        for data_ind in [all_data_low_freq, all_data_high_freq]:
            j += 1
            all_data = np.asarray(data_ind)  # convert to numpy array
            groups = np.zeros((76, 1))  # to be used for group LASSO - to be developed later

            x = np.asarray(all_data).T

            returns_subset = returns[int(lag + 29):int(model_days + lag + 29), :]

            # find coefficents for mean splines
            coef = np.linalg.lstsq(spline_basis_transform.T, returns_subset, rcond=None)[0]
            mean = np.matmul(coef.T, spline_basis_transform)  # calculate mean

            # calculate covariance regression matrices
            B_est, Psi_est = cov_reg_given_mean(A_est=coef, basis=spline_basis_transform,
                                                x=x[:, :-30], y=returns_subset.T,
                                                iterations=10, technique='ridge', max_iter=500, groups=groups)
            del groups

            # final output of covariance regression model - used to forecast
            variance_Model = Psi_est + np.matmul(np.matmul(B_est.T, x[:, -1]).astype(np.float64).reshape(-1, 1),
                                                 np.matmul(x[:, -1].T, B_est).astype(np.float64).reshape(1, -1)).astype(np.float64)
            weights_Model = rb_p_weights(variance_Model).x
            model_variance = global_obj_fun(weights_Model, variance_Model)
            model_returns = sum(weights_Model * returns[int(model_days + lag + 29), :])
            plt.scatter(np.sqrt(model_variance), model_returns, label=f'Model {low_high[j]}')

            # plot returns and variance of constituent stocks
            all_returns = returns[int(model_days + lag + 29), :]
            all_sd = np.sqrt(np.diag(variance_Model))
            plt.scatter(all_sd, all_returns, label=f'Stocks {low_high[j]}')

            realised_covariance = np.cov(returns_subset.T)

            # realised covariance weights
            weights_covariance = global_weights(realised_covariance)
            covariance_variance = global_obj_fun(weights_covariance, realised_covariance)
            covariance_returns = sum(weights_covariance * returns[int(model_days + lag + 29), :])
            if j == 1:
                plt.scatter(np.sqrt(covariance_variance), covariance_returns, label=f'Realised')

            # calculate global minimum variance portfolio weights, returns, and variance
            global_minimum_weights = global_weights(variance_Model)  # efficient frontier construction
            global_minimum_variance = global_obj_fun(global_minimum_weights, variance_Model)
            global_minimum_returns = sum(global_minimum_weights * returns[int(model_days + lag + 29), :])
            # plt.scatter(np.sqrt(global_minimum_variance), global_minimum_returns,
            #             label=f'Global {low_high[j]}', zorder=10, c='black')
            plt.scatter(np.sqrt(global_minimum_variance), global_minimum_returns,
                        zorder=10, c='black')

            # calculate maximum Sharpe ratio portfolio weights, returns, and variance
            sharpe_maximum_weights = sharpe_weights(variance_Model, returns[int(model_days + lag + 29), :], risk_free)
            sharpe_maximum_variance = global_obj_fun(sharpe_maximum_weights, variance_Model)
            sharpe_maximum_returns = sum(sharpe_maximum_weights * returns[int(model_days + lag + 29), :])
            if sharpe_maximum_returns < global_minimum_returns:  # reflect if negative
                sharpe_maximum_weights = 2 * global_minimum_weights - sharpe_maximum_weights
                sharpe_maximum_variance = global_obj_fun(sharpe_maximum_weights, variance_Model)
                sharpe_maximum_returns = sum(sharpe_maximum_weights * returns[int(model_days + lag + 29), :])
            # plt.scatter(np.sqrt(sharpe_maximum_variance), sharpe_maximum_returns,
            #             label=f'Sharpe {low_high[j]}', zorder=11, c='black')
            plt.scatter(np.sqrt(sharpe_maximum_variance), sharpe_maximum_returns,
                        zorder=11, c='black')

            # calculate efficient frontier
            efficient_frontier_sd = np.zeros(101)
            efficient_frontier_return = np.zeros(101)
            efficient_frontier_sd[0] = np.sqrt(global_minimum_variance)
            efficient_frontier_return[0] = global_minimum_returns
            for i in range(1, 100):
                efficient_frontier_sd[i] = \
                    np.sqrt(global_obj_fun(global_minimum_weights * (1 - (i / 100)) + (i / 100) * sharpe_maximum_weights,
                                    variance_Model))
                efficient_frontier_return[i] = (1 - (i / 100)) * global_minimum_returns + (i / 100) * sharpe_maximum_returns
            efficient_frontier_sd[-1] = np.sqrt(sharpe_maximum_variance)
            efficient_frontier_return[-1] = sharpe_maximum_returns
            plt.plot(efficient_frontier_sd, efficient_frontier_return, 'k-', Linewidth=int(2 * j + 1),
                     label=f'Efficient frontier {low_high[j]}')
            plt.plot(efficient_frontier_sd, 2 * efficient_frontier_return[0] - efficient_frontier_return, 'k--',
                     Linewidth=int(2 * j + 1))

            plt.title(f'Efficient Frontier')
            plt.ylabel('Expected returns')
            plt.xlabel('Expected variance of returns')
        box_0 = ax.get_position()
        ax.set_position([box_0.x0 - 0.05, box_0.y0, box_0.width * 0.85, box_0.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
        plt.show()

    weights[lag, :] = weights_Model

    risk_return_Model.append(np.exp(sum(weights_Model * returns[int(model_days + lag + 29), :])))
    risk_return_Covariance.append(np.exp(sum(weights_covariance * returns[int(model_days + lag + 29), :])))

plt.plot(np.cumprod(risk_return_Model), label='Model')
plt.plot(np.cumprod(risk_return_Covariance), label='Realised')
plt.legend(loc='best')
plt.show()

plt.plot(weights)
plt.show()

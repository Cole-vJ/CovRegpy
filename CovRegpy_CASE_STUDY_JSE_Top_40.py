
# Application
# EMD loop breaks - either resolve or drop JSE Top 40

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from AdvEMDpy import AdvEMDpy, emd_basis
from CovRegpy_PCA_functions import pca_func
from CovRegpy_portfolio_sharpe_ratio import sharpe_weights
from CovRegpy_covariance_regression_functions import cov_reg_given_mean
from CovRegpy_portfolio_weighting_functions import rb_p_weights, global_obj_fun, global_weights

# seed random number generation
np.random.seed(0)

# pull all close data
tickers_format = ['ABG.JO', 'AGL.JO', 'AMS.JO', 'ANG.JO', 'APN.JO', 'BHP.JO', 'BID.JO', 'BTI.JO', 'BVT.JO', 'CFR.JO',
                  'CLS.JO', 'CPI.JO', 'DSY.JO', 'EXX.JO', 'FSR.JO', 'GFI.JO', 'GLN.JO', 'GRT.JO', 'IMP.JO', 'INL.JO',
                  'INP.JO', 'MCG.JO', 'MNP.JO', 'MRP.JO', 'MTN.JO', 'NED.JO', 'NPH.JO', 'NPN.JO', 'NRP.JO', 'OMU.JO',
                  'PRX.JO', 'REM.JO', 'RNI.JO', 'SBK.JO', 'SHP.JO', 'SLM.JO', 'SOL.JO', 'SPP.JO', 'SSW.JO', 'VOD.JO',
                  'WHL.JO']
data = yf.download(tickers_format, start="2018-12-31", end="2022-01-01")
close_data = data['Close']
del data, tickers_format

# create date range and interpolate
date_index = pd.date_range(start='31/12/2018', end='01/01/2022')
close_data = close_data.reindex(date_index).interpolate()
close_data = close_data[::-1].interpolate()
close_data = close_data[::-1]
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
model_days = 731  # 2 years - less a month
knots_vector = np.linspace(0, model_days - 1, knots)
knots_vector = np.hstack((np.fliplr((knots_vector[0] - knots_vector[1:4]).reshape(1, -1)), knots_vector.reshape(1, -1),
                          np.fliplr(knots_vector[-1] + np.abs(knots_vector[-1] - knots_vector[-4:-1]).reshape(1, -1))))
forecast_days = np.shape(close_data)[0] - model_days - 30

# set up basis for mean extraction in CRC
spline_basis_transform = emd_basis.Basis(time_series=np.arange(model_days), time=np.arange(model_days))
spline_basis_transform = spline_basis_transform.cubic_b_spline(knots=knots_vector[0, :])

# store weights calculated throughout model
weights = np.zeros((forecast_days, np.shape(close_data)[1]))

# create list to which accumulated returns will be stored
risk_return_Model = [1]
risk_return_Equal = [1]
risk_return_Covariance = [1]
risk_return_pca = [1]

# having created storage vectors, etc - proceed with algorithm
for lag in range(forecast_days):
    print(lag)  # gauge progress of algorithm

    all_data = close_data.iloc[lag:int(model_days + lag + 1)]  # data window used to forecast

    if lag in [0, 30, 61, 91, 122, 153, 181, 212, 242, 273, 303, 334]:

        for j in range(np.shape(all_data)[1]):  # for individual stock in matrix

            # decompose price data
            emd = AdvEMDpy.EMD(time_series=np.asarray(all_data.iloc[:, j]), time=time[lag:int(model_days + lag + 1)])
            imfs, _, _, _, _, _, _ = \
                emd.empirical_mode_decomposition(knot_envelope=np.linspace(time[lag:int(model_days + lag + 1)][0],
                                                                           time[lag:int(model_days + lag + 1)][-1],
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
            except:
                all_data[f'{tickers[j]}_close_imf_{1}'] = imfs

        del _, imf, imfs, j
        # drop original price data
        for i in tickers:
            all_data = all_data.drop(f'{i}', axis=1)
        del i

        all_data = np.asarray(all_data)  # convert to numpy array
        groups = np.zeros((76, 1))  # to be used for group LASSO - to be developed later

        # normalise data
        x = np.asarray(all_data).T
        # x = (x - np.tile(x.mean(axis=1).reshape(-1, 1),
        #                  (1, np.shape(x)[1]))) / np.tile(x.std(axis=1).reshape(-1, 1), (1, np.shape(x)[1]))

        # extract returns one month ahead for forecasting
        returns_subset = returns[int(lag + 29):int(model_days + lag + 29), :]

        # calculation of realised covariance for comparison
        realised_covariance = np.cov(returns_subset.T)

        # find coefficents for mean splines
        returns_subset_forecast = returns[lag:int(model_days + lag), :]
        coef_forecast = np.linalg.lstsq(spline_basis_transform.T, returns_subset_forecast, rcond=None)[0]
        mean_forecast = np.matmul(coef_forecast.T, spline_basis_transform)  # calculate mean

        # calculate covariance regression using forecasted x
        B_est_forecast, Psi_est_forecast = cov_reg_given_mean(A_est=coef_forecast, basis=spline_basis_transform,
                                                              x=x[:, :-1], y=returns_subset_forecast.T,
                                                              iterations=10, technique='ridge', max_iter=500,
                                                              groups=groups)
        del groups

        forecast_x = np.zeros((len(x[:, -1]), 1))
        forecast_x[:, 0] = x[:, -1]

        # final output of covariance regression model - used to forecast
        variance_Model_forecast = \
            Psi_est_forecast + np.matmul(np.matmul(B_est_forecast.T, forecast_x).astype(np.float64).reshape(-1, 1),
                                         np.matmul(forecast_x.T, B_est_forecast).astype(np.float64).reshape(1, -1)).astype(np.float64)
        weights_Model_forecast = rb_p_weights(variance_Model_forecast).x
        model_variance_forecast = global_obj_fun(weights_Model_forecast, variance_Model_forecast)
        model_returns_forecast = sum(weights_Model_forecast * returns[int(model_days + lag + 29), :])
        plt.scatter(np.sqrt(model_variance_forecast), model_returns_forecast, label='Model forecast')

        # Gaussian Process - bottom

        # PCA portfolio - top
        pca_weights = pca_func(realised_covariance, 3)
        pca_total_weights = pca_weights.sum(axis=1)
        pca_variance = global_obj_fun(pca_total_weights, realised_covariance)
        pca_returns = sum(pca_total_weights * returns[int(model_days + lag + 29), :])
        plt.scatter(np.sqrt(pca_variance), pca_returns, label='PCA')
        # PCA portfolios - bottom

        # plot returns and variance of constituent stocks
        all_returns = returns[int(model_days + lag + 29), :]
        all_sd = np.sqrt(np.diag(variance_Model_forecast))
        plt.scatter(all_sd, all_returns, label='Stocks')

        # realised covariance weights
        weights_covariance = global_weights(realised_covariance)
        covariance_variance = global_obj_fun(weights_covariance, realised_covariance)
        covariance_returns = sum(weights_covariance * returns[int(model_days + lag + 29), :])
        plt.scatter(np.sqrt(covariance_variance), covariance_returns, label='Realised')

        # calculate global minimum variance portfolio weights, returns, and variance
        global_minimum_weights = global_weights(variance_Model_forecast)  # efficient frontier construction
        global_minimum_variance = global_obj_fun(global_minimum_weights, variance_Model_forecast)
        global_minimum_returns = sum(global_minimum_weights * returns[int(model_days + lag + 29), :])
        plt.scatter(np.sqrt(global_minimum_variance), global_minimum_returns, label='Global', zorder=10)

        # calculate maximum Sharpe ratio portfolio weights, returns, and variance
        sharpe_maximum_weights = sharpe_weights(variance_Model_forecast, returns[int(model_days + lag + 29), :], risk_free)
        sharpe_maximum_variance = global_obj_fun(sharpe_maximum_weights, variance_Model_forecast)
        sharpe_maximum_returns = sum(sharpe_maximum_weights * returns[int(model_days + lag + 29), :])
        if sharpe_maximum_returns < global_minimum_returns:  # reflect if negative
            sharpe_maximum_weights = 2 * global_minimum_weights - sharpe_maximum_weights
            sharpe_maximum_variance = global_obj_fun(sharpe_maximum_weights, variance_Model_forecast)
            sharpe_maximum_returns = sum(sharpe_maximum_weights * returns[int(model_days + lag + 29), :])
        plt.scatter(np.sqrt(sharpe_maximum_variance), sharpe_maximum_returns, label='Sharpe', zorder=11)

        # equal weighting comparison
        equal_variance = global_obj_fun(((1 / np.shape(close_data)[1]) * np.ones(np.shape(close_data)[1])),
                                        variance_Model_forecast)
        equal_returns = sum(((1 / np.shape(close_data)[1]) * np.ones(np.shape(close_data)[1]) *
                             returns[int(model_days + lag + 29), :]))
        plt.scatter(np.sqrt(equal_variance), equal_returns, label='Equal')

        # calculate efficient frontier
        efficient_frontier_sd = np.zeros(101)
        efficient_frontier_return = np.zeros(101)
        efficient_frontier_sd[0] = np.sqrt(global_minimum_variance)
        efficient_frontier_return[0] = global_minimum_returns
        for i in range(1, 100):
            efficient_frontier_sd[i] = \
                np.sqrt(global_obj_fun(global_minimum_weights * (1 - (i / 100)) + (i / 100) * sharpe_maximum_weights,
                                variance_Model_forecast))
            efficient_frontier_return[i] = (1 - (i / 100)) * global_minimum_returns + (i / 100) * sharpe_maximum_returns
        efficient_frontier_sd[-1] = np.sqrt(sharpe_maximum_variance)
        efficient_frontier_return[-1] = sharpe_maximum_returns
        plt.plot(efficient_frontier_sd, efficient_frontier_return, 'k-')
        plt.plot(efficient_frontier_sd, 2 * efficient_frontier_return[0] - efficient_frontier_return, 'k--')

        plt.title(f'Efficient Frontier')
        plt.ylabel('Expected returns')
        plt.xlabel('Expected variance of returns')
        plt.legend(loc='best')
        plt.show()

    weights[lag, :] = weights_Model_forecast

    risk_return_Model.append(np.exp(sum(weights_Model_forecast * returns[int(model_days + lag + 29), :])))
    risk_return_Covariance.append(np.exp(sum(weights_covariance * returns[int(model_days + lag + 29), :])))
    risk_return_Equal.append(np.exp(sum(((1 / np.shape(close_data)[1]) * np.ones(np.shape(close_data)[1])) *
                                        returns[int(model_days + lag + 29), :])))
    risk_return_pca.append(np.exp(np.sum(np.matmul(pca_weights.T,
                                                   returns[int(model_days + lag + 29), :].reshape(-1, 1)))))

plt.plot(np.cumprod(risk_return_Model), label='Model')
plt.plot(np.cumprod(risk_return_Covariance), label='Realised')
plt.plot(np.cumprod(risk_return_Equal), label='Equal')
plt.plot(np.cumprod(risk_return_pca), label='PCA')
plt.legend(loc='best')
plt.show()

plt.plot(weights)
plt.show()

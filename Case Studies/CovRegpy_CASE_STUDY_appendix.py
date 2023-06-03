
import textwrap
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
from scipy.stats import norm
import matplotlib.pyplot as plt
from CovRegpy_PCA import pca_func
from AdvEMDpy import AdvEMDpy, emd_basis
from CovRegpy_sharpe import sharpe_weights
from CovRegpy_RCR import cov_reg_given_mean
from CovRegpy_utilities import global_obj_fun, global_weights
from CovRegpy_RPP import equal_risk_parity_weights_summation_restriction

sns.set(style='darkgrid')

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

# daily risk-free rate - assumed low during period
risk_free = (0.00 / 365)

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
spline_basis_transform = spline_basis_transform.cubic_b_spline(knots=np.linspace(0, model_days, knots))

for lag in range(forecast_days):
    print(lag)  # gauge progress of algorithm

    all_data = close_data.iloc[lag:int(model_days + lag + 1)]  # data window used to forecast

    if lag in [0, 30, 61, 91, 122, 153, 181, 212, 242, 273, 303, 334]:  # 0 : 16-09-2021

        for j in range(np.shape(all_data)[1]):  # for individual stock in matrix

            # decompose price data
            emd = AdvEMDpy.EMD(time_series=np.asarray(all_data.iloc[:, j]), time=time[lag:int(model_days + lag + 1)])
            imfs, _, _, _, _, _, _ = \
                emd.empirical_mode_decomposition(knot_envelope=np.linspace(time[lag:int(model_days + lag + 1)][0],
                                                                           time[lag:int(model_days + lag + 1)][-1],
                                                                           knots),
                                                 matrix=True, verbose=False)

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
        x = np.asarray(all_data).T

        # extract returns one month ahead for forecasting
        realised_returns = returns[lag:int(model_days + lag), :]

        # calculation of realised covariance for comparison
        realised_covariance = np.cov(realised_returns.T)

        # find coefficents for mean splines
        coef = np.linalg.lstsq(spline_basis_transform.T, realised_returns, rcond=None)[0]
        mean = np.matmul(coef.T, spline_basis_transform)  # calculate mean

        # calculate covariance regression matrices
        B_est, Psi_est = cov_reg_given_mean(A_est=coef, basis=spline_basis_transform,
                                            x=x[:, :-1], y=realised_returns.T,
                                            iterations=10, technique='ridge', max_iter=500)

        if lag in [0]:
            plt.figure(figsize=(15, 9))
            plt.title(r'Efficient Frontier - {}'.format(str(close_data.index[int(model_days + lag + 1)])[:10]))

        # final output of covariance regression model - used to forecast
        variance_Model = Psi_est + np.matmul(np.matmul(B_est.T, x[:, -1]).astype(np.float64).reshape(-1, 1),
                                             np.matmul(x[:, -1].T, B_est).astype(np.float64).reshape(1, -1)).astype(np.float64)
        weights_Model = equal_risk_parity_weights_summation_restriction(variance_Model).x
        model_variance = global_obj_fun(weights_Model, variance_Model)
        model_returns = sum(weights_Model * np.mean(realised_returns, axis=0) * 365)
        plt.scatter(np.sqrt(model_variance), model_returns, label=r'EMD $l$-lagged RCR RPP portfolio')

        # calculate global minimum variance portfolio weights, returns, and variance
        global_minimum_weights = global_weights(realised_covariance)  # efficient frontier construction
        global_minimum_variance = global_obj_fun(global_minimum_weights, realised_covariance)
        global_minimum_returns = sum(global_minimum_weights * np.mean(realised_returns, axis=0) * 365)
        plt.scatter(np.sqrt(global_minimum_variance), global_minimum_returns,
                    label='Global minimum variance portfolio', zorder=10)

        # calculate maximum Sharpe ratio portfolio weights, returns, and variance
        sharpe_maximum_weights = sharpe_weights(realised_covariance, np.mean(realised_returns, axis=0) * 365, risk_free)
        sharpe_maximum_variance = global_obj_fun(sharpe_maximum_weights, realised_covariance)
        sharpe_maximum_returns = sum(sharpe_maximum_weights * np.mean(realised_returns, axis=0) * 365)
        if sharpe_maximum_returns < global_minimum_returns:  # reflect if negative
            sharpe_maximum_weights = 2 * global_minimum_weights - sharpe_maximum_weights
            sharpe_maximum_variance = global_obj_fun(sharpe_maximum_weights, realised_covariance)
            sharpe_maximum_returns = sum(sharpe_maximum_weights * np.mean(realised_returns, axis=0) * 365)
        plt.scatter(np.sqrt(sharpe_maximum_variance), sharpe_maximum_returns,
                    label='Maximum Sharpe ratio portfolio', zorder=11)

        # PCA portfolio - top
        pca_weights = pca_func(realised_covariance, 3)
        pca_total_weights = pca_weights.sum(axis=1)
        pca_variance = global_obj_fun(pca_total_weights, realised_covariance)
        pca_returns = sum(pca_total_weights * np.mean(realised_returns, axis=0) * 365)
        plt.scatter(np.sqrt(pca_variance), pca_returns, label='Principle component portfolio')
        # PCA portfolios - bottom

        # equal weighting comparison
        equal_variance = global_obj_fun(((1 / np.shape(close_data)[1]) * np.ones(np.shape(close_data)[1])),
                                        variance_Model)
        equal_returns = sum(((1 / np.shape(close_data)[1]) * np.ones(np.shape(close_data)[1]) *
                             np.mean(realised_returns, axis=0) * 365))
        plt.scatter(np.sqrt(equal_variance), equal_returns, label='Equally weighted portfolio')

        # calculate efficient frontier
        efficient_frontier_sd = np.zeros(101)
        efficient_frontier_return = np.zeros(101)
        efficient_frontier_sd[0] = np.sqrt(global_minimum_variance)
        efficient_frontier_return[0] = global_minimum_returns
        for i in range(1, 100):
            efficient_frontier_sd[i] = \
                np.sqrt(global_obj_fun(global_minimum_weights * (1 - (i / 100)) + (i / 100) * sharpe_maximum_weights,
                                realised_covariance))
            efficient_frontier_return[i] = (1 - (i / 100)) * global_minimum_returns + (i / 100) * sharpe_maximum_returns
        efficient_frontier_sd[-1] = np.sqrt(sharpe_maximum_variance)
        efficient_frontier_return[-1] = sharpe_maximum_returns
        plt.plot(efficient_frontier_sd, efficient_frontier_return, 'k-')
        plt.plot(efficient_frontier_sd, 2 * efficient_frontier_return[0] - efficient_frontier_return, 'k--')

        # plot returns and variance of constituent stocks
        stocks = ['AAPL', 'AMZN', 'GOOGL', 'MSFT', 'TSLA']
        all_returns = np.mean(realised_returns, axis=0) * 365
        all_sd = np.sqrt(np.diag(variance_Model))
        for m in np.arange(5):
            plt.scatter(all_sd[m], all_returns[m], label=stocks[m])

        plt.ylabel('Returns')
        plt.xlabel('Standard Deviation of Returns')
        plt.legend(loc='best')

        if lag in [0]:
            plt.savefig('../aas_figures/appendix_efficient_frontier.pdf')
        plt.show()

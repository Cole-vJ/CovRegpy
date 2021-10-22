
import numpy as np
import pandas as pd
from AdvEMDpy import AdvEMDpy, emd_basis
from Covariance_regression_functions import cov_reg_given_mean
from Portfolio_weighting_functions import rb_p_weights, global_obj_fun, global_weights
from Maximum_Sharpe_ratio_portfolio import sharpe_rb_p_weights
import yfinance as yf
import matplotlib.pyplot as plt

tickers = pd.read_csv('constituents.csv', header=0)
tickers_format = [f"{np.asarray(tickers)[i, 0].replace('.', '-')}" for i in range(np.shape(tickers)[0])]
# tickers_format = [f"{np.asarray(tickers)[i, 0].replace('.', '-')}" for i in range(50)]  # temp debugging
data = yf.download(tickers_format, start="2018-10-15", end="2021-10-16")
close_data = data['Close']
# close_data.to_csv('sp_500_close_3_year.csv')

# close_data = pd.read_csv('sp_500_close_3_year.csv', header=0)
# close_data['Date'] = pd.to_datetime(close_data['Date'])
# close_data = close_data.set_index('Date')
date_index = pd.date_range(start='16/10/2018', end='16/10/2021')
close_data = close_data.reindex(date_index).interpolate()
close_data = close_data = close_data[::-1].interpolate()
close_data = close_data = close_data[::-1]

risk_free = (0.02 / 365)  # daily risk free rate

time = np.arange(np.shape(close_data)[0])
knots = 70
returns = (np.log(np.asarray(close_data)[1:, :]) -
           np.log(np.asarray(close_data)[:-1, :]))

tickers = close_data.columns.values.tolist()
model_days = 701  # 2 years - less a month
forecast_days = np.shape(close_data)[0] - model_days

spline_basis_transform = emd_basis.Basis(time_series=np.arange(model_days), time=np.arange(model_days))
spline_basis_transform = spline_basis_transform.cubic_b_spline(knots=np.linspace(0, model_days - 1, knots))

weights = np.zeros((forecast_days, np.shape(close_data)[1]))

risk_return_Model = [1]
risk_return_Equal = [1]
risk_return_Covariance = [1]

for lag in range(forecast_days):
    print(lag)

    all_data = close_data.iloc[lag:int(model_days + lag + 1)]  # data window

    if lag in [0, 30, 61, 91, 122, 153, 181, 212, 242, 273, 303, 334]:  # 0 : 16-09-2021

        for j in range(np.shape(all_data)[1]):

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

        # drop original price data
        for i in tickers:
            all_data = all_data.drop(f'{i}', axis=1)

        all_data = np.asarray(all_data)  # convert to numpy array
        groups = np.zeros((76, 1))  # LATER
        returns_subset = returns[int(lag + 29):int(model_days + lag + 29), :]  # extract relevant returns

        actual_covariance = np.cov(returns_subset.T)  # calculation of realised covariance

        # # find coefficents for mean splines
        # coef = np.linalg.lstsq(spline_basis_transform.T, returns_subset, rcond=None)[0]
        # mean = np.matmul(coef.T, spline_basis_transform)  # calculate mean

        try:
            # find coefficents for mean splines
            coef = np.linalg.lstsq(spline_basis_transform.T, returns_subset, rcond=None)[0]
            mean = np.matmul(coef.T, spline_basis_transform)  # calculate mean
        except:
            for col in range(0, np.shape(returns_subset)[1], 20):
                coef = np.linalg.lstsq(spline_basis_transform.T,
                                       returns_subset[:, col:int(col + 20)],
                                       rcond=None)[0]
                if col == 0:
                    coef_all = coef
                else:
                    coef_all = np.hstack((coef_all, coef))
            mean = np.matmul(coef.T, spline_basis_transform)  # calculate mean

        # calculate covariance regression matrices
        B_est, Psi_est = cov_reg_given_mean(A_est=coef, basis=spline_basis_transform,
                                            x=np.asarray(all_data)[:-1, :].T, y=returns_subset.T,
                                            iterations=10, technique='elastic-net', max_iter=500, groups=groups)

        variance_Model = Psi_est + np.matmul(np.matmul(B_est.T,
                                                       np.asarray(all_data)[-1, :].T).astype(np.float64).reshape(-1, 1),
                                             np.matmul(np.asarray(all_data)[-1, :],
                                                       B_est).astype(np.float64).reshape(1, -1)).astype(np.float64)

        all_returns = returns[int(model_days + lag + 29), :]
        all_sd = np.sqrt(np.diag(variance_Model))

        # calculate global minimum variance portfolio and maximum Sharpe ratio
        weights_covariance = global_weights(actual_covariance)
        weights_Model = rb_p_weights(variance_Model).x

        global_minimum_weights = global_weights(variance_Model)
        global_minimum_variance = global_obj_fun(global_minimum_weights, variance_Model)
        global_minimum_returns = sum(global_minimum_weights * returns[int(model_days + lag + 29), :])

        sharpe_maximum_weights = sharpe_rb_p_weights(variance_Model,
                                                     returns[int(model_days + lag + 29), :],
                                                     risk_free).x
        sharpe_maximum_variance = global_obj_fun(sharpe_maximum_weights, variance_Model)
        sharpe_maximum_returns = sum(sharpe_maximum_weights * returns[int(model_days + lag + 29), :])

        covariance_variance = global_obj_fun(weights_covariance, variance_Model)
        covariance_returns = sum(weights_covariance * returns[int(model_days + lag + 29), :])
        plt.scatter(np.sqrt(covariance_variance), covariance_returns, label='Realised')

        equal_variance = global_obj_fun(((1 / np.shape(close_data)[1]) * np.ones(np.shape(close_data)[1])),
                                        variance_Model)
        equal_returns = sum(((1 / np.shape(close_data)[1]) * np.ones(np.shape(close_data)[1]) *
                             returns[int(model_days + lag + 29), :]))
        plt.scatter(np.sqrt(equal_variance), equal_returns, label='Equal')

        plt.scatter(np.sqrt(global_minimum_variance), global_minimum_returns, label='Global', zorder=10)
        plt.scatter(np.sqrt(sharpe_maximum_variance), sharpe_maximum_returns, label='Sharpe', zorder=11)

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
        plt.plot(efficient_frontier_sd, efficient_frontier_return, 'k-')

        plt.scatter(all_sd, all_returns)

        plt.title(f'Efficient Frontier')
        plt.ylabel('Expected returns')
        plt.xlabel('Expected variance of returns')
        plt.legend(loc='centre right')
        plt.show()

    weights[lag, :] = weights_Model

    risk_return_Model.append(np.exp(sum(weights_Model * returns[int(model_days + lag + 29), :])))
    risk_return_Covariance.append(np.exp(sum(weights_covariance * returns[int(model_days + lag + 29), :])))
    risk_return_Equal.append(np.exp(sum(((1 / np.shape(close_data)[1]) * np.ones(np.shape(close_data)[1])) *
                                        returns[int(model_days + lag + 29), :])))

temp = 0
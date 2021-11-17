
import numpy as np
import pandas as pd
import yfinance as yf
from AdvEMDpy import AdvEMDpy, emd_basis
from Covariance_regression_functions import cov_reg_given_mean
from Portfolio_weighting_functions import rb_p_weights, global_obj_fun, global_weights, global_weights_long
from Maximum_Sharpe_ratio_portfolio import sharpe_weights, sharpe_rb_p_weights
import matplotlib.pyplot as plt
from scipy.linalg import cholesky
from sklearn.linear_model import Lars, LassoLars, lars_path

# set seed for random number generation - consistent results
np.random.seed(0)

# create underlying structures - make structures easily distinguishable
time = np.arange(1098)
imfs_synth = np.zeros((15, 1098))
for row in range(15):
    imfs_synth[row, :] = 5 * np.cos(time * (3 * (row + 1) / 1097) * 2 * np.pi)  # easily distinguishable
del row

# daily risk free rate
risk_free = (0.02 / 365)

# establish number of knots
knots = 150

# construct explainable returns from stock factors - create independent returns and then correlate
returns = np.random.normal(risk_free, risk_free, size=(1097, 5))

# create base covariance using realised variance
tickers_format = ['MSFT', 'AAPL', 'GOOGL', 'AMZN', 'TSLA']
data = yf.download(tickers_format, start="2018-12-31", end="2021-11-14")
close_data = data['Close']
date_index = pd.date_range(start='31/12/2018', end='01/01/2022')
close_data = close_data.reindex(date_index).interpolate()
close_data = close_data = close_data[::-1].interpolate()
close_data = close_data = close_data[::-1]
base_covariance = np.cov((np.log(np.asarray(close_data)[1:, :]) - np.log(np.asarray(close_data)[:-1, :])).T)
del close_data, data, date_index, tickers_format

# create correlation structure
B = np.zeros((5, 15))
for i in range(5):
    for j in range(15):
        rand_uni = np.random.uniform(0, 1)
        B[i, j] = (-1) ** (i + j) if rand_uni > 1 / 2 else 0
del i, j, rand_uni
covariance_structure = np.zeros((5, 5, 1097))
for day in range(np.shape(covariance_structure)[2]):
    covariance_structure[:, :, day] = base_covariance + \
                                      np.matmul(np.matmul(B, imfs_synth[:, day]).reshape(-1, 1),
                                                np.matmul(imfs_synth[:, day].T, B.T).reshape(1, -1))
    # finally correlate returns
    returns[day, :] = np.dot(cholesky(covariance_structure[:, :, day], lower=True), returns[day, :])
del day

cumulative_returns = np.ones((1098, 5))
cumulative_returns[1:, :] = np.cumprod(np.exp(returns), axis=0)
date_index = pd.date_range(start='31/12/2018', end='01/01/2022')
tickers = ['A', 'B', 'C', 'D', 'E']
close_data = pd.DataFrame(columns=tickers, index=date_index)
close_data.iloc[:] = cumulative_returns

plt.plot(cumulative_returns)
plt.show()

# https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_lars.html#sphx-glr-auto-examples-linear-model-plot-lasso-lars-py
reg = Lars(normalize=False)
reg.fit(X=imfs_synth[:, :-1].T, y=returns)
coef_paths = reg.coef_path_
xx = np.sum(np.abs(coef_paths[0]), axis=0)
xx /= xx[-1]

plt.plot(xx, coef_paths[0].T)
for i in xx:
    plt.plot(i * np.ones(101), np.linspace(np.min(coef_paths[0]), np.max(coef_paths[0]), 101), '--', label=i)
plt.show()

model_days = 701  # 2 years - less a month
forecast_days = np.shape(close_data)[0] - model_days - 30

spline_basis_transform = emd_basis.Basis(time_series=np.arange(model_days), time=np.arange(model_days))
spline_basis_transform = spline_basis_transform.cubic_b_spline(knots=np.linspace(0, model_days - 1, knots))

risk_return_Model = [1]
risk_return_Equal = [1]
risk_return_Covariance = [1]

for lag in range(forecast_days):
    print(lag)

    all_data = close_data.iloc[lag:int(model_days + lag + 1)]  # data window

    if lag in [0, 30, 61, 91, 122, 153, 181, 212, 242, 273, 303, 334]:

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
        all_data = imfs_synth[:, lag:int(model_days + lag + 1)].T  # use actual underlying structures
        groups = np.zeros((76, 1))  # LATER
        returns_subset = returns[lag:int(model_days + lag), :]  # extract relevant returns

        realised_covariance = np.cov(returns_subset.T)  # calculation of realised covariance

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

        x = np.asarray(all_data).T

        # calculate covariance regression matrices
        B_est, Psi_est = cov_reg_given_mean(A_est=np.zeros_like(coef), basis=spline_basis_transform,
                                            x=x[:, :-1], y=returns_subset.T,
                                            iterations=10, technique='direct', max_iter=500, groups=groups)

        variance_Model = Psi_est + np.matmul(np.matmul(B_est.T, x[:, -1]).astype(np.float64).reshape(-1, 1),
                                             np.matmul(x[:, -1].T, B_est).astype(np.float64).reshape(1, -1)).astype(np.float64)

        all_returns = returns[int(model_days + lag), :]
        all_sd = np.sqrt(np.diag(variance_Model))

        # calculate global minimum variance portfolio and maximum Sharpe ratio
        weights_covariance = global_weights(realised_covariance)
        weights_Model = rb_p_weights(variance_Model).x

        global_minimum_weights = global_weights(variance_Model)  # efficient frontier construction
        # global_minimum_weights = global_weights_long(variance_Model).x
        global_minimum_variance = global_obj_fun(global_minimum_weights, variance_Model)
        global_minimum_returns = sum(global_minimum_weights * returns[int(model_days + lag + 29), :])

        sharpe_maximum_weights = sharpe_weights(variance_Model, returns[int(model_days + lag + 29), :], risk_free)
        sharpe_maximum_variance = global_obj_fun(sharpe_maximum_weights, variance_Model)
        sharpe_maximum_returns = sum(sharpe_maximum_weights * returns[int(model_days + lag + 29), :])
        if sharpe_maximum_returns < global_minimum_returns:
            sharpe_maximum_weights = 2 * global_minimum_weights - sharpe_maximum_weights
            sharpe_maximum_variance = global_obj_fun(sharpe_maximum_weights, variance_Model)
            sharpe_maximum_returns = sum(sharpe_maximum_weights * returns[int(model_days + lag + 29), :])

        covariance_variance = global_obj_fun(weights_covariance, realised_covariance)
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
        plt.plot(efficient_frontier_sd, 2 * efficient_frontier_return[0] - efficient_frontier_return, 'k--')

        plt.scatter(all_sd, all_returns)

        plt.title(f'Efficient Frontier')
        plt.ylabel('Expected returns')
        plt.xlabel('Expected variance of returns')
        plt.legend(loc='best')
        plt.show()

    risk_return_Model.append(np.exp(sum(weights_Model * returns[int(model_days + lag + 29), :])))
    risk_return_Covariance.append(np.exp(sum(weights_covariance * returns[int(model_days + lag + 29), :])))
    risk_return_Equal.append(np.exp(sum(((1 / np.shape(close_data)[1]) * np.ones(np.shape(close_data)[1])) *
                                        returns[int(model_days + lag + 29), :])))

plt.plot(np.cumprod(risk_return_Model), label='Model')
plt.plot(np.cumprod(risk_return_Covariance), label='Realised')
plt.plot(np.cumprod(risk_return_Equal), label='Equal')
plt.legend(loc='best')
plt.show()


import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from AdvEMDpy import emd_basis

np.random.seed(3)

# pull all close data
tickers_format = ['MSFT', 'AAPL', 'GOOGL', 'AMZN', 'TSLA']
data = yf.download(tickers_format, start="2018-12-31", end="2022-01-01")
close_data = data['Close']
del data, tickers_format

# create date range and interpolate
date_index = pd.date_range(start='31/12/2018', end='01/01/2022')
close_data = close_data.reindex(date_index).interpolate()
close_data = close_data[::-1].interpolate()
close_data = close_data[::-1]
del date_index

# calculate returns and realised covariance
returns = (np.log(np.asarray(close_data)[1:, :]) -
           np.log(np.asarray(close_data)[:-1, :]))
realised_covariance = np.cov(returns.T)
risk_free = (0.02 / 365)

# calculate knots and returns subset
model_days = 731
knots = 80
forecast_days = 30
knots_vector = np.linspace(0, model_days - 1, int(knots - 6))
knots_vector = np.linspace(-knots_vector[3], 2 * (model_days - 1) - knots_vector[-4], knots)
returns_subset_forecast = returns[:model_days, :]

# calculate spline basis and mean forecast
spline_basis_transform = emd_basis.Basis(time_series=np.arange(model_days), time=np.arange(model_days))
spline_basis_transform = spline_basis_transform.cubic_b_spline(knots=knots_vector)
coef_forecast = np.linalg.lstsq(spline_basis_transform.T, returns_subset_forecast, rcond=None)[0]
mean_forecast = np.matmul(coef_forecast.T, spline_basis_transform).T

# # plot returns and means
# for i in range(5):
#     plt.plot(returns_subset_forecast[:, i])
#     plt.plot(mean_forecast[:, i])
#     plt.show()

# set hyper-parameters
ndays = 10
P = 1
Q = 1
params = np.zeros(P + Q + 1)
modelled_variance = np.zeros_like(returns_subset_forecast)

# calculate garch parameters of each stock
# averaging the parameters does not work - parameters are wildly different!
for stock in range(5):
    returns_minus_mean = returns_subset_forecast[:, stock] - mean_forecast[:, stock]
    model = arch_model(returns_minus_mean, mean='Zero', vol='GARCH', p=P, q=Q)
    model_fit = model.fit()
    modelled_variance[:, stock] = model_fit.conditional_volatility
    params += np.asarray(model_fit.params)
    print(model_fit.params)

# # not necessary, but helps to follow process
# rt = returns_subset_forecast
# import mgarch
# vol = mgarch.mgarch()
# vol.fit(rt)
# ndays = 10 # volatility of nth day
# cov_nextday = vol.predict(ndays)

returns_minus_mean = returns_subset_forecast - mean_forecast


def covregpy_mgarch(returns_matrix, p=3, q=3, days=10, correlation=False):

    if np.shape(returns_matrix)[0] < np.shape(returns_matrix)[1]:
        returns_matrix = returns_matrix.T
    if p != q:
        p = np.min(p, q)
        q = np.min(p, q)

    modelled_variance = np.zeros_like(returns_matrix)

    for stock in range(np.shape(returns_matrix)[1]):
        model = arch_model(returns_matrix[:, stock], mean='Zero', vol='GARCH', p=p, q=q)
        model_fit = model.fit()
        modelled_variance[:, stock] = model_fit.conditional_volatility

    params = minimize(custom_loglike, (0.01, 0.94),
                      args=(returns_matrix, modelled_variance),
                      bounds=((1e-6, 1), (1e-6, 1)))
    a = params.x[0]
    b = params.x[1]

    t = np.shape(returns_matrix)[0]
    q_bar = np.cov(returns_matrix.T)
    q_t = np.zeros((t, np.shape(q_bar)[0], np.shape(q_bar)[1]))
    h_t = np.zeros((t, np.shape(q_bar)[0], np.shape(q_bar)[1]))
    for q_row in range(p):
        q_t[q_row] = np.matmul(returns_matrix[q_row, :].reshape(-1, 1) / 2,
                               returns_matrix[q_row, :].reshape(1, -1) / 2)
    dts = np.zeros((t, np.shape(q_bar)[0], np.shape(q_bar)[1]))
    for var in range(t):
        dts[var] = np.diag(modelled_variance[var, :])
    dts_inv = np.zeros((t, np.shape(q_bar)[0], np.shape(q_bar)[1]))
    for var in range(t):
        dts_inv[var] = np.linalg.inv(dts[var])
    et = np.zeros((t, np.shape(q_bar)[0], np.shape(q_bar)[1]))
    for var in range(t):
        et[var] = dts_inv[var] * returns_matrix[var, :].T
    qts = np.zeros((t, np.shape(q_bar)[0], np.shape(q_bar)[1]))
    r_t = np.zeros((t, np.shape(q_bar)[0], np.shape(q_bar)[1]))
    for var in range(1, t):
        q_t[var] = (1 - a - b) * q_bar + \
                   np.sum(a * (et[var] * et[var]), axis=0) + \
                   np.sum(b * q_t[var], axis=0)
        # q_t[var] = q_bar
        qts[var] = np.linalg.inv(np.sqrt(np.diag(np.diag(q_t[var]))))
        r_t[var] = np.matmul(qts[var], np.matmul(q_t[var], qts[var]))
        h_t[var] = np.matmul(dts[var], np.matmul(r_t[var], dts[var]))

    forecasted_covariance = h_t[-1] * np.sqrt(days)

    if correlation:
        corr = np.zeros_like(forecasted_covariance)
        for i in range(np.shape(corr)[0]):
            for j in range(np.shape(corr)[1]):
                corr[i, j] = forecasted_covariance[i, j] / (np.sqrt(forecasted_covariance[i, i]) *
                                                            np.sqrt(forecasted_covariance[j, j]))
        print(corr)

    return forecasted_covariance


def custom_loglike(params, returns_matrix, modelled_variance):

    t = np.shape(returns_matrix)[0]

    q_bar = np.cov(returns_matrix.T)
    q_t = np.zeros((t, np.shape(q_bar)[0], np.shape(q_bar)[1]))
    h_t = np.zeros((t, np.shape(q_bar)[0], np.shape(q_bar)[1]))

    dts = np.zeros((t, np.shape(q_bar)[0], np.shape(q_bar)[1]))
    dts_inv = np.zeros((t, np.shape(q_bar)[0], np.shape(q_bar)[1]))
    et = np.zeros((t, np.shape(q_bar)[0], np.shape(q_bar)[1]))
    qts = np.zeros((t, np.shape(q_bar)[0], np.shape(q_bar)[1]))
    r_t = np.zeros((t, np.shape(q_bar)[0], np.shape(q_bar)[1]))

    q_t[0] = np.matmul(returns_matrix[0, :].reshape(-1, 1) / 2, returns_matrix[0, :].reshape(1, -1) / 2)

    loglike = 0
    for var in range(1, t):
        dts[var] = np.diag(modelled_variance[var, :])
        dts_inv[var] = np.linalg.inv(dts[var])
        et[var] = dts_inv[var] * returns_matrix[var, :].T
        q_t[var] = (1 - sum(params)) * q_bar + \
                   np.sum(params[0] * (et[var] * et[var]), axis=0) + \
                   np.sum(params[1] * q_t[var], axis=0)
        qts[var] = np.linalg.inv(np.sqrt(np.diag(np.diag(q_t[var]))))
        r_t[var] = np.matmul(qts[var], np.matmul(q_t[var], qts[var]))
        h_t[var] = np.matmul(dts[var], np.matmul(r_t[var], dts[var]))

        loglike = loglike + np.shape(q_bar)[0] * np.log(2 * np.pi) + \
                  2 * np.log(modelled_variance[var].sum()) + \
                  np.log(np.linalg.det(r_t[var])) + \
                  np.matmul(returns_matrix[var], (np.matmul(np.linalg.inv(h_t[var]), returns_matrix[var].T)))

    return loglike


temp = covregpy_mgarch(returns_minus_mean, p=3, q=3, days=10, correlation=True)

print(temp)

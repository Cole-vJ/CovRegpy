
import mgarch
import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model
import matplotlib.pyplot as plt

from AdvEMDpy import emd_basis

np.random.seed(3)

# data = [np.random.normal(0, np.sqrt(i*0.01)) for i in range(0, 1000)]
# n_test = 100
# train, test = data[:-n_test], data[-n_test:]
# model = arch_model(train, mean='Zero', vol='GARCH', p=15, q=15)
# # model = arch_model(train, mean='Zero', vol='ARCH', p=15)
# model_fit = model.fit()
# yhat = model_fit.forecast(horizon=n_test)
# var = [i*0.01 for i in range(0, 1000)]
# plt.plot(var[-n_test:])
# plt.plot(yhat.variance.values[-1, :])
# plt.show()

# pull all close data
tickers_format = ['MSFT', 'AAPL', 'GOOGL', 'AMZN', 'TSLA']
data = yf.download(tickers_format, start="2018-12-31", end="2021-12-01")
close_data = data['Close']
del data, tickers_format

# create date range and interpolate
date_index = pd.date_range(start='31/12/2018', end='12/01/2021')
close_data = close_data.reindex(date_index).interpolate()
close_data = close_data[::-1].interpolate()
close_data = close_data[::-1]
del date_index

# calculate returns and realised covariance
returns = (np.log(np.asarray(close_data)[1:, :]) -
           np.log(np.asarray(close_data)[:-1, :]))
realised_covariance = np.cov(returns.T)
risk_free = (0.02 / 365)

model_days = 731
knots = 80
forecast_days = 30
knots_vector = np.linspace(0, model_days - 1, int(knots - 6))
knots_vector = np.linspace(-knots_vector[3], 2 * (model_days - 1) - knots_vector[-4], knots)
returns_subset_forecast = returns[:model_days, :]

spline_basis_transform = emd_basis.Basis(time_series=np.arange(model_days), time=np.arange(model_days))
spline_basis_transform = spline_basis_transform.cubic_b_spline(knots=knots_vector)
coef_forecast = np.linalg.lstsq(spline_basis_transform.T, returns_subset_forecast, rcond=None)[0]
mean_forecast = np.matmul(coef_forecast.T, spline_basis_transform).T

# for i in range(5):
#     plt.plot(returns_subset_forecast[:, i])
#     plt.plot(mean_forecast[:, i])
#     plt.show()

# vol = mgarch.mgarch()
# vol.fit(returns_subset_forecast)
ndays = 10
# cov_nextday = vol.predict(ndays)

P = 3
Q = 3
params = np.zeros(P + Q + 1)
modelled_variance = np.zeros_like(returns_subset_forecast)

for stock in range(5):
    returns_minus_mean = returns_subset_forecast[:, stock] - mean_forecast[:, stock]
    model = arch_model(returns_minus_mean, mean='Zero', vol='GARCH', p=P, q=Q)
    model_fit = model.fit()
    modelled_variance[:, stock] = model_fit.conditional_volatility
    params += np.asarray(model_fit.params)
params /= 5

returns_minus_mean = returns_subset_forecast - mean_forecast
t = np.shape(returns_minus_mean)[0]
q_bar = np.cov(returns_minus_mean.T)
q_t = np.zeros((t, np.shape(q_bar)[0], np.shape(q_bar)[1]))
r_t = np.zeros((t, np.shape(q_bar)[0], np.shape(q_bar)[1]))
h_t = np.zeros((t, np.shape(q_bar)[0], np.shape(q_bar)[1]))
for q_row in range(P):
    q_t[q_row] = np.matmul(returns_minus_mean[q_row, :].reshape(-1, 1) / 2,
                           returns_minus_mean[q_row, :].reshape(1, -1) / 2)
dts = np.zeros((t, np.shape(q_bar)[0], np.shape(q_bar)[1]))
for var in range(t):
    dts[var] = np.diag(modelled_variance[var, :])
dts_inv = np.zeros((t, np.shape(q_bar)[0], np.shape(q_bar)[1]))
for var in range(t):
    dts_inv[var] = np.linalg.inv(dts[var])
et = np.zeros((t, np.shape(q_bar)[0], np.shape(q_bar)[1]))
for var in range(t):
    et[var] = dts_inv[var] * returns_minus_mean[var, :].T
qts = np.zeros((t, np.shape(q_bar)[0], np.shape(q_bar)[1]))
r_t = np.zeros((t, np.shape(q_bar)[0], np.shape(q_bar)[1]))
params_a_matrix = np.ones((P, np.shape(q_bar)[0], np.shape(q_bar)[1]))
params_b_matrix = np.ones((P, np.shape(q_bar)[0], np.shape(q_bar)[1]))
for param in range(P):
    params_a_matrix[param] = params[int(param + 1)] * params_a_matrix[param]
    params_b_matrix[param] = params[int(param + P + 1)] * params_b_matrix[param]
q_bar_matrix = np.zeros((P, np.shape(q_bar)[0], np.shape(q_bar)[1]))
for var in range(P, t):
    q_t[var] = (1 - sum(params[1:])) * q_bar + \
               np.sum(params_a_matrix * (et[int(var - P):var] * et[int(var - P):var]), axis=0) + \
               np.sum(params_b_matrix * q_t[int(var - P):var], axis=0)
    qts[var] = np.linalg.inv(np.sqrt(np.diag(np.diag(q_t[var]))))
    r_t[var] = np.matmul(qts[var], np.matmul(q_t[var], qts[var]))
    h_t[var] = np.matmul(dts[var], np.matmul(r_t[var], dts[var]))

forecasted_covariance = h_t[-1] * np.sqrt(ndays)


def covregpy_mgarch(returns_matrix, p=3, q=3, days=10, correlation=False):

    if np.shape(returns_matrix)[0] < np.shape(returns_matrix)[1]:
        returns_matrix = returns_matrix.T
    if p != q:
        p = np.min(p, q)
        q = np.min(p, q)

    params = np.zeros(int(1 + p + q))
    modelled_variance = np.zeros_like(returns_matrix)

    for stock in range(np.shape(returns_matrix)[1]):
        model = arch_model(returns_matrix[:, stock], mean='Zero', vol='GARCH', p=p, q=q)
        model_fit = model.fit()
        modelled_variance[:, stock] = model_fit.conditional_volatility
        params += np.asarray(model_fit.params)
    params /= np.shape(returns_matrix)[1]

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
    params_a_matrix = np.ones((p, np.shape(q_bar)[0], np.shape(q_bar)[1]))
    params_b_matrix = np.ones((p, np.shape(q_bar)[0], np.shape(q_bar)[1]))
    for param in range(p):
        params_a_matrix[param] = params[int(param + 1)] * params_a_matrix[param]
        params_b_matrix[param] = params[int(param + q + 1)] * params_b_matrix[param]
    for var in range(p, t):
        q_t[var] = (1 - sum(params[1:])) * q_bar + \
                   np.sum(params_a_matrix * (et[int(var - p):var] * et[int(var - p):var]), axis=0) + \
                   np.sum(params_b_matrix * q_t[int(var - p):var], axis=0)
        # q_t[var] = q_bar
        qts[var] = np.linalg.inv(np.sqrt(np.diag(np.diag(q_t[var]))))
        r_t[var] = np.matmul(qts[var], np.matmul(q_t[var], qts[var]))
        h_t[var] = np.matmul(dts[var], np.matmul(r_t[var], dts[var]))

    forecasted_covariance = h_t[-1] * np.sqrt(days)

    if correlation:
        corr = np.zeros_like(forecasted_covariance)
        for i in range(np.shape(corr)[0]):
            for j in range(np.shape(corr)[1]):
                corr[i, j] = forecasted_covariance[i, j] / (np.sqrt(forecasted_covariance[i, i]) * np.sqrt(forecasted_covariance[j, j]))
        print(corr)

    return forecasted_covariance


temp = covregpy_mgarch(returns_minus_mean, p=3, q=3, days=10, correlation=True)

print(forecasted_covariance - temp)

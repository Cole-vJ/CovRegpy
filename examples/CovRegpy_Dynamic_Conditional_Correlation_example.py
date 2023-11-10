
#     ________
#            /
#      \    /
#       \  /
#        \/

# Note!!! Only allowed a certain number of requests to yfinance each hour

import numpy as np
import pandas as pd
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt

from AdvEMDpy import emd_basis

# uncomment if using installed CovRegpy package in custom environment
# from CovRegpy.CovRegpy_DCC import covregpy_dcc

# uncomment if using function directly from within this downloaded GitHub package
from CovRegpy_DCC import covregpy_dcc

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
returns = (np.log(np.asarray(close_data)[1:, :]) - np.log(np.asarray(close_data)[:-1, :]))
realised_covariance = np.cov(returns.T)

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
for i in range(5):
    plt.plot(returns_subset_forecast[:, i])
    plt.plot(mean_forecast[:, i])
    plt.show()

# # not necessary, but helps to follow process
# rt = returns_subset_forecast
# import mgarch
# vol = mgarch.mgarch()
# vol.fit(rt)
# ndays = 10 # volatility of nth day
# cov_nextday = vol.predict(ndays)

returns_minus_mean = returns_subset_forecast - mean_forecast

# forecast covariance of 5 assets 10 days into the future
# optionally print correlation
forecasted_covariance = covregpy_dcc(returns_minus_mean, p=3, q=3, days=10, print_correlation=True)
# print forecasted covariance
print(forecasted_covariance)

#     ________
#            /
#      \    /
#       \  /
#        \/

# Note!!! Only allowed a certain number of requests to yfinance each hour

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# uncomment if using installed CovRegpy package in custom environment
# from CovRegpy.CovRegpy_measures import covregpy_dcc

# uncomment if using function directly from within this downloaded GitHub package
from CovRegpy_add_scripts.CovRegpy_measures import (mean_return, variance_return, value_at_risk_return,
                                                    max_draw_down_return, omega_ratio_return, sortino_ratio_return,
                                                    sharpe_ratio_return, cumulative_return)

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
returns = (np.log(np.asarray(close_data)[1:, :]) - np.log(np.asarray(close_data)[:-1, :]))
realised_covariance = np.cov(returns.T)
risk_free = (0.01 / 365)

stock_weights = np.random.uniform(0, 1, (np.shape(returns)))
stock_weights = stock_weights.T / np.sum(stock_weights, axis=1)
window_width = 90

mean_returns = mean_return(weights=stock_weights, all_returns=returns, window=window_width)
plt.title(f'Mean Returns with {window_width} Day Window')
plt.plot(mean_returns)
plt.show()

variance_returns = variance_return(weights=stock_weights, all_returns=returns, window=window_width)
plt.title(f'Variance Returns with {window_width} Day Window')
plt.plot(variance_returns)
plt.show()

value_at_risk_returns = value_at_risk_return(weights=stock_weights, all_returns=returns, window=window_width)
plt.title(f'Value-at-Risk Returns with {window_width} Day Window')
plt.plot(value_at_risk_returns)
plt.show()

max_draw_down_returns = max_draw_down_return(weights=stock_weights, all_returns=returns, window=window_width)
plt.title(f'Maximum Draw Down Returns with {window_width} Day Window')
plt.plot(max_draw_down_returns)
plt.show()

omega_ratio_returns = omega_ratio_return(weights=stock_weights, all_returns=returns, window=window_width)
plt.title(f'Omega Ratio Returns with {window_width} Day Window')
plt.plot(omega_ratio_returns)
plt.show()

sortino_ratio_returns = sortino_ratio_return(weights=stock_weights, all_returns=returns, window=window_width)
plt.title(f'Sortino Ratio Returns with {window_width} Day Window')
plt.plot(sortino_ratio_returns)
plt.show()

sharpe_ratio_returns = sharpe_ratio_return(weights=stock_weights, all_returns=returns, window=window_width)
plt.title(f'Sharpe Ratio Returns with {window_width} Day Window')
plt.plot(sharpe_ratio_returns)
plt.show()

cumulative_returns = cumulative_return(weights=stock_weights, all_returns=returns)
plt.title(f'Cumulative Returns')
plt.plot(cumulative_returns)
plt.show()
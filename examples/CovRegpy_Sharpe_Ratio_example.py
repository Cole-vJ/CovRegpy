
#     ________
#            /
#      \    /
#       \  /
#        \/

import numpy as np
import pandas as pd
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style='darkgrid')

# uncomment if using installed CovRegpy package in custom environment
# from CovRegpy.CovRegpy_sharpe import sharpe_weights, sharpe_obj_fun

# uncomment if using function directly from within this downloaded GitHub package
from CovRegpy_sharpe import sharpe_weights, sharpe_obj_fun

# pull all close data
tickers_format = ['MSFT', 'AAPL']
# tickers_format = ['MSFT', 'AAPL', 'GOOGL', 'AMZN', 'TSLA']
data = yf.download(tickers_format, start="2018-10-15", end="2021-10-16")
close_data = data['Close']
del data, tickers_format

# create date range and interpolate
date_index = pd.date_range(start='16/10/2018', end='16/10/2021')
close_data = close_data.reindex(date_index).interpolate()
close_data = close_data[::-1].interpolate()
close_data = close_data[::-1]
del date_index

# calculate returns and realised covariance
returns = (np.log(np.asarray(close_data)[1:, :]) -
           np.log(np.asarray(close_data)[:-1, :]))
realised_covariance = np.cov(returns.T)
risk_free = (0.02 / 365)

# no restrictions on global minimum variance
weights = sharpe_weights(realised_covariance, returns[-1, :], risk_free)  # approx [0.25, 0.75]
msft_weight = np.linspace(-1, 2, 3001)
aapl_weight = 1 - msft_weight
sharpe_ratio = np.zeros(3001)
for i in range(3001):
    sharpe_ratio[i] = sharpe_obj_fun(np.asarray([msft_weight[i], aapl_weight[i]]),
                                     realised_covariance, returns[-1, :], risk_free)
plt.title('Maximum Sharpe Ratio with No Restrictions (Positive Correlation)')
plt.plot(msft_weight, sharpe_ratio)
plt.xlabel('MSFT Weights')
plt.ylabel('Sharpe Ratio')
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.scatter(weights[0], sharpe_obj_fun(weights, realised_covariance, returns[-1, :], risk_free), c='r')
plt.show()

# only long weights global minimum variance
# change correlation to negative for demonstration
negative_covariance = realised_covariance.copy()
negative_covariance[0, -1] = -negative_covariance[0, -1]
negative_covariance[-1, 0] = -negative_covariance[-1, 0]
weights = sharpe_weights(negative_covariance, returns[-1, :], risk_free)  # approx [0.47, 0.53]
msft_weight = np.linspace(0, 1, 1001)
aapl_weight = 1 - msft_weight
sharpe_ratio = np.zeros(1001)
for i in range(1001):
    sharpe_ratio[i] = sharpe_obj_fun(np.asarray([msft_weight[i], aapl_weight[i]]),
                                     negative_covariance, returns[-1, :], risk_free)
plt.title('Maximum Sharpe Ratio with No Restrictions (Negative Correlation)')
plt.plot(msft_weight, sharpe_ratio)
plt.xlabel('MSFT Weights')
plt.ylabel('Sharpe Ratio')
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.scatter(weights[0], sharpe_obj_fun(weights, negative_covariance, returns[-1, :], risk_free), c='r')
plt.show()

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# uncomment if using installed CovRegpy package in custom environment
# from CovRegpy_add_scripts.CovRegpy.CovRegpy_SSD import CovRegpy_ssd

# uncomment if using function directly from within this downloaded GitHub package
from CovRegpy_add_scripts.CovRegpy_utilities_efficient_frontier import (global_minimum_information,
                                                                        sharpe_information,
                                                                        efficient_frontier,
                                                                        pca_information)

# pull all close data
tickers_format = ['MSFT', 'AAPL', 'GOOGL', 'AMZN', 'TSLA']
data = yf.download(tickers_format, start="2018-12-31", end="2021-12-01")
close_data = data['Close']
del data, tickers_format

# create date range and interpolate
date_index = pd.date_range(start='31/12/2018', end='01/12/2021')
close_data = close_data.reindex(date_index).interpolate()
close_data = close_data[::-1].interpolate()
close_data = close_data[::-1]
del date_index

# calculate returns and realised covariance
returns = (np.log(np.asarray(close_data)[1:, :]) -
           np.log(np.asarray(close_data)[:-1, :]))
realised_covariance = np.cov(returns.T)
risk_free = (0.01 / 365)

# global minimum variance
global_minimum_weights, global_minimum_sd, global_minimum_returns = \
    global_minimum_information(realised_covariance, returns[-1, :])

# sharpe maximum ratio
sharpe_maximum_weights, sharpe_maximum_sd, sharpe_maximum_returns = \
    sharpe_information(realised_covariance, returns[-1, :],
                       global_minimum_weights, global_minimum_returns, risk_free)

# efficient frontier
efficient_frontier_sd, efficient_frontier_returns = \
    efficient_frontier(global_minimum_weights, global_minimum_returns, global_minimum_sd,
                       sharpe_maximum_weights, sharpe_maximum_returns, sharpe_maximum_sd,
                       realised_covariance, n=101)

# principle component analysis
pca_weights, pca_sd, pca_returns = pca_information(realised_covariance, returns[-1, :], factors=3)

# plots
plt.title('Efficient Frontier and Principle Portfolio')
plt.scatter(global_minimum_sd, global_minimum_returns, label='global minimum variance')
plt.scatter(sharpe_maximum_sd, sharpe_maximum_returns, label='maximum sharpe ratio')
plt.plot(efficient_frontier_sd, efficient_frontier_returns, label='efficient frontier')
plt.scatter(pca_sd, pca_returns, label='principle portfolio')
plt.ylabel('Returns')
plt.xlabel('Variance')
plt.legend(loc='lower right')
plt.show()
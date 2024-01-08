
#     ________
#            /
#      \    /
#       \  /
#        \/

# Note!!! Only allowed a certain number of requests to yfinance each hour

import textwrap
import numpy as np
import pandas as pd
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

sns.set(style='darkgrid')

# uncomment if using installed CovRegpy package in custom environment
# from CovRegpy_add_scripts.CovRegpy.CovRegpy_PCA import pca_func

# uncomment if using function directly from within this downloaded GitHub package
from CovRegpy_add_scripts.CovRegpy_PCA import pca_func

# store required stock tickers
tickers_format = ['MSFT', 'AAPL', 'GOOGL', 'AMZN', 'TSLA']
# send request for daily stock data between two date (if available otherwise NANs)
# can suppress progress output by changing corresponding paramter value to False
data = yf.download(tickers_format, start="2018-12-31", end="2021-12-01", progress=True)
month_ends = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
month_vector = np.asarray(['January', 'February', 'March', 'April', 'May', 'June',
                           'July', 'August', 'September', 'October', 'November', 'December', 'January'])
# isolate only the close data
close_data = data['Close']
# delete unnecessary variables
del data, tickers_format

# create pandas date range
date_index = pd.date_range(start='31/12/2018', end='31/01/2020')
# interpolate forwards
close_data = close_data.reindex(date_index).interpolate()
# reverse order, interpolate backwards, and reverse order again
close_data = close_data[::-1].interpolate()[::-1]
del date_index

# calculate logarithmic returns
returns = (np.log(np.asarray(close_data)[1:, :]) -
           np.log(np.asarray(close_data)[:-1, :]))

pca_storage = np.zeros((12, 5))

for annual_interval in np.arange(12):

    pca_11 = PCA(n_components=5)
    pca_11.fit(returns[month_ends[int(annual_interval)]:month_ends[int(annual_interval + 1)], :].T)
    pca_components_11 = pca_11.components_
    pca_singular_values_11 = pca_11.singular_values_
    # print(np.cumsum(pca_singular_values_11 ** 2) / np.sum(pca_singular_values_11 ** 2))

    pca_storage[annual_interval, :] = np.cumsum(pca_singular_values_11 ** 2) / np.sum(pca_singular_values_11 ** 2)

fig, axs = plt.subplots(1, 1, sharey=True, figsize=(10, 8))
plt.plot(pca_storage.T)
plt.plot(np.linspace(0, 4, 100), 0.80 * np.ones(100), 'k--')
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0%', '20%', '40%', '60%', '80%', '100%'])
plt.xticks([0, 1, 2, 3, 4], [1, 2, 3, 4, 5])
plt.title(textwrap.fill('Percentage of Cumulative Variation Explained by the Sequential Principal Components', 45),
          fontsize=16)
plt.ylabel('Percentage', fontsize=14)
plt.xlabel('Components', fontsize=14)
box_0 = axs.get_position()
axs.set_position([box_0.x0 - 0.02, box_0.y0 + 0.0, box_0.width * 1.14, box_0.height])
plt.show()

fig, axs = plt.subplots(1, 1, sharey=True, figsize=(10, 8))
plt.scatter(np.arange(1, 13), pca_storage[:, 1])
plt.plot(np.linspace(0, 12, 100), 0.80 * np.ones(100), 'k--')
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0%', '20%', '40%', '60%', '80%', '100%'])
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
           ['31-12-2018', '31-01-2019', '28-02-2019', '31-03-2019', '30-04-2019', '31-05-2019', '30-06-2019',
            '31-07-2019', '31-08-2019', '30-09-2019', '31-10-2019', '30-11-2019', '31-12-2019'], rotation=30)
plt.title(textwrap.fill('Percentage of Cumulative Variation Explained by the First Two Principal Components over the Previous Monthly Window', 60),
          fontsize=16)
plt.ylabel('Percentage', fontsize=14)
plt.xlabel('End of Annual Windows', fontsize=14)
box_0 = axs.get_position()
axs.set_position([box_0.x0 - 0.02, box_0.y0 + 0.02, box_0.width * 1.14, box_0.height])
plt.show()

# We showed above that in each month over 80% of the variation is explained by the first two principal components
# We now calculate the weights for each month using the pca_func script

for annual_interval in np.arange(12):

    # calculate realised covariance of logarithmic returns
    realised_covariance = np.cov(returns[month_ends[int(annual_interval)]:month_ends[int(annual_interval + 1)], :].T)
    # calculate weight per stock per component
    pca_weights = pca_func(cov=realised_covariance, n_components=2)
    if annual_interval == 11:
        print('Weights for {} 2020 based on PCA of {} 2019 using two '
              'components: '.format(month_vector[int(annual_interval + 1)],
                                    month_vector[int(annual_interval)]))
    else:
        print('Weights for {} 2019 based on PCA of {} 2019 using two '
              'components: '.format(month_vector[int(annual_interval + 1)],
                                    month_vector[int(annual_interval)]))
    # print total weight
    print(np.sum(pca_weights, axis=1))

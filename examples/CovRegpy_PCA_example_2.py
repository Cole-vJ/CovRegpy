
#     ________
#            /
#      \    /
#       \  /
#        \/

import textwrap
import numpy as np
import pandas as pd
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

sns.set(style='darkgrid')

# uncomment if using installed CovRegpy package in custom environment
# from CovRegpy.CovRegpy_PCA import pca_func

# uncomment if using function directly from within this downloaded GitHub package
from CovRegpy_PCA import pca_func

# load 11 sector indices
sector_11_indices = pd.read_csv('../S&P500_Data/sp_500_11_sector_indices.csv', header=0)
sector_11_indices = sector_11_indices.set_index(['Unnamed: 0'])

# sector numpy array
sector_11_indices_array = np.vstack((np.zeros((1, 11)), np.asarray(sector_11_indices)))

# construct portfolios
end_of_month_vector = np.asarray([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31,
                                  31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31,
                                  31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31,
                                  31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31,
                                  31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
end_of_month_vector_cumsum = np.cumsum(end_of_month_vector)
month_vector = np.asarray(['January', 'February', 'March', 'April', 'May', 'June',
                           'July', 'August', 'September', 'October', 'November', 'December'])

pca_storage = np.zeros((48, 11))

# how many months to consider when calculating realised covariance
months = 12

for day in range(len(end_of_month_vector_cumsum[:-int(months + 1)])):

    pca_11 = PCA(n_components=11)
    pca_11.fit(sector_11_indices_array[end_of_month_vector_cumsum[int(day)]:
                                       end_of_month_vector_cumsum[int(day + 1)], :].T)
    pca_components_11 = pca_11.components_
    pca_singular_values_11 = pca_11.singular_values_
    print(np.cumsum(pca_singular_values_11 ** 2) / np.sum(pca_singular_values_11 ** 2))

    pca_storage[day, :] = np.cumsum(pca_singular_values_11 ** 2) / np.sum(pca_singular_values_11 ** 2)

fig, axs = plt.subplots(1, 1, sharey=True, figsize=(10, 8))
plt.plot(pca_storage.T)
plt.plot(np.linspace(0, 10, 100), 0.65 * np.ones(100), 'k--')
plt.yticks([0, 0.2, 0.4, 0.6, 0.65, 0.8, 1.0], ['0%', '20%', '40%', '60%', '65%', '80%', '100%'])
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
plt.title(textwrap.fill('Percentage of Cumulative Variation Explained by the Sequential Principal Components', 45),
          fontsize=16)
plt.ylabel('Percentage', fontsize=14)
plt.xlabel('Components', fontsize=14)
box_0 = axs.get_position()
axs.set_position([box_0.x0 - 0.02, box_0.y0 + 0.0, box_0.width * 1.14, box_0.height])
plt.show()

fig, axs = plt.subplots(1, 1, sharey=True, figsize=(10, 8))
plt.scatter(np.arange(48), pca_storage[:, 2])
plt.plot(np.linspace(0, 47, 100), 0.65 * np.ones(100), 'k--')
plt.yticks([0, 0.2, 0.4, 0.6, 0.65, 0.8, 1.0], ['0%', '20%', '40%', '60%', '65%', '80%', '100%'])
plt.xticks([0, 12, 24, 36, 47], ['31-12-2017', '31-12-2018', '31-12-2019', '31-12-2020', '30-11-2021'])
plt.title(textwrap.fill('Percentage of Cumulative Variation Explained by the First Three Principal Components over the Previous Annual Window', 60),
          fontsize=16)
plt.ylabel('Percentage', fontsize=14)
plt.xlabel('End of Annual Windows', fontsize=14)
box_0 = axs.get_position()
axs.set_position([box_0.x0 - 0.02, box_0.y0 + 0.0, box_0.width * 1.14, box_0.height])
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

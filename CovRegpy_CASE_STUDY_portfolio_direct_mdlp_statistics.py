
# Case Study - MDLP compare

import csv
import textwrap
import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from CovRegpy_utilities import efficient_frontier, global_minimum_forward_applied_information, \
    sharpe_forward_applied_information, pca_forward_applied_information, \
    global_minimum_forward_applied_information_long, \
    sharpe_forward_applied_information_summation_restriction

from CovRegpy_measures import cumulative_return, mean_return, variance_return, value_at_risk_return, \
    max_draw_down_return, omega_ratio_return, sortino_ratio_return, sharpe_ratio_return

from CovRegpy_RPP import risk_parity_weights_summation_restriction

from CovRegpy_DCC import covregpy_dcc

np.random.seed(0)

sns.set(style='darkgrid')

# create S&P 500 index
sp500_close = pd.read_csv('S&P500_Data/sp_500_close_5_year.csv', header=0)
sp500_close = sp500_close.set_index(['Unnamed: 0'])
sp500_market_cap = pd.read_csv('S&P500_Data/sp_500_market_cap_5_year.csv', header=0)
sp500_market_cap = sp500_market_cap.set_index(['Unnamed: 0'])

sp500_returns = np.log(np.asarray(sp500_close)[1:, :] / np.asarray(sp500_close)[:-1, :])
weights = np.asarray(sp500_market_cap) / np.tile(np.sum(np.asarray(sp500_market_cap), axis=1).reshape(-1, 1), (1, 505))
sp500_returns = np.sum(sp500_returns * weights[:-1, :], axis=1)[365:]
sp500_proxy = np.append(1, np.exp(np.cumsum(sp500_returns)))

# load 11 sector indices
sector_11_indices = pd.read_csv('S&P500_Data/sp_500_11_sector_indices.csv', header=0)
sector_11_indices = sector_11_indices.set_index(['Unnamed: 0'])

# approximate daily treasury par yield curve rates for 3 year bonds
risk_free = (0.01 / 365)  # daily risk free rate

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
year_vector = np.asarray(['2017', '2018', '2019', '2020', '2021'])

# minimum variance portfolio
sector_11_indices_array = sector_11_indices_array[1:, :]

months = 12

sp500_close = pd.read_csv('S&P500_Data/sp_500_close_5_year.csv', header=0)
sp500_close = sp500_close.set_index(['Unnamed: 0'])
sp500_market_cap = pd.read_csv('S&P500_Data/sp_500_market_cap_5_year.csv', header=0)
sp500_market_cap = sp500_market_cap.set_index(['Unnamed: 0'])

sp500_returns = np.log(np.asarray(sp500_close)[1:, :] / np.asarray(sp500_close)[:-1, :])
weights_sp500 = np.asarray(sp500_market_cap) / np.tile(np.sum(np.asarray(sp500_market_cap), axis=1).reshape(-1, 1),
                                                       (1, 505))

weight_matrix_global_minimum = np.zeros_like(sector_11_indices_array)
weight_matrix_maximum_sharpe_ratio = np.zeros_like(sector_11_indices_array)
weight_matrix_pca = np.zeros_like(sector_11_indices_array)
weight_matrix_dcc = np.zeros_like(sector_11_indices_array)
weight_matrix_realised = np.zeros_like(sector_11_indices_array)

values = 250

for i in range(values):

    weight_matrix_high = pd.read_csv(
        '/home/cole/Desktop/Cole/Cole Documents/CovRegpy/CovRegpy/weights/direct_high_weights_{}.csv'.format(int(i)),
        header=0)
    weight_matrix_high = np.asarray(weight_matrix_high.set_index(['Unnamed: 0']))
    weight_matrix_mid = pd.read_csv(
        '/home/cole/Desktop/Cole/Cole Documents/CovRegpy/CovRegpy/weights/direct_mid_weights_{}.csv'.format(int(i)),
        header=0)
    weight_matrix_mid = np.asarray(weight_matrix_mid.set_index(['Unnamed: 0']))
    weight_matrix_low = pd.read_csv(
        '/home/cole/Desktop/Cole/Cole Documents/CovRegpy/CovRegpy/weights/direct_low_weights_{}.csv'.format(int(i)),
        header=0)
    weight_matrix_low = np.asarray(weight_matrix_low.set_index(['Unnamed: 0']))

    cumulative_returns_mdlp_high = cumulative_return(weight_matrix_high[:end_of_month_vector_cumsum[48]].T,
                                                     sector_11_indices_array[end_of_month_vector_cumsum[12]:].T)
    cumulative_returns_mdlp_mid = cumulative_return(weight_matrix_mid[:end_of_month_vector_cumsum[48]].T,
                                                    sector_11_indices_array[end_of_month_vector_cumsum[12]:].T)
    cumulative_returns_mdlp_low = cumulative_return(weight_matrix_low[:end_of_month_vector_cumsum[48]].T,
                                                    sector_11_indices_array[end_of_month_vector_cumsum[12]:].T)

    try:
        high_freq_cum_returns = np.vstack((high_freq_cum_returns, cumulative_returns_mdlp_high))
    except:
        high_freq_cum_returns = cumulative_returns_mdlp_high.copy()
    try:
        mid_freq_cum_returns = np.vstack((mid_freq_cum_returns, cumulative_returns_mdlp_mid))
    except:
        mid_freq_cum_returns = cumulative_returns_mdlp_mid.copy()
    try:
        low_freq_cum_returns = np.vstack((low_freq_cum_returns, cumulative_returns_mdlp_low))
    except:
        low_freq_cum_returns = cumulative_returns_mdlp_low.copy()

x = np.linspace(0.05, 0.15, 100)
realised_variance_returns = np.log(1.4849082851150919)/4
dcc_returns = np.log(1.4668328786939422)/4
sp500_proxy_returns = np.log(1.6552954098543056)/4
minimum_var_returns = np.log(1.3587817911842188)/4
pca_returns = np.log(1.4897923114940013)/4

def normal(mu, sigma, x):
    return (1/np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(-((mu - x)**2)/(sigma ** 2))

ax = plt.subplot(111)
plt.gcf().subplots_adjust(bottom=0.15)
plt.title(textwrap.fill('Annualised Returns of Benchmark Portfolios versus Risk Parity Portfolios', 45))
plt.plot(realised_variance_returns * np.ones(100), np.linspace(0, 25, 100), '--', label='Realised covariance',
         Linewidth=2)
plt.plot(dcc_returns * np.ones(100), np.linspace(0, 25, 100), '--', label='DDC-GARCH', Linewidth=2)
plt.plot(sp500_proxy_returns * np.ones(100), np.linspace(0, 25, 100), '--', label='S&P 500 Proxy', Linewidth=2)
plt.plot(minimum_var_returns * np.ones(100), np.linspace(0, 25, 100), '--', label='Minimum variance', Linewidth=2)
plt.plot(pca_returns * np.ones(100), np.linspace(0, 25, 100), '--', label='PCA', Linewidth=2)
plt.plot(x, normal(np.log(np.mean(high_freq_cum_returns, axis=0)[-1]) / 4,
                   np.std(np.log(high_freq_cum_returns) / 4, axis=0)[-1], x), label=textwrap.fill('High frequency long/short', 10))
plt.plot(x, normal(np.log(np.mean(mid_freq_cum_returns, axis=0)[-1]) / 4,
                   np.std(np.log(mid_freq_cum_returns) / 4, axis=0)[-1], x), label=textwrap.fill('Mid frequencies long/short', 10))
plt.plot(x, normal(np.log(np.mean(low_freq_cum_returns, axis=0)[-1]) / 4,
                   np.std(np.log(low_freq_cum_returns) / 4, axis=0)[-1], x), label=textwrap.fill('Low frequency long/short', 10))
plt.xlabel('Annualised Returns')
plt.ylabel('Density')
box_0 = ax.get_position()
ax.set_position([box_0.x0 - 0.04, box_0.y0, box_0.width * 0.84, box_0.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
plt.savefig('figures/Annualised_returns.png')
plt.show()

print('High frequency returns mean: {}'.format(np.log(np.mean(high_freq_cum_returns, axis=0)[-1]) / 4))
print('High frequency returns sd: {}'.format(np.std(np.log(high_freq_cum_returns) / 4, axis=0)[-1]))
print('High frequency returns skewness: {}'.format(sp.stats.skew(np.log(high_freq_cum_returns) / 4, axis=0)[-1]))
print('Mid frequency returns mean: {}'.format(np.log(np.mean(mid_freq_cum_returns, axis=0)[-1]) / 4))
print('Mid frequency returns sd: {}'.format(np.std(np.log(mid_freq_cum_returns) / 4, axis=0)[-1]))
print('Mid frequency returns skewness: {}'.format(sp.stats.skew(np.log(mid_freq_cum_returns) / 4, axis=0)[-1]))
print('Low frequency returns mean: {}'.format(np.log(np.mean(low_freq_cum_returns, axis=0)[-1]) / 4))
print('Low frequency returns sd: {}'.format(np.std(np.log(low_freq_cum_returns) / 4, axis=0)[-1]))
print('Low frequency returns skewness: {}'.format(sp.stats.skew(np.log(low_freq_cum_returns) / 4, axis=0)[-1]))

plt.plot(np.mean(high_freq_cum_returns, axis=0), 'b-')
plt.plot(np.mean(mid_freq_cum_returns, axis=0), c='orange')
plt.plot(np.mean(low_freq_cum_returns, axis=0), 'g-')
plt.plot(np.mean(high_freq_cum_returns, axis=0) + 2 * np.std(high_freq_cum_returns, axis=0), 'b--')
plt.plot(np.mean(high_freq_cum_returns, axis=0) - 2 * np.std(high_freq_cum_returns, axis=0), 'b--')
plt.plot(np.mean(mid_freq_cum_returns, axis=0) + 2 * np.std(mid_freq_cum_returns, axis=0), '--', c='orange')
plt.plot(np.mean(mid_freq_cum_returns, axis=0) - 2 * np.std(mid_freq_cum_returns, axis=0), '--', c='orange')
plt.plot(np.mean(low_freq_cum_returns, axis=0) + 2 * np.std(low_freq_cum_returns, axis=0), 'g--')
plt.plot(np.mean(low_freq_cum_returns, axis=0) - 2 * np.std(low_freq_cum_returns, axis=0), 'g--')
plt.show()



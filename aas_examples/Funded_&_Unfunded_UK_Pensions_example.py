
#     ________
#            /
#      \    /
#       \  /
#        \/

import numpy as np
import pandas as pd
from pandas import DatetimeIndex
import yfinance as yf
import seaborn as sns
from AdvEMDpy import AdvEMDpy
import pandas_datareader as pdr
import matplotlib.pyplot as plt

from CovRegpy import cov_reg_given_mean
from CovRegpy_RPP import equal_risk_parity_weights_long_restriction

sns.set(style='darkgrid')

# All uncommented code below constructs the indices - saved and reloaded

# uk_ftse100_tickers = pd.read_html('https://en.wikipedia.org/wiki/FTSE_100_Index')[4]
# sectors = uk_ftse100_tickers['FTSE Industry Classification Benchmark sector[14]']

# tickers_format = [col + '.L' for col in np.asarray(uk_ftse100_tickers['Ticker'])]
# data = yf.download(tickers_format, start="2003-09-30", end="2023-11-01")
# close_data = data['Close']
# del data, tickers_format, uk_ftse100_tickers

# date_index = pd.date_range(start='30/09/2003', end='31/10/2023')
# close_data = close_data.reindex(date_index).interpolate()
# sectors_subset = sectors[np.invert(np.asarray(close_data.isna().iloc[0]))]
# close_data = close_data.dropna(axis=1)
# date_index = pd.date_range(start='30/09/2003', end='31/10/2023')
# close_data = close_data.reindex(date_index)
# close_data = close_data[::-1].interpolate()
# close_data = close_data[::-1]
# del date_index, sectors

# returns = (np.log(np.asarray(close_data)[1:, :]) - np.log(np.asarray(close_data)[:-1, :]))
# unique_sectors = np.unique(np.asarray(sectors_subset))
# index_returns = np.zeros((np.shape(returns)[0], len(unique_sectors)))

# count_col = 0

# for sector_unique in unique_sectors:
#     count_stocks_in_subset = 0
#     count_stocks = 0
#     for sector in np.asarray(sectors_subset):
#         if sector_unique == sector:
#             index_returns[:, count_col] += returns[:, count_stocks]
#             count_stocks_in_subset += 1
#         count_stocks += 1
#     index_returns[:, count_col] = index_returns[:, count_col] / count_stocks_in_subset
#     count_col += 1
# returns_dates = close_data.index[1:]
# index_returns = pd.DataFrame(index_returns, columns=unique_sectors, index=returns_dates)
# del close_data, count_col, count_stocks, count_stocks_in_subset, returns, sector, sector_unique, sectors_subset, (
#     unique_sectors, returns_dates)

# index_returns.to_csv('../FTSE_100_Data/ftse_indices.csv')

index_returns = pd.read_csv('../FTSE_100_Data/ftse_indices.csv', header=0)
index_returns = index_returns.set_index('Unnamed: 0')
index_returns.index = pd.to_datetime(index_returns.index)

uk_gdp = pdr.fred.FredReader(symbols='UKNGDP', start="2003-09-30", end="2023-11-01")
uk_gdp = uk_gdp.read()
uk_gdp_returns = np.log(np.asarray(uk_gdp)[1:, :]) - np.log(np.asarray(uk_gdp)[:-1, :])
quarter_dates = uk_gdp.index

# pull FTSE 100 sector indices

# knots = 20
#
# x_high = {}
# x_mid = {}
# x_low = {}
#
# returns_high_storage = np.zeros((int(len(quarter_dates) - 3), 1))
# returns_mid_storage = np.zeros((int(len(quarter_dates) - 3), 1))
# returns_low_storage = np.zeros((int(len(quarter_dates) - 3), 1))
# returns_equal_storage = np.zeros((int(len(quarter_dates) - 3), 1))
# returns_gdp_storage = np.zeros((int(len(quarter_dates) - 3), 1))
#
# for i in range(int(len(quarter_dates) - 2)):
#
#     del x_high, x_mid, x_low
#
#     subset_closing_data = index_returns.reindex(pd.date_range(start=quarter_dates[i], end=quarter_dates[int(i + 1)]))
#
#     for signal in range(np.shape(np.asarray(index_returns))[1]):
#         emd = AdvEMDpy.EMD(time_series=np.asarray(np.asarray(subset_closing_data)[:, signal]),
#                            time=np.arange(np.shape(np.asarray(subset_closing_data))[0]))
#         imfs, _, _, _, _, _, _ = \
#             emd.empirical_mode_decomposition(
#                 knot_envelope=np.linspace(-12, np.shape(np.asarray(subset_closing_data))[0] + 12, knots),
#                 matrix=True, stop_crit_threshold=0.1, max_imfs=3, max_internal_iter=10, verbose=False)
#
#         print(signal)
#
#         try:
#             imfs = imfs[1:, :]
#         except:
#             pass
#         try:
#             x_high = np.vstack((imfs[0, :], x_high))
#         except:
#             x_high = imfs[0, :].copy()
#         try:
#             x_mid = np.vstack((imfs[1, :], x_mid))
#         except:
#             x_mid = imfs[1, :].copy()
#         try:
#             x_low = np.vstack((imfs[2, :], x_low))
#         except:
#             try:
#                 x_low = imfs[2, :].copy()
#             except:
#                 try:
#                     x_low = np.vstack((1e-5 * np.ones_like(imfs[0, :]), x_low))
#                 except:
#                     x_low = 1e-5 * np.ones_like(imfs[0, :].copy())
#
#     # calculate y - same for both imf and ssa
#     y = index_returns.reindex(pd.date_range(start=quarter_dates[int(i + 1)], end=quarter_dates[int(i + 2)]))
#     y = y.T
#
#     y = np.asarray(y)
#
#     diff = 0
#     # make extended model 'x' and 'y' same size:
#     if np.shape(x_high)[1] != np.shape(y)[1]:
#         diff = int(np.abs(np.shape(y)[1] - np.shape(x_high)[1]))
#         if np.shape(x_high)[1] < np.shape(y)[1]:
#             y = y[:, :np.shape(x_high)[1]]
#         elif np.shape(y)[1] < np.shape(x_high)[1]:
#             x_high = x_high[:, :np.shape(y)[1]]
#             x_mid = x_mid[:, :np.shape(y)[1]]
#             x_low = x_low[:, :np.shape(y)[1]]
#
#     if i > 0:
#
#         cov_forecast_high = np.zeros((np.shape(x_high)[0], np.shape(x_high)[0], np.shape(x_high)[1]))
#         cov_forecast_mid = np.zeros((np.shape(x_mid)[0], np.shape(x_mid)[0], np.shape(x_mid)[1]))
#         cov_forecast_low = np.zeros((np.shape(x_low)[0], np.shape(x_low)[0], np.shape(x_low)[1]))
#
#         for dt in range(np.shape(x_high)[1]):
#             cov_forecast_high[:, :, dt] = Psi_est_high + np.matmul(np.matmul(B_est_high, x_high[:, dt].reshape(-1, 1)),
#                                                                    np.matmul(B_est_high, x_high[:, dt].reshape(-1, 1)).T)
#             cov_forecast_mid[:, :, dt] = Psi_est_mid + np.matmul(np.matmul(B_est_mid, x_mid[:, dt].reshape(-1, 1)),
#                                                                  np.matmul(B_est_mid, x_mid[:, dt].reshape(-1, 1)).T)
#             cov_forecast_low[:, :, dt] = Psi_est_low + np.matmul(np.matmul(B_est_low, x_low[:, dt].reshape(-1, 1)),
#                                                                  np.matmul(B_est_low, x_low[:, dt].reshape(-1, 1)).T)
#
#         median_cov_forecast_high = np.median(cov_forecast_high, axis=2)
#         median_cov_forecast_mid = np.median(cov_forecast_mid, axis=2)
#         median_cov_forecast_low = np.median(cov_forecast_low, axis=2)
#
#         rpp_weights_high = equal_risk_parity_weights_long_restriction(median_cov_forecast_high).x
#         rpp_weights_mid = equal_risk_parity_weights_long_restriction(median_cov_forecast_mid).x
#         rpp_weights_low = equal_risk_parity_weights_long_restriction(median_cov_forecast_low).x
#
#         returns_high = np.matmul(rpp_weights_high, np.sum(y, axis=1))
#         returns_mid = np.matmul(rpp_weights_mid, np.sum(y, axis=1))
#         returns_low = np.matmul(rpp_weights_low, np.sum(y, axis=1))
#         returns_equal = np.matmul(np.ones_like(rpp_weights_low) / len(rpp_weights_low), np.sum(y, axis=1))
#         returns_gdp = uk_gdp_returns[int(i + 1), 0]
#
#         returns_high_storage[int(i - 1)] = returns_high
#         returns_mid_storage[int(i - 1)] = returns_mid
#         returns_low_storage[int(i - 1)] = returns_low
#         returns_equal_storage[int(i - 1)] = returns_equal
#         returns_gdp_storage[int(i - 1)] = returns_gdp
#
#     # calculate 'A_est'
#     A_est = np.zeros((2, np.shape(y)[0]))
#     spline_basis = np.zeros((2, np.shape(x_high)[1]))
#
#     B_est_high, Psi_est_high = \
#         cov_reg_given_mean(A_est=A_est, basis=spline_basis, x=x_high, y=np.asarray(y), iterations=10)
#     B_est_mid, Psi_est_mid = \
#         cov_reg_given_mean(A_est=A_est, basis=spline_basis, x=x_mid, y=np.asarray(y), iterations=10)
#     B_est_low, Psi_est_low = \
#         cov_reg_given_mean(A_est=A_est, basis=spline_basis, x=x_low, y=np.asarray(y), iterations=10)
#
# returns_high_storage = pd.DataFrame(returns_high_storage, index=uk_gdp.index[3:], columns=['Quaterly returns'])
# returns_mid_storage = pd.DataFrame(returns_mid_storage, index=uk_gdp.index[3:], columns=['Quaterly returns'])
# returns_low_storage = pd.DataFrame(returns_low_storage, index=uk_gdp.index[3:], columns=['Quaterly returns'])
# returns_equal_storage = pd.DataFrame(returns_equal_storage, index=uk_gdp.index[3:], columns=['Quaterly returns'])
# returns_gdp_storage = pd.DataFrame(returns_gdp_storage, index=uk_gdp.index[3:], columns=['Quaterly returns'])
# returns_high_storage.to_csv('../FTSE_100_Data/ftse_indices_high_freq_weights.csv')
# returns_mid_storage.to_csv('../FTSE_100_Data/ftse_indices_mid_freq_weights.csv')
# returns_low_storage.to_csv('../FTSE_100_Data/ftse_indices_low_freq_weights.csv')
# returns_equal_storage.to_csv('../FTSE_100_Data/ftse_indices_equal_freq_weights.csv')
# returns_gdp_storage.to_csv('../FTSE_100_Data/ftse_indices_gdp_freq_weights.csv')

returns_high_storage = pd.read_csv('../FTSE_100_Data/ftse_indices_high_freq_weights.csv', header=0, index_col='DATE')
returns_mid_storage = pd.read_csv('../FTSE_100_Data/ftse_indices_mid_freq_weights.csv', header=0, index_col='DATE')
returns_low_storage = pd.read_csv('../FTSE_100_Data/ftse_indices_low_freq_weights.csv', header=0, index_col='DATE')
returns_equal_storage = pd.read_csv('../FTSE_100_Data/ftse_indices_equal_freq_weights.csv', header=0, index_col='DATE')
returns_gdp_storage = pd.read_csv('../FTSE_100_Data/ftse_indices_gdp_freq_weights.csv', header=0, index_col='DATE')

print('High-frequency mean returns: {}'.format(np.round(np.mean(np.asarray(returns_high_storage)), 4)))
print('Mid-frequency mean returns: {}'.format(np.round(np.mean(np.asarray(returns_mid_storage)), 4)))
print('Low-frequency mean returns: {}'.format(np.round(np.mean(np.asarray(returns_low_storage)), 4)))
print('Equal weighting mean returns: {}'.format(np.round(np.mean(np.asarray(returns_equal_storage)), 4)))
print('FTSE 100 mean returns: {}'.format(np.round(np.mean(np.asarray(returns_gdp_storage)), 4)))

print('High-frequency variance returns: {}'.format(np.round(np.var(np.asarray(returns_high_storage)), 8)))
print('Mid-frequency variance returns: {}'.format(np.round(np.var(np.asarray(returns_mid_storage)), 8)))
print('Low-frequency variance returns: {}'.format(np.round(np.var(np.asarray(returns_low_storage)), 8)))
print('Equal weighting variance returns: {}'.format(np.round(np.var(np.asarray(returns_equal_storage)), 8)))
print('FTSE 100 variance returns: {}'.format(np.round(np.var(np.asarray(returns_gdp_storage)), 8)))

print('High-frequency VaR returns: {}'.format(np.round(np.quantile(np.asarray(returns_high_storage), 0.05), 8)))
print('Mid-frequency VaR returns: {}'.format(np.round(np.quantile(np.asarray(returns_mid_storage), 0.05), 8)))
print('Low-frequency VaR returns: {}'.format(np.round(np.quantile(np.asarray(returns_low_storage), 0.05), 8)))
print('Equal weighting VaR returns: {}'.format(np.round(np.quantile(np.asarray(returns_equal_storage), 0.05), 8)))
print('FTSE 100 VaR returns: {}'.format(np.round(np.quantile(np.asarray(returns_gdp_storage), 0.05), 8)))

print('High-frequency CVaR returns: {}'.format(np.round(np.mean(
    returns_high_storage[np.quantile(np.asarray(returns_high_storage),
                                     0.05) > returns_high_storage]), 8)))
print('Mid-frequency CVaR returns: {}'.format(np.round(np.mean(
    returns_mid_storage[np.quantile(np.asarray(returns_mid_storage),
                                    0.05) > returns_mid_storage]), 8)))
print('Low-frequency CVaR returns: {}'.format(np.round(np.mean(
    returns_low_storage[np.quantile(np.asarray(returns_low_storage),
                                    0.05) > returns_low_storage]), 8)))
print('Equal weighting CVaR returns: {}'.format(np.round(np.mean(
    returns_equal_storage[np.quantile(np.asarray(returns_equal_storage),
                                      0.05) > returns_equal_storage]), 8)))
print('FTSE 100 CVaR returns: {}'.format(np.round(np.mean(
    returns_gdp_storage[np.quantile(np.asarray(returns_gdp_storage),
                                    0.05) > returns_gdp_storage]), 8)))

print('High-frequency MDD returns: {}'.format(np.round(
    (np.min(returns_high_storage) - np.max(returns_high_storage)) / np.max(returns_high_storage), 8)))
print('Mid-frequency MDD returns: {}'.format(np.round(
    (np.min(returns_mid_storage) - np.max(returns_mid_storage)) / np.max(returns_mid_storage), 8)))
print('Low-frequency MDD returns: {}'.format(np.round(
    (np.min(returns_low_storage) - np.max(returns_low_storage)) / np.max(returns_low_storage), 8)))
print('Equal weighting MDD returns: {}'.format(np.round(
    (np.min(returns_equal_storage) - np.max(returns_equal_storage)) / np.max(returns_equal_storage), 8)))
print('FTSE 100 MDD returns: {}'.format(np.round(
    (np.min(returns_gdp_storage) - np.max(returns_gdp_storage)) / np.max(returns_gdp_storage), 8)))

fig, axs = plt.subplots(1, 1, sharey=True, figsize=(15, 12))
fig.suptitle('Cumulative Returns of Funded and Unfunded Pension Schemes', fontsize=30)
plt.plot(np.exp(np.cumsum(np.append(np.asarray([0]), np.asarray(returns_gdp_storage)[:, 0]))),
         label='FTSE 100', linewidth=3)
plt.plot(np.exp(np.cumsum(np.append(np.asarray([0]), np.asarray(returns_high_storage)[:, 0]))),
         label='High-frequency', linewidth=3)
plt.plot(np.exp(np.cumsum(np.append(np.asarray([0]), np.asarray(returns_mid_storage)[:, 0]))),
         label='Mid-frequency', linewidth=3)
plt.plot(np.exp(np.cumsum(np.append(np.asarray([0]), np.asarray(returns_low_storage)[:, 0]))),
         label='Low-frequency', linewidth=3)
plt.plot(np.exp(np.cumsum(np.append(np.asarray([0]), np.asarray(returns_equal_storage)[:, 0]))),
         label='Equal weighting', linewidth=3)
plt.ylabel('Cumulative Returns', fontsize=24)
plt.xlabel('Dates', fontsize=24)
plt.xticks(np.append([0], np.arange(1, int(len(np.asarray(returns_high_storage)[:, 0]) + 1))[np.arange(2, 76, 4)]),
           np.append(['2004-04-01'], returns_high_storage.index[np.arange(2, 76, 4)]), rotation=45, fontsize=16)
plt.yticks(np.arange(1, 6), fontsize=20)
box_1 = axs.get_position()
axs.set_position([box_1.x0 - 0.07, box_1.y0 + 0.03, box_1.width, box_1.height])
axs.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14)
plt.savefig('../aas_figures/Funded_and_unfunded_pensions.pdf')
plt.show()

gamma = 1.2
print('gamma = {}'.format(gamma))
opt_high_weight = ((np.mean(np.asarray(returns_high_storage)[:, 0]) - np.mean(np.asarray(returns_gdp_storage)[:, 0]) +
                   gamma * (np.var(np.asarray(returns_gdp_storage)[:, 0]) -
                            np.cov(np.asarray(returns_high_storage)[:, 0], np.asarray(returns_gdp_storage)[:, 0])[0, 1])) /
                   (gamma * (np.var(np.asarray(returns_high_storage)[:, 0]) + np.var(np.asarray(returns_gdp_storage)[:, 0]) -
                             2 * np.cov(np.asarray(returns_high_storage)[:, 0], np.asarray(returns_gdp_storage)[:, 0])[0, 1])))
print('Optimal weighting for high-frequency forecast of covariance structure = {}%'.format(opt_high_weight))

print('Hybrid High-frequency mean returns: {}'.format(np.round(opt_high_weight * np.mean(np.asarray(returns_high_storage)) +
                                                               (1 - opt_high_weight) * np.mean(np.asarray(returns_gdp_storage)), 4)))
print('Hybrid High-frequency variance returns: {}'.format(np.round(np.var(opt_high_weight * np.asarray(returns_high_storage) +
                                                               (1 - opt_high_weight) * np.asarray(returns_gdp_storage)), 8)))
print('Hybrid High-frequency VaR returns: {}'.format(np.round(np.quantile(opt_high_weight * np.asarray(returns_high_storage) +
                                                               (1 - opt_high_weight) * np.asarray(returns_gdp_storage), 0.05), 8)))
hybrid_high = (opt_high_weight * np.asarray(returns_high_storage) +
               (1 - opt_high_weight) * np.asarray(returns_gdp_storage))
print('Hybrid High-frequency CVaR returns: {}'.format(
    np.round(np.mean(hybrid_high[np.quantile(np.asarray(hybrid_high),0.05) > hybrid_high]), 8)))
print('Hybrid High-frequency MDD returns: {}'.format(np.round(
    (np.min(hybrid_high) - np.max(hybrid_high)) / np.max(hybrid_high), 8)))

opt_mid_weight = ((np.mean(np.asarray(returns_mid_storage)[:, 0]) - np.mean(np.asarray(returns_gdp_storage)[:, 0]) +
                   gamma * (np.var(np.asarray(returns_gdp_storage)[:, 0]) -
                            np.cov(np.asarray(returns_mid_storage)[:, 0], np.asarray(returns_gdp_storage)[:, 0])[0, 1])) /
                   (gamma * (np.var(np.asarray(returns_mid_storage)[:, 0]) + np.var(np.asarray(returns_gdp_storage)[:, 0]) -
                             2 * np.cov(np.asarray(returns_mid_storage)[:, 0], np.asarray(returns_gdp_storage)[:, 0])[0, 1])))
print('Optimal weighting for mid-frequency forecast of covariance structure = {}%'.format(opt_mid_weight))

print('Hybrid Mid-frequency mean returns: {}'.format(np.round(opt_mid_weight * np.mean(np.asarray(returns_mid_storage)) +
                                                               (1 - opt_mid_weight) * np.mean(np.asarray(returns_gdp_storage)), 4)))
print('Hybrid Mid-frequency variance returns: {}'.format(np.round(np.var(opt_mid_weight * np.asarray(returns_mid_storage) +
                                                               (1 - opt_mid_weight) * np.asarray(returns_gdp_storage)), 8)))
print('Hybrid Mid-frequency VaR returns: {}'.format(np.round(np.quantile(opt_mid_weight * np.asarray(returns_mid_storage) +
                                                               (1 - opt_mid_weight) * np.asarray(returns_gdp_storage), 0.05), 8)))
hybrid_mid = (opt_mid_weight * np.asarray(returns_mid_storage) +
               (1 - opt_mid_weight) * np.asarray(returns_gdp_storage))
print('Hybrid Mid-frequency CVaR returns: {}'.format(
    np.round(np.mean(hybrid_mid[np.quantile(np.asarray(hybrid_mid),0.05) > hybrid_mid]), 8)))
print('Hybrid Mid-frequency MDD returns: {}'.format(np.round(
    (np.min(hybrid_mid) - np.max(hybrid_mid)) / np.max(hybrid_mid), 8)))

opt_low_weight = ((np.mean(np.asarray(returns_low_storage)[:, 0]) - np.mean(np.asarray(returns_gdp_storage)[:, 0]) +
                   gamma * (np.var(np.asarray(returns_gdp_storage)[:, 0]) -
                            np.cov(np.asarray(returns_low_storage)[:, 0], np.asarray(returns_gdp_storage)[:, 0])[0, 1])) /
                   (gamma * (np.var(np.asarray(returns_low_storage)[:, 0]) + np.var(np.asarray(returns_gdp_storage)[:, 0]) -
                             2 * np.cov(np.asarray(returns_low_storage)[:, 0], np.asarray(returns_gdp_storage)[:, 0])[0, 1])))
print('Optimal weighting for low-frequency forecast of covariance structure = {}%'.format(opt_low_weight))

print('Hybrid Low-frequency mean returns: {}'.format(np.round(opt_low_weight * np.mean(np.asarray(returns_low_storage)) +
                                                               (1 - opt_low_weight) * np.mean(np.asarray(returns_gdp_storage)), 4)))
print('Hybrid Low-frequency variance returns: {}'.format(np.round(np.var(opt_low_weight * np.asarray(returns_low_storage) +
                                                               (1 - opt_low_weight) * np.asarray(returns_gdp_storage)), 8)))
print('Hybrid Low-frequency VaR returns: {}'.format(np.round(np.quantile(opt_low_weight * np.asarray(returns_low_storage) +
                                                               (1 - opt_low_weight) * np.asarray(returns_gdp_storage), 0.05), 8)))
hybrid_low = (opt_low_weight * np.asarray(returns_low_storage) +
               (1 - opt_low_weight) * np.asarray(returns_gdp_storage))
print('Hybrid Low-frequency CVaR returns: {}'.format(
    np.round(np.mean(hybrid_low[np.quantile(np.asarray(hybrid_low),0.05) > hybrid_low]), 8)))
print('Hybrid Low-frequency MDD returns: {}'.format(np.round(
    (np.min(hybrid_low) - np.max(hybrid_low)) / np.max(hybrid_low), 8)))

gamma = 1.4
print('gamma = {}'.format(gamma))
opt_high_weight = ((np.mean(np.asarray(returns_high_storage)[:, 0]) - np.mean(np.asarray(returns_gdp_storage)[:, 0]) +
                   gamma * (np.var(np.asarray(returns_gdp_storage)[:, 0]) -
                            np.cov(np.asarray(returns_high_storage)[:, 0], np.asarray(returns_gdp_storage)[:, 0])[0, 1])) /
                   (gamma * (np.var(np.asarray(returns_high_storage)[:, 0]) + np.var(np.asarray(returns_gdp_storage)[:, 0]) -
                             2 * np.cov(np.asarray(returns_high_storage)[:, 0], np.asarray(returns_gdp_storage)[:, 0])[0, 1])))
print('Optimal weighting for high-frequency forecast of covariance structure = {}%'.format(opt_high_weight))

print('Hybrid High-frequency mean returns: {}'.format(np.round(opt_high_weight * np.mean(np.asarray(returns_high_storage)) +
                                                               (1 - opt_high_weight) * np.mean(np.asarray(returns_gdp_storage)), 4)))
print('Hybrid High-frequency variance returns: {}'.format(np.round(np.var(opt_high_weight * np.asarray(returns_high_storage) +
                                                               (1 - opt_high_weight) * np.asarray(returns_gdp_storage)), 8)))
print('Hybrid High-frequency VaR returns: {}'.format(np.round(np.quantile(opt_high_weight * np.asarray(returns_high_storage) +
                                                               (1 - opt_high_weight) * np.asarray(returns_gdp_storage), 0.05), 8)))
hybrid_high = (opt_high_weight * np.asarray(returns_high_storage) +
               (1 - opt_high_weight) * np.asarray(returns_gdp_storage))
print('Hybrid High-frequency CVaR returns: {}'.format(
    np.round(np.mean(hybrid_high[np.quantile(np.asarray(hybrid_high),0.05) > hybrid_high]), 8)))
print('Hybrid High-frequency MDD returns: {}'.format(np.round(
    (np.min(hybrid_high) - np.max(hybrid_high)) / np.max(hybrid_high), 8)))

opt_mid_weight = ((np.mean(np.asarray(returns_mid_storage)[:, 0]) - np.mean(np.asarray(returns_gdp_storage)[:, 0]) +
                   gamma * (np.var(np.asarray(returns_gdp_storage)[:, 0]) -
                            np.cov(np.asarray(returns_mid_storage)[:, 0], np.asarray(returns_gdp_storage)[:, 0])[0, 1])) /
                   (gamma * (np.var(np.asarray(returns_mid_storage)[:, 0]) + np.var(np.asarray(returns_gdp_storage)[:, 0]) -
                             2 * np.cov(np.asarray(returns_mid_storage)[:, 0], np.asarray(returns_gdp_storage)[:, 0])[0, 1])))
print('Optimal weighting for mid-frequency forecast of covariance structure = {}%'.format(opt_mid_weight))

print('Hybrid Mid-frequency mean returns: {}'.format(np.round(opt_mid_weight * np.mean(np.asarray(returns_mid_storage)) +
                                                               (1 - opt_mid_weight) * np.mean(np.asarray(returns_gdp_storage)), 4)))
print('Hybrid Mid-frequency variance returns: {}'.format(np.round(np.var(opt_mid_weight * np.asarray(returns_mid_storage) +
                                                               (1 - opt_mid_weight) * np.asarray(returns_gdp_storage)), 8)))
print('Hybrid Mid-frequency VaR returns: {}'.format(np.round(np.quantile(opt_mid_weight * np.asarray(returns_mid_storage) +
                                                               (1 - opt_mid_weight) * np.asarray(returns_gdp_storage), 0.05), 8)))
hybrid_mid = (opt_mid_weight * np.asarray(returns_mid_storage) +
               (1 - opt_mid_weight) * np.asarray(returns_gdp_storage))
print('Hybrid Mid-frequency CVaR returns: {}'.format(
    np.round(np.mean(hybrid_mid[np.quantile(np.asarray(hybrid_mid),0.05) > hybrid_mid]), 8)))
print('Hybrid Mid-frequency MDD returns: {}'.format(np.round(
    (np.min(hybrid_mid) - np.max(hybrid_mid)) / np.max(hybrid_mid), 8)))

opt_low_weight = ((np.mean(np.asarray(returns_low_storage)[:, 0]) - np.mean(np.asarray(returns_gdp_storage)[:, 0]) +
                   gamma * (np.var(np.asarray(returns_gdp_storage)[:, 0]) -
                            np.cov(np.asarray(returns_low_storage)[:, 0], np.asarray(returns_gdp_storage)[:, 0])[0, 1])) /
                   (gamma * (np.var(np.asarray(returns_low_storage)[:, 0]) + np.var(np.asarray(returns_gdp_storage)[:, 0]) -
                             2 * np.cov(np.asarray(returns_low_storage)[:, 0], np.asarray(returns_gdp_storage)[:, 0])[0, 1])))
print('Optimal weighting for low-frequency forecast of covariance structure = {}%'.format(opt_low_weight))

print('Hybrid Low-frequency mean returns: {}'.format(np.round(opt_low_weight * np.mean(np.asarray(returns_low_storage)) +
                                                               (1 - opt_low_weight) * np.mean(np.asarray(returns_gdp_storage)), 4)))
print('Hybrid Low-frequency variance returns: {}'.format(np.round(np.var(opt_low_weight * np.asarray(returns_low_storage) +
                                                               (1 - opt_low_weight) * np.asarray(returns_gdp_storage)), 8)))
print('Hybrid Low-frequency VaR returns: {}'.format(np.round(np.quantile(opt_low_weight * np.asarray(returns_low_storage) +
                                                               (1 - opt_low_weight) * np.asarray(returns_gdp_storage), 0.05), 8)))
hybrid_low = (opt_low_weight * np.asarray(returns_low_storage) +
               (1 - opt_low_weight) * np.asarray(returns_gdp_storage))
print('Hybrid Low-frequency CVaR returns: {}'.format(
    np.round(np.mean(hybrid_low[np.quantile(np.asarray(hybrid_low),0.05) > hybrid_low]), 8)))
print('Hybrid Low-frequency MDD returns: {}'.format(np.round(
    (np.min(hybrid_low) - np.max(hybrid_low)) / np.max(hybrid_low), 8)))

gamma = 1.6
print('gamma = {}'.format(gamma))
opt_high_weight = ((np.mean(np.asarray(returns_high_storage)[:, 0]) - np.mean(np.asarray(returns_gdp_storage)[:, 0]) +
                   gamma * (np.var(np.asarray(returns_gdp_storage)[:, 0]) -
                            np.cov(np.asarray(returns_high_storage)[:, 0], np.asarray(returns_gdp_storage)[:, 0])[0, 1])) /
                   (gamma * (np.var(np.asarray(returns_high_storage)[:, 0]) + np.var(np.asarray(returns_gdp_storage)[:, 0]) -
                             2 * np.cov(np.asarray(returns_high_storage)[:, 0], np.asarray(returns_gdp_storage)[:, 0])[0, 1])))
print('Optimal weighting for high-frequency forecast of covariance structure = {}%'.format(opt_high_weight))

print('Hybrid High-frequency mean returns: {}'.format(np.round(opt_high_weight * np.mean(np.asarray(returns_high_storage)) +
                                                               (1 - opt_high_weight) * np.mean(np.asarray(returns_gdp_storage)), 4)))
print('Hybrid High-frequency variance returns: {}'.format(np.round(np.var(opt_high_weight * np.asarray(returns_high_storage) +
                                                               (1 - opt_high_weight) * np.asarray(returns_gdp_storage)), 8)))
print('Hybrid High-frequency VaR returns: {}'.format(np.round(np.quantile(opt_high_weight * np.asarray(returns_high_storage) +
                                                               (1 - opt_high_weight) * np.asarray(returns_gdp_storage), 0.05), 8)))
hybrid_high = (opt_high_weight * np.asarray(returns_high_storage) +
               (1 - opt_high_weight) * np.asarray(returns_gdp_storage))
print('Hybrid High-frequency CVaR returns: {}'.format(
    np.round(np.mean(hybrid_high[np.quantile(np.asarray(hybrid_high),0.05) > hybrid_high]), 8)))
print('Hybrid High-frequency MDD returns: {}'.format(np.round(
    (np.min(hybrid_high) - np.max(hybrid_high)) / np.max(hybrid_high), 8)))

opt_mid_weight = ((np.mean(np.asarray(returns_mid_storage)[:, 0]) - np.mean(np.asarray(returns_gdp_storage)[:, 0]) +
                   gamma * (np.var(np.asarray(returns_gdp_storage)[:, 0]) -
                            np.cov(np.asarray(returns_mid_storage)[:, 0], np.asarray(returns_gdp_storage)[:, 0])[0, 1])) /
                   (gamma * (np.var(np.asarray(returns_mid_storage)[:, 0]) + np.var(np.asarray(returns_gdp_storage)[:, 0]) -
                             2 * np.cov(np.asarray(returns_mid_storage)[:, 0], np.asarray(returns_gdp_storage)[:, 0])[0, 1])))
print('Optimal weighting for mid-frequency forecast of covariance structure = {}%'.format(opt_mid_weight))

print('Hybrid Mid-frequency mean returns: {}'.format(np.round(opt_mid_weight * np.mean(np.asarray(returns_mid_storage)) +
                                                               (1 - opt_mid_weight) * np.mean(np.asarray(returns_gdp_storage)), 4)))
print('Hybrid Mid-frequency variance returns: {}'.format(np.round(np.var(opt_mid_weight * np.asarray(returns_mid_storage) +
                                                               (1 - opt_mid_weight) * np.asarray(returns_gdp_storage)), 8)))
print('Hybrid Mid-frequency VaR returns: {}'.format(np.round(np.quantile(opt_mid_weight * np.asarray(returns_mid_storage) +
                                                               (1 - opt_mid_weight) * np.asarray(returns_gdp_storage), 0.05), 8)))
hybrid_mid = (opt_mid_weight * np.asarray(returns_mid_storage) +
               (1 - opt_mid_weight) * np.asarray(returns_gdp_storage))
print('Hybrid Mid-frequency CVaR returns: {}'.format(
    np.round(np.mean(hybrid_mid[np.quantile(np.asarray(hybrid_mid),0.05) > hybrid_mid]), 8)))
print('Hybrid Mid-frequency MDD returns: {}'.format(np.round(
    (np.min(hybrid_mid) - np.max(hybrid_mid)) / np.max(hybrid_mid), 8)))

opt_low_weight = ((np.mean(np.asarray(returns_low_storage)[:, 0]) - np.mean(np.asarray(returns_gdp_storage)[:, 0]) +
                   gamma * (np.var(np.asarray(returns_gdp_storage)[:, 0]) -
                            np.cov(np.asarray(returns_low_storage)[:, 0], np.asarray(returns_gdp_storage)[:, 0])[0, 1])) /
                   (gamma * (np.var(np.asarray(returns_low_storage)[:, 0]) + np.var(np.asarray(returns_gdp_storage)[:, 0]) -
                             2 * np.cov(np.asarray(returns_low_storage)[:, 0], np.asarray(returns_gdp_storage)[:, 0])[0, 1])))
print('Optimal weighting for low-frequency forecast of covariance structure = {}%'.format(opt_low_weight))

print('Hybrid Low-frequency mean returns: {}'.format(np.round(opt_low_weight * np.mean(np.asarray(returns_low_storage)) +
                                                               (1 - opt_low_weight) * np.mean(np.asarray(returns_gdp_storage)), 4)))
print('Hybrid Low-frequency variance returns: {}'.format(np.round(np.var(opt_low_weight * np.asarray(returns_low_storage) +
                                                               (1 - opt_low_weight) * np.asarray(returns_gdp_storage)), 8)))
print('Hybrid Low-frequency VaR returns: {}'.format(np.round(np.quantile(opt_low_weight * np.asarray(returns_low_storage) +
                                                               (1 - opt_low_weight) * np.asarray(returns_gdp_storage), 0.05), 8)))
hybrid_low = (opt_low_weight * np.asarray(returns_low_storage) +
               (1 - opt_low_weight) * np.asarray(returns_gdp_storage))
print('Hybrid Low-frequency CVaR returns: {}'.format(
    np.round(np.mean(hybrid_low[np.quantile(np.asarray(hybrid_low),0.05) > hybrid_low]), 8)))
print('Hybrid Low-frequency MDD returns: {}'.format(np.round(
    (np.min(hybrid_low) - np.max(hybrid_low)) / np.max(hybrid_low), 8)))

temp = 0

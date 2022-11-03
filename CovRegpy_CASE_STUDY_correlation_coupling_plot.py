
# Case Study - construction of correlation coupling plot

import textwrap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from CovRegpy_RCR import cov_reg_given_mean, cubic_b_spline

from AdvEMDpy import AdvEMDpy

sns.set(style='darkgrid')

# load 11 sector indices
sector_11_indices = pd.read_csv('S&P500_Data/sp_500_11_sector_indices.csv', header=0)
sector_11_indices = sector_11_indices.set_index(['Unnamed: 0'])

# approximate daily treasury par yield curve rates for 3 year bonds
risk_free = (0.01 / 365)  # daily risk free rate

# sector numpy array
sector_11_indices_array = np.vstack((np.zeros((1, 11)), np.asarray(sector_11_indices)))

for col, sector in enumerate(sector_11_indices.columns):
    plt.plot(np.asarray(np.cumprod(np.exp(sector_11_indices_array[:, col]))), label=sector)
ax = plt.subplot(111)
plt.gcf().subplots_adjust(bottom=0.18)
plt.title(textwrap.fill('Cumulative Returns of Eleven Market Cap Weighted Sector Indices of S&P 500 from 1 January 2017 to 31 December 2021', 60),
          fontsize=10)
plt.xticks([0, 365, 730, 1095, 1461, 1826],
           ['31-12-2016', '31-12-2017', '31-12-2018', '31-12-2019', '31-12-2020', '31-12-2021'],
           fontsize=8, rotation=-30)
plt.plot(640 * np.ones(100), np.linspace(0.5, 2, 100), 'k--', Linewidth=2)
plt.plot(722 * np.ones(100), np.linspace(0.5, 2, 100), 'k--', Linewidth=2,
         label=textwrap.fill('Final quarter 2018 bear market', 18))
plt.plot(1144 * np.ones(100), np.linspace(0.1, 2.5, 100), 'k--')
plt.plot(1177 * np.ones(100), np.linspace(0.1, 2.5, 100), 'k--', label='SARS-CoV-2')
plt.legend(loc='upper left', fontsize=7)
plt.xlabel('Days', fontsize=10)
plt.ylabel('Cumulative Returns', fontsize=10)
plt.yticks([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5], ['0.5', '1.0', '1.5', '2.0', '2.5', '3.0', '3.5'], fontsize=8)
plt.savefig('figures/S&P 500 - 11 Sectors/Sector_11_indices.png')
del sector, col
plt.show()

price_signal = np.cumprod(np.exp(sector_11_indices_array), axis=0)[1:, :]

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

# calculate 'spline_basis'
knots = 180  # arbitray - can adjust
end_time = 1825
spline_basis = cubic_b_spline(knots=np.linspace(-12, end_time + 12, knots),
                              time=np.linspace(0, end_time, end_time + 1))

# calculate 'A_est'
A_est = np.linalg.lstsq(spline_basis.transpose(), sector_11_indices_array, rcond=None)[0]

# calculate 'x'
# decompose price data
for signal in range(np.shape(price_signal)[1]):
    emd = AdvEMDpy.EMD(time_series=np.asarray(price_signal[:, signal]),
                       time=np.linspace(0, end_time, int(end_time + 1)))
    imfs, _, _, _, _, _, _ = \
        emd.empirical_mode_decomposition(knot_envelope=np.linspace(-12, end_time + 12, knots),
                                         matrix=True)

    # deal with constant last IMF and insert IMFs in dataframe
    # deal with different frequency structures here
    try:
        imfs = imfs[1:, :]
        if np.isclose(imfs[-1, 0], imfs[-1, -1]):
            imfs[-2, :] += imfs[-1, :]
            imfs = imfs[:-1, :]
    except:
        pass
    try:
        x = np.vstack((imfs, x))
    except:
        x = imfs.copy()

    # calculate y - same for both imf and ssa
    y = sector_11_indices_array
    y = y.T

# calculate B_est and Psi_est - direct application
B_est_direct, Psi_est_direct = \
    cov_reg_given_mean(A_est=A_est, basis=spline_basis, x=x, y=y, iterations=100)

# empty forecasted variance storage matrix - direct
variance_Model_forecast_direct = np.zeros((1826, np.shape(B_est_direct)[1], np.shape(B_est_direct)[1]))

# iteratively calculate variance
for var_day in range(1826):

    variance_Model_forecast_direct[var_day] = \
        Psi_est_direct + np.matmul(np.matmul(B_est_direct.T, x[:, var_day]).astype(np.float64).reshape(-1, 1),
                                   np.matmul(x[:, var_day].T,
                                             B_est_direct).astype(np.float64).reshape(1, -1)).astype(np.float64)

x, y = np.meshgrid(np.arange(11), np.arange(11))

# ax = plt.subplot(111)
# plt.title('Forecast Covariance at Start of Period 1')
# plt.pcolormesh(x, y, variance_Model_forecast_direct[639, :, :], cmap='gist_rainbow')
# plt.xticks(np.arange(11), [textwrap.fill(col, 15) for col in sector_11_indices.columns], fontsize=8, rotation='45')
# plt.yticks(np.arange(11), [textwrap.fill(col, 15) for col in sector_11_indices.columns], fontsize=8)
# plt.colorbar()
# box_0 = ax.get_position()
# ax.set_position([box_0.x0 + 0.05, box_0.y0 + 0.075, box_0.width * 0.9, box_0.height * 0.9])
# plt.savefig('figures/forecast_start_1.png')
# plt.show()
# ax = plt.subplot(111)
# plt.title('Forecast Covariance at End of Period 1')
# plt.pcolormesh(x, y, variance_Model_forecast_direct[721, :, :], cmap='gist_rainbow')
# plt.xticks(np.arange(11), [textwrap.fill(col, 15) for col in sector_11_indices.columns], fontsize=8, rotation='45')
# plt.yticks(np.arange(11), [textwrap.fill(col, 15) for col in sector_11_indices.columns], fontsize=8)
# plt.colorbar()
# box_0 = ax.get_position()
# ax.set_position([box_0.x0 + 0.05, box_0.y0 + 0.075, box_0.width * 0.9, box_0.height * 0.9])
# plt.savefig('figures/forecast_end_1.png')
# plt.show()
# ax = plt.subplot(111)
# plt.title('Forecast Covariance at Start of Period 2')
# plt.pcolormesh(x, y, variance_Model_forecast_direct[1143, :, :], cmap='gist_rainbow')
# plt.xticks(np.arange(11), [textwrap.fill(col, 15) for col in sector_11_indices.columns], fontsize=8, rotation='45')
# plt.yticks(np.arange(11), [textwrap.fill(col, 15) for col in sector_11_indices.columns], fontsize=8)
# plt.colorbar()
# box_0 = ax.get_position()
# ax.set_position([box_0.x0 + 0.05, box_0.y0 + 0.075, box_0.width * 0.9, box_0.height * 0.9])
# plt.savefig('figures/forecast_start_2.png')
# plt.show()
# ax = plt.subplot(111)
# plt.title('Forecast Covariance at End of Period 2')
# plt.pcolormesh(x, y, variance_Model_forecast_direct[1176, :, :], cmap='gist_rainbow')
# plt.xticks(np.arange(11), [textwrap.fill(col, 15) for col in sector_11_indices.columns], fontsize=8, rotation='45')
# plt.yticks(np.arange(11), [textwrap.fill(col, 15) for col in sector_11_indices.columns], fontsize=8)
# plt.colorbar()
# box_0 = ax.get_position()
# ax.set_position([box_0.x0 + 0.05, box_0.y0 + 0.075, box_0.width * 0.9, box_0.height * 0.9])
# plt.savefig('figures/forecast_end_2.png')
# plt.show()
#
# window_1 = 80
# window_2 = 30
# ax = plt.subplot(111)
# plt.title('Realised Covariance at Start of Period 1')
# plt.pcolormesh(x, y, np.cov(price_signal[int(639 - window_1):639, :].T), cmap='gist_rainbow')
# plt.xticks(np.arange(11), [textwrap.fill(col, 15) for col in sector_11_indices.columns], fontsize=8, rotation='45')
# plt.yticks(np.arange(11), [textwrap.fill(col, 15) for col in sector_11_indices.columns], fontsize=8)
# plt.colorbar()
# box_0 = ax.get_position()
# ax.set_position([box_0.x0 + 0.05, box_0.y0 + 0.075, box_0.width * 0.9, box_0.height * 0.9])
# plt.savefig('figures/realised_start_1.png')
# plt.show()
# ax = plt.subplot(111)
# plt.title('Realised Covariance at End of Period 1')
# plt.pcolormesh(x, y, np.cov(price_signal[int(721 - window_1):721, :].T), cmap='gist_rainbow')
# plt.xticks(np.arange(11), [textwrap.fill(col, 15) for col in sector_11_indices.columns], fontsize=8, rotation='45')
# plt.yticks(np.arange(11), [textwrap.fill(col, 15) for col in sector_11_indices.columns], fontsize=8)
# plt.colorbar()
# box_0 = ax.get_position()
# ax.set_position([box_0.x0 + 0.05, box_0.y0 + 0.075, box_0.width * 0.9, box_0.height * 0.9])
# plt.savefig('figures/realised_end_1.png')
# plt.show()
# ax = plt.subplot(111)
# plt.title('Realised Covariance After Recession of Period 1')
# plt.pcolormesh(x, y, np.cov(price_signal[721:int(721 + window_1), :].T), cmap='gist_rainbow')
# plt.xticks(np.arange(11), [textwrap.fill(col, 15) for col in sector_11_indices.columns], fontsize=8, rotation='45')
# plt.yticks(np.arange(11), [textwrap.fill(col, 15) for col in sector_11_indices.columns], fontsize=8)
# plt.colorbar()
# box_0 = ax.get_position()
# ax.set_position([box_0.x0 + 0.05, box_0.y0 + 0.075, box_0.width * 0.9, box_0.height * 0.9])
# plt.savefig('figures/realised_after_1.png')
# plt.show()
# ax = plt.subplot(111)
# plt.title('Realised Covariance at Start of Period 2')
# plt.pcolormesh(x, y, np.cov(price_signal[int(1143 - window_2):1143, :].T), cmap='gist_rainbow')
# plt.xticks(np.arange(11), [textwrap.fill(col, 15) for col in sector_11_indices.columns], fontsize=8, rotation='45')
# plt.yticks(np.arange(11), [textwrap.fill(col, 15) for col in sector_11_indices.columns], fontsize=8)
# plt.colorbar()
# box_0 = ax.get_position()
# ax.set_position([box_0.x0 + 0.05, box_0.y0 + 0.075, box_0.width * 0.9, box_0.height * 0.9])
# plt.savefig('figures/realised_start_2.png')
# plt.show()
# ax = plt.subplot(111)
# plt.title('Realised Covariance at End of Period 2')
# plt.pcolormesh(x, y, np.cov(price_signal[int(1176 - window_2):1176, :].T), cmap='gist_rainbow')
# plt.xticks(np.arange(11), [textwrap.fill(col, 15) for col in sector_11_indices.columns], fontsize=8, rotation='45')
# plt.yticks(np.arange(11), [textwrap.fill(col, 15) for col in sector_11_indices.columns], fontsize=8)
# plt.colorbar()
# box_0 = ax.get_position()
# ax.set_position([box_0.x0 + 0.05, box_0.y0 + 0.075, box_0.width * 0.9, box_0.height * 0.9])
# plt.savefig('figures/realised_end_2.png')
# plt.show()
# ax = plt.subplot(111)
# plt.title('Realised Covariance After Recession of Period 2')
# plt.pcolormesh(x, y, np.cov(price_signal[1176:int(1176 + window_2), :].T), cmap='gist_rainbow')
# plt.xticks(np.arange(11), [textwrap.fill(col, 15) for col in sector_11_indices.columns], fontsize=8, rotation='45')
# plt.yticks(np.arange(11), [textwrap.fill(col, 15) for col in sector_11_indices.columns], fontsize=8)
# plt.colorbar()
# box_0 = ax.get_position()
# ax.set_position([box_0.x0 + 0.05, box_0.y0 + 0.075, box_0.width * 0.9, box_0.height * 0.9])
# plt.savefig('figures/realised_after_2.png')
# plt.show()

fig, axs = plt.subplots(1, 3)
plt.suptitle('Correlation Structure Relative to Energy Sector')

for col, sector in enumerate(sector_11_indices.columns):
    if col != 3:
        if col == 0 or col == 1 or col == 2 or col == 7:
            axs[0].plot((variance_Model_forecast_direct[:, col, 3] /
                         np.sqrt(variance_Model_forecast_direct[:, 3, 3] *
                                 variance_Model_forecast_direct[:, col, col])), label=textwrap.fill(sector, 14))
            axs[1].plot((variance_Model_forecast_direct[:, col, 3] /
                         np.sqrt(variance_Model_forecast_direct[:, 3, 3] *
                                 variance_Model_forecast_direct[:, col, col])), label=textwrap.fill(sector, 14))
            axs[2].plot((variance_Model_forecast_direct[:, col, 3] /
                         np.sqrt(variance_Model_forecast_direct[:, 3, 3] *
                                 variance_Model_forecast_direct[:, col, col])), label=textwrap.fill(sector, 14))
        else:
            axs[0].plot((variance_Model_forecast_direct[:, col, 3] /
                         np.sqrt(variance_Model_forecast_direct[:, 3, 3] *
                                 variance_Model_forecast_direct[:, col, col])), label=sector)
            axs[1].plot((variance_Model_forecast_direct[:, col, 3] /
                         np.sqrt(variance_Model_forecast_direct[:, 3, 3] *
                                 variance_Model_forecast_direct[:, col, col])), label=sector)
            axs[2].plot((variance_Model_forecast_direct[:, col, 3] /
                         np.sqrt(variance_Model_forecast_direct[:, 3, 3] *
                                 variance_Model_forecast_direct[:, col, col])), label=sector)
    else:
        pass

axs[0].set_xticks([0, 364, 729, 1094, 1460, 1825])
axs[0].set_xticklabels(['01-01-2017', '31-12-2017', '31-12-2018', '31-12-2019', '31-12-2020', '31-12-2021'],
                       fontsize=8, rotation=-30)
gap = 20
axs[0].set_title(textwrap.fill('Full Correlation Structure', 11))
axs[1].set_title(textwrap.fill('Market Down-Turn 2018', 9))
axs[1].set_xticks([639, 721])
axs[1].set_xticklabels(['02-10-2018', '23-12-2018'], fontsize=8, rotation=-30)
axs[1].set_xlim(639 - gap, 721 + gap)
axs[2].set_title(textwrap.fill('SARS-CoV-2 Pandemic 2020', 10))
axs[2].set_xticks([1143, 1176])
axs[2].set_xticklabels(['28-02-2020', '01-04-2020'], fontsize=8, rotation=-30)
axs[2].set_xlim(1143 - gap, 1176 + gap)

axs[0].set_yticks([-0.5, 0.0, 0.5, 1.0])
axs[0].set_yticklabels(['-0.5', '0.0', '0.5', '1.0'], fontsize=8)
axs[1].set_yticks([-0.5, 0.0, 0.5, 1.0])
axs[1].set_yticklabels(['', '', '', ''], fontsize=8)
axs[2].set_yticks([-0.5, 0.0, 0.5, 1.0])
axs[2].set_yticklabels(['', '', '', ''], fontsize=8)
axs[0].plot(639 * np.ones(100), np.linspace(-0.3, 1.1, 100), 'k--', Linewidth=2)
axs[0].plot(721 * np.ones(100), np.linspace(-0.3, 1.1, 100), 'k--', Linewidth=2,
            label=textwrap.fill('Final quarter 2018 bear market', 14))
axs[0].plot(1143 * np.ones(100), np.linspace(-0.3, 1.1, 100), 'k--')
axs[0].plot(1176 * np.ones(100), np.linspace(-0.3, 1.1, 100), 'k--', label='SARS-CoV-2')
axs[1].plot(639 * np.ones(100), np.linspace(-0.3, 1.1, 100), 'k--', Linewidth=2)
axs[1].plot(721 * np.ones(100), np.linspace(-0.3, 1.1, 100), 'k--', Linewidth=2,
            label=textwrap.fill('Final quarter 2018 bear market', 14))
axs[1].plot(1143 * np.ones(100), np.linspace(-0.3, 1.1, 100), 'k--')
axs[1].plot(1176 * np.ones(100), np.linspace(-0.3, 1.1, 100), 'k--', label='SARS-CoV-2')
axs[2].plot(639 * np.ones(100), np.linspace(-0.3, 1.1, 100), 'k--', Linewidth=2)
axs[2].plot(721 * np.ones(100), np.linspace(-0.3, 1.1, 100), 'k--', Linewidth=2,
            label=textwrap.fill('Final quarter 2018 bear market', 14))
axs[2].plot(1143 * np.ones(100), np.linspace(-0.3, 1.1, 100), 'k--')
axs[2].plot(1176 * np.ones(100), np.linspace(-0.3, 1.1, 100), 'k--', label='SARS-CoV-2')
axs[0].set_xlabel('Days', fontsize=10)
axs[1].set_xlabel('Days', fontsize=10)
axs[2].set_xlabel('Days', fontsize=10)
axs[0].set_ylabel('Correlation', fontsize=10)
plt.subplots_adjust(wspace=0.1, top=0.8, bottom=0.16, left=0.08)
box_0 = axs[0].get_position()
axs[0].set_position([box_0.x0, box_0.y0, box_0.width * 0.9, box_0.height * 1.0])
box_1 = axs[1].get_position()
axs[1].set_position([box_1.x0 - 0.045, box_1.y0, box_1.width * 0.9, box_1.height * 1.0])
box_2 = axs[2].get_position()
axs[2].set_position([box_2.x0 - 0.09, box_2.y0, box_2.width * 0.9, box_2.height * 1.0])
axs[2].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
plt.savefig('figures/S&P 500 - 11 Sectors/Sector_11_indices_correlation_structure.png')
plt.show()

x = np.linspace(1, 10, 10)
y = np.linspace(1, 10, 10)
Z, Y = np.meshgrid(x, y)
X = np.random.uniform(0, 1, np.shape(Z))

temp_1 = np.random.normal(0, 1, (10, 10))
temp_2 = np.random.normal(0, 1, (10, 10))
temp_3 = np.random.normal(0, 1, (10, 10))
temp_1 = np.corrcoef(temp_1)
temp_2 = np.corrcoef(temp_2)
temp_3 = np.corrcoef(temp_3)

fig = plt.figure()
fig.set_size_inches(8, 10)
ax = plt.axes(projection='3d')
ax.set_title('Correlation Structure Through Time')
cov_plot = ax.plot_surface(Z, Y, temp_1, rstride=1, cstride=1, cmap='gist_rainbow', edgecolor='none')
ax.plot_surface(Z, Y, temp_2 + 40, rstride=1, cstride=1, cmap='gist_rainbow', edgecolor='none')
ax.plot_surface(Z, Y, temp_3 + 80, rstride=1, cstride=1, cmap='gist_rainbow', edgecolor='none')
# ax.set_xlabel('Asset')
ax.set_xticks(ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
ax.set_xticklabels(labels=['BTC', 'ETH', 'BNB', 'ADA', 'XRP', 'DOGE', 'DOT', 'UNI', 'LINK', 'SOL'], rotation=20,
                   fontsize=8, ha="left", rotation_mode="anchor")
# ax.set_zlim(0, 10)
# ax.set_ylabel('Asset')
ax.set_yticks(ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
ax.set_yticklabels(labels=['BTC', 'ETH', 'BNB', 'ADA', 'XRP', 'DOGE', 'DOT', 'UNI', 'LINK', 'SOL'], rotation=0,
                   fontsize=8)
ax.set_zticks(ticks=[0, 40, 80])
ax.set_zticklabels(['01-10-2020', '02-10-2020', '03-10-2020'], rotation=-60, fontsize=8)
cbar = plt.colorbar(cov_plot)
cbar.set_label("Covariance")
plt.savefig('figures/covariance_example.png')
plt.show()

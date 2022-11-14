
# median breaks correlation structure

import textwrap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from CovRegpy_RCR import cov_reg_given_mean, cubic_b_spline
from CovRegpy_SSD import CovRegpy_ssd

from AdvEMDpy import AdvEMDpy

np.random.seed(0)
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
knots = 20  # arbitray - can adjust
d1 = int(721-639)
spline_basis = cubic_b_spline(knots=np.linspace(-12, d1 + 12, knots), time=np.arange(0, d1, 1))

# calculate 'A_est'
A_est = np.linalg.lstsq(spline_basis.transpose(), sector_11_indices_array[639:721, :], rcond=None)[0]

variance_forecast_1 = np.zeros((d1, 11, 11))
# variance_forecast_2 = np.zeros((d2, 11, 11))
ssd_vectors = np.zeros(11)

min_kl_distance = 1e6

B_est_20 = pd.read_csv('B and Psi Estimates/B_est_[20].csv', header=0)
B_est_20 = np.asarray(B_est_20.set_index('Unnamed: 0'))
Psi_est_20 = pd.read_csv('B and Psi Estimates/Psi_est_[20].csv', header=0)

B_est_1213 = pd.read_csv('B and Psi Estimates/B_est_CoV_[12, 13].csv', header=0)
B_est_1213 = np.asarray(B_est_1213.set_index('Unnamed: 0'))
Psi_est_1213 = pd.read_csv('B and Psi Estimates/Psi_est_CoV_[12, 13].csv', header=0)

x = sector_11_indices_array.T

for var_day in range(d1):

    variance_forecast_1[var_day] = \
        B_est_20 + np.matmul(np.matmul(B_est_20.T, x[:, int(var_day + 639 - d1)].reshape(-1, 1).T).astype(np.float64).reshape(-1, 1),
                                   np.matmul(x[:, int(var_day + 639 - d1)].reshape(-1, 1),
                                             B_est_20).astype(np.float64).reshape(1, -1)).astype(np.float64)

fig, axs = plt.subplots(1, 3)
plt.suptitle('Correlation Structure Relative to Energy Sector')

for col, sector in enumerate(sector_11_indices.columns):
    if col != 3:
        if col == 0 or col == 1 or col == 2 or col == 7:
            for ax in len(axs):
                axs[ax].plot((variance_forecast_1[:, col, 3] /
                              np.sqrt(variance_forecast_1[:, 3, 3] *
                                      variance_forecast_1[:, col, col])), label=textwrap.fill(sector, 14))
        else:
            for ax in len(axs):
                axs[ax].plot((variance_forecast_1[:, col, 3] /
                              np.sqrt(variance_forecast_1[:, 3, 3] *
                                      variance_forecast_1[:, col, col])), label=sector)
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
plt.show()

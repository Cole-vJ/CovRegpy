
#     ________
#            /
#      \    /
#       \  /
#        \/

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

colours = ['grey', 'red', 'orange', 'gold', 'olive', 'cyan', 'green', 'blue', 'darkviolet', 'deeppink', 'magenta']

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
d1 = int(1176 - 1143)
spline_basis = cubic_b_spline(knots=np.linspace(-12, d1 + 12, knots), time=np.arange(0, d1, 1))

# calculate 'A_est'
A_est = np.linalg.lstsq(spline_basis.transpose(), sector_11_indices_array[1143:1176, :], rcond=None)[0]

variance_forecast_1 = np.zeros((d1, 11, 11))
x, y, x_ssa = 0, 0, 0
ssd_vectors = np.zeros(11)

min_kl_distance = 1e6

for i in np.arange(-30, 1, 30):
    del x, y, x_ssa
    for signal in range(np.shape(price_signal)[1]):
        emd = AdvEMDpy.EMD(time_series=np.asarray(price_signal[int(1143 + i):int(1176 + i), signal]),
                           time=np.arange(0, d1, 1))
        imfs, _, _, _, _, _, _ = \
            emd.empirical_mode_decomposition(knot_envelope=np.linspace(-12, d1 + 12, knots),
                                             matrix=True, mean_threshold=1e-3, initial_smoothing=False)
        ssd_comps = CovRegpy_ssd(np.asarray(price_signal[int(1143 + i):int(1176 + i), signal]),
                                 initial_trend_ratio=10, nmse_threshold=0.01, plot=False, debug=False, method='l2')
        ssd_vectors[signal] = np.shape(ssd_comps)[0]
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
        if ssd_comps[-1, 0] == ssd_comps[-1, -1]:
            ssd_comps = ssd_comps[-2, :].reshape(1, -1)
        else:
            ssd_comps = ssd_comps[-1, :].reshape(1, -1)
        try:
            x_ssa = np.vstack((ssd_comps, x_ssa))
        except:
            x_ssa = ssd_comps.copy()

    sets = np.asarray([[i] for i in range(22)] + [[i * 2, i * 2 + 1] for i in range(11)])

    if i == 0:
        for subset in sets:
            B_est = pd.read_csv('B and Psi Estimates/B_est_CoV_{}.csv'.format(str(subset)), header=0)
            B_est = np.asarray(B_est.set_index('Unnamed: 0'))
            Psi_est = pd.read_csv('B and Psi Estimates/Psi_est_CoV_{}.csv'.format(str(subset)), header=0)
            Psi_est = np.asarray(Psi_est.set_index('Unnamed: 0'))
            variance_forecast_1 = np.zeros((d1, 11, 11))
            Psi = np.zeros((d1, 11, 11))
            BxxB = np.zeros((d1, 11, 11))

            for var_day in range(d1):
                Psi_day = Psi_est
                BxxB_day = np.matmul(np.matmul(B_est.T,
                                               x[subset, var_day]).astype(np.float64).reshape(-1, 1),
                                     np.matmul(x[subset,
                                               var_day].T, B_est).astype(np.float64).reshape(1, -1)).astype(np.float64)
                Psi[var_day] = Psi_day
                BxxB[var_day] = BxxB_day
                variance_forecast_1[var_day] = Psi_day + BxxB_day

            x_mesh, y_mesh = np.meshgrid(np.arange(11), np.arange(11))

            # B_est_20 = pd.read_csv('B and Psi Estimates/B_est_[20].csv', header=0)
            # B_est_20 = np.asarray(B_est_20.set_index('Unnamed: 0'))
            # Psi_est_20 = pd.read_csv('B and Psi Estimates/Psi_est_[20].csv', header=0)
            # Psi_est_20 = np.asarray(Psi_est_20.set_index('Unnamed: 0'))
            #
            # B_est_1213 = pd.read_csv('B and Psi Estimates/B_est_CoV_[12, 13].csv', header=0)
            # B_est_1213 = np.asarray(B_est_1213.set_index('Unnamed: 0'))
            # Psi_est_1213 = pd.read_csv('B and Psi Estimates/Psi_est_CoV_[12, 13].csv', header=0)
            # Psi_est_1213 = np.asarray(Psi_est_1213.set_index('Unnamed: 0'))
            #
            # for var_day in range(d1):
            #     variance_forecast_1[var_day] = \
            #         Psi_est_1213 + np.matmul(np.matmul(B_est_1213.T, x[12:14, var_day].T),
            #                                  np.matmul(x[12:14, var_day], B_est_1213)).astype(np.float64)

            # fig, axs = plt.subplots(1, 2)
            # plt.suptitle('Correlation Structure Relative to Energy Sector')
            #
            # for col, sector in enumerate(sector_11_indices.columns):
            #     if col != 3:
            #         if col == 0 or col == 1 or col == 2 or col == 7:
            #             for ax in range(2):
            #                 axs[ax].plot(np.arange(33) + 1143, (variance_forecast_1[:, col, 3] /
            #                               np.sqrt(variance_forecast_1[:, 3, 3] *
            #                                       variance_forecast_1[:, col, col])), label=textwrap.fill(sector, 14))
            #         else:
            #             for ax in range(2):
            #                 axs[ax].plot(np.arange(33) + 1143, (variance_forecast_1[:, col, 3] /
            #                               np.sqrt(variance_forecast_1[:, 3, 3] *
            #                                       variance_forecast_1[:, col, col])), label=sector)
            #         for day in np.arange(-10, 33):
            #             real_cov = np.cov(price_signal[int(day + 1123):int(day + 1143), :].T)
            #             axs[ax].scatter(day + 1143, real_cov[3, col] / np.sqrt(real_cov[3, 3] * real_cov[col, col]),
            #                             c=colours[col])
            #         if col == 0 or col == 1 or col == 2 or col == 7:
            #             axs[1].scatter(day + 1143, real_cov[3, col] / np.sqrt(real_cov[3, 3] * real_cov[col, col]),
            #                            c=colours[col], label=textwrap.fill('{}'.format(sector), 14))
            #             axs[0].scatter(day + 1143, real_cov[3, col] / np.sqrt(real_cov[3, 3] * real_cov[col, col]),
            #                            c=colours[col], label=textwrap.fill('{}'.format(sector), 14))
            #         else:
            #             axs[1].scatter(day + 1143, real_cov[3, col] / np.sqrt(real_cov[3, 3] * real_cov[col, col]),
            #                            c=colours[col], label='{}'.format(sector))
            #             axs[0].scatter(day + 1143, real_cov[3, col] / np.sqrt(real_cov[3, 3] * real_cov[col, col]),
            #                            c=colours[col], label='{}'.format(sector))
            #     else:
            #         pass
            #
            # gap = 10
            # axs[0].set_title(textwrap.fill('Market Down-Turn 2018', 9))
            # axs[0].set_xticks([639, 721])
            # axs[0].set_xticklabels(['02-10-2018', '23-12-2018'], fontsize=8, rotation=-30)
            # axs[0].set_xlim(639 - gap, 721 + gap)
            # axs[1].set_title(textwrap.fill('SARS-CoV-2 Pandemic 2020', 10))
            # axs[1].set_xticks([1143, 1176])
            # axs[1].set_xticklabels(['28-02-2020', '01-04-2020'], fontsize=8, rotation=-30)
            # axs[1].set_xlim(1143 - 12, 1176 + 2)
            #
            # axs[0].set_yticklabels(['-0.5', '0.0', '0.5', '1.0'], fontsize=8)
            # axs[0].set_yticks([-0.5, 0.0, 0.5, 1.0])
            # axs[1].set_yticks([-0.5, 0.0, 0.5, 1.0])
            # axs[1].set_yticklabels(['', '', '', ''], fontsize=8)
            # axs[0].set_ylim(-0.35, 1.15)
            # axs[1].set_ylim(-0.35, 1.15)
            # # axs[0].plot(639 * np.ones(100), np.linspace(-0.3, 1.1, 100), 'k--', Linewidth=2)
            # # axs[0].plot(721 * np.ones(100), np.linspace(-0.3, 1.1, 100), 'k--', Linewidth=2,
            # #             label=textwrap.fill('Final quarter 2018 bear market', 14))
            # # axs[0].plot(1143 * np.ones(100), np.linspace(-0.3, 1.1, 100), 'k--')
            # # axs[0].plot(1176 * np.ones(100), np.linspace(-0.3, 1.1, 100), 'k--', label='SARS-CoV-2')
            # # axs[1].plot(639 * np.ones(100), np.linspace(-0.3, 1.1, 100), 'k--', Linewidth=2)
            # # axs[1].plot(721 * np.ones(100), np.linspace(-0.3, 1.1, 100), 'k--', Linewidth=2,
            # #             label=textwrap.fill('Final quarter 2018 bear market', 14))
            # # axs[1].plot(1143 * np.ones(100), np.linspace(-0.3, 1.1, 100), 'k--')
            # # axs[1].plot(1176 * np.ones(100), np.linspace(-0.3, 1.1, 100), 'k--', label='SARS-CoV-2')
            # axs[0].set_xlabel('Days', fontsize=10)
            # axs[1].set_xlabel('Days', fontsize=10)
            # plt.subplots_adjust(wspace=0.1, top=0.8, bottom=0.16, left=0.08)
            # box_0 = axs[0].get_position()
            # axs[0].set_position([box_0.x0, box_0.y0, box_0.width * 0.92, box_0.height * 1.0])
            # box_1 = axs[1].get_position()
            # axs[1].set_position([box_1.x0 - 0.025, box_1.y0, box_1.width * 0.92, box_1.height * 1.0])
            # axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=6)
            # plt.show()

            A = variance_forecast_1[-1, :, :]
            B = np.cov(price_signal[int(1176 - d1):1176, :].T)
            kl_distance = (1 / 2) * (np.sum(np.diag(np.matmul(np.linalg.inv(A), B))) -
                                     np.log(np.linalg.det(B) / np.linalg.det(A)))
            if kl_distance < min_kl_distance:
                min_kl_distance = kl_distance
                subset_min = subset
            print(str(subset) + ' = {}'. format(kl_distance))

            if subset == [0] or subset == [15] or subset == [17] or subset == [21] or \
                    subset == [2, 3] or subset == [10, 11] or subset == [12, 13] or subset == [14, 15]:
                ax = plt.subplot(111)
                if subset == [0]:
                    plt.title(textwrap.fill(
                        'Optimal Unattributable Variance Forecast using High Frequency Communication Services Implicit Factor', 55))
                elif subset == [15]:
                    plt.title(textwrap.fill(
                        'Optimal Unattributable Variance Forecast using Low Frequency Information Technology Implicit Factor', 50))
                elif subset == [17]:
                    plt.title(textwrap.fill(
                        'Optimal Unattributable Variance Forecast using Low Frequency Materials Implicit Factor', 50))
                elif subset == [21]:
                    plt.title(textwrap.fill(
                        'Optimal Unattributable Variance Forecast using Low Frequency Utilities Implicit Factor', 50))
                elif subset == [2, 3]:
                    plt.title(textwrap.fill(
                        'Optimal Unattributable Variance Forecast using High Frequency and Low Frequency Consumer '
                        'Discretionary Implicit Factors',
                        65))
                elif subset == [10, 11]:
                    plt.title(textwrap.fill(
                        'Optimal Unattributable Variance Forecast using High Frequency and Low Frequency Health Care'
                        ' Implicit Factors',
                        65))
                elif subset == [12, 13]:
                    plt.title(textwrap.fill(
                        'Optimal Unattributable Variance Forecast using High Frequency and Low Frequency Industrials'
                        ' Implicit Factors',
                        60))
                elif subset == [14, 15]:
                    plt.title(textwrap.fill(
                        'Optimal Unattributable Variance Forecast using High Frequency and Low Frequency Information '
                        'Technology Implicit Factors',
                        50))

                plt.pcolormesh(x_mesh, y_mesh, Psi_day, cmap='gist_rainbow')
                plt.xticks(np.arange(11), [textwrap.fill(col, 15) for col in sector_11_indices.columns], fontsize=8,
                           rotation='45')
                plt.yticks(np.arange(11), [textwrap.fill(col, 15) for col in sector_11_indices.columns], fontsize=8)
                plt.colorbar()
                box_0 = ax.get_position()
                ax.set_position([box_0.x0 + 0.05, box_0.y0 + 0.075, box_0.width * 0.9, box_0.height * 0.9])
                plt.savefig('B and Psi Estimates/Psi_forecast_CoV_{}'.format(subset))
                plt.show()
                ax = plt.subplot(111)
                if subset == [0]:
                    plt.title(textwrap.fill(
                        'Optimal Attributable Variance Forecast using High Frequency Communication Services Implicit Factor',
                        50))
                elif subset == [15]:
                    plt.title(textwrap.fill(
                        'Optimal Attributable Variance Forecast using Low Frequency Information Technology Implicit Factor',
                        50))
                elif subset == [17]:
                    plt.title(textwrap.fill(
                        'Optimal Attributable Variance Forecast using Low Frequency Materials Implicit Factor', 50))
                elif subset == [21]:
                    plt.title(textwrap.fill(
                        'Optimal Attributable Variance Forecast using Low Frequency Utilities Implicit Factor', 50))
                elif subset == [2, 3]:
                    plt.title(textwrap.fill(
                        'Optimal Attributable Variance Forecast using High Frequency and Low Frequency Consumer '
                        'Discretionary Implicit Factors',
                        65))
                elif subset == [10, 11]:
                    plt.title(textwrap.fill(
                        'Optimal Attributable Variance Forecast using High Frequency and Low Frequency Health Care'
                        ' Implicit Factors',
                        60))
                elif subset == [12, 13]:
                    plt.title(textwrap.fill(
                        'Optimal Attributable Variance Forecast using High Frequency and Low Frequency Industrials'
                        ' Implicit Factors',
                        60))
                elif subset == [14, 15]:
                    plt.title(textwrap.fill(
                        'Optimal Attributable Variance Forecast using High Frequency and Low Frequency Information '
                        'Technology Implicit Factors',
                        65))
                plt.pcolormesh(x_mesh, y_mesh, BxxB_day, cmap='gist_rainbow')
                plt.xticks(np.arange(11), [textwrap.fill(col, 15) for col in sector_11_indices.columns], fontsize=8,
                           rotation='45')
                plt.yticks(np.arange(11), [textwrap.fill(col, 15) for col in sector_11_indices.columns], fontsize=8)
                plt.colorbar()
                box_0 = ax.get_position()
                ax.set_position([box_0.x0 + 0.05, box_0.y0 + 0.075, box_0.width * 0.9, box_0.height * 0.9])
                plt.savefig('B and Psi Estimates/BxxB_forecast_CoV_{}'.format(subset))
                plt.show()
                ax = plt.subplot(111)
                if subset == [0]:
                    plt.title(textwrap.fill(
                        'Optimal Variance Forecast using High Frequency Communication Services Implicit Factor',
                        50))
                elif subset == [15]:
                    plt.title(textwrap.fill(
                        'Optimal Variance Forecast using Low Frequency Information Technology Implicit Factor',
                        50))
                elif subset == [17]:
                    plt.title(textwrap.fill(
                        'Optimal Variance Forecast using Low Frequency Materials Implicit Factor', 50))
                elif subset == [21]:
                    plt.title(textwrap.fill(
                        'Optimal Variance Forecast using Low Frequency Utilities Implicit Factor', 50))
                elif subset == [2, 3]:
                    plt.title(textwrap.fill(
                        'Optimal Variance Forecast using High Frequency and Low Frequency Consumer '
                        'Discretionary Implicit Factors',
                        55))
                elif subset == [10, 11]:
                    plt.title(textwrap.fill(
                        'Optimal Variance Forecast using High Frequency and Low Frequency Health Care'
                        ' Implicit Factors',
                        50))
                elif subset == [12, 13]:
                    plt.title(textwrap.fill(
                        'Optimal Variance Forecast using High Frequency and Low Frequency Industrials'
                        ' Implicit Factors',
                        50))
                elif subset == [14, 15]:
                    plt.title(textwrap.fill(
                        'Optimal Variance Forecast using High Frequency and Low Frequency Information '
                        'Technology Implicit Factors',
                        55))
                plt.pcolormesh(x_mesh, y_mesh, variance_forecast_1[var_day], cmap='gist_rainbow')
                plt.xticks(np.arange(11), [textwrap.fill(col, 15) for col in sector_11_indices.columns], fontsize=8,
                           rotation='45')
                plt.yticks(np.arange(11), [textwrap.fill(col, 15) for col in sector_11_indices.columns], fontsize=8)
                plt.colorbar()
                box_0 = ax.get_position()
                ax.set_position([box_0.x0 + 0.05, box_0.y0 + 0.075, box_0.width * 0.9, box_0.height * 0.9])
                plt.savefig('B and Psi Estimates/Sigma_forecast_CoV_{}'.format(subset))
                plt.show()
                ax = plt.subplot(111)
                if subset == [0]:
                    plt.title('High Frequency Communication Services Implicit Factor')
                    plt.plot(np.arange(d1), x[0, :], label=textwrap.fill('High Frequency Communication Services IMF', 13))
                    plt.plot(np.arange(d1), price_signal[int(1176 - d1):int(1176), 0] - x[1, :],
                             label=textwrap.fill('Price minus trend', 13))
                elif subset == [15]:
                    plt.title('Low Frequency Information Technology Implicit Factor')
                    plt.plot(np.arange(d1), x[15, :], label=textwrap.fill('Low Frequency Information Technology IMF', 13))
                    plt.plot(np.arange(d1), price_signal[int(1176 - d1):int(1176), 7],
                             label=textwrap.fill('Price', 11))
                elif subset == [17]:
                    plt.title('Low Frequency Materials Implicit Factor')
                    plt.plot(np.arange(d1), x[17, :],
                             label=textwrap.fill('Low Frequency Materials IMF', 9))
                    plt.plot(np.arange(d1), price_signal[int(1176 - d1):int(1176), 8],
                             label=textwrap.fill('Price', 11))
                elif subset == [21]:
                    plt.title('Low Frequency Utilities Implicit Factor')
                    plt.plot(np.arange(d1), x[21, :],
                             label=textwrap.fill('Low Frequency Utilities IMF', 9))
                    plt.plot(np.arange(d1), price_signal[int(1176 - d1):int(1176), 10],
                             label=textwrap.fill('Price', 11))
                elif subset == [2, 3]:
                    plt.title(textwrap.fill('High and Low Frequency Consumer Discretionary Implicit Factor', 35))
                    plt.plot(np.arange(d1), x[subset[0], :],
                             label=textwrap.fill('High Frequency Consumer Discretionary IMF', 15))
                    plt.plot(np.arange(d1), x[subset[1], :],
                             label=textwrap.fill('Low Frequency Consumer Discretionary IMF', 15))
                    plt.plot(np.arange(d1), price_signal[int(1176 - d1):int(1176), 1],
                             label=textwrap.fill('Price', 11))
                elif subset == [10, 11]:
                    plt.title('High and Low Frequency Health Care Implicit Factor')
                    plt.plot(np.arange(d1), x[subset[0], :],
                             label=textwrap.fill('High Frequency Health Care IMF', 15))
                    plt.plot(np.arange(d1), x[subset[1], :],
                             label=textwrap.fill('Low Frequency Health Care IMF', 15))
                    plt.plot(np.arange(d1), price_signal[int(1176 - d1):int(1176), 5],
                             label=textwrap.fill('Price', 11))
                elif subset == [12, 13]:
                    plt.title('High and Low Frequency Industrials Implicit Factor')
                    plt.plot(np.arange(d1), x[subset[0], :],
                             label=textwrap.fill('High Frequency Industrials IMF', 15))
                    plt.plot(np.arange(d1), x[subset[1], :],
                             label=textwrap.fill('Low Frequency Industrials IMF', 15))
                    plt.plot(np.arange(d1), price_signal[int(1176 - d1):int(1176), 6],
                             label=textwrap.fill('Price', 11))
                elif subset == [14, 15]:
                    plt.title(textwrap.fill('High and Low Frequency Information Technology Implicit Factor', 45))
                    plt.plot(np.arange(d1), x[subset[0], :],
                             label=textwrap.fill('High Frequency Information Technology IMF', 15))
                    plt.plot(np.arange(d1), x[subset[1], :],
                             label=textwrap.fill('Low Frequency Information Technology IMF', 15))
                    plt.plot(np.arange(d1), price_signal[int(1176 - d1):int(1176), 7],
                             label=textwrap.fill('Price', 11))
                plt.xticks([0, 32], ['27 January 2020', '28 February 2020'])
                box_0 = ax.get_position()
                ax.set_position([box_0.x0 - 0.01, box_0.y0 + 0.025, box_0.width * 0.86, box_0.height * 0.98])
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
                plt.savefig('B and Psi Estimates/IMF_CoV_{}'.format(subset))
                plt.show()

            pd.DataFrame(np.median(variance_forecast_1, axis=0)).to_csv('B and Psi Estimates/cov_{}.csv'.format(str(subset)))

        for row in np.arange(np.shape(x_ssa)[0]):
            B_est_ssd = pd.read_csv('B and Psi Estimates/B_est_ssd_CoV_{}.csv'.format(row), header=0)
            B_est_ssd = np.asarray(B_est_ssd.set_index('Unnamed: 0'))
            Psi_est_ssd = pd.read_csv('B and Psi Estimates/Psi_est_ssd_CoV_{}.csv'.format(row), header=0)
            Psi_est_ssd = np.asarray(Psi_est_ssd.set_index('Unnamed: 0'))
            variance_forecast_ssd_1 = np.zeros((d1, 11, 11))
            Psi_ssd = np.zeros((d1, 11, 11))
            BxxB_ssd = np.zeros((d1, 11, 11))

            for var_day in range(d1):
                Psi_day = Psi_est_ssd
                BxxB_day = np.matmul(B_est_ssd.T * x_ssa[row, var_day],
                                     x_ssa[row, var_day] * B_est_ssd).astype(np.float64)
                Psi_ssd[var_day] = Psi_day
                BxxB_ssd[var_day] = BxxB_day
                variance_forecast_ssd_1[var_day] = Psi_day + BxxB_day

            x_mesh, y_mesh = np.meshgrid(np.arange(11), np.arange(11))

            A = variance_forecast_ssd_1[-1, :, :]
            B = np.cov(price_signal[int(1176 - d1):1176, :].T)
            kl_distance = (1 / 2) * (np.sum(np.diag(np.matmul(np.linalg.inv(A), B))) -
                                     np.log(np.linalg.det(B) / np.linalg.det(A)))
            if kl_distance < min_kl_distance:
                min_kl_distance = kl_distance
                row_min = row
            if kl_distance < 98.0:
                print(str(row) + ' = {}'.format(kl_distance))

            if row == 3:
                ax = plt.subplot(111)
                plt.title(textwrap.fill(
                    'Optimal Unattributable Variance Forecast using Communication Services SSD Implicit Factor', 50))

                plt.pcolormesh(x_mesh, y_mesh, Psi_day, cmap='gist_rainbow')
                plt.xticks(np.arange(11), [textwrap.fill(col, 15) for col in sector_11_indices.columns], fontsize=8,
                           rotation='45')
                plt.yticks(np.arange(11), [textwrap.fill(col, 15) for col in sector_11_indices.columns], fontsize=8)
                plt.colorbar()
                box_0 = ax.get_position()
                ax.set_position([box_0.x0 + 0.05, box_0.y0 + 0.075, box_0.width * 0.9, box_0.height * 0.9])
                plt.savefig('B and Psi Estimates/Psi_forecast_ssd_CoV_{}'.format(row))
                plt.show()
                ax = plt.subplot(111)
                plt.title(textwrap.fill(
                    'Optimal Attributable Variance Forecast using Communication Services SSD Implicit Factor', 50))
                plt.pcolormesh(x_mesh, y_mesh, BxxB_day, cmap='gist_rainbow')
                plt.xticks(np.arange(11), [textwrap.fill(col, 15) for col in sector_11_indices.columns], fontsize=8,
                           rotation='45')
                plt.yticks(np.arange(11), [textwrap.fill(col, 15) for col in sector_11_indices.columns], fontsize=8)
                plt.colorbar()
                box_0 = ax.get_position()
                ax.set_position([box_0.x0 + 0.05, box_0.y0 + 0.075, box_0.width * 0.9, box_0.height * 0.9])
                plt.savefig('B and Psi Estimates/BxxB_forecast_ssd_CoV_{}'.format(row))
                plt.show()
                ax = plt.subplot(111)
                plt.title(textwrap.fill(
                    'Optimal Variance Forecast using Communication Services SSD Implicit Factor', 50))
                plt.pcolormesh(x_mesh, y_mesh, variance_forecast_1[var_day], cmap='gist_rainbow')
                plt.xticks(np.arange(11), [textwrap.fill(col, 15) for col in sector_11_indices.columns], fontsize=8,
                           rotation='45')
                plt.yticks(np.arange(11), [textwrap.fill(col, 15) for col in sector_11_indices.columns], fontsize=8)
                plt.colorbar()
                box_0 = ax.get_position()
                ax.set_position([box_0.x0 + 0.05, box_0.y0 + 0.075, box_0.width * 0.9, box_0.height * 0.9])
                plt.savefig('B and Psi Estimates/Sigma_forecast_ssd_CoV_{}'.format(row))
                plt.show()
                ax = plt.subplot(111)
                plt.title('Communication Services SSD Implicit Factor')
                plt.plot(np.arange(d1), x_ssa[row, :], label=textwrap.fill('Communication Services SSD', 13))
                plt.plot(np.arange(d1), x[int(row + 1), :],
                         label=textwrap.fill(' IMF', 11))
                plt.xticks([0, 32], ['27 January 2020', '28 February 2020'])
                box_0 = ax.get_position()
                ax.set_position([box_0.x0 + 0.00, box_0.y0 + 0.025, box_0.width * 0.86, box_0.height * 0.98])
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
                plt.savefig('B and Psi Estimates/SSD_CoV_{}'.format(row))
                plt.show()

            pd.DataFrame(np.median(variance_forecast_1, axis=0)).to_csv(
                'B and Psi Estimates/cov_ssd_row_{}.csv'.format(row))

        print('Minimum KL Distance: ' + str(row) + ' = {}'. format(min_kl_distance))
        A_realised = np.cov(price_signal[int(1176 - 2 * d1):int(1176 - d1), :].T)
        kl_distance_realised = (1 / 2) * (np.sum(np.diag(np.matmul(np.linalg.inv(A_realised), B))) -
                                          np.log(np.linalg.det(B) / np.linalg.det(A_realised)))
        print('Realised covariance KL DIstance: = {}'.format(kl_distance_realised))

        for row in np.arange(np.shape(x_ssa)[0]):

            B_est_ssd = pd.read_csv('B and Psi Estimates/B_est_ssd_CoV_{}.csv'.format(row), header=0)
            B_est_ssd = np.asarray(B_est_ssd.set_index('Unnamed: 0'))
            Psi_est_ssd = pd.read_csv('B and Psi Estimates/Psi_est_ssd_CoV_{}.csv'.format(row), header=0)
            Psi_est_ssd = np.asarray(Psi_est_ssd.set_index('Unnamed: 0'))
            variance_forecast_ssd_1 = np.zeros((d1, 11, 11))
            Psi_ssd = np.zeros((d1, 11, 11))
            BxxB_ssd = np.zeros((d1, 11, 11))

    y = sector_11_indices_array[int(1143 + i + d1):int(1176 + i + d1), :]
    y = y.T

    for subset in sets:
        B_est_direct, Psi_est_direct = \
            cov_reg_given_mean(A_est=A_est, basis=spline_basis, x=x[subset, :], y=y, iterations=100,
                               technique='elastic-net', alpha=0.0001, l1_ratio_or_reg=0.00001)
        # print(x[subset, :])
        B_est_direct = pd.DataFrame(B_est_direct)
        Psi_est_direct = pd.DataFrame(Psi_est_direct)
        B_est_direct.to_csv('B and Psi Estimates/B_est_CoV_{}.csv'.format(str(subset)))
        Psi_est_direct.to_csv('B and Psi Estimates/Psi_est_CoV_{}.csv'.format(str(subset)))

    for row in np.arange(np.shape(x_ssa)[0]):
        B_est_direct_ssd, Psi_est_direct_ssd = \
            cov_reg_given_mean(A_est=A_est, basis=spline_basis, x=x_ssa[row, :].reshape(1, -1), y=y, iterations=100,
                               technique='ridge', alpha=0.0001, l1_ratio_or_reg=0.00001)
        # print(x[subset, :])
        B_est_direct_ssd = pd.DataFrame(B_est_direct_ssd)
        Psi_est_direct_ssd = pd.DataFrame(Psi_est_direct_ssd)
        B_est_direct_ssd.to_csv('B and Psi Estimates/B_est_ssd_CoV_{}.csv'.format(str(row)))
        Psi_est_direct_ssd.to_csv('B and Psi Estimates/Psi_est_ssd_CoV_{}.csv'.format(str(row)))

# Kullback–Leibler divergence
kl_distance = (1 / 2) * (np.sum(np.diag(np.matmul(np.linalg.inv(np.median(variance_forecast_1, axis=0)),
                                                  np.cov(price_signal[int(1176 - d1):1176, :].T)))) -
                         np.log((np.linalg.det(np.cov(price_signal[int(1176 - d1):1176, :].T)))/
                                (np.linalg.det(np.median(variance_forecast_1, axis=0)))))
# print(kl_distance)
x, y = np.meshgrid(np.arange(11), np.arange(11))

window_1 = 82
window_2 = 30
ax = plt.subplot(111)
plt.title('Realised Covariance at Start of Period 1')
plt.pcolormesh(x, y, np.cov(price_signal[int(721 - 2 * window_1):int(721 - window_1), :].T), cmap='gist_rainbow')
plt.xticks(np.arange(11), [textwrap.fill(col, 15) for col in sector_11_indices.columns], fontsize=8, rotation='45')
plt.yticks(np.arange(11), [textwrap.fill(col, 15) for col in sector_11_indices.columns], fontsize=8)
plt.colorbar()
box_0 = ax.get_position()
ax.set_position([box_0.x0 + 0.05, box_0.y0 + 0.075, box_0.width * 0.9, box_0.height * 0.9])
plt.show()
ax = plt.subplot(111)
plt.title('Realised Covariance at End of Period 1')
plt.pcolormesh(x, y, np.cov(price_signal[int(721 - window_1):int(721), :].T), cmap='gist_rainbow')
plt.xticks(np.arange(11), [textwrap.fill(col, 15) for col in sector_11_indices.columns], fontsize=8, rotation='45')
plt.yticks(np.arange(11), [textwrap.fill(col, 15) for col in sector_11_indices.columns], fontsize=8)
plt.colorbar()
box_0 = ax.get_position()
ax.set_position([box_0.x0 + 0.05, box_0.y0 + 0.075, box_0.width * 0.9, box_0.height * 0.9])
plt.show()
ax = plt.subplot(111)
plt.title('Realised Covariance After Recession of Period 1')
plt.pcolormesh(x, y, np.cov(price_signal[int(721):int(721 + window_1), :].T), cmap='gist_rainbow')
plt.xticks(np.arange(11), [textwrap.fill(col, 15) for col in sector_11_indices.columns], fontsize=8, rotation='45')
plt.yticks(np.arange(11), [textwrap.fill(col, 15) for col in sector_11_indices.columns], fontsize=8)
plt.colorbar()
box_0 = ax.get_position()
ax.set_position([box_0.x0 + 0.05, box_0.y0 + 0.075, box_0.width * 0.9, box_0.height * 0.9])
plt.show()
ax = plt.subplot(111)
plt.title('Realised Covariance at Start of Period 2')
plt.pcolormesh(x, y, np.cov(price_signal[int(1143 - window_2):1143, :].T), cmap='gist_rainbow')
plt.xticks(np.arange(11), [textwrap.fill(col, 15) for col in sector_11_indices.columns], fontsize=8, rotation='45')
plt.yticks(np.arange(11), [textwrap.fill(col, 15) for col in sector_11_indices.columns], fontsize=8)
plt.colorbar()
box_0 = ax.get_position()
ax.set_position([box_0.x0 + 0.05, box_0.y0 + 0.075, box_0.width * 0.9, box_0.height * 0.9])
plt.show()
ax = plt.subplot(111)
plt.title('Realised Covariance at End of Period 2')
plt.pcolormesh(x, y, np.cov(price_signal[int(1176 - window_2):1176, :].T), cmap='gist_rainbow')
plt.xticks(np.arange(11), [textwrap.fill(col, 15) for col in sector_11_indices.columns], fontsize=8, rotation='45')
plt.yticks(np.arange(11), [textwrap.fill(col, 15) for col in sector_11_indices.columns], fontsize=8)
plt.colorbar()
box_0 = ax.get_position()
ax.set_position([box_0.x0 + 0.05, box_0.y0 + 0.075, box_0.width * 0.9, box_0.height * 0.9])
plt.show()
ax = plt.subplot(111)
plt.title('Realised Covariance After Recession of Period 2')
plt.pcolormesh(x, y, np.cov(price_signal[1176:int(1176 + window_2), :].T), cmap='gist_rainbow')
plt.xticks(np.arange(11), [textwrap.fill(col, 15) for col in sector_11_indices.columns], fontsize=8, rotation='45')
plt.yticks(np.arange(11), [textwrap.fill(col, 15) for col in sector_11_indices.columns], fontsize=8)
plt.colorbar()
box_0 = ax.get_position()
ax.set_position([box_0.x0 + 0.05, box_0.y0 + 0.075, box_0.width * 0.9, box_0.height * 0.9])
plt.show()

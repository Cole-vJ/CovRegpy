
#     ________
#            /
#      \    /
#       \  /
#        \/

# study coefficient extraction using synthetic structures (sinusoids) and known B coefficients

import numpy as np
import pandas as pd
import yfinance as yf
from AdvEMDpy import emd_basis
from CovRegpy_RCR import cov_reg_given_mean
import matplotlib.pyplot as plt
from scipy.linalg import cholesky
from mpl_toolkits.mplot3d import Axes3D

# set seed for random number generation - consistent results
np.random.seed(0)

# create underlying structures - make structures easily distinguishable
time = np.arange(1098)
imfs_synth = np.zeros((15, 1098))
for row in range(15):
    imfs_synth[row, :] = np.cos(time * (3 * (row + 1) / 1097) * 2 * np.pi +
                                2 * np.pi * np.random.uniform(0, 1))  # easily distinguishable
del row

# plot of synthetic underlying structures
plt.plot(imfs_synth.T)
plt.show()

# daily risk free rate
risk_free = (0.01 / 365)

# establish number of knots
knots = 150

# construct explainable returns from stock factors - create independent returns and then correlate
returns = np.random.normal(risk_free, risk_free, size=(1097, 5))

# create base covariance using realised variance
tickers_format = ['MSFT', 'AAPL', 'GOOGL', 'AMZN', 'TSLA']
data = yf.download(tickers_format, start="2018-12-31", end="2022-01-01")
close_data = data['Close']
date_index = pd.date_range(start='31/12/2018', end='01/01/2022')
close_data = close_data.reindex(date_index).interpolate()
close_data = close_data = close_data[::-1].interpolate()
close_data = close_data = close_data[::-1]
base_covariance = np.cov((np.log(np.asarray(close_data)[1:, :]) - np.log(np.asarray(close_data)[:-1, :])).T)
del close_data, data, date_index, tickers_format

x = np.arange(5)
y = np.arange(5)
x, y = np.meshgrid(x, y)

fig = plt.figure()
fig.set_size_inches(8, 6)
ax = plt.axes(projection='3d')
ax.view_init(30, -70)
ax.set_title('Base Covariance for Five Synthetic Assets')
cov_plot = ax.plot_surface(x, y, base_covariance, rstride=1, cstride=1, cmap='gist_rainbow',
                           edgecolor='none')
ax.set_xticks(ticks=[0, 1, 2, 3, 4])
ax.set_xticklabels(labels=['A (MSFT)', 'B (AAPL)', 'C (GOOGL)', 'D (AMZN)', 'E (TSLA)'], fontsize=8, ha="left", rotation_mode="anchor")
# ax.set_xlabel('Assets', fontsize=8)
ax.set_yticks(ticks=[0, 1, 2, 3, 4])
ax.set_yticklabels(labels=['A (MSFT)', 'B (AAPL)', 'C (GOOGL)', 'D (AMZN)', 'E (TSLA)'], rotation=0, fontsize=8)
# ax.set_ylabel('Assets', fontsize=8)
ax.set_zticks(ticks=[0, 0.001])
ax.set_zticklabels([0, 0.001], fontsize=8)
ax.set_zlabel('Covariance', fontsize=8)
cbar = plt.colorbar(cov_plot)
box_0 = ax.get_position()
ax.set_position([box_0.x0 - 0.05, box_0.y0, box_0.width, box_0.height])
plt.savefig('../aas_figures/Synthetic_base_covariance')
plt.show()

# create correlation structure - random structure
B = np.zeros((5, 15))
for i in range(5):
    for j in range(15):
        rand_uni = np.random.uniform(0, 1)
        B[i, j] = 1 * (-1) ** (i + j) if rand_uni > 1 / 2 else 0
# del i, j, rand_uni

# LARS test - best results
# B[:, 2] = 1
# B[:, 5] = -1
# B[:, 8] = 1
# B[:, 11] = -1
# B[:, 14] = 1

# LARS test - structures too close to be isolated
# B[:, 2] = 1
# B[:, 3] = -1
# B[:, 4] = 1
# B[:, 5] = -1
# B[:, 6] = 1

covariance_structure = np.zeros((5, 5, 1097))
for day in range(np.shape(covariance_structure)[2]):
    covariance_structure[:, :, day] = base_covariance + \
                                      np.matmul(np.matmul(B, imfs_synth[:, day]).reshape(-1, 1),
                                                np.matmul(imfs_synth[:, day].T, B.T).reshape(1, -1))
    # finally correlate returns
    returns[day, :] = cholesky(covariance_structure[:, :, day], lower=True).dot(returns[day, :])
del day

fig = plt.figure()
fig.set_size_inches(8, 6)
ax = plt.subplot(111)
ax.set_title('Covariance for Five Synthetic Assets', fontsize=16)
plt.plot(np.arange(1097), covariance_structure[0, 0, :] * np.ones(1097), label='Asset A')
plt.plot(np.arange(1097), covariance_structure[1, 1, :] * np.ones(1097), label='Asset B')
plt.plot(np.arange(1097), covariance_structure[2, 2, :] * np.ones(1097), label='Asset C')
plt.plot(np.arange(1097), covariance_structure[3, 3, :] * np.ones(1097), label='Asset D')
plt.plot(np.arange(1097), covariance_structure[4, 4, :] * np.ones(1097), label='Asset E')
ax.set_xticks(ticks=[0, 365, 731, 1096])
ax.set_xticklabels(labels=['01-01-2019', '01-01-2020', '01-01-2021', '01-01-2022'], fontsize=8, ha="left",
                   rotation_mode="anchor", rotation=-30)
ax.set_yticks(ticks=[0, 5, 10, 15, 20, 25])
ax.set_yticklabels(labels=[0, 5, 10, 15, 20, 25], fontsize=8)
ax.set_xlabel('Time', fontsize=10)
ax.set_ylabel('Covariance', fontsize=10)
box_0 = ax.get_position()
ax.set_position([box_0.x0 - 0.05, box_0.y0 + 0.025, box_0.width, box_0.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
plt.savefig('../aas_figures/Synthetic_correlation')
plt.show()

cumulative_returns = np.ones((1098, 5))
cumulative_returns[1:, :] = np.cumprod(np.exp(returns), axis=0)
date_index = pd.date_range(start='31/12/2018', end='01/01/2022')
tickers = ['A', 'B', 'C', 'D', 'E']
close_data = pd.DataFrame(columns=tickers, index=date_index)
close_data.iloc[:] = cumulative_returns

fig = plt.figure()
fig.set_size_inches(8, 6)
ax = plt.subplot(111)
ax.set_title('Cumulative Returns for Five Synthetic Assets', fontsize=16)
plt.plot(np.arange(1098), cumulative_returns[:, 0], label='Asset A')
plt.plot(np.arange(1098), cumulative_returns[:, 1], label='Asset B')
plt.plot(np.arange(1098), cumulative_returns[:, 2], label='Asset C')
plt.plot(np.arange(1098), cumulative_returns[:, 3], label='Asset D')
plt.plot(np.arange(1098), cumulative_returns[:, 4], label='Asset E')
ax.set_xticks(ticks=[0, 366, 732, 1097])
ax.set_xticklabels(labels=['31-12-2018', '01-01-2020', '01-01-2021', '01-01-2022'], fontsize=8, ha="left",
                   rotation_mode="anchor", rotation=-30)
ax.set_yticks(ticks=[0.98, 0.99, 1, 1.01, 1.02, 1.03, 1.04, 1.05])
ax.set_yticklabels(labels=[0.98, 0.99, 1, 1.01, 1.02, 1.03, 1.04, 1.05], fontsize=8)
ax.set_ylim(0.975, 1.055)
ax.set_xlabel('Time', fontsize=10)
ax.set_ylabel('Cumulative returns', fontsize=10)
box_0 = ax.get_position()
ax.set_position([box_0.x0 - 0.05, box_0.y0 + 0.025, box_0.width, box_0.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
plt.savefig('../aas_figures/Synthetic_cumulative_returns')
plt.show()

model_days = 701  # 2 years - less a month
forecast_days = np.shape(close_data)[0] - model_days - 30

spline_basis_transform = emd_basis.Basis(time_series=np.arange(model_days), time=np.arange(model_days))
spline_basis_transform = spline_basis_transform.cubic_b_spline(knots=np.linspace(0, model_days - 1, knots))

# test direct estimation recovery of true B coefficients

for lag in range(forecast_days):
    print(lag)

    if lag in [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]:  # 0 : 31-12-2020 # 365 : 31-12-2021

        all_data = imfs_synth[:, lag:int(model_days + lag + 1)].T  # use actual underlying structures
        groups = np.zeros((76, 1))  # LATER
        returns_subset = returns[lag:int(model_days + lag), :]  # extract relevant returns

        realised_covariance = np.cov(returns_subset.T)  # calculation of realised covariance

        coef = np.linalg.lstsq(spline_basis_transform.T, returns_subset, rcond=None)[0]
        mean = np.matmul(coef.T, spline_basis_transform)  # calculate mean

        x = np.asarray(all_data).T

        # calculate covariance regression matrices
        B_est, Psi_est = cov_reg_given_mean(A_est=np.zeros_like(coef), basis=spline_basis_transform,
                                            x=x[:, :-1], y=returns_subset.T,
                                            iterations=10, technique='direct')

        fig, axs = plt.subplots(5, 1)
        assets = ['Asset A', 'Asset B', 'Asset C', 'Asset D', 'Asset E']
        plt.suptitle('True Underlying Coefficients versus Estimated Coefficients')
        for i in range(5):
            if i == 0:
                axs[i].set_title('Note sign is not important - signs must always be identically opposite')
            axs[i].plot(np.arange(1, 16, 1), -B[i, :],
                        label=f'True coefficients underlying returns of asset {assets[i]}')
            axs[i].plot(np.arange(1, 16, 1), (1 / risk_free) * B_est[:, i].T, '--',
                        label=f'Estimate coefficients underlying returns of asset {assets[i]}')
            axs[i].set_ylabel(f'{assets[i]}', rotation=90, fontsize=10)
            axs[i].set_xticks((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15))
            if i == 4:
                axs[i].set_xticklabels(('1', '2', '3', '4', '5',
                                        '6', '7', '8', '9', '10',
                                        '11', '12', '13', '14', '15'), fontsize=8)
                axs[i].set_xlabel('Sinusoidal structures', fontsize=10)
        if lag == 0:
            plt.savefig('../aas_figures/Synthetic_case_study.png')
        plt.show()

        variance_Model = Psi_est + np.matmul(np.matmul(B_est.T,
                                                       x[:, -1]).astype(np.float64).reshape(-1, 1),
                                             np.matmul(x[:, -1].T,
                                                       B_est).astype(np.float64).reshape(1, -1)).astype(np.float64)

# test lasso estimation recovery of true B coefficients

for lag in range(forecast_days):
    print(lag)

    if lag in [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]:  # 0 : 31-12-2020 # 365 : 31-12-2021

        all_data = imfs_synth[:, lag:int(model_days + lag + 1)].T  # use actual underlying structures
        groups = np.zeros((76, 1))  # LATER
        returns_subset = returns[lag:int(model_days + lag), :]  # extract relevant returns

        realised_covariance = np.cov(returns_subset.T)  # calculation of realised covariance

        coef = np.linalg.lstsq(spline_basis_transform.T, returns_subset, rcond=None)[0]
        mean = np.matmul(coef.T, spline_basis_transform)  # calculate mean

        x = np.asarray(all_data).T

        # calculate covariance regression matrices
        B_est, Psi_est = cov_reg_given_mean(A_est=np.zeros_like(coef), basis=spline_basis_transform,
                                            x=x[:, :-1], y=returns_subset.T,
                                            iterations=10, technique='lasso', max_iter=500,
                                            groups=groups, LARS=False, true_coefficients=B, test_lasso=True, alpha=1e-8)

        x_mesh = np.arange(5)
        y = np.arange(5)
        x_mesh, y = np.meshgrid(x_mesh, y)

        if lag == 0 or lag == 31:
            fig = plt.figure()
            fig.set_size_inches(8, 6)
            ax = plt.axes(projection='3d')
            ax.view_init(30, -70)
            ax.set_title('Base Covariance Lasso Estimate for Five Synthetic Assets')
            cov_plot = ax.plot_surface(x_mesh, y, (1 / (risk_free ** 2)) * Psi_est, rstride=1, cstride=1, cmap='gist_rainbow',
                                       edgecolor='none')
            ax.set_xticks(ticks=[0, 1, 2, 3, 4])
            ax.set_xticklabels(labels=['A', 'B', 'C', 'D', 'E'], fontsize=8, ha="left", rotation_mode="anchor")
            ax.set_xlabel('Assets', fontsize=8)
            ax.set_yticks(ticks=[0, 1, 2, 3, 4])
            ax.set_yticklabels(labels=['A', 'B', 'C', 'D', 'E'], rotation=0, fontsize=8)
            ax.set_ylabel('Assets', fontsize=8)
            if lag == 0:
                ax.set_zticks(ticks=[0, 2])
                ax.set_zticklabels([0, 2], fontsize=8)
            if lag == 31:
                ax.set_zticks(ticks=[0, 0.001])
                ax.set_zticklabels([0, 0.001], fontsize=8)
            ax.set_zlabel('Covariance', fontsize=8)
            cbar = plt.colorbar(cov_plot)
            box_0 = ax.get_position()
            ax.set_position([box_0.x0 - 0.05, box_0.y0, box_0.width, box_0.height])
            if lag == 0:
                plt.savefig('../aas_figures/lasso_correlation_zero.png')
            if lag == 31:
                plt.savefig('../aas_figures/lasso_correlation_accurate.png')
            plt.show()

        fig, axs = plt.subplots(5, 1)
        assets = ['Asset A', 'Asset B', 'Asset C', 'Asset D', 'Asset E']
        plt.suptitle('True Underlying Coefficients versus Lasso Estimated Coefficients', fontsize=14)
        for i in range(5):
            if i == 0:
                axs[i].set_title('Note sign is not important - signs must always be identically opposite')
            axs[i].plot(np.arange(1, 16, 1), B[i, :],
                        label=f'True coefficients underlying returns of asset {assets[i]}')
            axs[i].plot(np.arange(1, 16, 1), (1 / risk_free) * B_est[:, i].T, '--',
                        label=f'Estimate coefficients underlying returns of asset {assets[i]}')
            axs[i].set_ylabel(f'{assets[i]}', rotation=90, fontsize=10)
            axs[i].set_xticks((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15))
            if i == 4:
                axs[i].set_xticklabels(('1', '2', '3', '4', '5',
                                        '6', '7', '8', '9', '10',
                                        '11', '12', '13', '14', '15'), fontsize=8)
                axs[i].set_xlabel('Sinusoidal structures', fontsize=10)
        if lag == 0:
            plt.savefig('../aas_figures/Synthetic_case_study_lasso_zero.png')
        elif lag == 31:
            plt.savefig('../aas_figures/Synthetic_case_study_lasso_accurate.png')
        plt.show()

        variance_Model = Psi_est + np.matmul(np.matmul(B_est.T,
                                                       x[:, -1]).astype(np.float64).reshape(-1, 1),
                                             np.matmul(x[:, -1].T,
                                                       B_est).astype(np.float64).reshape(1, -1)).astype(np.float64)


import numpy as np
import pandas as pd
import yfinance as yf
from AdvEMDpy import emd_basis
from CovRegpy_covariance_regression_functions import cov_reg_given_mean
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
# plt.plot(imfs_synth.T)
# plt.show()

# daily risk free rate
risk_free = (0.02 / 365)

# establish number of knots
knots = 150

# construct explainable returns from stock factors - create independent returns and then correlate
returns = np.random.normal(risk_free, risk_free, size=(1097, 5))

# create base covariance using realised variance
tickers_format = ['MSFT', 'AAPL', 'GOOGL', 'AMZN', 'TSLA']
data = yf.download(tickers_format, start="2018-12-31", end="2021-11-14")
close_data = data['Close']
date_index = pd.date_range(start='31/12/2018', end='01/01/2022')
close_data = close_data.reindex(date_index).interpolate()
close_data = close_data = close_data[::-1].interpolate()
close_data = close_data = close_data[::-1]
base_covariance = np.cov((np.log(np.asarray(close_data)[1:, :]) - np.log(np.asarray(close_data)[:-1, :])).T)
del close_data, data, date_index, tickers_format

# create correlation structure - random structure
B = np.zeros((5, 15))
# for i in range(5):
#     for j in range(15):
#         rand_uni = np.random.uniform(0, 1)
#         B[i, j] = 1 * (-1) ** (i + j) if rand_uni > 1 / 2 else 0
# del i, j, rand_uni

# LARS test - best results
B[:, 2] = 1
B[:, 5] = -1
B[:, 8] = 1
B[:, 11] = -1
B[:, 14] = 1
B = 0.01 * B

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
    returns[day, :] = np.dot(cholesky(covariance_structure[:, :, day], lower=True), returns[day, :])
del day

print(np.mean(np.mean(np.mean(np.abs(covariance_structure)))) / np.mean(np.mean(np.abs(base_covariance))))
plt.plot(np.mean(np.mean(np.abs(covariance_structure), axis=0), axis=0) / np.mean(np.mean(np.abs(base_covariance))))
plt.show()

cumulative_returns = np.ones((1098, 5))
cumulative_returns[1:, :] = np.cumprod(np.exp(returns), axis=0)
date_index = pd.date_range(start='31/12/2018', end='01/01/2022')
tickers = ['A', 'B', 'C', 'D', 'E']
close_data = pd.DataFrame(columns=tickers, index=date_index)
close_data.iloc[:] = cumulative_returns

plt.plot(cumulative_returns)
plt.show()

model_days = 701  # 2 years - less a month
forecast_days = np.shape(close_data)[0] - model_days - 30

spline_basis_transform = emd_basis.Basis(time_series=np.arange(model_days), time=np.arange(model_days))
spline_basis_transform = spline_basis_transform.cubic_b_spline(knots=np.linspace(0, model_days - 1, knots))

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
                                            iterations=10, technique='direct', max_iter=500,
                                            groups=groups, LARS=False, true_coefficients=B)

        fig, axs = plt.subplots(5, 1)
        assets = ['Asset A', 'Asset B', 'Asset C', 'Asset D', 'Asset E']
        plt.suptitle('True Underlying Coefficients versus Estimated Coefficents')
        for i in range(5):
            axs[i].plot(np.arange(1, 16, 1), B[i, :],
                        label=f'True coefficents underlying returns of asset {assets[i]}')
            axs[i].plot(np.arange(1, 16, 1), (1 / risk_free) * -B_est[:, i].T, '--',
                        label=f'Estimate coefficents underlying returns of asset {assets[i]}')
            axs[i].set_ylabel(f'{assets[i]}', rotation=90, fontsize=10)
            axs[i].set_xticks((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15))
            # axs[i].set_yticks((-1, 0, 1))
            # axs[i].set_yticklabels(('-1', '0', '1'), fontsize=8)
            if i == 4:
                axs[i].set_xticklabels(('1', '2', '3', '4', '5',
                                        '6', '7', '8', '9', '10',
                                        '11', '12', '13', '14', '15'), fontsize=8)
                axs[i].set_xlabel('Sinusoidal structures', fontsize=10)
        plt.savefig('figures/Synthetic_case_study.png')
        plt.show()

        # x = np.linspace(1, 15, 15)
        # y = np.linspace(1, 5, 5)
        # X, Y = np.meshgrid(x, y)
        # ax = plt.axes(projection='3d')
        # cov_plot = ax.plot_surface(X, Y, B, rstride=1, cstride=1, cmap='gist_rainbow', edgecolor='none')
        # ax.plot_surface(X, Y, -B_est.T * (1 / risk_free) + 100, rstride=1, cstride=1, cmap='gist_rainbow', edgecolor='none')
        # cbar = plt.colorbar(cov_plot)
        # plt.show()

        variance_Model = Psi_est + np.matmul(np.matmul(B_est.T,
                                                       x[:, -1]).astype(np.float64).reshape(-1, 1),
                                             np.matmul(x[:, -1].T,
                                                       B_est).astype(np.float64).reshape(1, -1)).astype(np.float64)


import numpy as np
import seaborn as sns
import textwrap
import matplotlib.pyplot as plt

sns.set(style='darkgrid')

points = 1001

time = np.linspace(0, 1, points)
label = list()
label.append('Strongly Positive')
label.append('Mildly Positive')
label.append('No')
label.append('Mildly Negative')
label.append('Strongly Negative')


np.random.seed(0)
corr = 0.5
points_asset = 304
time_asset = np.arange(304)

# L = np.linalg.cholesky(np.array([[1, corr], [corr, 1]]))
time_series = np.random.normal(0, 1, (100, points_asset)) * 0.0001
time_series[:, 0] = 0
# time_series = np.matmul(L, time_series)
fig = plt.gcf()
fig.set_size_inches(8, 4.5)
plt.title(textwrap.fill(r'Cumulative Returns of 100 Assets', 45))
plt.plot(time_asset, -np.exp(np.cumsum(time_series[:, :], axis=1)).T + 2)
# plt.plot(time_asset, -np.exp(np.cumsum(time_series[1, :])) + 2, c='blue')
plt.plot(int(points_asset - 1) * np.ones(101), -np.linspace(0.993, 1.006, 101) + 2, 'k--')
plt.plot(395 * np.ones(101), -np.linspace(0.993, 1.006, 101) + 2, 'k--')
plt.fill(np.append(np.linspace(int(points_asset - 1), 395, 101), np.linspace(395, int(points_asset - 1), 101)),
         -np.append(0.993 * np.ones(101), 1.006 * np.ones(101)) + 2, c='lightcyan')
plt.text(338, 1.0009, '?', fontsize=32)
plt.savefig('/home/cole/Desktop/Cole/Cole Documents/CovRegpy/CovRegpy/figures/covariance_demonstration/corr_asset')
plt.ylabel('Cumulative Returns')
plt.xlabel('Days')
plt.show()

i = 0
for corr in [0.9, 0.5, 0, -0.5, -0.9]:
    L = np.linalg.cholesky(np.array([[1, corr], [corr, 1]]))
    time_series = np.random.normal(0, 1, (2, points)) * 0.0001
    time_series = np.matmul(L, time_series)
    plt.title(textwrap.fill(r'Cumulative Returns of Assets with {} Correlation ($\rho = {}$)'.format(label[i], corr), 45))
    i += 1
    plt.plot(time, np.exp(np.cumsum(time_series[0, :])), c='darkred')
    plt.plot(time, np.exp(np.cumsum(time_series[1, :])), c='blue')
    plt.savefig('/home/cole/Desktop/Cole/Cole Documents/CovRegpy/CovRegpy/figures/covariance_demonstration/corr_{}'.format(i))
    plt.show()

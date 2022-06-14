
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

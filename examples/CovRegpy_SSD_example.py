
#     ________
#            /
#      \    /
#       \  /
#        \/

import textwrap
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
import matplotlib.pyplot as plt

sns.set(style='darkgrid')

# uncomment if using installed CovRegpy package in custom environment
# from CovRegpy.CovRegpy_SSD import CovRegpy_ssd

# uncomment if using function directly from within this downloaded GitHub package
from CovRegpy_SSD import CovRegpy_ssd

begin = 0
end = 1
points = int(7.5 * 512)
x = np.linspace(begin, end, points)

signal_1 = np.sin(250 * np.pi * x ** 2)
signal_2 = np.sin(80 * np.pi * x ** 2)

signal = signal_1 + signal_2

decomposition = CovRegpy_ssd(signal, plot=True, debug=True)

# pull all close data
tickers_format = ['MSFT', 'AAPL', 'GOOGL', 'AMZN', 'TSLA']
data = yf.download(tickers_format, start="2018-10-15", end="2021-10-16")
close_data = data['Close']
del data, tickers_format

# create date range and interpolate
date_index = pd.date_range(start='16/10/2018', end='16/10/2021')
close_data = close_data.reindex(date_index).interpolate()
close_data = close_data[::-1].interpolate()
close_data = close_data[::-1]
del date_index

# singular spectrum decomposition
test = CovRegpy_ssd(np.asarray(close_data['MSFT'][-100:]), nmse_threshold=0.05, plot=True)
plt.plot(test.T)
plt.show()

# figures for paper

np.random.seed(0)

x11_time = np.linspace(0, 120, 121)
x11_trend_cycle = (1 / 100) * (x11_time - 10) * (x11_time - 60) * (x11_time - 110) + 1000
x11_seasonal = 100 * np.sin((2 * np.pi / 12) * x11_time)
x11_noise = 100 * np.random.normal(0, 1, 121)
x11_time_series = x11_trend_cycle + x11_seasonal + x11_noise

plt.plot(x11_time, x11_time_series)
plt.title('Additive X11 Example Time Series')
plt.xticks([0, 20, 40, 60, 80, 100, 120], fontsize=8)
plt.yticks([400, 600, 800, 1000, 1200, 1400, 1600], fontsize=8)
plt.show()

ssd_decomp = CovRegpy_ssd(x11_time_series, initial_trend_ratio=10.0, plot=True)

fig, axs = plt.subplots(3, 1)
plt.subplots_adjust(hspace=0.3)
fig.suptitle('Additive Decomposition SSA Demonstration')
axs[0].plot(x11_time, x11_trend_cycle, label='Component')
axs[0].plot(x11_time, ssd_decomp[0, :], 'r--', label='SSD component 1')
axs[0].set_xticks([0, 20, 40, 60, 80, 100, 120])
axs[0].set_xticklabels(['', '', '', '', '', '', ''], fontsize=8)
axs[0].set_yticks([500, 1000, 1500])
axs[0].set_yticklabels(['500', '1000', '1500'], fontsize=8)
axs[0].set_title('Trend-Cycle Component')
box_0 = axs[0].get_position()
axs[0].set_position([box_0.x0 - 0.05, box_0.y0, box_0.width * 0.84, box_0.height])
axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
axs[1].plot(x11_time, x11_seasonal, label='Component')
axs[1].plot(x11_time, ssd_decomp[1, :], 'r--', label='SSD component 2')
axs[1].set_xticks([0, 20, 40, 60, 80, 100, 120])
axs[1].set_xticklabels(['', '', '', '', '', '', ''], fontsize=8)
axs[1].set_yticks([-100, 0, 100])
axs[1].set_yticklabels(['-100', '0', '100'], fontsize=8)
axs[1].set_ylim(-175, 175)
axs[1].set_title('Seasonal Component')
box_1 = axs[1].get_position()
axs[1].set_position([box_1.x0 - 0.05, box_1.y0, box_1.width * 0.84, box_1.height])
axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
axs[2].plot(x11_time, x11_noise, label='Component')
axs[2].plot(x11_time, np.sum(ssd_decomp[2:, :], axis=0), 'r--',
            label=textwrap.fill('SSD component 3 onwards summed', 15))
axs[2].set_xticks([0, 20, 40, 60, 80, 100, 120])
axs[2].set_xticklabels(['0', '20', '40', '60', '80', '100', '120'], fontsize=8)
axs[2].set_yticks([-200, 0, 200])
axs[2].set_yticklabels(['-200', '0', '200'], fontsize=8)
axs[2].set_ylim(-250, 250)
axs[2].set_title('Random Error')
box_2 = axs[2].get_position()
axs[2].set_position([box_2.x0 - 0.05, box_2.y0, box_2.width * 0.84, box_2.height])
axs[2].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
# axs[2].set_xlabel('Months')
plt.savefig('../aas_figures/Example_ssd_decomposition')
plt.show()

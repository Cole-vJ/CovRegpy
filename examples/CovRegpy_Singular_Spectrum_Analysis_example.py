
#     ________
#            /
#      \    /
#       \  /
#        \/

import textwrap
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style='darkgrid')

# uncomment if using installed CovRegpy package in custom environment
# from CovRegpy.CovRegpy_SSA import CovRegpy_ssa

# uncomment if using function directly from within this downloaded GitHub package
from CovRegpy_SSA import CovRegpy_ssa

np.random.seed(0)

time = np.linspace(0, 120, 121)
trend_cycle = (1 / 100) * (time - 10) * (time - 60) * (time - 110) + 1000
seasonal = 100 * np.sin((2 * np.pi / 12) * time)
noise = 100 * np.random.normal(0, 1, 121)
time_series = trend_cycle + seasonal + noise

plt.plot(time, time_series)
plt.title('Additive Example Time Series')
plt.xticks([0, 20, 40, 60, 80, 100, 120], fontsize=8)
plt.yticks([400, 600, 800, 1000, 1200, 1400, 1600], fontsize=8)
plt.show()

ssa_decomp = CovRegpy_ssa(time_series, L=10, est=8)

fig, axs = plt.subplots(3, 1)
plt.subplots_adjust(hspace=0.3)
fig.suptitle('Additive Decomposition SSA Demonstration')
axs[0].plot(time, trend_cycle, label='Component')
axs[0].plot(time, ssa_decomp[0], 'r--', label='SSA trend')
axs[0].plot(time, ssa_decomp[1][0, :], 'g--', label=textwrap.fill('D-SSA first component', 12))
axs[0].set_xticks([0, 20, 40, 60, 80, 100, 120])
axs[0].set_xticklabels(['', '', '', '', '', '', ''], fontsize=8)
axs[0].set_yticks([500, 1000, 1500])
axs[0].set_yticklabels(['500', '1000', '1500'], fontsize=8)
axs[0].set_title('Trend-Cycle Component')
box_0 = axs[0].get_position()
axs[0].set_position([box_0.x0 - 0.05, box_0.y0, box_0.width * 0.84, box_0.height])
axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
axs[1].plot(time, seasonal, label='Component')
axs[1].plot(time, ssa_decomp[1][1, :], 'g--', label=textwrap.fill('D-SSA second component', 12))
axs[1].plot(time, ssa_decomp[1][1, :] + ssa_decomp[1][2, :], 'r--',
            label=textwrap.fill('D-SSA second and third components', 12))
axs[1].set_xticks([0, 20, 40, 60, 80, 100, 120])
axs[1].set_xticklabels(['', '', '', '', '', '', ''], fontsize=8)
axs[1].set_yticks([-100, 0, 100])
axs[1].set_yticklabels(['-100', '0', '100'], fontsize=8)
axs[1].set_ylim(-175, 175)
axs[1].set_title('Seasonal Component')
box_1 = axs[1].get_position()
axs[1].set_position([box_1.x0 - 0.05, box_1.y0, box_1.width * 0.84, box_1.height])
axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
axs[2].plot(time, noise, label='Component')
axs[2].plot(time, ssa_decomp[1][3, :], 'g--', label=textwrap.fill('D-SSA fourth component', 12))
axs[2].plot(time, np.sum(ssa_decomp[1][3:, :], axis=0), 'r--',
            label=textwrap.fill('D-SSA fourth and onwards components', 12))
axs[2].set_xticks([0, 20, 40, 60, 80, 100, 120])
axs[2].set_xticklabels(['0', '20', '40', '60', '80', '100', '120'], fontsize=8)
axs[2].set_yticks([-200, 0, 200])
axs[2].set_yticklabels(['-200', '0', '200'], fontsize=8)
axs[2].set_ylim(-250, 250)
axs[2].set_title('Random Error')
box_2 = axs[2].get_position()
axs[2].set_position([box_2.x0 - 0.05, box_2.y0, box_2.width * 0.84, box_2.height])
axs[2].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
plt.savefig('../aas_figures/Example_ssa_decomposition')
plt.show()

ssa_decomp = CovRegpy_ssa(time_series, L=10, est=8, KS_test=True, plot_KS_test=False)

plt.plot(time, trend_cycle, label='Component')
plt.plot(time, ssa_decomp[0], 'r--', label='KS-SSA trend')
plt.xticks([0, 20, 40, 60, 80, 100, 120], ['0', '20', '40', '60', '80', '100', '120'], fontsize=8)
plt.xlabel('Months', fontsize=10)
plt.yticks([500, 1000, 1500], ['500', '1000', '1500'], fontsize=8)
plt.title('Trend-Cycle Component and KS-SSA Trend Estimate')
plt.legend(loc='best', fontsize=8)
plt.savefig('../aas_figures/Example_ks_ssa')
plt.show()

# Kolmogorov–Smirnov assisted SSA experiment

# adequate KS_scale_limit - need to find intelligent way of automating
ssa_decomp = CovRegpy_ssa(time_series, L=10, est=8, KS_test=True, plot_KS_test=False,
                          KS_start=4, KS_interval=2, KS_end=30, plot=True, KS_scale_limit=100.0)
# KS_scale_limit too small - trend collapses to time series
ssa_decomp = CovRegpy_ssa(time_series, L=10, est=8, KS_test=True, plot_KS_test=False,
                          KS_start=4, KS_interval=2, KS_end=30, plot=True, KS_scale_limit=1.0)

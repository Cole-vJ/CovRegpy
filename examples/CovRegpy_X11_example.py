
#     ________
#            /
#      \    /
#       \  /
#        \/

import textwrap
import numpy as np
import cvxpy as cvx
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style='darkgrid')

# uncomment if using installed CovRegpy package in custom environment
# from CovRegpy.CovRegpy_X11 import henderson_weights, CovRegpy_X11

# uncomment if using function directly from within this downloaded GitHub package
from CovRegpy_X11 import henderson_weights, CovRegpy_X11

time = np.linspace(0, 120, 121)
time_series = \
    time + (1 / 1000) * (time * (time - 60) * (time - 110)) + 10 * np.sin(((2 * np.pi) / 12) * time) + \
    np.random.normal(0, 5, 121)

# Henderson symmetric filter calculation from first principles - reproduces values exactly
# Closed form solution to problem exists

henderson_13 = henderson_weights(13)
index = np.asarray((-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6))
vx = cvx.Variable(19)
objective = cvx.Minimize(cvx.norm(vx[3:] - 3 * vx[2:-1] + 3 * vx[1:-2] - vx[:-3]))
constraints = []
constraints += [sum(vx) == 1]
constraints += [vx[3:-3].T * index == 0]
constraints += [vx[3:-3].T * index ** 2 == 0]
constraints += [vx[0] == 0]
constraints += [vx[1] == 0]
constraints += [vx[2] == 0]
constraints += [vx[-3] == 0]
constraints += [vx[-2] == 0]
constraints += [vx[-1] == 0]
prob = cvx.Problem(objective, constraints)
result = prob.solve(verbose=True, solver=cvx.ECOS)
filtered_signal = np.array(vx.value)
plt.plot(henderson_13, label='Henderson function test')
plt.plot(filtered_signal[3:-3], '--', label='Direct calculation test')
plt.legend(loc='best')
plt.title('Henderson 13 Points Weights Calculation Test')
plt.show()

filtered_time_series = CovRegpy_X11(time, time_series, trend_window_width_3=23)

plt.plot(time, time_series, label='Time series')
plt.plot(time, filtered_time_series[0], label='Trend estimate')
plt.plot(time, filtered_time_series[1], label='Seasonal estimate')
plt.plot(time, filtered_time_series[2], label='Error estimate')
plt.legend(loc='best')
plt.title(textwrap.fill('Example X11 Time Series, Trend Estimate, Seasonal Estimate, and Error Estimate', 40))
plt.show()

# figures for paper

np.random.seed(0)

x11_time = np.linspace(0, 120, 121)
x11_trend_cycle = (1 / 100) * (x11_time - 10) * (x11_time - 60) * (x11_time - 110) + 1000
x11_seasonal = 100 * np.sin((2 * np.pi / 12) * x11_time)
x11_noise = 100 * np.random.normal(0, 1, 121)
x11_time_series = x11_trend_cycle + x11_seasonal + x11_noise

plt.plot(x11_time, x11_time_series)
plt.title('Additive Synthetic Time Series')
plt.xticks([0, 20, 40, 60, 80, 100, 120], fontsize=8)
plt.yticks([400, 600, 800, 1000, 1200, 1400, 1600], fontsize=8)
plt.ylabel('Numeraire')
plt.xlabel('t')
plt.savefig('../aas_figures/Example_time_series')
plt.show()

x11_decomp = CovRegpy_X11(x11_time, x11_time_series, seasonal_factor='3x3',
                          trend_window_width_1=13, trend_window_width_2=13, trend_window_width_3=13)

fig, axs = plt.subplots(3, 1)
plt.subplots_adjust(hspace=0.3)
fig.suptitle('Additive X11 Decomposition Demonstration')
axs[0].plot(x11_time, x11_trend_cycle)
axs[0].plot(x11_time, x11_decomp[0], 'r--')
axs[0].set_xticks([0, 20, 40, 60, 80, 100, 120])
axs[0].set_xticklabels(['', '', '', '', '', '', ''])
axs[0].set_yticks([500, 1000, 1500])
axs[0].set_yticklabels(['500', '1000', '1500'], fontsize=8)
axs[0].set_ylim(250, 1750)
axs[0].set_title('Trend-Cycle Component')
box_0 = axs[0].get_position()
axs[0].set_position([box_0.x0 - 0.05, box_0.y0, box_0.width * 0.95, box_0.height])
axs[1].plot(x11_time, x11_seasonal, label='Component')
axs[1].plot(x11_time, x11_decomp[1], 'r--', label='X11 estimate')
axs[1].set_xticks([0, 20, 40, 60, 80, 100, 120])
axs[1].set_xticklabels(['', '', '', '', '', '', ''])
axs[1].set_yticks([-100, 0, 100])
axs[1].set_yticklabels(['-100', '0', '100'], fontsize=8)
axs[1].set_ylim(-180, 180)
axs[1].set_title('Seasonal Component')
box_1 = axs[1].get_position()
axs[1].set_position([box_1.x0 - 0.05, box_1.y0, box_1.width * 0.95, box_1.height])
axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
axs[2].plot(x11_time, x11_noise)
axs[2].plot(x11_time, x11_decomp[2], 'r--')
axs[2].set_xticks([0, 20, 40, 60, 80, 100, 120])
axs[2].set_xticklabels(['0', '20', '40', '60', '80', '100', '120'], fontsize=8)
axs[2].set_yticks([-200, 0, 200])
axs[2].set_yticklabels(['-200', '0', '200'], fontsize=8)
axs[2].set_ylim(-250, 250)
axs[2].set_xlabel('Months', fontsize=10)
axs[2].set_title('Random Error')
box_2 = axs[2].get_position()
axs[2].set_position([box_2.x0 - 0.05, box_2.y0, box_2.width * 0.95, box_2.height])
plt.savefig('../aas_figures/Example_x11_decomposition')
plt.show()

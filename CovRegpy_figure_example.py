
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='darkgrid')

ax = plt.subplot(111)
plt.gcf().subplots_adjust(bottom=0.15)
for i in range(14):
    plt.plot(i * np.ones(100), np.linspace(0, 1, 100), '--', c='whitesmoke', zorder=1)
plt.title('Demonstrating Covariate and Response Variable Delay')
plt.fill(np.append(0.0 * np.ones(100), 0.5 * np.ones(100)),
         np.append(np.linspace(0.7, 0.6, 100), np.linspace(0.6, 0.7, 100)),
         c='darkmagenta', zorder=2, label='IMFs discarded')
plt.fill(np.append(0.5 * np.ones(100), 11 * np.ones(100)),
         np.append(np.linspace(0.7, 0.6, 100), np.linspace(0.6, 0.7, 100)),
         c='b', zorder=2, label='IMFs for fitting (X)')
plt.fill(np.append(11 * np.ones(100), 12 * np.ones(100)),
         np.append(np.linspace(0.7, 0.6, 100), np.linspace(0.6, 0.7, 100)),
         c='deepskyblue', zorder=2, label='IMFs for forecasting')
plt.plot(np.zeros(100), np.linspace(0.55, 0.75, 100), '--', c='k', zorder=3)
plt.plot(11 * np.ones(100), np.linspace(0.55, 0.75, 100), '--', c='k', zorder=3)
plt.plot(12 * np.ones(100), np.linspace(0.55, 0.75, 100), '--', c='k', zorder=3)
plt.text(5.30, 0.90, r'$T$')
plt.plot(np.linspace(0.5, 11, 100), 0.875 * np.ones(100), 'k-')
plt.plot(0.5 * np.ones(100), np.linspace(0.86, 0.89, 100), 'k-')
plt.plot(11 * np.ones(100), np.linspace(0.86, 0.89, 100), 'k-')
plt.text(5.20, 0.80, r'$T_X$')
plt.plot(np.linspace(0, 11, 100), 0.775 * np.ones(100), 'k-')
plt.plot(np.zeros(100), np.linspace(0.76, 0.79, 100), 'k-')
plt.plot(11 * np.ones(100), np.linspace(0.76, 0.79, 100), 'k-')
plt.fill(np.append(np.ones(100), 12 * np.ones(100)), np.append(np.linspace(0.4, 0.3, 100), np.linspace(0.3, 0.4, 100)),
         c='g', zorder=2, label='Returns for fitting (Y)')
plt.fill(np.append(12 * np.ones(100), 13 * np.ones(100)), np.append(np.linspace(0.4, 0.3, 100), np.linspace(0.3, 0.4, 100)),
         c='lime', zorder=2, label='Forecasted')
plt.plot(np.ones(100), np.linspace(0.25, 0.45, 100), '--', c='k', zorder=3)
plt.plot(12 * np.ones(100), np.linspace(0.25, 0.45, 100), '--', c='k', zorder=3)
plt.plot(13 * np.ones(100), np.linspace(0.25, 0.45, 100), '--', c='k', zorder=3)
plt.text(6.20, 0.15, r'$T_Y$')
plt.plot(np.linspace(1, 12, 100), 0.225 * np.ones(100), 'k-')
plt.plot(np.ones(100), np.linspace(0.21, 0.24, 100), 'k-')
plt.plot(12 * np.ones(100), np.linspace(0.21, 0.24, 100), 'k-')
plt.text(6.30, 0.05, r'$T$')
plt.plot(np.linspace(1, 12, 100), 0.125 * np.ones(100), 'k-')
plt.plot(np.ones(100), np.linspace(0.11, 0.14, 100), 'k-')
plt.plot(12 * np.ones(100), np.linspace(0.11, 0.14, 100), 'k-')
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
plt.xlabel('Months', fontsize=10)
plt.yticks([0.65, 0.35], ['IMFs', 'Returns'], fontsize=10, rotation=60)
plt.grid(b=None)
box_0 = ax.get_position()
ax.set_position([box_0.x0 - 0.04, box_0.y0, box_0.width * 0.84, box_0.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
plt.savefig('figures/Inteval_plot.png')
plt.show()

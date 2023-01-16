
#     ________
#            /
#      \    /
#       \  /
#        \/

import textwrap
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from CovRegpy_forecasting import gp_forecast
from CovRegpy_IFF import CovRegpy_IMF_IFF
from sklearn.gaussian_process.kernels import ExpSineSquared, RBF, RationalQuadratic
from sklearn.linear_model import Ridge

sns.set(style='darkgrid')

def CovRegpy_neural_network(time_series, no_sample=300, fit_window=200, alpha=1.0):

    X = np.zeros((no_sample, fit_window))
    y = np.zeros(no_sample)

    for model in range(no_sample):
        X[model, :] = time_series[-int(no_sample + fit_window - model):-int(no_sample - model)]
        y[model] = time_series[-int(no_sample - model)]

    clf = Ridge(alpha=alpha)
    clf.fit(X, y)

    forecast_time_series = np.zeros(fit_window)

    for model in range(fit_window):
        if model == 0:
            forecast_time_series[model] = clf.predict(time_series[-int(fit_window - model):].reshape(1, -1))
        else:
            forecast_time_series[model] = clf.predict(np.append(time_series[-int(fit_window - model):],
                                                                forecast_time_series[:model]).reshape(1, -1))

    return forecast_time_series

time = np.linspace(0, 5 * np.pi, 1001)
time_extended = np.linspace(5 * np.pi, 6 * np.pi, 201)[1:]
time_series = np.linspace(1, 5, 1001) * np.sin(np.append(0, np.cumsum(np.linspace(1, 2, 1000) * np.diff(time))))

ax = plt.subplot(111)
plt.title(textwrap.fill('Example Time Series Modulated Amplitude and Modulated Frequency', 40), fontsize=16)
plt.plot(time, time_series)
plt.plot(5 * np.pi * np.ones(101), np.linspace(-6, 6, 101), 'k--')
plt.plot(6 * np.pi * np.ones(101), np.linspace(-6, 6, 101), 'k--')
plt.fill(np.append(np.linspace(5 * np.pi, 6 * np.pi, 101), np.linspace(6 * np.pi, 5 * np.pi, 101)),
         np.append(-6 * np.ones(101), 6 * np.ones(101)), c='lightcyan')
plt.text(5.4 * np.pi, 0, '?', fontsize=16)
ax.set_xticks([0, np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi, 5 * np.pi, 6 * np.pi])
ax.set_xticklabels(['0', r'$\pi$', r'$2\pi$', r'$3\pi$', r'$4\pi$', r'$5\pi$', r'$6\pi$'])
ax.set_xlim(-0.25 * np.pi, 6.25 * np.pi)
plt.savefig('aas_figures/forecasting_time_series.png')
plt.show()

np.random.seed(3)

edge = 304

time_asset = np.arange(edge)
time_extended_asset = np.arange(int(edge - 1), 395)[1:]
returns = np.random.normal(0.0001, 0.001, edge)
returns[0] = 0
time_series_asset = (np.cumprod([np.exp(returns)]) - 1) * (0.03 / 0.04) + 1

ax = plt.subplot(111)
plt.title(textwrap.fill('Forecasting Financial Instruments', 40), fontsize=16)
plt.plot(time_asset, time_series_asset)
plt.plot(int(edge - 1) * np.ones(101), np.linspace(1.02, 1.04, 101), 'k--')
plt.plot(395 * np.ones(101), np.linspace(1.02, 1.04, 101), 'k--')
plt.fill(np.append(np.linspace(int(edge - 1), 395, 101), np.linspace(395, int(edge - 1), 101)),
         np.append(1.02 * np.ones(101), 1.04 * np.ones(101)), c='lightcyan')
plt.text(339, 1.028, '?', fontsize=32)
plt.ylabel('Cumulative Returns')
plt.xlabel('Days')
plt.savefig('aas_figures/forecasting_asset.png')
plt.show()

ax = plt.subplot(111)
plt.gcf().subplots_adjust(bottom=0.15)
for i in range(14):
    plt.plot(i * np.ones(100), np.linspace(0, 1, 100), '--', c='whitesmoke', zorder=1)
plt.title('Lagged Effects Model', fontsize=16)
plt.fill(np.append(0.0 * np.ones(100), 11 * np.ones(100)),
         np.append(np.linspace(0.7, 0.6, 100), np.linspace(0.6, 0.7, 100)),
         c='b', zorder=2, label=textwrap.fill('Dependent variable for fitting', 20))
plt.fill(np.append(11 * np.ones(100), 12 * np.ones(100)),
         np.append(np.linspace(0.7, 0.6, 100), np.linspace(0.6, 0.7, 100)),
         c='deepskyblue', zorder=2, label=textwrap.fill('Dependent variable for forecasting', 20))
plt.plot(np.zeros(100), np.linspace(0.55, 0.75, 100), '--', c='k', zorder=3)
plt.plot(11 * np.ones(100), np.linspace(0.55, 0.75, 100), '--', c='k', zorder=3)
plt.plot(12 * np.ones(100), np.linspace(0.55, 0.75, 100), '--', c='k', zorder=3)
plt.text(5.20, 0.80, r'$T_X$')
plt.plot(np.linspace(0, 11, 100), 0.775 * np.ones(100), 'k-')
plt.plot(np.zeros(100), np.linspace(0.76, 0.79, 100), 'k-')
plt.plot(11 * np.ones(100), np.linspace(0.76, 0.79, 100), 'k-')
plt.fill(np.append(np.ones(100), 12 * np.ones(100)), np.append(np.linspace(0.4, 0.3, 100), np.linspace(0.3, 0.4, 100)),
         c='g', zorder=2, label=textwrap.fill('Independent variable for fitting', 20))
plt.fill(np.append(12 * np.ones(100), 13 * np.ones(100)), np.append(np.linspace(0.4, 0.3, 100), np.linspace(0.3, 0.4, 100)),
         c='lime', zorder=2, label=textwrap.fill('Independent variable forecasted', 20))
plt.plot(np.ones(100), np.linspace(0.25, 0.45, 100), '--', c='k', zorder=3)
plt.plot(12 * np.ones(100), np.linspace(0.25, 0.45, 100), '--', c='k', zorder=3)
plt.plot(13 * np.ones(100), np.linspace(0.25, 0.45, 100), '--', c='k', zorder=3)
plt.text(6.20, 0.15, r'$T_Y$')
plt.plot(np.linspace(1, 12, 100), 0.225 * np.ones(100), 'k-')
plt.plot(np.ones(100), np.linspace(0.21, 0.24, 100), 'k-')
plt.plot(12 * np.ones(100), np.linspace(0.21, 0.24, 100), 'k-')
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
plt.xlabel('Time', fontsize=12)
plt.yticks([0.95, 0.45], ['Independent Variable', 'Dependent Variable'], fontsize=10, rotation=90)
plt.grid(visible=None)
box_0 = ax.get_position()
ax.set_position([box_0.x0 - 0.04, box_0.y0, box_0.width * 0.84, box_0.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
plt.savefig('aas_figures/forecasting_interval.png')
plt.show()

# interval plot

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
plt.grid(visible=None)
box_0 = ax.get_position()
ax.set_position([box_0.x0 - 0.04, box_0.y0, box_0.width * 0.84, box_0.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
plt.savefig('figures/Inteval_plot.png')
plt.show()

neural_network_forecast = CovRegpy_neural_network(time_series)

ax = plt.subplot(111)
plt.gcf().subplots_adjust(bottom=0.15)
plt.title('Neural Network Forecasting', fontsize=16)
plt.plot(time, time_series, label='Time series')
plt.plot(time_extended, neural_network_forecast, 'k-', label=textwrap.fill('Time series forecast', 12))
plt.plot(4 * np.pi * np.ones(101), np.linspace(-8, 6, 101), 'r--')
plt.plot(5 * np.pi * np.ones(101), np.linspace(-8, 6, 101), 'r--', label=textwrap.fill('Forecasting window', 12))
ax.set_xticks([0, np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi, 5 * np.pi, 6 * np.pi])
ax.set_xticklabels(['0', r'$\pi$', r'$2\pi$', r'$3\pi$', r'$4\pi$', r'$5\pi$', r'$6\pi$'])
ax.set_xlim(-0.25 * np.pi, 6.25 * np.pi)
box_0 = ax.get_position()
ax.set_position([box_0.x0 - 0.04, box_0.y0, box_0.width * 0.94, box_0.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
plt.savefig('aas_figures/neural_network.png')
plt.show()

# capture varying frequency and amplitude
k1 = RBF(length_scale=90.0) * ExpSineSquared(length_scale=1.3, periodicity=1.0)
k2 = RationalQuadratic(length_scale=1.2, alpha=0.78) * ExpSineSquared(length_scale=1.3, periodicity=1.0)
k3 = ExpSineSquared(length_scale=1.0, periodicity=1.0)
kernel = k1 + k2 + k3

y_forecast, sigma, y_forecast_upper, y_forecast_lower = \
    gp_forecast(time, time_series, time_extended, kernel, 0.95, plot=False)

ax = plt.subplot(111)
plt.gcf().subplots_adjust(bottom=0.15)
plt.title('Gaussian Process Forecasting', fontsize=16)
plt.plot(time, time_series, label='Time series')
plt.plot(time_extended, y_forecast, 'k', label=textwrap.fill('Time series forecast', 12))
plt.plot(time_extended, y_forecast_upper, 'r', label=textwrap.fill('95% upper bound', 12))
plt.plot(time_extended, y_forecast_lower, 'blue', label=textwrap.fill('95% lower bound', 12))
plt.fill(np.append(time_extended, time_extended[::-1]), np.append(y_forecast_upper, y_forecast[::-1]), c='lightcoral',
         label=textwrap.fill('95% upper bound area', 12))
plt.fill(np.append(time_extended, time_extended[::-1]), np.append(y_forecast_lower, y_forecast[::-1]), c='c',
         label=textwrap.fill('95% lower bound area', 12))
ax.set_xticks([0, np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi, 5 * np.pi, 6 * np.pi])
ax.set_xticklabels(['0', r'$\pi$', r'$2\pi$', r'$3\pi$', r'$4\pi$', r'$5\pi$', r'$6\pi$'])
ax.set_xlim(-0.25 * np.pi, 6.25 * np.pi)
box_0 = ax.get_position()
ax.set_position([box_0.x0 - 0.04, box_0.y0, box_0.width * 0.94, box_0.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
plt.savefig('aas_figures/gaussian_process.png')
plt.show()

y_forecast_iff = CovRegpy_IMF_IFF(time, time_series, type='linear', optimisation='l2',
                                  components=1, fit_window=500, forecast_window=200, debug=True)

ax = plt.subplot(111)
plt.gcf().subplots_adjust(bottom=0.15)
plt.title('Instantaneous Frequency Forecasting', fontsize=16)
plt.plot(time, time_series,
         label=textwrap.fill('Time series', 12))
plt.plot(y_forecast_iff[1], y_forecast_iff[0], 'k',
         label=textwrap.fill('Time series forecast', 12))
ax.set_xticks([0, np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi, 5 * np.pi, 6 * np.pi])
ax.set_xticklabels(['0', r'$\pi$', r'$2\pi$', r'$3\pi$', r'$4\pi$', r'$5\pi$', r'$6\pi$'])
ax.set_xlim(-0.25 * np.pi, 6.25 * np.pi)
box_0 = ax.get_position()
ax.set_position([box_0.x0 - 0.04, box_0.y0, box_0.width * 0.94, box_0.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
plt.savefig('aas_figures/iff.png')
plt.show()

sigma = 1
l = 1
tau = 5
alpha = 1
p = 1


def rbf(time, sigma, l, tau):

    kernel = (sigma ** 2) * np.exp(-((time - tau) ** 2) / (2 * l ** 2))

    return kernel


def rq(time, sigma, l, tau, alpha):

    kernel = (sigma ** 2) * (1 + ((time - tau) ** 2) / (2 * alpha * l ** 2)) ** (-alpha)

    return kernel


def ess(time, sigma, l, tau, p):

    kernel = (sigma ** 2) * np.exp(-2 * np.sin(np.pi * np.abs(time - tau) / p) ** 2 / (l ** 2))

    return kernel


time = np.linspace(0, 10, 1001)

ax = plt.subplot(111)
plt.gcf().subplots_adjust(bottom=0.15)
plt.title('Compound Kernels with Decaying Periodicity', fontsize=16)
plt.plot(time, rbf(time, sigma, l, tau), '--', label='RBF')
plt.plot(time, rq(time, sigma, l, tau, alpha), '--', label='RQ')
plt.plot(time, rbf(time, sigma, l, tau) * ess(time, sigma, l, tau, p), label='RBF x ESS', linewidth=2)
plt.plot(time, rq(time, sigma, l, tau, alpha) * ess(time, sigma, l, tau, p), label='RQ x ESS', linewidth=2)
plt.legend(loc='upper left')
plt.savefig('aas_figures/kernels.png')
plt.show()

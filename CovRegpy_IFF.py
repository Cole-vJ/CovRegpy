
#     ________
#            /
#      \    /
#       \  /
#        \/

import textwrap
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

from AdvEMDpy import AdvEMDpy, emd_utils
from AdvEMDpy.emd_hilbert import hilbert_spectrum, Hilbert, theta, omega


def CovRegpy_fit(time, time_series, time_forecast, time_series_forecast, alpha=0.1, debug=True):
    """
    Function created to force fit function owing to uncertainty
    in phase once translated from frequency domain to time domain.

    Parameters
    ----------
    time : real ndarray
        Time used in fitting model.

    time_series : real ndarray
        Time series used in fitting model.

    time_forecast : real ndarray
        Independent variable used in forecasting.

    time_series_forecast : real ndarray
        Forecasted time series.

    alpha : real
        Limit for fitting time series and forecasted time series using curvature difference.

    debug : boolean
        Debugging printing successive curvature values.

    Returns
    -------
    time_forecast_fit : real ndarray
        Truncated and fitted time.

    time_series_forecast_fit : real ndarray
        Truncated and fitted time series.

    Notes
    -----

    """
    time_series_forecast_fit = time_series_forecast - (time_series_forecast[0] - time_series[-1])
    time_forecast_fit = time_forecast

    slope_1 = (time_series[-2] - time_series[-1]) / (time[-2] - time[-1])
    slope_2 = (time_series[-1] - time_series_forecast_fit[0]) / (time[-1] - time_forecast_fit[0])
    slope_3 = (time_series_forecast_fit[0] - time_series_forecast_fit[1]) / (time_forecast_fit[0] - time_forecast_fit[1])

    curvature = np.abs(np.diff(np.diff([slope_1, slope_2, slope_3])))
    if debug:
        print(curvature)

    while curvature > alpha:

        time_series_forecast_fit = time_series_forecast_fit[1:]
        time_series_forecast_fit = time_series_forecast_fit - (time_series_forecast_fit[0] - time_series[-1])
        time_forecast_fit = time_forecast_fit - (time_forecast_fit[0] - time[-1])
        time_forecast_fit = time_forecast_fit[1:]

        # if debug:
        #     plt.plot(time, time_series)
        #     plt.plot(time_forecast_fit, time_series_forecast_fit)
        #     plt.show()

        slope_1 = (time_series[-2] - time_series[-1]) / (time[-2] - time[-1])
        slope_2 = (time_series[-1] - time_series_forecast_fit[0]) / (time[-1] - time_forecast_fit[0])
        slope_3 = (time_series_forecast_fit[0] - time_series_forecast_fit[1]) / (
                    time_forecast_fit[0] - time_forecast_fit[1])

        curvature_prev = curvature.copy()
        curvature = np.abs(np.diff(np.diff([slope_1, slope_2, slope_3])))
        if debug:
            print(curvature)

        if curvature_prev < curvature:
            time_forecast_fit = time_forecast[-int(len(time_forecast_fit) + 2):]
            time_series_forecast_fit = time_series_forecast[-int(len(time_series_forecast_fit) + 1):]

            time_series_forecast_fit = time_series_forecast_fit - (time_series_forecast_fit[0] - time_series[-1])
            time_forecast_fit = time_forecast_fit - (time_forecast_fit[0] - time[-1])
            time_forecast_fit = time_forecast_fit[1:]
            break

    return time_forecast_fit, time_series_forecast_fit


def CovRegpy_IMF_IFF(time, time_series, type='linear', optimisation='l1', alpha=0.1,
                     components=2, fit_window=200, forecast_window=50, force_fit=False, debug=True):
    """
    Function created to forecast time series (assumed, but not necessary, to be an IMF) by translating time series
    into frequency domain, forecasting sparse information in frequency domain, then translating back into temporal
    domain.

    Parameters
    ----------
    time : real ndarray
        Time to be used in fitting model.

    time_series : real ndarray
        Time series to be used in fitting model.

    type : string
        Type of forecasting to be used in frequency domain.
        Only 'linear' available for now, but can be greatly expanded.

    optimisation : string
        Type of regularisation to be used in fitting model.
        Only 'l1' and 'l2' available for now.

    alpha : real
        Limit for fitting time series and forecasted time series using curvature difference.

    components : positive integer
        Number of components to estimate/isolate and to use in forecasting.

    fit_window : positive integer
        Number of points to use when fitting model.

    forecast_window : positive integer
        Number of points to forecast.

    force_fit : boolean
        If true then use CovRegpy_fit() function to fit time series and forecasted time series using curvatures.

    debug : boolean
        Plot intermediate result.

    Returns
    -------
    forecast_time : real ndarray
        Forecasted time - simply extended time.

    imf_forecast : real ndarray
        Forecasted time series.

    Notes
    -----

    """
    advemdpy = AdvEMDpy.EMD(time=time, time_series=time_series)
    emd = advemdpy.empirical_mode_decomposition()

    for comp in range(components):

        full_freq = emd[2][int(comp + 1), :]
        full_amp = np.sqrt(emd[0][int(comp + 1), :] ** 2 + emd[1][int(comp + 1), :] ** 2)

        if debug:
            plt.title('Full Frequency and Amplitude')
            plt.plot(time[1:], full_freq)
            plt.plot(time, full_amp)
            plt.show()

        if type == 'linear':

            forecast_time = \
                emd_utils.time_extension(time)[int(2 * len(time) - 1):int(int(2 * len(time) - 1) + forecast_window)]

            if optimisation == 'l2':
                freq_coef = \
                    np.linalg.lstsq(np.hstack((np.ones(fit_window).reshape(-1, 1), time[-fit_window:].reshape(-1, 1))),
                                    full_freq[-fit_window:], rcond=None)[0]
                amp_coef = \
                    np.linalg.lstsq(np.hstack((np.ones(fit_window).reshape(-1, 1), time[-fit_window:].reshape(-1, 1))),
                                    full_amp[-fit_window:], rcond=None)[0]
            elif optimisation == 'l1':
                freq_lasso = linear_model.Lasso(alpha=alpha, fit_intercept=False)
                freq_lasso.fit(np.hstack((np.ones(fit_window).reshape(-1, 1), time[-fit_window:].reshape(-1, 1))),
                               full_freq[-fit_window:])
                freq_coef = freq_lasso.coef_
                amp_lasso = linear_model.Lasso(alpha=alpha, fit_intercept=False)
                amp_lasso.fit(np.hstack((np.ones(fit_window).reshape(-1, 1), time[-fit_window:].reshape(-1, 1))),
                               full_amp[-fit_window:])
                amp_coef = amp_lasso.coef_

            freq_forecast = np.matmul(freq_coef.reshape(1, -1), np.vstack((np.ones(forecast_window).reshape(1, -1),
                                                                           forecast_time.reshape(1, -1))))
            freq_linear = np.matmul(freq_coef.reshape(1, -1), np.vstack((np.ones_like(time[-fit_window:]).reshape(1, -1),
                                                                         time[-fit_window:].reshape(1, -1))))
            amp_forecast = np.matmul(amp_coef.reshape(1, -1), np.vstack((np.ones(forecast_window).reshape(1, -1),
                                                                         forecast_time.reshape(1, -1))))
            amp_linear = np.matmul(amp_coef.reshape(1, -1), np.vstack((np.ones_like(time[-fit_window:]).reshape(1, -1),
                                                                       time[-fit_window:].reshape(1, -1))))

            # calculated directly from phase - approximate phase - need to refine
            phase = np.arcsin(np.sin(np.mean(full_freq[-fit_window:]) * time[-fit_window:])[-1])

            if np.sin(np.mean(full_freq[-fit_window:]) * time[-fit_window:])[-1] < \
                    np.sin(np.mean(full_freq[-fit_window:]) * time[-fit_window:])[-2]:
                if phase < 0:
                    phase = - np.pi - phase
                else:
                    phase = np.pi - phase

            imf_forecast =\
                amp_forecast[0, :] * \
                np.sin(phase + np.cumsum((freq_forecast[0, :] * np.diff(np.append(time[-1], forecast_time)))))

            if debug:
                plt.title('IMF Component and Fitted Component')
                plt.plot(time[-fit_window:], emd[0][int(comp + 1), :][-fit_window:])
                plt.plot(time[-fit_window:], np.mean(full_amp[-fit_window:]) *
                         np.sin(np.mean(full_freq[-fit_window:]) * time[-fit_window:]))
                plt.scatter(time[-1], np.mean(full_amp[-fit_window:]) * np.sin(phase), c='r')
                plt.show()
                ax = plt.subplot(111)
                plt.gcf().subplots_adjust(bottom=0.10)
                plt.title(textwrap.fill('Hilbert Transform of Amplitude and Frequency Modulated Time Series', 35),
                          fontsize=16)
                # plt.plot(time[:-1], np.linspace(1, 2, 1000), label='True frequency')
                x_hs, y, z = hilbert_spectrum(time, np.vstack((time, emd[0][int(comp + 1), :])),
                                              np.vstack((time, emd[1][int(comp + 1), :])),
                                              np.vstack((time[:-1], emd[2][int(comp + 1), :])),
                                              max_frequency=7, which_imfs=[1], plot=False)
                ax.pcolormesh(x_hs, y, np.abs(z), cmap='gist_rainbow', vmin=0, vmax=np.abs(z).max())
                plt.xticks([0, np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi],
                           ['0', r'$\pi$', r'$2\pi$', r'$3\pi$', r'$4\pi$'])
                # plt.legend(loc='upper left')
                plt.savefig('../aas_figures/iff_ht.png')
                plt.show()
                ax = plt.subplot(111)
                plt.gcf().subplots_adjust(bottom=0.15)
                plt.title('Instantaneous Frequency Fit and Forecast', fontsize=16)
                plt.scatter(time[-fit_window:], full_freq[-fit_window:],
                            label=textwrap.fill('Instantaneous frequency', 15))
                plt.scatter(forecast_time, freq_forecast, label=textwrap.fill('Forecast instantaneous frequency', 15))
                plt.plot(np.append(time[-fit_window:], forecast_time),
                         np.append(freq_linear, freq_forecast), 'r-',
                         label=textwrap.fill('Linear regression', 12))
                plt.xticks([3 * np.pi, 4 * np.pi, 5 * np.pi, 6 * np.pi],
                           [r'$3\pi$', r'$4\pi$', r'$5\pi$', r'$6\pi$'])
                plt.ylim(-0.5, 8.5)
                box_0 = ax.get_position()
                ax.set_position([box_0.x0 - 0.04, box_0.y0, box_0.width * 0.90, box_0.height])
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
                plt.savefig('../aas_figures/iff_freq.png')
                plt.show()
                ax = plt.subplot(111)
                plt.gcf().subplots_adjust(bottom=0.15)
                plt.title('Instantaneous Amplitude Fit and Forecast', fontsize=16)
                plt.scatter(time[-fit_window:], full_amp[-fit_window:],
                            label=textwrap.fill('Instantaneous amplitude', 15))
                plt.scatter(forecast_time, amp_forecast, label=textwrap.fill('Forecast instantaneous amplitude', 15))
                plt.plot(np.append(time[-fit_window:], forecast_time),
                         np.append(amp_linear, amp_forecast), 'r-',
                         label=textwrap.fill('Linear regression', 12))
                plt.xticks([3 * np.pi, 4 * np.pi, 5 * np.pi, 6 * np.pi],
                           [r'$3\pi$', r'$4\pi$', r'$5\pi$', r'$6\pi$'])
                box_0 = ax.get_position()
                ax.set_position([box_0.x0 - 0.04, box_0.y0, box_0.width * 0.92, box_0.height])
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
                plt.savefig('../aas_figures/iff_amp.png')
                plt.show()
                ax = plt.subplot(111)
                plt.gcf().subplots_adjust(bottom=0.15)
                plt.title('Unfitted Instantaneous Frequency Forecast', fontsize=16)
                plt.plot(time, emd[0][int(comp + 1), :], label=textwrap.fill('Time series', 12))
                plt.plot(forecast_time, imf_forecast, 'k-', label=textwrap.fill('Unfitted time series forecast', 12))
                plt.xticks([0, np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi, 5 * np.pi, 6 * np.pi],
                           ['0', r'$\pi$', r'$2\pi$', r'$3\pi$', r'$4\pi$', r'$5\pi$', r'$6\pi$'])
                box_0 = ax.get_position()
                ax.set_position([box_0.x0 - 0.04, box_0.y0, box_0.width * 0.94, box_0.height])
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
                plt.savefig('../aas_figures/iff_unfit.png')
                plt.show()

        try:
            imf_forecast_mat = np.vstack((imf_forecast.reshape(1, -1), imf_forecast_mat.reshape(1, -1)))
        except:
            imf_forecast_mat = imf_forecast.copy()

    if force_fit:
        forecast_time, imf_forecast = CovRegpy_fit(time, time_series, forecast_time,
                                                   imf_forecast, alpha=0.1, debug=True)

    return imf_forecast, forecast_time


def fit_sinusoid(time, time_series, time_forecast, time_series_forecast, forecast_window):

    slope_1 = (time_series[:, -2] - time_series[:, -1]) / (time[-2] - time[-1])
    slope_2 = (time_series[:, -1] - time_series_forecast[:, 0]) / (time[-1] - time_forecast[0])
    slope_3 = (time_series_forecast[:, 0] - time_series_forecast[:, 1]) / (time_forecast[0] - time_forecast[1])

    time_series_forecast_fit = time_series_forecast - \
        np.tile(((((slope_1 + slope_3) / 2) - slope_2) * (time[-1] - time_forecast[0])).reshape(-1, 1), (1, forecast_window))

    return time_forecast, time_series_forecast_fit


def CovRegpy_IFF(time, imfs, hts, ifs, type='linear', optimisation='l1', alpha=0.1,
                 fit_window=200, forecast_window=50, fit=True):

    for comp in range(np.shape(imfs)[0]):

        full_freq = ifs[comp, :]
        full_amp = np.sqrt(imfs[comp, :] ** 2 + hts[comp, :] ** 2)

        if type == 'linear':

            forecast_time = \
                emd_utils.time_extension(time)[int(2 * len(time) - 1):int(int(2 * len(time) - 1) + forecast_window)]

            if optimisation == 'l2':
                freq_coef = \
                    np.linalg.lstsq(np.hstack((np.ones(fit_window).reshape(-1, 1), time[-fit_window:].reshape(-1, 1))),
                                    full_freq[-fit_window:], rcond=None)[0]
                amp_coef = \
                    np.linalg.lstsq(np.hstack((np.ones(fit_window).reshape(-1, 1), time[-fit_window:].reshape(-1, 1))),
                                    full_amp[-fit_window:], rcond=None)[0]
            elif optimisation == 'l1':
                freq_lasso = linear_model.Lasso(alpha=alpha, fit_intercept=False)
                freq_lasso.fit(np.hstack((np.ones(fit_window).reshape(-1, 1), time[-fit_window:].reshape(-1, 1))),
                               full_freq[-fit_window:])
                freq_coef = freq_lasso.coef_
                amp_lasso = linear_model.Lasso(alpha=alpha, fit_intercept=False)
                amp_lasso.fit(np.hstack((np.ones(fit_window).reshape(-1, 1), time[-fit_window:].reshape(-1, 1))),
                               full_amp[-fit_window:])
                amp_coef = amp_lasso.coef_

            freq_forecast = np.matmul(freq_coef.reshape(1, -1), np.vstack((np.ones(forecast_window).reshape(1, -1),
                                                                           forecast_time.reshape(1, -1))))
            amp_forecast = np.matmul(amp_coef.reshape(1, -1), np.vstack((np.ones(forecast_window).reshape(1, -1),
                                                                         forecast_time.reshape(1, -1))))
            # trigonometric phase symmetry
            if -1 <= imfs[comp, :][-1] / np.mean(full_amp[-fit_window:]) <= 1:
                phase = np.arcsin(imfs[comp, :][-1] / np.mean(full_amp[-fit_window:]))
                if 0 < imfs[comp, :][-1] < imfs[comp, :][-2]:
                    phase = np.pi - phase
                elif imfs[comp, :][-1] < imfs[comp, :][-2] < 0:
                    phase = - np.pi + phase
            elif imfs[comp, :][-1] / np.mean(full_amp[-fit_window:]) > 1:
                phase = np.pi * 0.5
            elif imfs[comp, :][-1] / np.mean(full_amp[-fit_window:]) < -1:
                phase = - np.pi * 0.5

            imf_forecast =\
                amp_forecast[0, :] * \
                np.sin(phase + np.cumsum((freq_forecast[0, :] * np.diff(np.append(time[-1], forecast_time)))))

        try:
            imf_forecast_mat = np.vstack((imf_forecast_mat.reshape(comp, -1),
                                          imf_forecast.reshape(1, -1)))
        except:
            imf_forecast_mat = imf_forecast.copy()

    if fit:
        forecast_time, imf_forecast_mat = fit_sinusoid(time, imfs, forecast_time, imf_forecast_mat, forecast_window)

    return imf_forecast_mat, forecast_time

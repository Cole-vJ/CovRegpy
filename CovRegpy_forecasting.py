
#     ________
#            /
#      \    /
#       \  /
#        \/

import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.gaussian_process import GaussianProcessRegressor


def gp_forecast(x_fit, y_fit, x_forecast, kernel, confidence_level, plot=False):
    """
    Gaussian Process forecasting.

    Parameters
    ----------
    x_fit : real ndarray
        Independent variable for fitting.

    y_fit : real ndarray
        Dependent variable for fitting.

    x_forecast : real ndarray
        Independent variable for forecasting i.e. x_fit and y_fit are used to fit the model and
        x_forecast is then used to approximate the corresponding "y_forecast".

    kernel : sklearn.gaussian_process.kernels object
        Kernel to be used in Gaussian Process model - these must be imported from sklearn.gaussian_process.kernels.

    confidence_level : real
        Confidence level must be such that 0.00 < confidence_level < 1.00 -
        confidence interval is fitted about mean where model is confidence_level x 100% value will be within boundary.

    plot : boolean
        Debugging through plotting.

    Returns
    -------
    y_forecast : real ndarray
        Forecasted dependent variable vector.

    sigma : real ndarray
        Forecasted sigma variable vector.

    y_forecast_upper : real ndarray
        Forecasted upper boundary such that = y_forecast + norm.ppf(1 - (1 - confidence_level) / 2) * sigma

    y_forecast_lower : real ndarray
        Forecasted lower boundary such that = y_forecast - norm.ppf(1 - (1 - confidence_level) / 2) * sigma

    Notes
    -----

    """

    if not isinstance(x_fit, (type(np.asarray([1.0, 2.0])), type(pd.DataFrame(np.asarray([1.0, 2.0]))))):
        raise TypeError('Independent variable for fitting must be of type np.ndarray and pd.Dataframe.')
    if pd.isnull(np.asarray(x_fit)).any():
        raise TypeError('Independent variable for fitting must not contain nans.')
    if not (np.array(x_fit).dtype != np.array(np.arange(11.0)).dtype or
            np.array(x_fit).dtype != np.array(np.arange(11)).dtype):
        raise TypeError('Independent variable for fitting must only contain floats or integers.')
    if not isinstance(y_fit, (type(np.asarray([1.0, 2.0])), type(pd.DataFrame(np.asarray([1.0, 2.0]))))):
        raise TypeError('Dependent variable for fitting must be of type np.ndarray and pd.Dataframe.')
    if pd.isnull(np.asarray(y_fit)).any():
        raise TypeError('Dependent variable for fitting must not contain nans.')
    if np.array(y_fit).dtype != np.array(np.arange(11.0)).dtype:
        raise TypeError('Dependent variable for fitting must only contain floats.')

    # reshape variables
    subset_x = np.atleast_2d(x_fit).T
    x = np.atleast_2d(x_forecast).T
    y = y_fit.ravel()

    # instantiate Gaussian Process
    gaus_proc = GaussianProcessRegressor(kernel=kernel, alpha=1 ** 2, n_restarts_optimizer=0)

    # fit process to time and time series subset
    gaus_proc.fit(subset_x, y)

    # predict full signal over full time sets
    y_forecast, sigma = gaus_proc.predict(np.atleast_2d(x), return_std=True)

    # calculate confidence bounds
    bounds = norm.ppf(1 - (1 - confidence_level) / 2)

    # calculate lower and upper bounds
    y_forecast_upper = y_forecast + bounds * sigma
    y_forecast_lower = y_forecast - bounds * sigma

    # plot
    if plot:
        plt.plot(x_fit, y_fit)
        plt.plot(x_forecast, y_forecast)
        plt.fill(np.concatenate([x_forecast, x_forecast[::-1]]),
                 np.concatenate([y_forecast_lower, y_forecast_upper[::-1]]), alpha=.5, fc='b', ec='None')
        plt.plot(x_forecast, y_forecast_upper, '--')
        plt.plot(x_forecast, y_forecast_lower, '--')
        plt.show()

    return y_forecast, sigma, y_forecast_upper, y_forecast_lower


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

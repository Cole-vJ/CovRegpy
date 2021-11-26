
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor


def gp_forecast(x_fit, y_fit, x_forecast, kernel, confidence_level, plot=False):

    # reshape variables
    subset_x = np.atleast_2d(x_fit).T
    x = np.atleast_2d(x_forecast).T
    y = y_fit.ravel()

    # instantiate Gaussian Process
    gaus_proc = GaussianProcessRegressor(kernel=kernel, alpha=1 ** 2, n_restarts_optimizer=10)

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

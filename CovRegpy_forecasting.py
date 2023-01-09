
#     ________
#            /
#      \    /
#       \  /
#        \/

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from AdvEMDpy import emd_basis
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared, WhiteKernel, RBF, RationalQuadratic


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


if __name__ == "__main__":

    # simple sinusoid
    time_full = np.linspace(0, 120, 1201)
    time = time_full[:1002]
    time_series = np.sin((1 / 10) * time) + np.cos((1 / 5) * time)
    sinusiod_kernel = RBF(length_scale=10.0) * ExpSineSquared(length_scale=1.3, periodicity=1.0)

    y_forecast, sigma, y_forecast_upper, y_forecast_lower = \
        gp_forecast(time, time_series, time_full, sinusiod_kernel, 0.95, plot=False)
    plt.plot(time, time_series)
    plt.plot(time_full, y_forecast, '--')
    plt.show()

    # pull all close data
    tickers_format = ['MSFT', 'AAPL', 'GOOGL', 'AMZN', 'TSLA']
    data = yf.download(tickers_format, start="2018-12-31", end="2021-12-01")
    close_data = data['Close']
    del data, tickers_format

    # create date range and interpolate
    date_index = pd.date_range(start='31/12/2018', end='12/01/2021')
    close_data = close_data.reindex(date_index).interpolate()
    close_data = close_data[::-1].interpolate()
    close_data = close_data[::-1]
    del date_index

    close_data = np.asarray(close_data).T
    model_days = np.shape(close_data)[1]
    knots = 114
    knots_vector = np.linspace(0, model_days - 1, int(knots - 6))
    knots_vector = np.linspace(-knots_vector[3], 2 * (model_days - 1) - knots_vector[-4], knots)
    time = np.arange(model_days)
    time_extended = np.arange(int(model_days + 50))

    spline_basis_transform = emd_basis.Basis(time_series=time, time=time)
    spline_basis_transform = spline_basis_transform.cubic_b_spline(knots=knots_vector)
    coef_forecast = np.linalg.lstsq(spline_basis_transform.T, close_data.T, rcond=None)[0]
    mean_forecast = np.matmul(coef_forecast.T, spline_basis_transform)

    plt.plot(close_data.T)
    plt.plot(mean_forecast.T)
    plt.show()

    # long term smooth rising trend
    k1 = 66.0 ** 2 * RBF(length_scale=67.0)
    # seasonal component
    k2 = (2.4 ** 2 * RBF(length_scale=90.0) * ExpSineSquared(length_scale=1.3, periodicity=1.0))
    # medium term irregularity
    k3 = 0.66 ** 2 * RationalQuadratic(length_scale=1.2, alpha=0.78)
    # noise terms
    k4 = 0.18 ** 2 * RBF(length_scale=0.134) + WhiteKernel(noise_level=0.19 ** 2)

    kernel = k1 + k2 + k3 + k4

    lag = 200

    for time_series in range(np.shape(close_data)[0]):
        # forecast time series
        y_forecast, sigma, y_forecast_upper, y_forecast_lower = \
            gp_forecast(time[int(model_days - lag - 1):],
                        close_data[time_series, int(model_days - lag - 1):],
                        time_extended[int(model_days - lag - 1):],
                        kernel, 0.95, plot=False)

        plt.plot(time[int(model_days - lag - 1):], close_data[time_series, int(model_days - lag - 1):])
        plt.plot(time_extended[int(model_days - lag - 1):], y_forecast)
        plt.fill(np.concatenate([time_extended[int(model_days - lag - 1):],
                                 time_extended[int(model_days - lag - 1):][::-1]]),
                 np.concatenate([y_forecast_lower, y_forecast_upper[::-1]]), alpha=.5, fc='b', ec='None')
        plt.plot(time_extended[int(model_days - lag - 1):], y_forecast_upper, '--')
        plt.plot(time_extended[int(model_days - lag - 1):], y_forecast_lower, '--')
        plt.show()

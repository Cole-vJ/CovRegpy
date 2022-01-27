
# formatted

import textwrap
import scipy as sp
import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib import mlab
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from CovRegpy_singular_spectrum_analysis import ssa
from scipy.fft import fft

# Bonizzi, P., Karel, J., Meste, O., & Peeters, R. (2014).
# Singular Spectrum Decomposition: A New Method for Time Series Decomposition.
# Advances in Adaptive Data Analysis, 6(04), 1450011 (1-34). World Scientific.


def gaussian(f, A, mu, sigma):
    return A * np.exp(- (f - mu) ** 2 / (2 * sigma ** 2))


def max_bool(time_series):
    max_bool_order_1 = np.r_[False, time_series[1:] >= time_series[:-1]] & \
                       np.r_[time_series[:-1] > time_series[1:], False]
    return max_bool_order_1


def obj_func(theta, f, mu_1, mu_2, mu_3, spectrum):
    return sum(np.abs(theta[0] * np.exp(- (f - mu_1) ** 2 / (2 * theta[3] ** 2)) +
                      theta[1] * np.exp(- (f - mu_2) ** 2 / (2 * theta[4] ** 2)) +
                      theta[2] * np.exp(- (f - mu_3) ** 2 / (2 * theta[5] ** 2)) - spectrum))


def cons_long_only_weight(x):
    return x


def gaus_param(w0, f, mu_1, mu_2, mu_3, spectrum):
    cons = ({'type': 'ineq', 'fun': cons_long_only_weight})
    return minimize(obj_func, x0=w0, args=(f, mu_1, mu_2, mu_3, spectrum), method='nelder-mead', constraints=cons,
                    bounds=[(0, None), (0, None), (0, None), (0, None), (0, None), (0, None)])


def scaling_factor_obj_func(a, residual_time_series, trend_estimate):
    return sum((residual_time_series - a * trend_estimate) ** 2)


def scaling_factor(residual_time_series, trend_estimate):
    cons = ({'type': 'ineq', 'fun': cons_long_only_weight})
    return minimize(scaling_factor_obj_func, x0=np.asarray(1), args=(residual_time_series, trend_estimate),
                    method='nelder-mead', constraints=cons)


def ssd(time_series, nmse_threshold=0.01, plot=False):

    # make duplicate of original time series
    time_series_orig = time_series.copy()
    time_series_resid = time_series.copy()

    # first test for 'SIZEABLE' trend
    dt = (24 * 60 * 60)
    s, f = mlab.psd(np.asarray(time_series_resid), Fs=1 / dt)
    if plot:
        plt.plot(f * dt, s)
        plt.show()

    if np.log10(s)[1] == np.max(np.log10(s)):
        trend_est = ssa(time_series=np.asarray(time_series_resid), L=int(len(np.asarray(time_series_resid)) / 3), est=1)
        if plot:
            plt.plot(np.asarray(time_series_resid))
            plt.plot(trend_est, '--')
            plt.show()
        time_series_resid -= trend_est
        if plot:
            plt.plot(np.asarray(time_series_resid))
            plt.show()
            s, f = mlab.psd(np.asarray(time_series_resid), Fs=1 / dt)
            plt.plot(f * dt, s)
            plt.show()

    try:
        time_series_est_mat = trend_est.reshape(1, -1)
    except:
        pass

    nmse = nmse_threshold + 0.01

    while nmse > nmse_threshold:
        print(nmse)

        s, f = mlab.psd(np.asarray(time_series_resid), Fs=1 / dt)

        mu_1 = f[s == max(s)] * dt
        A_1 = (1 / 2) * s[s == max(s)]
        sigma_1 = (2 / 3) * f[((2 / 3) * s[s == max(s)] - s) == min((2 / 3) * s[s == max(s)] - s)] * dt
        gaus_1 = gaussian(f * dt, A_1, mu_1, sigma_1)

        mu_2 = f[s == np.sort(s[max_bool(s)])[-2]] * dt
        A_2 = (1 / 2) * s[s == np.sort(s[max_bool(s)])[-2]]
        sigma_2 = (2 / 3) * f[((2 / 3) * s[s == np.sort(s[max_bool(s)])[-2]] - s)
                              == min((2 / 3) * s[s == np.sort(s[max_bool(s)])[-2]] - s)] * dt
        gaus_2 = gaussian(f * dt, A_2, mu_2, sigma_2)

        mu_3 = (mu_1 + mu_2) / 2
        A_3 = (1 / 4) * s[np.abs(f * dt - mu_3) == min(np.abs(f * dt - mu_3))][0]
        sigma_3 = 4 * np.abs(mu_1 - mu_2)
        gaus_3 = gaussian(f * dt, A_3, mu_3, sigma_3)

        x0 = np.zeros(6)
        x0[0] = A_1
        x0[1] = A_2
        x0[2] = A_3
        x0[3] = sigma_1
        x0[4] = sigma_2
        x0[5] = sigma_3

        thetas = gaus_param(x0, f * dt, mu_1, mu_2, mu_3, s).x
        f_range = [(mu_1 - 2.5 * thetas[3])[0], (mu_1 + 2.5 * thetas[3])[0]]

        if plot:
            plt.plot(np.asarray(time_series_resid))
            plt.show()
            s, f = mlab.psd(np.asarray(time_series_resid), Fs=1 / dt)
            plt.title('Gaussian Function Initialisation')
            plt.plot(f * dt, s, label='Power-spectral density')
            plt.plot(f * dt, gaus_1, '--', label=r'$g_1(f) = A_1^{(0)}e^{\frac{(f-\mu_1)^2}{\sigma^{(0)2}_1}}$', Linewidth=2)
            plt.plot(f * dt, gaus_2, '--', label=r'$g_2(f) = A_2^{(0)}e^{\frac{(f-\mu_2)^2}{\sigma^{(0)2}_2}}$', Linewidth=2)
            plt.plot(f * dt, gaus_3, '--', label=r'$g_3(f) = A_3^{(0)}e^{\frac{(f-\mu_3)^2}{\sigma^{(0)2}_3}}$', Linewidth=2)
            plt.plot(f * dt, gaus_1 + gaus_2 + gaus_3, '--',
                     label=r'$\sum_{i=1}^{3}A_i^{(0)}e^{\frac{(f-\mu_i)^2}{\sigma^{(0)2}_i}}$', Linewidth=2)
            plt.plot(mu_1 * np.ones(100), np.linspace(np.min(s), 1.1 * np.max(s), 100), '--', label=r'$\mu_1$')
            plt.plot(mu_2 * np.ones(100), np.linspace(np.min(s), 1.1 * np.max(s), 100), '--', label=r'$\mu_2$')
            plt.plot(mu_3 * np.ones(100), np.linspace(np.min(s), 1.1 * np.max(s), 100), '--', label=r'$\mu_3$')
            plt.legend(loc='best')
            plt.xlabel('Standardised Frequency')
            plt.ylabel('Spectral Density')
            plt.xlim(-0.005, 0.255)
            plt.xticks([0, 0.05, 0.1, 0.15, 0.2, 0.25])
            plt.show()
            plt.title('Gaussian Function Optimised')
            plt.plot(f * dt, s, label='Power-spectral density')
            plt.plot(f * dt, gaussian(f * dt, thetas[0], mu_1, thetas[3]),
                     '--', label=r'$g^{opt}_1(f) = A_1^{opt}e^{\frac{(f-\mu_1)^2}{\sigma^{opt2}_1}}$', Linewidth=2)
            plt.plot(f * dt, gaussian(f * dt, thetas[1], mu_2, thetas[4]),
                     '--', label=r'$g^{opt}_2(f) = A_2^{opt}e^{\frac{(f-\mu_2)^2}{\sigma^{opt2}_2}}$', Linewidth=2)
            plt.plot(f * dt, gaussian(f * dt, thetas[2], mu_3, thetas[5]),
                     '--', label=r'$g^{opt}_3(f) = A_3^{opt}e^{\frac{(f-\mu_3)^2}{\sigma^{opt2}_3}}$', Linewidth=2)
            plt.plot(f * dt, gaussian(f * dt, thetas[0], mu_1, thetas[3]) +
                     gaussian(f * dt, thetas[1], mu_2, thetas[4]) +
                     gaussian(f * dt, thetas[2], mu_3, thetas[5]), '--',
                     label=r'$\sum_{i=1}^{3}A_i^{opt}e^{\frac{(f-\mu_i)^2}{\sigma^{opt2}_i}}$', Linewidth=2)
            plt.plot(f_range[0] * np.ones(101), np.linspace(0, 1.1 * 2 * A_1, 101), 'k--')
            plt.plot(f_range[1] * np.ones(101), np.linspace(0, 1.1 * 2 * A_1, 101), 'k--', label='Frequency bounds')
            plt.legend(loc='best')
            plt.xlabel('Standardised Frequency')
            plt.ylabel('Spectral Density')
            plt.xlim(-0.005, 0.255)
            plt.xticks([0, 0.05, 0.1, 0.15, 0.2, 0.25])
            plt.show()
        L = int(1.2 * (1 / dt) * (1 / f[np.max(s) == s]))
        fft_spectrum = sp.fft.fft(time_series_resid)
        fft_freq = sp.fft.fftfreq(len(time_series_resid))
        if f_range[0] < 0:
            fft_spectrum_band = fft_spectrum * (fft_freq < f_range[1])
            fft_spectrum_band = fft_spectrum_band * (fft_freq > -f_range[1])
        else:
            fft_spectrum_band = fft_spectrum * (np.r_[fft_freq < f_range[1]] & np.r_[fft_freq > f_range[0]])
            fft_spectrum_band += fft_spectrum * (np.r_[fft_freq > -f_range[1]] & np.r_[fft_freq < -f_range[0]])
        trend_est_1 = sp.fft.ifft(fft_spectrum_band)
        trend_est_1 = np.real(trend_est_1)

        # decomposition - MODIFIED embedding
        X = np.zeros((len(np.asarray(trend_est_1)), len(np.asarray(trend_est_1)) - L + 1))
        for col in range(len(np.asarray(trend_est_1)) - L + 1):
            X[:, col] = np.hstack((np.asarray(trend_est_1)[col:int(len(np.asarray(trend_est_1) + col))], np.asarray(trend_est_1)[:col]))

        # decomposition - singular value decomposition (eigen-vectors in columns)
        eigen_values, eigen_vectors = np.linalg.eig(np.matmul(X, X.T))
        V_storage = np.real(np.matmul(X.T, eigen_vectors[:, 0].reshape(-1, 1)) / np.sqrt(eigen_values[0]))
        X_estimate = np.real(np.sqrt(eigen_values[0]) * np.matmul(eigen_vectors[:, 0].reshape(-1, 1), V_storage.T))

        # reconstruction - averaging
        trend_est_2 = X_estimate[:, 0].copy()
        averaging_vector = (len(time_series_resid) - L + 1) * np.ones_like(trend_est_2)
        for col in range(1, len(time_series_resid) - L + 1):
            trend_est_2[col:] += X_estimate[:int(len(trend_est_2[col:])), col]
            trend_est_2[:col] += X_estimate[int(len(trend_est_2[col:])):, col]
        trend_est_2 /= averaging_vector
        trend_est_2 = np.real(trend_est_2)

        if plot:
            plt.plot(time_series_resid)
            plt.plot(trend_est_1, label='estimate 1')
            plt.plot(trend_est_2, label='estimate 2')
            plt.legend(loc='upper left')
            plt.show()

        if sum(trend_est_2 * (time_series_resid - trend_est_2)) > 0:
            trend_est = trend_est_2
        else:
            trend_est = trend_est_1

        # scaling factor - top

        a_opt = scaling_factor(residual_time_series=time_series_resid, trend_estimate=trend_est).x
        if plot:
            plt.title('Optimisation of Scaling Factor')
            plt.plot(np.asarray(time_series_resid), label='Residual time series')
            plt.plot(trend_est, '--', label='Unscaled trend')
            plt.plot(a_opt * trend_est, '--', label='Scaled trend')
            plt.legend(loc='best')
            plt.show()
        trend_est *= a_opt

        # scaling factor - bottom

        time_series_resid -= trend_est

        try:
            time_series_est_mat = np.vstack((time_series_est_mat, trend_est))
        except:
            time_series_est_mat = trend_est.reshape(1, -1)

        nmse = sum(time_series_resid ** 2) / sum((time_series_orig - np.mean(time_series_orig)) ** 2)

    time_series_est_mat = np.vstack((time_series_est_mat, time_series_resid))

    return time_series_est_mat


if __name__ == "__main__":

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
    test = ssd(np.asarray(close_data['MSFT'][-100:]), nmse_threshold=0.05, plot=True)
    plt.plot(test.T)
    plt.show()

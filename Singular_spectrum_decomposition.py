
import scipy as sp
import numpy as np
from matplotlib import mlab
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from Singular_spectrum_analysis import ssa

# Bonizzi, P., Karel, J., Meste, O., & Peeters, R. (2014).
# Singular Spectrum Decomposition: A New Method for Time Series Decomposition.
# Advances in Adaptive Data Analysis, 6(04), 1450011-1 - 1450011-34. World Scientific.


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


def gaus_param(f, w0, mu_1, mu_2, mu_3, spectrum):
    cons = ({'type': 'ineq', 'fun': cons_long_only_weight})
    return minimize(obj_func, w0, args=(f, mu_1, mu_2, mu_3, spectrum), method='SLSQP', constraints=cons)


def ssd(time_series, plot=False):

    # first test for 'SIZEABLE' trend
    dt = (24 * 60 * 60)
    s, f = mlab.psd(np.asarray(time_series), Fs=1 / dt)
    if plot:
        plt.plot(f * dt, s)
        plt.show()

    if np.log10(s)[1] == np.max(np.log10(s)):
        trend_est = ssa(time_series=np.asarray(time_series), L=int(len(np.asarray(time_series)) / 3), est=1)
        if plot:
            plt.plot(np.asarray(time_series))
            plt.plot(trend_est, '--')
            plt.show()
        time_series -= trend_est
        if plot:
            plt.plot(np.asarray(time_series))
            plt.show()
            s, f = mlab.psd(np.asarray(time_series), Fs=1 / dt)
            plt.plot(f * dt, s)
            plt.show()

    try:
        time_series_est_mat = trend_est.reshape(1, -1)
    except:
        pass

    for i in range(5):
        s, f = mlab.psd(np.asarray(time_series), Fs=1 / dt)

        mu_1 = f[s == max(s)] * dt
        A_1 = (1 / 2) * s[s == max(s)]
        sigma_1 = (2 / 3) * f[((2 / 3) * s[s == max(s)] - s) == min((2 / 3) * s[s == max(s)] - s)] * dt
        gaus_1 = gaussian(f * dt, A_1, mu_1, sigma_1)

        mu_2 = f[s == np.sort(s[max_bool(s)])[-2]] * dt
        A_2 = (1 / 2) * s[s == np.sort(s[max_bool(s)])[-2]]
        sigma_2 = (2 / 3) * f[((2 / 3) * s[s == np.sort(s[max_bool(s)])[-2]] - s) == min((2 / 3) * s[s == np.sort(s[max_bool(s)])[-2]] - s)] * dt
        gaus_2 = gaussian(f * dt, A_2, mu_2, sigma_2)

        mu_3 = (mu_1 + mu_2) / 2
        A_3 = (1 / 4) * s[np.abs(f * dt - mu_3) == min(np.abs(f * dt - mu_3))]
        sigma_3 = 4 * np.abs(mu_1 - mu_2)
        gaus_3 = gaussian(f * dt, A_3, mu_3, sigma_3)

        thetas = gaus_param(f * dt, [A_1, A_2, A_3, sigma_1, sigma_2, sigma_3], mu_1, mu_2, mu_3, 2 * gaus_1).x
        f_range = [(mu_1 - 2.5 * thetas[3])[0], (mu_1 + 2.5 * thetas[3])[0]]

        if plot:
            plt.plot(np.asarray(time_series))
            plt.show()
            s, f = mlab.psd(np.asarray(time_series), Fs=1 / dt)
            plt.plot(f * dt, s)
            plt.plot(f * dt, gaus_1, '--')
            plt.plot(f * dt, gaus_2, '--')
            plt.plot(f * dt, gaus_3, '--')
            plt.plot(f * dt, gaus_1 + gaus_2 + gaus_3, '--')
            plt.plot(f * dt, gaussian(f * dt, thetas[0], mu_1, thetas[3]) +
                     gaussian(f * dt, thetas[1], mu_2, thetas[4]) +
                     gaussian(f * dt, thetas[2], mu_3, thetas[5]), '--')
            plt.plot(f_range[0] * np.ones(101), np.linspace(0, 2 * A_1, 101), 'k--')
            plt.plot(f_range[1] * np.ones(101), np.linspace(0, 2 * A_1, 101), 'k--')
            plt.show()
        L = int(1.2 * (1 / dt) * (1 / f[np.max(s) == s]))
        fft_spectrum = sp.fft.fft(time_series)
        fft_freq = sp.fft.fftfreq(len(time_series))
        if f_range[0] < 0:
            fft_spectrum_band = fft_spectrum * (fft_freq < f_range[1])
            fft_spectrum_band = fft_spectrum_band * (fft_freq > -f_range[1])
        else:
            fft_spectrum_band = fft_spectrum * (np.r_[fft_freq < f_range[1]] & np.r_[fft_freq > f_range[0]])
            fft_spectrum_band = fft_spectrum_band * (np.r_[fft_freq > -f_range[1]] & np.r_[fft_freq < -f_range[0]])
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
        averaging_vector = (len(time_series) - L + 1) * np.ones_like(trend_est_2)
        for col in range(1, len(time_series) - L + 1):
            trend_est_2[col:] += X_estimate[col:, col]
            trend_est_2[:col] += X_estimate[:col, col]
        trend_est_2 /= averaging_vector
        trend_est_2 = np.real(trend_est_2)

        if plot:
            plt.plot(time_series)
            plt.plot(trend_est_1, label='estimate 1')
            plt.plot(trend_est_2, label='estimate 2')
            plt.legend(loc='upper left')
            plt.show()

        if sum(trend_est_2 * ((time_series - trend_est_2) - trend_est_2)) > 0:
            trend_est = trend_est_2
        else:
            trend_est = trend_est_1

        time_series -= trend_est

        try:
            time_series_est_mat = np.vstack((time_series_est_mat, trend_est))
        except:
            time_series_est_mat = trend_est.reshape(1, -1)

    return time_series_est_mat

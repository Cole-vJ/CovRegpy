
#     ________
#            /
#      \    /
#       \  /
#        \/

# Main reference: Bonizzi, Karel, Meste, & Peeters (2014)
# Bonizzi, P., Karel, J., Meste, O., & Peeters, R. (2014).
# Singular Spectrum Decomposition: A New Method for Time Series Decomposition.
# Advances in Adaptive Data Analysis, 6(04), 1450011 (1-34). World Scientific.

import scipy as sp
import numpy as np
import seaborn as sns
from matplotlib import mlab
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from CovRegpy_SSA import CovRegpy_ssa
from scipy.fft import fft

np.random.seed(0)

sns.set(style='darkgrid')


def gaussian(f, A, mu, sigma):
    """
    Gaussian distribution to be fitted to power-spectral density.

    Parameters
    ----------
    f : real ndarray
        Frequency over which distribution will be fitted.

    A : float
        Amplitude of Gaussian distribution.

    mu : float
        Mean of Gaussian distribution.

    sigma : float
        Standard deviation of Gaussian distribution.

    Returns
    -------
    distribution : real ndarray
        Gaussian distribution over domain 'f'.

    Notes
    -----
    Used in calculation of downsampling range.

    """

    if not isinstance(f, (type(np.asarray([[1.0, 2.0], [3.0, 4.0]])))):
        raise TypeError('\'f\' must be of type np.ndarray.')
    if np.array(f).dtype != np.array([[1., 1., 1., 1.]]).dtype:
        raise TypeError('\'f\' must only contain floats.')
    try:
        if (not isinstance(A[0], float)) or A[0] < 0:
            raise ValueError('\'A\' must be a non-negative float.')
        if (not isinstance(mu[0], float)) or mu[0] <= 0:
            raise ValueError('\'mu\' must be a positive float.')
        if (not isinstance(sigma[0], float)) or sigma[0] <= 0:
            raise ValueError('\'sigma\' must be a positive float.')
    except (IndexError, TypeError):
        if (not isinstance(A, float)) or A < 0:
            raise ValueError('\'A\' must be a non-negative float.')
        if (not isinstance(mu, float)) or mu <= 0:
            raise ValueError('\'mu\' must be a positive float.')
        if (not isinstance(sigma, float)) or sigma <= 0:
            raise ValueError('\'sigma\' must be a positive float.')

    distribution = A * np.exp(- (f - mu) ** 2 / (2 * sigma ** 2))

    return distribution


def max_bool(time_series):
    """
    Calculate maximum boolean of time series.

    Parameters
    ----------
    time_series : real ndarray
        Time series where maximums are to be calculated.

    Returns
    -------
    max_bool : real ndarray
        Boolean array with location of maximums.

    Notes
    -----
    Used to calculate centres of each Gaussian distributions.

    """

    if not isinstance(time_series, (type(np.asarray([[1.0, 2.0], [3.0, 4.0]])))):
        raise TypeError('\'time_series\' must be of type np.ndarray.')
    if np.array(time_series).dtype != np.array([[1., 1., 1., 1.]]).dtype:
        raise TypeError('\'time_series\' must only contain floats.')

    max_bool = np.r_[False, time_series[1:] >= time_series[:-1]] & np.r_[time_series[:-1] > time_series[1:], False]

    return max_bool


def spectral_obj_func_l1(theta, f, mu_1, mu_2, mu_3, spectrum):
    """
    Calculate 3 Gaussian functions.

    Parameters
    ----------
    theta : real ndarray
        A vector of shape (6,) containing amplitudes and sigmas.

    f : real ndarray
        Frequency over which Gaussian distributions and frequency spectrum are fitted.

    mu_1 : float
        Mean of the highest frequency peak.

    mu_2 : float
         Mean of the second-highest frequency peak.

    mu_3 : float
        Mean of remaining spectra.

    spectrum : real ndarray
        Spectrum to be fitted to Gaussian distributions.

    Returns
    -------
    objective_function : real ndarray
        The l1 norm of difference between spectrum and Gaussian distributions.

    Notes
    -----
    Should experiment with different norms.

    """

    if not isinstance(theta, (type(np.asarray([[1.0, 2.0], [3.0, 4.0]])))):
        raise TypeError('\'theta\' must be of type np.ndarray.')
    if np.array(theta).dtype != np.array([[1., 1., 1., 1.]]).dtype:
        raise TypeError('\'theta\' must only contain floats.')
    if len(theta) != 6:
        raise ValueError('\'theta\' must be of length 6.')
    if not isinstance(f, (type(np.asarray([[1.0, 2.0], [3.0, 4.0]])))):
        raise TypeError('\'f\' must be of type np.ndarray.')
    if np.array(f).dtype != np.array([[1., 1., 1., 1.]]).dtype:
        raise TypeError('\'f\' must only contain floats.')
    try:
        if (not isinstance(mu_1[0], float)) or mu_1[0] <= 0:
            raise ValueError('\'mu_1\' must be a positive float.')
        if (not isinstance(mu_2[0], float)) or mu_2[0] <= 0:
            raise ValueError('\'mu_2\' must be a positive float.')
        if (not isinstance(mu_3[0], float)) or mu_3[0] <= 0:
            raise ValueError('\'mu_3\' must be a positive float.')
    except (IndexError, TypeError):
        if (not isinstance(mu_1, float)) or mu_1 <= 0:
            raise ValueError('\'mu_1\' must be a positive float.')
        if (not isinstance(mu_2, float)) or mu_2 <= 0:
            raise ValueError('\'mu_2\' must be a positive float.')
        if (not isinstance(mu_3, float)) or mu_3 <= 0:
            raise ValueError('\'mu_3\' must be a positive float.')
    if not isinstance(spectrum, (type(np.asarray([[1.0, 2.0], [3.0, 4.0]])))):
        raise TypeError('\'spectrum\' must be of type np.ndarray.')
    if np.array(spectrum).dtype != np.array([[1., 1., 1., 1.]]).dtype:
        raise TypeError('\'spectrum\' must only contain floats.')
    if len(f) != len(spectrum):
        raise ValueError('\'f\' and \'spectrum\' are incompatible lengths.')

    objective_function = sum(np.abs(theta[0] * np.exp(- (f - mu_1) ** 2 / (2 * theta[3] ** 2)) +
                                    theta[1] * np.exp(- (f - mu_2) ** 2 / (2 * theta[4] ** 2)) +
                                    theta[2] * np.exp(- (f - mu_3) ** 2 / (2 * theta[5] ** 2)) - spectrum))

    return objective_function


def spectral_obj_func_l2(theta, f, mu_1, mu_2, mu_3, spectrum):
    """
    Calculate 3 Gaussian functions.

    Parameters
    ----------
    theta : real ndarray
        A vector of shape (6,) containing amplitudes and sigmas.

    f : real ndarray
        Frequency over which Gaussian distributions and frequency spectrum are fitted.

    mu_1 : float
        Mean of the highest frequency peak.

    mu_2 : float
         Mean of the second-highest frequency peak.

    mu_3 : float
        Mean of remaining spectra.

    spectrum : real ndarray
        Spectrum to be fitted to Gaussian distributions.

    Returns
    -------
    objective_function : real ndarray
        The l2 norm of difference between spectrum and Gaussian distributions.

    Notes
    -----
    Should experiment with different norms.

    """

    if not isinstance(theta, (type(np.asarray([[1.0, 2.0], [3.0, 4.0]])))):
        raise TypeError('\'theta\' must be of type np.ndarray.')
    if np.array(theta).dtype != np.array([[1., 1., 1., 1.]]).dtype:
        raise TypeError('\'theta\' must only contain floats.')
    if len(theta) != 6:
        raise ValueError('\'theta\' must be of length 6.')
    if not isinstance(f, (type(np.asarray([[1.0, 2.0], [3.0, 4.0]])))):
        raise TypeError('\'f\' must be of type np.ndarray.')
    if np.array(f).dtype != np.array([[1., 1., 1., 1.]]).dtype:
        raise TypeError('\'f\' must only contain floats.')
    try:
        if (not isinstance(mu_1[0], float)) or mu_1[0] <= 0:
            raise ValueError('\'mu_1\' must be a positive float.')
        if (not isinstance(mu_2[0], float)) or mu_2[0] <= 0:
            raise ValueError('\'mu_2\' must be a positive float.')
        if (not isinstance(mu_3[0], float)) or mu_3[0] <= 0:
            raise ValueError('\'mu_3\' must be a positive float.')
    except (IndexError, TypeError):
        if (not isinstance(mu_1, float)) or mu_1 <= 0:
            raise ValueError('\'mu_1\' must be a positive float.')
        if (not isinstance(mu_2, float)) or mu_2 <= 0:
            raise ValueError('\'mu_2\' must be a positive float.')
        if (not isinstance(mu_3, float)) or mu_3 <= 0:
            raise ValueError('\'mu_3\' must be a positive float.')
    if not isinstance(spectrum, (type(np.asarray([[1.0, 2.0], [3.0, 4.0]])))):
        raise TypeError('\'spectrum\' must be of type np.ndarray.')
    if np.array(spectrum).dtype != np.array([[1., 1., 1., 1.]]).dtype:
        raise TypeError('\'spectrum\' must only contain floats.')
    if len(f) != len(spectrum):
        raise ValueError('\'f\' and \'spectrum\' are incompatible lengths.')

    objective_function = sum((theta[0] * np.exp(- (f - mu_1) ** 2 / (2 * theta[3] ** 2)) +
                              theta[1] * np.exp(- (f - mu_2) ** 2 / (2 * theta[4] ** 2)) +
                              theta[2] * np.exp(- (f - mu_3) ** 2 / (2 * theta[5] ** 2)) - spectrum) ** 2)

    return objective_function


def constraint_positive(x):
    """
    Constraint on amplitudes and sigmas being positive.

    Parameters
    ----------
    x : real ndarray
        Parameters to be constrained to be positive.

    Returns
    -------
    x : real ndarray
        Positive constrained parameters.

    Notes
    -----
    Constraint function.

    """
    return x


def gaus_param(w0, f, mu_1, mu_2, mu_3, spectrum, method='l1'):
    """
    Function that calculates optimal amplitudes and sigmas.

    Parameters
    ----------
    w0 : real ndarray
        Vectors of starting parameters.

    f : real ndarray
        Frequency over which parameters are optimised.

    mu_1 : float
        Mean of the highest frequency peak.

    mu_2 : float
         Mean of the second-highest frequency peak.

    mu_3 : float
        Mean of remaining spectra.

    spectrum : real ndarray
        Spectrum to be fitted to Gaussian distributions.

    method : string
        Whether to us l1 norm or l2 norm.

    Returns
    -------
    OptimizeResult : OptimizeResult
        Returns optimised objective function as well as parameters.

    Notes
    -----

    """

    if not isinstance(w0, (type(np.asarray([[1.0, 2.0], [3.0, 4.0]])))):
        raise TypeError('\'w0\' must be of type np.ndarray.')
    if np.array(w0).dtype != np.array([[1., 1., 1., 1.]]).dtype:
        raise TypeError('\'w0\' must only contain floats.')
    if len(w0) != 6:
        raise ValueError('\'w0\' must be of length 6.')
    if not isinstance(f, (type(np.asarray([[1.0, 2.0], [3.0, 4.0]])))):
        raise TypeError('\'f\' must be of type np.ndarray.')
    if np.array(f).dtype != np.array([[1., 1., 1., 1.]]).dtype:
        raise TypeError('\'f\' must only contain floats.')
    try:
        if (not isinstance(mu_1[0], float)) or mu_1[0] <= 0:
            raise ValueError('\'mu_1\' must be a positive float.')
        if (not isinstance(mu_2[0], float)) or mu_2[0] <= 0:
            raise ValueError('\'mu_2\' must be a positive float.')
        if (not isinstance(mu_3[0], float)) or mu_3[0] <= 0:
            raise ValueError('\'mu_3\' must be a positive float.')
    except (IndexError, TypeError):
        if (not isinstance(mu_1, float)) or mu_1 <= 0:
            raise ValueError('\'mu_1\' must be a positive float.')
        if (not isinstance(mu_2, float)) or mu_2 <= 0:
            raise ValueError('\'mu_2\' must be a positive float.')
        if (not isinstance(mu_3, float)) or mu_3 <= 0:
            raise ValueError('\'mu_3\' must be a positive float.')
    if not isinstance(spectrum, (type(np.asarray([[1.0, 2.0], [3.0, 4.0]])))):
        raise TypeError('\'spectrum\' must be of type np.ndarray.')
    if np.array(spectrum).dtype != np.array([[1., 1., 1., 1.]]).dtype:
        raise TypeError('\'spectrum\' must only contain floats.')
    if len(f) != len(spectrum):
        raise ValueError('\'f\' and \'spectrum\' are incompatible lengths.')
    if method not in {'l1', 'l2'}:
        raise ValueError('\'method\' not an acceptable value.')

    cons = ({'type': 'ineq', 'fun': constraint_positive})
    if method == 'l1':
        return minimize(spectral_obj_func_l1, x0=w0,
                        args=(f, mu_1, mu_2, mu_3, spectrum), method='nelder-mead',
                        constraints=cons, bounds=[(0, None), (0, None), (0, None), (0, None), (0, None), (0, None)])
    elif method == 'l2':
        return minimize(spectral_obj_func_l2, x0=w0,
                        args=(f, mu_1, mu_2, mu_3, spectrum), method='nelder-mead',
                        constraints=cons, bounds=[(0, None), (0, None), (0, None), (0, None), (0, None), (0, None)])


def scaling_factor_obj_func(a, residual_time_series, trend_estimate):
    """
    Scaling factor calculation.

    Parameters
    ----------
    a : float
        Scaling factor to be optimised.

    residual_time_series : real ndarray
        Residual time series to which scaled trend estimate is fitted.

    trend_estimate : real ndarray
        Component estimate to be fitted with scaling factor.

    Returns
    -------
    l2_error : float
        L2 error of scaled result and residual time series.

    Notes
    -----

    """

    try:
        if (not isinstance(a[0], float)) or a[0] <= 0:
            raise ValueError('\'a\' must be a positive float.')
    except (IndexError, TypeError):
        if (not isinstance(a, float)) or a <= 0:
            raise ValueError('\'a\' must be a positive float.')
    if not isinstance(residual_time_series, (type(np.asarray([[1.0, 2.0], [3.0, 4.0]])))):
        raise TypeError('\'residual_time_series\' must be of type np.ndarray.')
    if np.array(residual_time_series).dtype != np.array([[1., 1., 1., 1.]]).dtype:
        raise TypeError('\'residual_time_series\' must only contain floats.')
    if not isinstance(trend_estimate, (type(np.asarray([[1.0, 2.0], [3.0, 4.0]])))):
        raise TypeError('\'trend_estimate\' must be of type np.ndarray.')
    if np.array(trend_estimate).dtype != np.array([[1., 1., 1., 1.]]).dtype:
        raise TypeError('\'trend_estimate\' must only contain floats.')
    if len(residual_time_series) != len(trend_estimate):
        raise ValueError('\'residual_time_series\' and \'trend_estimate\' are incompatible lengths.')


    l2_error = sum((residual_time_series - a * trend_estimate) ** 2)

    return l2_error


def scaling_factor(residual_time_series, trend_estimate):
    """
    Scaling factor calculation.

    Parameters
    ----------
    residual_time_series : real ndarray
        Residual time series to which scaled trend estimate is fitted.

    trend_estimate : real ndarray
        Component estimate to be fitted with scaling factor.

    Returns
    -------
    OptimizeResult : OptimizeResult
        Optimised result and optimised scale factor.

    Notes
    -----

    """

    if not isinstance(residual_time_series, (type(np.asarray([[1.0, 2.0], [3.0, 4.0]])))):
        raise TypeError('\'residual_time_series\' must be of type np.ndarray.')
    if np.array(residual_time_series).dtype != np.array([[1., 1., 1., 1.]]).dtype:
        raise TypeError('\'residual_time_series\' must only contain floats.')
    if not isinstance(trend_estimate, (type(np.asarray([[1.0, 2.0], [3.0, 4.0]])))):
        raise TypeError('\'trend_estimate\' must be of type np.ndarray.')
    if np.array(trend_estimate).dtype != np.array([[1., 1., 1., 1.]]).dtype:
        raise TypeError('\'trend_estimate\' must only contain floats.')
    if len(residual_time_series) != len(trend_estimate):
        raise ValueError('\'residual_time_series\' and \'trend_estimate\' are incompatible lengths.')

    cons = ({'type': 'ineq', 'fun': constraint_positive})
    return minimize(scaling_factor_obj_func, x0=np.asarray(1),
                    args=(residual_time_series, trend_estimate),
                    method='nelder-mead', constraints=cons)


def CovRegpy_ssd(time_series, initial_trend_ratio=3.0, nmse_threshold=0.01, plot=False, debug=False, method='l2'):
    """
    Singular Spectrum Decomposition based on Bonizzi, Karel, Meste, & Peeters (2014).

    Parameters
    ----------
    time_series : real ndarray
        Time series to decompose.

    initial_trend_ratio : positive float
        Recommended initial_trend_ratio=3.0 in Bonizzi, Karel, Meste, & Peeters (2014).
        Not appropriate for some trends as demonstrated in example.

    nmse_threshold : positive float
        Normalised Mean-Squared Error stopping criterion from Bonizzi, Karel, Meste, & Peeters (2014).
        Should explore additional stopping criteria.

    plot : boolean
        Plot all intermediate stages as part of debugging.

    debug : boolean
        Print incremental residual nmse value - while loop continues until drops below threshold of nmse_threshold.

    method : string
        Whether to use l1 or l2 optimisation in gaus_param() function.

    Returns
    -------
    time_series_est_mat : real ndarray
        Matrix containing decomposition of time series - iterates until nmse_threshold is met.

    Notes
    -----
    Many expansions possible. We explore some of these in attempt to make more robust.

    """

    if not isinstance(time_series, (type(np.asarray([[1.0, 2.0], [3.0, 4.0]])))):
        raise TypeError('\'time_series\' must be of type np.ndarray.')
    if np.array(time_series).dtype != np.array([[1., 1., 1., 1.]]).dtype:
        raise TypeError('\'time_series\' must only contain floats.')
    if (not isinstance(initial_trend_ratio, float)) or initial_trend_ratio <= 0:
        raise ValueError('\'initial_trend_ratio\' must be a positive float.')
    if (not isinstance(nmse_threshold, float)) or nmse_threshold <= 0:
        raise ValueError('\'nmse_threshold\' must be a positive float.')
    if not isinstance(plot, bool):
        raise TypeError('\'plot\' must be boolean.')
    if not isinstance(debug, bool):
        raise TypeError('\'debug\' must be boolean.')
    if method not in {'l1', 'l2'}:
        raise ValueError('\'method\' not an acceptable value.')

    # make duplicate of original time series
    time_series_orig = time_series.copy()
    time_series_resid = time_series.copy()

    # first test for 'SIZEABLE' trend
    dt = (24 * 60 * 60)
    s, f = mlab.psd(np.asarray(time_series_resid), Fs=1 / dt)
    if plot:
        plt.title('Power Spectral Density of Original Time Series')
        plt.plot(f * dt, s)
        plt.show()

    # if power is concentrated at first or second frequency band i.e. if time series is transient in nature.
    if np.log10(s)[1] == np.max(np.log10(s)) or np.log10(s)[0] == np.max(np.log10(s)):
        trend_est = \
            CovRegpy_ssa(time_series=np.asarray(time_series_resid),
                         L=int(len(np.asarray(time_series_resid)) / initial_trend_ratio), est=1)[0]
        if plot:
            plt.title('Original Transient Time Series and Trend Estimate')
            plt.plot(np.asarray(time_series_resid), label='Transient time series')
            plt.plot(trend_est, '--', label='Trend estimate')
            plt.legend(loc='best')
            plt.show()
        time_series_resid -= trend_est
        if plot:
            plt.title('Residual Time Series after Initial Transient Trend Removed')
            plt.plot(np.asarray(time_series_resid))
            plt.show()
            s, f = mlab.psd(np.asarray(time_series_resid), Fs=1 / dt)
            plt.title('Power Spectral Density after Initial Transient Trend Removed')
            plt.plot(f * dt, s)
            plt.show()

    try:
        time_series_est_mat = trend_est.reshape(1, -1)
    except:
        pass

    # Ensure initial nmse value initialises while loop
    nmse = nmse_threshold + 0.01

    while nmse > nmse_threshold:
        if debug:
            print(f'Normalised Mean-Squared Error: {nmse}')

        s, f = mlab.psd(np.asarray(time_series_resid), Fs=1 / dt)

        mu_1 = f[s == max(s)] * dt
        A_1 = (1 / 2) * s[s == max(s)]
        sigma_1 = (2 / 3) * f[((2 / 3) * s[s == max(s)] - s) == min((2 / 3) * s[s == max(s)] - s)] * dt
        gaus_1 = gaussian(f * dt, A_1, mu_1, sigma_1)

        try:
            mu_2 = f[s == np.sort(s[max_bool(s)])[-2]] * dt
            A_2 = (1 / 2) * s[s == np.sort(s[max_bool(s)])[-2]]
            sigma_2 = (2 / 3) * f[((2 / 3) * s[s == np.sort(s[max_bool(s)])[-2]] - s)
                                  == min((2 / 3) * s[s == np.sort(s[max_bool(s)])[-2]] - s)] * dt
            gaus_2 = gaussian(f * dt, A_2, mu_2, sigma_2)
        except:
            gaus_2 = gaussian(f * dt, A_1, mu_1, sigma_1)


        mu_3 = (mu_1 + mu_2) / 2
        A_3 = np.asarray([(1 / 4) * s[np.abs(f * dt - mu_3) == min(np.abs(f * dt - mu_3))][0]])
        sigma_3 = 4 * np.abs(mu_1 - mu_2)
        gaus_3 = gaussian(f * dt, A_3, mu_3, sigma_3)

        x0 = np.zeros(6)
        x0[0] = A_1
        x0[1] = A_2
        x0[2] = A_3
        x0[3] = sigma_1
        x0[4] = sigma_2
        x0[5] = sigma_3

        thetas = gaus_param(x0, f * dt, mu_1, mu_2, mu_3, s, method=method).x
        f_range = [(mu_1 - 2.5 * thetas[3])[0], (mu_1 + 2.5 * thetas[3])[0]]

        if plot:
            plt.plot(np.asarray(time_series_resid))
            plt.show()
            s, f = mlab.psd(np.asarray(time_series_resid), Fs=1 / dt)
            plt.title('Gaussian Function Initialisation')
            plt.plot(f * dt, s, label='Power-spectral density')
            plt.plot(f * dt, gaus_1, '--', label=r'$g_1(f) = A_1^{(0)}e^{\frac{(f-\mu_1)^2}{\sigma^{(0)2}_1}}$', linewidth=2)
            plt.plot(f * dt, gaus_2, '--', label=r'$g_2(f) = A_2^{(0)}e^{\frac{(f-\mu_2)^2}{\sigma^{(0)2}_2}}$', linewidth=2)
            plt.plot(f * dt, gaus_3, '--', label=r'$g_3(f) = A_3^{(0)}e^{\frac{(f-\mu_3)^2}{\sigma^{(0)2}_3}}$', linewidth=2)
            plt.plot(f * dt, gaus_1 + gaus_2 + gaus_3, '--',
                     label=r'$\sum_{i=1}^{3}g_i(f)$', linewidth=2)
            plt.plot(mu_1 * np.ones(100), np.linspace(np.min(s), 1.1 * np.max(s), 100), '--', label=r'$\mu_1$')
            plt.plot(mu_2 * np.ones(100), np.linspace(np.min(s), 1.1 * np.max(s), 100), '--', label=r'$\mu_2$')
            plt.plot(mu_3 * np.ones(100), np.linspace(np.min(s), 1.1 * np.max(s), 100), '--', label=r'$\mu_3$')
            plt.legend(loc='best', fontsize=8)
            plt.xlabel('Standardised Frequency')
            plt.ylabel('Spectral Density')
            plt.xlim(-0.005, 0.255)
            plt.xticks([0, 0.05, 0.1, 0.15, 0.2, 0.25])
            plt.show()
            plt.title('Gaussian Function Optimised')
            plt.plot(f * dt, s, label='Power-spectral density')
            plt.plot(f * dt, gaussian(f * dt, thetas[0], mu_1[0], thetas[3]),
                     '--', label=r'$g^{opt}_1(f) = A_1^{opt}e^{\frac{(f-\mu_1)^2}{\sigma^{opt2}_1}}$', linewidth=2)
            plt.plot(f * dt, gaussian(f * dt, thetas[1], mu_2[0], thetas[4]),
                     '--', label=r'$g^{opt}_2(f) = A_2^{opt}e^{\frac{(f-\mu_2)^2}{\sigma^{opt2}_2}}$', linewidth=2)
            plt.plot(f * dt, gaussian(f * dt, thetas[2], mu_3[0], thetas[5]),
                     '--', label=r'$g^{opt}_3(f) = A_3^{opt}e^{\frac{(f-\mu_3)^2}{\sigma^{opt2}_3}}$', linewidth=2)
            plt.plot(f * dt, gaussian(f * dt, thetas[0], mu_1[0], thetas[3]) +
                     gaussian(f * dt, thetas[1], mu_2[0], thetas[4]) +
                     gaussian(f * dt, thetas[2], mu_3[0], thetas[5]), '--',
                     label=r'$\sum_{i=1}^{3}g^{opt}_i(f)$', linewidth=2)
            plt.plot(f_range[0] * np.ones(101), np.linspace(0, 1.1 * 2 * A_1, 101), 'k--')
            plt.plot(f_range[1] * np.ones(101), np.linspace(0, 1.1 * 2 * A_1, 101), 'k--', label='Frequency bounds')
            plt.legend(loc='best', fontsize=8)
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
            plt.plot(time_series_resid, label='Detrended time series')
            plt.title('Downsampled and Modified Embedding Estimates')
            plt.plot(trend_est_1, '--', label='Downsampled estimate', linewidth=2)
            plt.plot(trend_est_2, '--', label='Modified embedding estimate', linewidth=2)
            plt.legend(loc='best')
            # plt.xlabel('Months')
            # plt.ylim(-390, 390)
            plt.show()

        # if sum(trend_est_2 * (time_series_resid - trend_est_2)) > 0:
        if sum((time_series_resid - trend_est_2) ** 2) < sum((time_series_resid - trend_est_1) ** 2):
            trend_est = trend_est_2
        else:
            trend_est = trend_est_1

        # scaling factor - top

        a_opt = scaling_factor(residual_time_series=time_series_resid, trend_estimate=trend_est).x
        if plot:
            plt.title('Optimisation of Scaling Factor')
            plt.plot(np.asarray(time_series_resid), label=r'$v_i(t)$')
            plt.plot(trend_est, '--', label=r'$g_i(t)$', linewidth=2)
            plt.plot(a_opt * trend_est, '--', label=r'$\tilde{g}_i(t)$', linewidth=2)
            plt.legend(loc='best')
            # plt.xlabel('Months')
            # plt.ylim(-390, 390)
            plt.show()
        trend_est *= a_opt

        # scaling factor - bottom

        time_series_resid -= trend_est

        try:
            if sum(np.abs(trend_est)) == 0:
                time_series_est_mat = np.vstack((time_series_est_mat, time_series_resid))
                time_series_resid = np.zeros_like(time_series_resid)
            else:
                time_series_est_mat = np.vstack((time_series_est_mat, trend_est))
        except:
            time_series_est_mat = trend_est.reshape(1, -1)

        nmse = sum(time_series_resid ** 2) / sum((time_series_orig - np.mean(time_series_orig)) ** 2)

    time_series_est_mat = np.vstack((time_series_est_mat, time_series_resid))

    return time_series_est_mat

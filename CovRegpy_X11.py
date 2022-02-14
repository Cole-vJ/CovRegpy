
# Document Strings Publication

import textwrap
import numpy as np
import cvxpy as cvx
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d

sns.set(style='darkgrid')


def henderson_kernel(order=13, start=np.nan, end=np.nan):
    """
    Henderson Kernel.

    Parameters
    ----------
    order : odd positive integer
        Order of filter.

    start : negative integer or numpy.nan
        Used in asymmetric weighting. If numpy.nan then symmetric weighting.

    end : positive integer or numpy.nan
        Used in asymmetric weighting. If numpy.nan then symmetric weighting.

    Returns
    -------
    y : real ndarray
        Weights for Henderson filter.

    Notes
    -----
    Has easily calculable asymmetric weighting.
    Exact Henderson Kernel - differs slightly from classical Henderson smoother.
    Renormalised when incomplete - does nothing when complete as weights sum to one.

    """
    if np.isnan(start):
        start = -int((order - 1) / 2)
    if np.isnan(end):
        end = int((order - 1) / 2)
    t = np.asarray(range(start, end + 1)) / (int((order - 1) / 2) + 1)
    y = (15 / 79376) * (5184 - 12289 * t ** 2 + 9506 * t ** 4 - 2401 * t ** 6) * ((2175 / 1274) - (1372 / 265) * t ** 2)
    y = y / sum(y)

    return y


def henderson_weights(order=13, start=np.nan, end=np.nan):
    """
    Classical Henderson weights.

    Parameters
    ----------
    order : odd positive integer
        Order of filter.

    start : negative integer or numpy.nan
        Used in asymmetric weighting. If numpy.nan then symmetric weighting.

    end : positive integer or numpy.nan
        Used in asymmetric weighting. If numpy.nan then symmetric weighting.

    Returns
    -------
    y : real ndarray
        Weights for classical Henderson weights.

    Notes
    -----
    Does not have easily calculable asymmetric weighting.
    Renormalised when incomplete - does nothing when complete as weights sum to one.

    """
    if np.isnan(start):
        start = -int((order - 1) / 2)
    if np.isnan(end):
        end = int((order - 1) / 2)
    p = int((order - 1) / 2)
    n = p + 2
    vector = np.asarray(range(start, (end + 1)))
    y = (315 * ((n - 1) ** 2 - vector ** 2) * (n ** 2 - vector ** 2) * ((n + 1) ** 2 - vector ** 2) *
         (3 * n ** 2 - 16 - 11 * vector ** 2)) / \
        (8 * n * (n ** 2 - 1) * (4 * n ** 2 - 1) * (4 * n ** 2 - 9) * (4 * n ** 2 - 25))
    y = y / sum(y)

    return y


def henderson_ma(time_series, order=13, method='kernel'):
    """
    Henderson filter.

    Parameters
    ----------
    time_series : real ndarray
        Time series to be Henderson filtered.

    order : odd positive integer
        Order of Henderson filter.

    method : string
        Technique to be used in calculation of weights.

    Returns
    -------
    henderson_filtered_time_series : real ndarray
        Henderson filtered time series.

    Notes
    -----
    Only two options. Classical weights and kernel weights.

    Require asymmetric weights that sum to approximately one on the edges - multiple options:
        (1) use asymmetric filter (truncate and renormalise) - 'renormalise'.
        (2) extrapolate and use symmetric filter (X11ARIMA) - not yet added.
        (3) Reproducing Kernel Hilbert Space Method - 'kernel'.
        (4) Classical asymmetric results (unknown calculation) - not yet added.

    """
    henderson_filtered_time_series = np.zeros_like(time_series)

    if method == 'renormalise':
        weights = henderson_weights(order=order)
    elif method == 'kernel':
        weights = henderson_kernel(order=order)

    for k in range(len(time_series)):
        if k < ((order - 1) / 2):
            if method == 'renormalise':
                asymmetric_weights = henderson_weights(order=order, start=(0 - k))
            elif method == 'kernel':
                asymmetric_weights = henderson_kernel(order=order, start=(0 - k))
            henderson_filtered_time_series[k] = \
                np.sum(asymmetric_weights * time_series[:int(k + ((order - 1) / 2) + 1)])
        elif k > len(time_series) - ((order - 1) / 2) - 1:
            if method == 'renormalise':
                asymmetric_weights = henderson_weights(order=order, end=(len(time_series) - k - 1))
            elif method == 'kernel':
                asymmetric_weights = henderson_kernel(order=order, end=(len(time_series) - k - 1))
            henderson_filtered_time_series[k] = \
                np.sum(asymmetric_weights * time_series[int(k - ((order - 1) / 2)):])
        else:
            henderson_filtered_time_series[k] = \
                np.sum(weights * time_series[int(k - ((order - 1) / 2)):int(k + ((order - 1) / 2) + 1)])

    return henderson_filtered_time_series


def seasonal_ma(time_series, factors='3x3', seasonality='monthly'):
    """
    Seasonal moving-average filter.

    Parameters
    ----------
    time_series : real ndarray
        Time series to be seasonally filtered.

    factors : string
        Seasonal filter to be applied.

    seasonality : string
        Seasonality to be used - either monthly or quarterly.

    Returns
    -------
    seasonal_filtered_time_series : real ndarray
        Seasonally filtered time series.

    Notes
    -----
    Need to make an automated factor-weighting calculation.

    """
    seasonal_filtered_time_series = np.zeros_like(time_series)

    if factors == '3x3':
        weighting = np.asarray((1 / 9, 2 / 9, 1 / 3, 2 / 9, 1 / 9))
        season_window_width = 5
    elif factors == '3x5':
        weighting = np.asarray((1 / 15, 2 / 15, 1 / 5, 1 / 5, 1 / 5, 2 / 15, 1 / 15))
        season_window_width = 7
    elif factors == '3x7':
        weighting = np.asarray((1 / 21, 2 / 21, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 2 / 21, 1 / 21))
        season_window_width = 9
    elif factors == '3x9':
        weighting = np.asarray((1 / 27, 2 / 27, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 2 / 27, 1 / 27))
        season_window_width = 11

    if seasonality == 'monthly':
        for month in range(12):

            month_bool = (np.asarray(range(len(time_series))) % 12 - month == 0)
            index_values = np.asarray(range(len(time_series)))[month_bool]

            for point in np.asarray(range(len(index_values)))[
                         int((season_window_width - 1) / 2):int(len(index_values) - ((season_window_width - 1) / 2))]:
                relative_points = \
                    time_series[index_values[int(point - ((season_window_width - 1) / 2)):int(
                        point + ((season_window_width + 1) / 2))]]

                seasonal_filtered_time_series[index_values[point]] = \
                    sum(weighting * relative_points)

        seasonal_filtered_time_series[:int(((season_window_width - 1) / 2) * 12)] = np.append(
            seasonal_filtered_time_series[int(((season_window_width - 1) / 2) * 12):int(((season_window_width + 1) / 2) * 12)],
            seasonal_filtered_time_series[int(((season_window_width - 1) / 2) * 12):int(((season_window_width + 1) / 2) * 12)])

        seasonal_filtered_time_series[int(len(seasonal_filtered_time_series) - ((season_window_width - 1) / 2) * 12):] = np.append(
            seasonal_filtered_time_series[int(len(seasonal_filtered_time_series) - ((season_window_width + 1) / 2) * 12):int(
                len(seasonal_filtered_time_series) - ((season_window_width - 1) / 2) * 12)], seasonal_filtered_time_series[int(
                len(seasonal_filtered_time_series) - ((season_window_width + 1) / 2) * 12):int(
                len(seasonal_filtered_time_series) - ((season_window_width - 1) / 2) * 12)])

    elif seasonality == 'quarterly':
        for quarter in range(4):

            month_bool = (np.asarray(range(len(time_series))) % 4 - quarter == 0)
            index_values = np.asarray(range(len(time_series)))[month_bool]

            for point in np.asarray(range(len(index_values)))[
                         int((season_window_width - 1) / 2):int(len(index_values) - ((season_window_width - 1) / 2))]:
                relative_points = \
                    time_series[index_values[int(point - ((season_window_width - 1) / 2)):int(
                        point + ((season_window_width + 1) / 2))]]

                seasonal_filtered_time_series[index_values[point]] = \
                    sum(weighting * relative_points)

        seasonal_filtered_time_series[:int(((season_window_width - 1) / 2) * 4)] = np.append(
            seasonal_filtered_time_series[
            int(((season_window_width - 1) / 2) * 4):int(((season_window_width + 1) / 2) * 4)],
            seasonal_filtered_time_series[
            int(((season_window_width - 1) / 2) * 4):int(((season_window_width + 1) / 2) * 4)])

        seasonal_filtered_time_series[
        int(len(seasonal_filtered_time_series) - ((season_window_width - 1) / 2) * 4):] = np.append(
            seasonal_filtered_time_series[
            int(len(seasonal_filtered_time_series) - ((season_window_width + 1) / 2) * 4):int(
                len(seasonal_filtered_time_series) - ((season_window_width - 1) / 2) * 4)],
            seasonal_filtered_time_series[int(
                len(seasonal_filtered_time_series) - ((season_window_width + 1) / 2) * 4):int(
                len(seasonal_filtered_time_series) - ((season_window_width - 1) / 2) * 4)])

    return seasonal_filtered_time_series


def x11(time, time_series, model_type='additive', seasonality='monthly',
        seasonal_factor='3x3', trend_window_width_1=13, trend_window_width_2=13, trend_window_width_3=13):
    """
    Standard X11 decomposition method.

    Parameters
    ----------
    time : real ndarray
        Time over which time series is defined.

    time_series : real ndarray
        Time series to be filtered.

    model_type : string
        Only additive model for now.

    seasonality : string
        Seasonality to be used - either monthly or quarterly.

    seasonal_factor : string
        Seasonal weights to be used.

    trend_window_width_1 : odd positive integer
        Trend window width to be used on first iteration.

    trend_window_width_2 : odd positive integer
        Trend window width to be used on second iteration.

    trend_window_width_3 : odd positive integer
        Trend window width to be used on third iteration.

    Returns
    -------
    final_estimate_trend : real ndarray
        Trend estimate of X11 filtered method.

    final_estimate_season : real ndarray
        Seasonal estimate of X11 filtered method.

    final_estimate_irregular : real ndarray
        Irregular estimate (error terms) of X11 filtered method.

    Notes
    -----
    Standard X11 method.

    """
    if model_type == 'additive':

        # step 1
        # initial estimate of trend-cycle
        first_estimate_trend = np.zeros_like(time_series)

        for point in \
                range(len(time_series))[int((trend_window_width_1 - 1) / 2):int(len(time_series) -
                                                                                ((trend_window_width_1 - 1) / 2))]:

            first_estimate_trend[point] = \
                np.mean(time_series[int(point -
                                        ((trend_window_width_1 - 1) / 2)):int(point +
                                                                              ((trend_window_width_1 - 1) / 2))])

        # interpolate edges
        # relevant to remove some trend at edges
        interpolation = \
            interp1d(time[int((trend_window_width_1 - 1) / 2): int(len(time_series) -
                                                                   ((trend_window_width_1 - 1) / 2))],
                     first_estimate_trend[int((trend_window_width_1 - 1) / 2): int(len(time_series) -
                                                                                   ((trend_window_width_1 - 1) / 2))],
                     fill_value='extrapolate')

        left_extrapolate = interpolation(time[:int((trend_window_width_1 - 1) / 2)])
        right_extrapolate = interpolation(time[int(len(time_series) - ((trend_window_width_1 - 1) / 2)):])

        first_estimate_trend[:int((trend_window_width_1 - 1) / 2)] = left_extrapolate
        first_estimate_trend[int(len(time_series) - ((trend_window_width_1 - 1) / 2)):] = right_extrapolate

        # time series without trend-cycle component
        no_trend_cycle = time_series - first_estimate_trend

        # step 2
        # initial estimate of seasonality
        # assume monthly for now
        first_estimate_season = seasonal_ma(no_trend_cycle, factors=seasonal_factor, seasonality=seasonality)

        # step 3
        # estimate seasonality adjusted data
        # time series without seasonality component (with trend-cycle component)
        no_seasonality = time_series - first_estimate_season

        # step 4
        # better estimate of trend
        next_estimate_trend = henderson_ma(no_seasonality, trend_window_width_2)
        no_trend_cycle_2 = time_series - next_estimate_trend

        # step 5
        final_estimate_season = seasonal_ma(no_trend_cycle_2, factors=seasonal_factor, seasonality=seasonality)

        # step 6
        no_seasonality_2 = time_series - final_estimate_season

        # step 7
        final_estimate_trend = henderson_ma(no_seasonality_2, trend_window_width_3)

        # step 8
        final_estimate_irregular = no_seasonality_2 - final_estimate_trend

    return final_estimate_trend, final_estimate_season, final_estimate_irregular


if __name__ == "__main__":

    time = np.linspace(0, 120, 121)
    time_series = time + (1 / 1000) * (time * (time - 60) * (time - 110)) + \
                  10 * np.sin(((2 * np.pi) / 12) * time) + np.random.normal(0, 5, 121)
    plt.plot(time, time_series)
    plt.title('Example X11 Time Series')
    plt.show()

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
    # print(henderson_13)
    # print(sum(henderson_13))
    # print(sum(henderson_13 * index))
    # print(sum(henderson_13 * index ** 2))
    # print(filtered_signal[3:-3])
    # print(sum(filtered_signal[3:-3]))
    # print(sum(filtered_signal[3:-3] * index))
    # print(sum(filtered_signal[3:-3] * index ** 2))
    plt.plot(henderson_13, label='Henderson function test')
    plt.plot(filtered_signal[3:-3], '--', label='Direct calculation test')
    plt.legend(loc='best')
    plt.title('Henderson 13 Points Weights Calculation Test')
    plt.show()

    filtered_time_series = x11(time, time_series, trend_window_width_3=23)
    plt.plot(time, time_series, label='Time series')
    plt.plot(time, filtered_time_series[0], label='Trend estimate')
    plt.legend(loc='best')
    plt.title('Example X11 Time Series and Trend Estimate')
    plt.show()

    plt.plot(time, time_series, label='Time series')
    plt.plot(time, filtered_time_series[0], label='Trend estimate')
    plt.plot(time, filtered_time_series[1], label='Seasonal estimate')
    plt.plot(time, filtered_time_series[2], label='Error estimate')
    plt.legend(loc='best')
    plt.title(textwrap.fill('Example X11 Time Series, Trend Estimate, Seasonal Estimate, and Error Estimate', 40))
    plt.show()

    time = np.linspace(-4, 4, 100)
    henderson_13 = time * (time - 2) * (time + 2)

    plt.title('Cubic Function and Henderson Filtered Cubic Function')
    plt.plot(time, henderson_13, label='Cubic function')
    plt.plot(time, henderson_ma(henderson_13), '--', label='Henderson filtered function')
    plt.legend(loc='best')
    plt.show()


#     ________
#            /
#      \    /
#       \  /
#        \/

import numpy as np
from CovRegpy_PCA import pca_func
from CovRegpy_sharpe import (sharpe_weights, sharpe_weights_individual_weight_restriction,
                             sharpe_weights_summation_weight_restriction)
from CovRegpy_RPP import global_obj_fun, global_weights, global_weights_long


def efficient_frontier(global_minimum_weights, global_minimum_returns, global_minimum_sd,
                       maximum_sharpe_weights, maximum_sharpe_returns, maximum_sharpe_sd, variance, n=101):
    """
    Calculates efficient frontier by extrapolating between weights of
    global minimum variance portfolio and maximum Sharpe ratio portfolios.

    Parameters
    ----------
    global_minimum_weights : real ndarray
        Global minimum variance portfolio weights.

    global_minimum_returns : real ndarray
        Global minimum variance portfolio returns.

    global_minimum_sd : real ndarray
        Global minimum variance portfolio standard deviation.

    maximum_sharpe_weights : real ndarray
        Maximum Sharpe ratio portfolio weights.

    maximum_sharpe_returns : real ndarray
        Maximum Sharpe ratio portfolio returns.

    maximum_sharpe_sd : real ndarray
        Maximum Sharpe ratio portfolio standard deviation.

    variance : real ndarray
        Variance of assets in model universe.

    n : integer
        Number of points to use in efficient frontier estimation.

    Returns
    -------
    efficient_frontier_sd : real ndarray
        Efficient frontier standard deviations.

    efficient_frontier_returns : real ndarray
        Efficient frontier returns.

    Notes
    -----

    """
    efficient_frontier_sd = np.zeros(n)
    efficient_frontier_returns = np.zeros(n)

    efficient_frontier_sd[0] = global_minimum_sd
    efficient_frontier_returns[0] = global_minimum_returns
    for i in range(1, int(n - 1)):
        efficient_frontier_sd[i] = \
            np.sqrt(global_obj_fun(global_minimum_weights * (1 - (i / (n - 1))) + (i / (n - 1)) *
                                   maximum_sharpe_weights, variance))
        efficient_frontier_returns[i] = (1 - (i / (n - 1))) * \
                                            global_minimum_returns + (i / 100) * maximum_sharpe_returns

    efficient_frontier_sd[-1] = maximum_sharpe_sd
    efficient_frontier_returns[-1] = maximum_sharpe_returns

    return efficient_frontier_sd, efficient_frontier_returns


def global_minimum_information(variance, returns):
    """
    Calculates global minimum variance portfolio using variance and returns of asset model universe.

    Parameters
    ----------
    variance : real ndarray
        Variance of portfolio universe.

    returns : real ndarray
        Returns of portfolio universe.

    Returns
    -------
    global_minimum_weights : real ndarray
        Global minimum variance portfolio weights.

    global_minimum_sd : real ndarray
        Global minimum variance portfolio standard deviation.

    global_minimum_returns : real ndarray
        Global minimum variance portfolio returns.

    Notes
    -----

    """
    global_minimum_weights = global_weights(variance)
    global_minimum_sd = np.sqrt(global_obj_fun(global_minimum_weights, variance))
    global_minimum_returns = sum(global_minimum_weights * returns)

    return global_minimum_weights, global_minimum_sd, global_minimum_returns


def global_minimum_forward_applied_information(weight_calculating_variance, forward_looking_variance,
                                               forward_looking_returns):
    """
    Calculates global minimum variance portfolio using variance and returns of asset model universe.
    Applies weights to realised covariance and returns without look-forward bias.

    Parameters
    ----------
    weight_calculating_variance : real ndarray
        Variance used to calculate global minimum variance portfolio weights.

    forward_looking_variance : real ndarray
        Variance on which calculated weights are applied to calculate realised portfolio statistics.

    forward_looking_returns : real ndarray
        Returns on which calculated weights are applied to calculate realised portfolio statistics.

    Returns
    -------
    global_minimum_weights : real ndarray
        Global minimum variance portfolio weights.

    global_minimum_sd : real ndarray
        Global minimum variance portfolio standard deviation.

    global_minimum_returns : real ndarray
        Global minimum variance portfolio returns.

    Notes
    -----
    Weights calculated on realised covariance, applied to monthly covariance and returns looking forward.

    """
    global_minimum_weights = global_weights(weight_calculating_variance)
    global_minimum_sd = np.sqrt(global_obj_fun(global_minimum_weights, forward_looking_variance))
    global_minimum_returns = sum(global_minimum_weights * forward_looking_returns)

    return global_minimum_weights, global_minimum_sd, global_minimum_returns


def global_minimum_forward_applied_information_long(weight_calculating_variance, forward_looking_variance,
                                                    forward_looking_returns):
    """
    Calculates global minimum variance portfolio using variance and returns of asset model universe.
    Applies weights to realised covariance and returns without look-forward bias.
    With additional long restrictions.

    Parameters
    ----------
    weight_calculating_variance : real ndarray
        Variance used to calculate global minimum variance portfolio weights.

    forward_looking_variance : real ndarray
        Variance on which calculated weights are applied to calculate realised portfolio statistics.

    forward_looking_returns : real ndarray
        Returns on which calculated weights are applied to calculate realised portfolio statistics.

    Returns
    -------
    global_minimum_weights : real ndarray
        Global minimum variance portfolio weights.

    global_minimum_sd : real ndarray
        Global minimum variance portfolio standard deviation.

    global_minimum_returns : real ndarray
        Global minimum variance portfolio returns.

    Notes
    -----
    Weights calculated on realised covariance, applied to monthly covariance and returns looking forward.

    """
    global_minimum_weights = global_weights_long(weight_calculating_variance).x
    global_minimum_sd = np.sqrt(global_obj_fun(global_minimum_weights, forward_looking_variance))
    global_minimum_returns = sum(global_minimum_weights * forward_looking_returns)

    return global_minimum_weights, global_minimum_sd, global_minimum_returns


def sharpe_information(variance, returns, global_minimum_weights, global_minimum_returns, risk_free=(0.01 / 365)):
    """
    Maximum Sharpe ratio portfolio.

    Parameters
    ----------
    variance : real ndarray
        Variance of portfolio universe.

    returns : real ndarray
        Returns of portfolio universe.

    global_minimum_weights : real ndarray
        Weights used to reflect Sharpe ratio if negative as cone is symmetric.

    global_minimum_returns : real ndarray
        Returns used to reflect Sharpe ratio if negative as cone is symmetric.

    risk_free : float
        Risk-free rate used in calculation of Sharpe ratio.

    Returns
    -------
    sharpe_maximum_weights : real ndarray
        Maximum Sharpe ratio portfolio weights.

    sharpe_maximum_sd : real ndarray
        Maximum Sharpe ratio portfolio standard deviation.

    sharpe_maximum_returns : real ndarray
        Maximum Sharpe ratio portfolio returns.

    Notes
    -----

    """
    sharpe_maximum_weights = sharpe_weights(variance, returns, risk_free)
    sharpe_maximum_sd = np.sqrt(global_obj_fun(sharpe_maximum_weights, variance))
    sharpe_maximum_returns = sum(sharpe_maximum_weights * returns)

    # reflect if negative
    if sharpe_maximum_returns < global_minimum_returns:
        sharpe_maximum_weights = 2 * global_minimum_weights - sharpe_maximum_weights
        sharpe_maximum_sd = np.sqrt(global_obj_fun(sharpe_maximum_weights, variance))
        sharpe_maximum_returns = sum(sharpe_maximum_weights * returns)

    return sharpe_maximum_weights, sharpe_maximum_sd, sharpe_maximum_returns


def sharpe_forward_applied_information(weight_calculating_variance, weight_calculating_returns,
                                       forward_looking_variance, forward_looking_returns,
                                       global_minimum_weights, global_minimum_returns, risk_free=(0.01 / 365)):
    """
    Calculates maximum Sharpe ratio portfolio using variance and returns of asset model universe.
    Applies weights to realised covariance and returns without look-forward bias.

    Parameters
    ----------
    weight_calculating_variance : real ndarray
        Variance used to calculate global minimum variance portfolio weights.

    weight_calculating_returns : real ndarray
        Variance used to calculate global minimum variance portfolio weights.

    forward_looking_variance : real ndarray
        Variance on which calculated weights are applied to calculate realised portfolio statistics.

    forward_looking_returns : real ndarray
        Returns on which calculated weights are applied to calculate realised portfolio statistics.

    global_minimum_weights : real ndarray
        Weights used to reflect Sharpe ratio if negative as cone is symmetric.

    global_minimum_returns : real ndarray
        Returns used to reflect Sharpe ratio if negative as cone is symmetric.

    risk_free : float
        Risk-free rate used in calculation of Sharpe ratio.

    Returns
    -------
    sharpe_maximum_weights : real ndarray
        Maximum Sharpe ratio portfolio weights.

    sharpe_maximum_sd : real ndarray
        Maximum Sharpe ratio portfolio standard deviation.

    sharpe_maximum_returns : real ndarray
        Maximum Sharpe ratio portfolio returns.

    Notes
    -----
    Weights calculated on realised covariance, applied to monthly covariance and returns looking forward.

    """
    sharpe_maximum_weights = sharpe_weights(weight_calculating_variance, weight_calculating_returns, risk_free)
    sharpe_maximum_sd = np.sqrt(global_obj_fun(sharpe_maximum_weights, forward_looking_variance))
    sharpe_maximum_returns = sum(sharpe_maximum_weights * forward_looking_returns)

    # reflect if negative
    if sharpe_maximum_returns < global_minimum_returns:
        sharpe_maximum_weights = 2 * global_minimum_weights - sharpe_maximum_weights
        sharpe_maximum_sd = np.sqrt(global_obj_fun(sharpe_maximum_weights, forward_looking_variance))
        sharpe_maximum_returns = sum(sharpe_maximum_weights * forward_looking_returns)

    return sharpe_maximum_weights, sharpe_maximum_sd, sharpe_maximum_returns


def sharpe_forward_applied_information_individual_restriction(weight_calculating_variance, weight_calculating_returns,
                                                              forward_looking_variance, forward_looking_returns,
                                                              risk_free=(0.01 / 365), short_limit=0.3):
    """
    Calculates maximum Sharpe ratio portfolio using variance and returns of asset model universe.
    Applies weights to realised covariance and returns without look-forward bias.
    Restrictions are applied to individual weights.

    Parameters
    ----------
    weight_calculating_variance : real ndarray
        Variance used to calculate global minimum variance portfolio weights.

    weight_calculating_returns : real ndarray
        Variance used to calculate global minimum variance portfolio weights.

    forward_looking_variance : real ndarray
        Variance on which calculated weights are applied to calculate realised portfolio statistics.

    forward_looking_returns : real ndarray
        Returns on which calculated weights are applied to calculate realised portfolio statistics.

    risk_free : float
        Risk-free rate used in calculation of Sharpe ratio.

    short_limit : float
        Restriction on individual weights in portfolio.

    Returns
    -------
    sharpe_maximum_weights : real ndarray
        Maximum Sharpe ratio portfolio weights.

    sharpe_maximum_sd : real ndarray
        Maximum Sharpe ratio portfolio standard deviation.

    sharpe_maximum_returns : real ndarray
        Maximum Sharpe ratio portfolio returns.

    Notes
    -----
    Weights calculated on realised covariance, applied to monthly covariance and returns looking forward.

    """
    sharpe_maximum_weights = sharpe_weights_individual_weight_restriction(weight_calculating_variance,
                                                                          weight_calculating_returns, risk_free,
                                                                          short_limit=short_limit).x
    sharpe_maximum_sd = np.sqrt(global_obj_fun(sharpe_maximum_weights, forward_looking_variance))
    sharpe_maximum_returns = sum(sharpe_maximum_weights * forward_looking_returns)

    # # reflect if negative - not appropriate as weights have restrictions
    # if sharpe_maximum_returns < global_minimum_returns:
    #     sharpe_maximum_weights = 2 * global_minimum_weights - sharpe_maximum_weights
    #     sharpe_maximum_sd = np.sqrt(global_obj_fun(sharpe_maximum_weights, forward_looking_variance))
    #     sharpe_maximum_returns = sum(sharpe_maximum_weights * weight_calculating_returns)

    return sharpe_maximum_weights, sharpe_maximum_sd, sharpe_maximum_returns


def sharpe_forward_applied_information_summation_restriction(weight_calculating_variance, weight_calculating_returns,
                                                             forward_looking_variance, forward_looking_returns,
                                                             risk_free=(0.01 / 365), short_limit=0.3, long_limit=1.3):
    """
    Calculates maximum Sharpe ratio portfolio using variance and returns of asset model universe.
    Applies weights to realised covariance and returns without look-forward bias.
    Restrictions are applied to weight summations.

    Parameters
    ----------
    weight_calculating_variance : real ndarray
        Variance used to calculate global minimum variance portfolio weights.

    weight_calculating_returns : real ndarray
        Variance used to calculate global minimum variance portfolio weights.

    forward_looking_variance : real ndarray
        Variance on which calculated weights are applied to calculate realised portfolio statistics.

    forward_looking_returns : real ndarray
        Returns on which calculated weights are applied to calculate realised portfolio statistics.

    risk_free : float
        Risk-free rate used in calculation of Sharpe ratio.

    short_limit : float
        Restriction on weight summation in portfolio - short limit.

    long_limit : float
        Restriction on weight summation in portfolio - long limit.

    Returns
    -------
    sharpe_maximum_weights : real ndarray
        Maximum Sharpe ratio portfolio weights.

    sharpe_maximum_sd : real ndarray
        Maximum Sharpe ratio portfolio standard deviation.

    sharpe_maximum_returns : real ndarray
        Maximum Sharpe ratio portfolio returns.

    Notes
    -----
    Weights calculated on realised covariance, applied to monthly covariance and returns looking forward.

    """
    sharpe_maximum_weights = sharpe_weights_summation_weight_restriction(weight_calculating_variance,
                                                                         weight_calculating_returns,
                                                                         risk_free, short_limit=short_limit,
                                                                         long_limit=long_limit).x
    sharpe_maximum_sd = np.sqrt(global_obj_fun(sharpe_maximum_weights, forward_looking_variance))
    sharpe_maximum_returns = sum(sharpe_maximum_weights * forward_looking_returns)

    # # reflect if negative - not appropriate as weights have restrictions
    # if sharpe_maximum_returns < global_minimum_returns:
    #     sharpe_maximum_weights = 2 * global_minimum_weights - sharpe_maximum_weights
    #     sharpe_maximum_sd = np.sqrt(global_obj_fun(sharpe_maximum_weights, forward_looking_variance))
    #     sharpe_maximum_returns = sum(sharpe_maximum_weights * forward_looking_returns)

    return sharpe_maximum_weights, sharpe_maximum_sd, sharpe_maximum_returns


def pca_information(variance, returns, factors=3):
    """
    Calculates principal component weighting.

    Parameters
    ----------
    variance : real ndarray
        Variance of assets in model universe.

    returns : real ndarray
        Returns of assets in model universe.

    factors : integer
        Number of factors to use in principal portfolio weighting calculation.

    Returns
    -------
    pca_weights : real ndarray
        Principal portfolio weights.

    pca_sd : real ndarray
        Principal portfolio standard deviation.

    pca_returns : real ndarray
        Principal portfolio returns.

    Notes
    -----

    """
    pca_weights = pca_func(variance, factors)
    pca_weights = pca_weights.sum(axis=1)
    pca_sd = np.sqrt(global_obj_fun(pca_weights, variance))
    pca_returns = sum(pca_weights * returns)

    return pca_weights, pca_sd, pca_returns


def pca_forward_applied_information(weight_calculating_variance, forward_looking_variance,
                                    forward_looking_returns, factors=3):
    """
    Calculates principal component weighting.

    Parameters
    ----------
    weight_calculating_variance : real ndarray
        Variance used to calculate global minimum variance portfolio weights.

    forward_looking_variance : real ndarray
        Variance on which calculated weights are applied to calculate realised portfolio statistics.

    forward_looking_returns : real ndarray
        Returns on which calculated weights are applied to calculate realised portfolio statistics.

    factors : integer
        Number of factors to use in principal portfolio weighting calculation.

    Returns
    -------
    pca_weights : real ndarray
        Principal portfolio weights.

    pca_sd : real ndarray
        Principal portfolio standard deviation.

    pca_returns : real ndarray
        Principal portfolio returns.

    Notes
    -----
    Weights calculated on realised covariance, applied to monthly covariance and returns looking forward.

    """
    pca_weights = pca_func(weight_calculating_variance, factors)
    pca_weights = pca_weights.sum(axis=1)
    pca_sd = np.sqrt(global_obj_fun(pca_weights, forward_looking_variance))
    pca_returns = sum(pca_weights * forward_looking_returns)

    return pca_weights, pca_sd, pca_returns

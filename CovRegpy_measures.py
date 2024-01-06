
#     ________
#            /
#      \    /
#       \  /
#        \/

import numpy as np


def portfolio_return(weights, all_returns):
    """
    Calculate daily returns of portfolio.

    Parameters
    ----------
    weights : real ndarray
        Matrix containing weights.

    all_returns : real ndarray
        Matrix containing daily returns.

    Returns
    -------
    portfolio_returns : real ndarray
        Vector containing portfolio daily returns.

    Notes
    -----
    Utility function used in other portfolio measures.

    """
    if np.shape(weights)[0] > np.shape(weights)[1]:
        weights = weights.T
    if np.shape(all_returns)[0] > np.shape(all_returns)[1]:
        all_returns = all_returns.T
    portfolio_returns = np.sum(weights * all_returns, axis=0)

    return portfolio_returns


def mean_return(weights, all_returns, window):
    """
    Calculate mean of daily returns of portfolio.

    Parameters
    ----------
    weights : real ndarray
        Matrix containing weights.

    all_returns : real ndarray
        Matrix containing daily returns.

    window : positive integer
        Window over which measure is calculated.

    Returns
    -------
    mean_returns : real ndarray
        Vector containing mean of portfolio daily returns using specified window.

    Notes
    -----

    """
    portfolio_returns = portfolio_return(weights, all_returns)
    mean_returns = np.zeros(int(np.shape(weights)[1] - window + 1))
    for col in range(len(mean_returns)):
        mean_returns[col] = np.mean(portfolio_returns[col:int(col + window)])

    return mean_returns


def variance_return(weights, all_returns, window):
    """
    Calculate variance of daily returns of portfolio.

    Parameters
    ----------
    weights : real ndarray
        Matrix containing weights.

    all_returns : real ndarray
        Matrix containing daily returns.

    window : positive integer
        Window over which measure is calculated.

    Returns
    -------
    variance_returns : real ndarray
        Vector containing variance of portfolio daily returns using specified window.

    Notes
    -----

    """
    portfolio_returns = portfolio_return(weights, all_returns)
    variance_returns = np.zeros(int(np.shape(weights)[1] - window + 1))
    for col in range(len(variance_returns)):
        variance_returns[col] = np.var(portfolio_returns[col:int(col + window)])

    return variance_returns


def value_at_risk_return(weights, all_returns, window):
    """
    Calculate value-at-risk of daily returns of portfolio.

    Parameters
    ----------
    weights : real ndarray
        Matrix containing weights.

    all_returns : real ndarray
        Matrix containing daily returns.

    window : positive integer
        Window over which measure is calculated.

    Returns
    -------
    variance_returns : real ndarray
        Vector containing value-at-risk of portfolio daily returns using specified window.

    Notes
    -----

    """
    portfolio_returns = portfolio_return(weights, all_returns)
    value_at_risk_returns = np.zeros(int(np.shape(weights)[1] - window + 1))
    for col in range(len(value_at_risk_returns)):
        value_at_risk_returns[col] = np.quantile(portfolio_returns[col:int(col + window)], 0.05)

    return value_at_risk_returns


def c_value_at_risk_return(weights, all_returns, window):
    """
    Calculate c-value-at-risk (c-VaR) of daily returns of portfolio. No assumptions of Normality.

    Parameters
    ----------
    weights : real ndarray
        Matrix containing weights.

    all_returns : real ndarray
        Matrix containing daily returns.

    window : positive integer
        Window over which measure is calculated.

    Returns
    -------
    variance_returns : real ndarray
        Vector containing value-at-risk of portfolio daily returns using specified window.

    Notes
    -----

    """
    portfolio_returns = portfolio_return(weights, all_returns)
    c_value_at_risk_returns = np.zeros(int(np.shape(weights)[1] - window + 1))
    for col in range(len(c_value_at_risk_returns)):
        quant = np.quantile(portfolio_returns[col:int(col + window)], 0.05)
        c_value_at_risk_returns[col] = np.mean(portfolio_returns[col:int(col + window)][portfolio_returns[col:int(col + window)] < quant])

    return c_value_at_risk_returns


def max_draw_down_return(weights, all_returns, window):
    """
    Calculate maximum drawdown of daily returns of portfolio.

    Parameters
    ----------
    weights : real ndarray
        Matrix containing weights.

    all_returns : real ndarray
        Matrix containing daily returns.

    window : positive integer
        Window over which measure is calculated.

    Returns
    -------
    variance_returns : real ndarray
        Vector containing maximum drawdown of portfolio daily returns using specified window.

    Notes
    -----

    """
    portfolio_returns = portfolio_return(weights, all_returns)
    max_draw_down_returns = np.zeros(int(np.shape(weights)[1] - window + 1))
    for col in range(len(max_draw_down_returns)):
        max_draw_down_returns[col] = (np.min(portfolio_returns[col:int(col + window)]) -
                                      np.max(portfolio_returns[col:int(col + window)])) / np.max(
            portfolio_returns[col:int(col + window)])

    return max_draw_down_returns


def omega_ratio_return(weights, all_returns, window):
    """
    Calculate omega ratio of daily returns of portfolio.

    Parameters
    ----------
    weights : real ndarray
        Matrix containing weights.

    all_returns : real ndarray
        Matrix containing daily returns.

    window : positive integer
        Window over which measure is calculated.

    Returns
    -------
    variance_returns : real ndarray
        Vector containing omega ratio of portfolio daily returns using specified window.

    Notes
    -----

    """
    portfolio_returns = portfolio_return(weights, all_returns)
    omega_ratio_returns = np.zeros(int(np.shape(weights)[1] - window + 1))
    for col in range(len(omega_ratio_returns)):
        omega_ratio_returns[col] = np.mean(portfolio_returns[col:int(col + window)] *
                                           (portfolio_returns[col:int(col + window)] > 0)) / \
                                   np.mean(-portfolio_returns[col:int(col + window)] *
                                           (portfolio_returns[col:int(col + window)] <= 0))

    return omega_ratio_returns


def sortino_ratio_return(weights, all_returns, window, risk_free=(0.01/365)):
    """
    Calculate Sortino ratio of daily returns of portfolio.

    Parameters
    ----------
    weights : real ndarray
        Matrix containing weights.

    all_returns : real ndarray
        Matrix containing daily returns.

    window : positive integer
        Window over which measure is calculated.

    risk_free : float
        Risk-free rate assumed to be 0.01. This is converted to a daily exponential rate.

    Returns
    -------
    variance_returns : real ndarray
        Vector containing Sortino ratio of portfolio daily returns using specified window.

    Notes
    -----

    """
    portfolio_returns = portfolio_return(weights, all_returns)
    sortino_ratio_returns = np.zeros(int(np.shape(weights)[1] - window + 1))
    for col in range(len(sortino_ratio_returns)):
        sortino_ratio_returns[col] = (np.mean(portfolio_returns[col:int(col + window)]) - risk_free) / \
                                     np.std(portfolio_returns[col:int(col + window)] *
                                            (portfolio_returns[col:int(col + window)] < 0))

    return sortino_ratio_returns


def sharpe_ratio_return(weights, all_returns, window, risk_free=(0.01/365)):
    """
    Calculate Sharpe ratio of daily returns of portfolio.

    Parameters
    ----------
    weights : real ndarray
        Matrix containing weights.

    all_returns : real ndarray
        Matrix containing daily returns.

    window : positive integer
        Window over which measure is calculated.

    risk_free : float
        Risk-free rate assumed to be 0.01. This is converted to a daily exponential rate.

    Returns
    -------
    variance_returns : real ndarray
        Vector containing Sharpe ratio of portfolio daily returns using specified window.

    Notes
    -----

    """
    portfolio_returns = portfolio_return(weights, all_returns)
    sharpe_ratio_returns = np.zeros(int(np.shape(weights)[1] - window + 1))
    for col in range(len(sharpe_ratio_returns)):
        sharpe_ratio_returns[col] = (np.mean(portfolio_returns[col:int(col + window)]) - risk_free) / \
                                     np.std(portfolio_returns[col:int(col + window)])

    return sharpe_ratio_returns


def cumulative_return(weights, all_returns):
    """
    Calculate cumulative returns of daily returns of portfolio.

    Parameters
    ----------
    weights : real ndarray
        Matrix containing weights.

    all_returns : real ndarray
        Matrix containing daily returns.

    Returns
    -------
    variance_returns : real ndarray
        Vector containing cumulative returns of portfolio daily returns.

    Notes
    -----

    """
    portfolio_returns = portfolio_return(weights, all_returns)
    cumulative_returns = np.zeros(int(len(portfolio_returns) + 1))
    cumulative_returns[0] = 1
    for col in range(int(len(cumulative_returns) - 1)):
        cumulative_returns[int(col + 1)] = cumulative_returns[col] * np.exp(portfolio_returns[col])

    return cumulative_returns

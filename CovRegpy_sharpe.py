
#     ________
#            /
#      \    /
#       \  /
#        \/

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# risk budgeting approach
def sharpe_obj_fun(x, p_cov, returns, risk_free=(0.01 / 365)):
    """
    Maximum Sharpe ratio portfolio objective function for use in optimisation function.

    Parameters
    ----------
    x : real ndarray
        Weights of assets in portfolio.

    p_cov : real ndarray
        Covariance matrix of portfolio.

    returns : real ndarray
        Returns matrix of portfolio.

    risk_free : float
        Risk free rate assumed to be 1% - annualised.

    Returns
    -------
    obj : real ndarray
        Maximum Sharpe ratio portfolio objective function.

    Notes
    -----

    """
    obj = -(sum(x * returns) - risk_free) / np.sqrt(np.sum(x * np.dot(p_cov, x)))

    return obj


# equality constraint: = 0
def cons_sum_weight(x):
    """
    Constraint function - weights must sum to one.

    Parameters
    ----------
    x : real ndarray
        Weights of assets in portfolio.

    Returns
    -------
    weights_sum_zero : float
        Weights summed and one subtracted from them - constraint must be zero.

    Notes
    -----
    Equality constraint: = 0

    """
    weights_sum_zero = np.sum(x) - 1

    return weights_sum_zero


# inequality constraint: > 0
def cons_long_only_weight(x):
    """
    Constraint function - weights must all be non-negative.

    Parameters
    ----------
    x : real ndarray
        Weights of assets in portfolio.

    Returns
    -------
    x : real ndarray
        Weights must all be non-negative.

    Notes
    -----
    Inequality constraint: > 0

    """
    return x


# shorting constraint: > k
def cons_short_limit_weight(x, k):
    """
    Constraint function - weights must all be individually greater than -k.

    Parameters
    ----------
    x : real ndarray
        Weights of assets in portfolio.

    k : positive float
        Individual weights must each be greater than -k.

    Returns
    -------
    y : real ndarray
        Weights must all be greater than -k.

    Notes
    -----
    Inequality constraint: > -k.
    Require a summation restriction as with a large number of assets this can grow unreasonably large.

    """
    y = x + k

    return y  # some limit to shorting weight in portfolio


# shorting summation constraint: > k
def cons_short_limit_sum_weight(x, k):
    """
    Constraint function - combined shorting weights must be greater than -k.

    Parameters
    ----------
    x : real ndarray
        Weights of assets in portfolio.

    k : positive float
        Summed negative weights must be greater than -k.

    Returns
    -------
    y : real ndarray
        Summed negative weights must be greater than -k.

    Notes
    -----
    Inequality constraint: > -k.

    """
    y = np.sum(x[x < 0]) + k

    return y  # some limit to shorting weight in portfolio


# long summation constraint: > k
def cons_long_limit_sum_weight(x, k):
    """
    Constraint function - combined long weights must be less than k.

    Parameters
    ----------
    x : real ndarray
        Weights of assets in portfolio.

    k : positive float
        Summed positive weights must be lesser than k.

    Returns
    -------
    y : real ndarray
        Summed positive weights must be less than k.

    Notes
    -----
    Inequality constraint: < k.

    """
    y = np.sum(x[x > 0]) - k

    return y  # some limit to long weight in portfolio


# risk budgeting weighting
def sharpe_weights_long(cov, returns, risk_free=(0.01 / 365)):
    """
    Maximum Sharpe ratio with long restriction.

    Parameters
    ----------
    cov : real ndarray
        Covariance matrix of portfolio.

    returns : real ndarray
        Returns matrix of portfolio.

    risk_free : float
        Risk free rate assumed to be 1% - annualised.

    Returns
    -------
    OptimizeResult : OptimizeResult
        Optimised result and optimised weights.

    Notes
    -----
    Maximum Sharpe ratio portfolio with long restriction.
    Constrained optimisation - 'SLSQP' won't change if variance is too low - must change 'ftol' to smaller value.

    """
    w0 = np.ones((np.shape(cov)[0], 1)) / np.shape(cov)[0]
    cons = ({'type': 'eq', 'fun': cons_sum_weight}, {'type': 'ineq', 'fun': cons_long_only_weight})
    return minimize(sharpe_obj_fun, w0, args=(cov, returns, risk_free),
                    method='SLSQP', constraints=cons, options={'ftol': 1e-9})

# risk budgeting weighting
def sharpe_weights_individual_weight_restriction(cov, returns, risk_free=(0.01 / 365), short_limit=0.3):
    """
    Maximum Sharpe ratio with shorting restriction.

    Parameters
    ----------
    cov : real ndarray
        Covariance matrix of portfolio.

    returns : real ndarray
        Returns matrix of portfolio.

    risk_free : float
        Risk free rate assumed to be 1% - annualised.

    short_limit : float
        Individual weights must each be greater than short_limit.

    Returns
    -------
    OptimizeResult : OptimizeResult
        Optimised result and optimised weights.

    Notes
    -----
    Maximum Sharpe ratio portfolio with shorting restriction.
    Constrained optimisation - 'SLSQP' won't change if variance is too low - must change 'ftol' to smaller value.

    """
    w0 = np.ones((np.shape(cov)[0], 1)) / np.shape(cov)[0]
    cons = ({'type': 'eq', 'fun': cons_sum_weight},
            {'type': 'ineq', 'fun': cons_short_limit_weight, 'args': [short_limit]})
    return minimize(sharpe_obj_fun, w0, args=(cov, returns, risk_free),
                    method='SLSQP', constraints=cons, options={'ftol': 1e-9})


# risk budgeting weighting
def sharpe_weights_summation_weight_restriction(cov, returns, risk_free=(0.01 / 365), short_limit=0.3, long_limit=1.3):
    """
    Maximum Sharpe ratio with short and long summation restrictions.

    Parameters
    ----------
    cov : real ndarray
        Covariance matrix of portfolio.

    returns : real ndarray
        Returns matrix of portfolio.

    risk_free : float
        Risk free rate assumed to be 1% - annualised.

    short_limit : float
        Summation of negative weights must be greater than short_limit.

    long_limit : float
        Summation of positive weights must be less than long_limit.

    Returns
    -------
    OptimizeResult : OptimizeResult
        Optimised result and optimised weights.

    Notes
    -----
    Maximum Sharpe ratio portfolio with short and long summation restrictions.
    Constrained optimisation - 'SLSQP' won't change if variance is too low - must change 'ftol' to smaller value.

    """
    w0 = np.ones((np.shape(cov)[0], 1)) / np.shape(cov)[0]
    cons = ({'type': 'eq', 'fun': cons_sum_weight},
            {'type': 'ineq', 'fun': cons_short_limit_sum_weight, 'args': [short_limit]},
            {'type': 'ineq', 'fun': cons_long_limit_sum_weight, 'args': [long_limit]})
    return minimize(sharpe_obj_fun, w0, args=(cov, returns, risk_free),
                    method='SLSQP', constraints=cons, options={'ftol': 1e-9})


# global minimum weights
def sharpe_weights(cov, returns, risk_free=(0.01 / 365)):
    """
    Maximum Sharpe ratio portfolio direct calculation.

    Parameters
    ----------
    cov : real ndarray
        Covariance matrix of portfolio.

    returns : real ndarray
        Returns matrix of portfolio.

    risk_free : float
        Risk free rate assumed to be 1% - annualised.

    Returns
    -------
    weights : float
        Maximum Sharpe ratio portfolio weights.

    Notes
    -----
    Maximum Sharpe ratio portfolio direct calculation.

    """
    try:
        weights = (np.matmul(np.linalg.inv(cov),
                             (returns - risk_free * np.ones_like(returns)).reshape(-1, 1)) /
                   np.matmul(np.ones_like(returns).reshape(1, -1),
                             np.matmul(np.linalg.inv(cov),
                                       (returns - risk_free * np.ones_like(returns)).reshape(-1, 1))))[:, 0]

        return weights
    except:
        weights = (np.matmul(np.linalg.pinv(cov),
                             (returns - risk_free * np.ones_like(returns)).reshape(-1, 1)) /
                   np.matmul(np.ones_like(returns).reshape(1, -1),
                             np.matmul(np.linalg.pinv(cov),
                                       (returns - risk_free * np.ones_like(returns)).reshape(-1, 1))))[:, 0]

        return weights


if __name__ == "__main__":

    # pull all close data
    tickers_format = ['MSFT', 'AAPL']
    # tickers_format = ['MSFT', 'AAPL', 'GOOGL', 'AMZN', 'TSLA']
    data = yf.download(tickers_format, start="2018-10-15", end="2021-10-16")
    close_data = data['Close']
    del data, tickers_format

    # create date range and interpolate
    date_index = pd.date_range(start='16/10/2018', end='16/10/2021')
    close_data = close_data.reindex(date_index).interpolate()
    close_data = close_data[::-1].interpolate()
    close_data = close_data[::-1]
    del date_index

    # calculate returns and realised covariance
    returns = (np.log(np.asarray(close_data)[1:, :]) -
               np.log(np.asarray(close_data)[:-1, :]))
    realised_covariance = np.cov(returns.T)
    risk_free = (0.02 / 365)

    # no restrictions on global minimum variance
    weights = sharpe_weights(realised_covariance, returns[-1, :], risk_free)  # approx [0.25, 0.75]
    msft_weight = np.linspace(-1, 2, 3001)
    aapl_weight = 1 - msft_weight
    sharpe_ratio = np.zeros(3001)
    for i in range(3001):
        sharpe_ratio[i] = sharpe_obj_fun(np.asarray([msft_weight[i], aapl_weight[i]]),
                                         realised_covariance, returns[-1, :], risk_free)
    plt.title('Maximum Sharpe Ratio with No Restrictions (Positive Correlation)')
    plt.plot(msft_weight, sharpe_ratio)
    plt.xlabel('MSFT Weights')
    plt.ylabel('Sharpe Ratio')
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.scatter(weights[0], sharpe_obj_fun(weights, realised_covariance, returns[-1, :], risk_free), c='r')
    plt.show()

    # only long weights global minimum variance
    # change correlation to negative for demonstration
    negative_covariance = realised_covariance.copy()
    negative_covariance[0, -1] = -negative_covariance[0, -1]
    negative_covariance[-1, 0] = -negative_covariance[-1, 0]
    weights = sharpe_weights(negative_covariance, returns[-1, :], risk_free)  # approx [0.47, 0.53]
    msft_weight = np.linspace(0, 1, 1001)
    aapl_weight = 1 - msft_weight
    sharpe_ratio = np.zeros(1001)
    for i in range(1001):
        sharpe_ratio[i] = sharpe_obj_fun(np.asarray([msft_weight[i], aapl_weight[i]]),
                                         negative_covariance, returns[-1, :], risk_free)
    plt.title('Maximum Sharpe Ratio with No Restrictions (Negative Correlation)')
    plt.plot(msft_weight, sharpe_ratio)
    plt.xlabel('MSFT Weights')
    plt.ylabel('Sharpe Ratio')
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.scatter(weights[0], sharpe_obj_fun(weights, negative_covariance, returns[-1, :], risk_free), c='r')
    plt.show()
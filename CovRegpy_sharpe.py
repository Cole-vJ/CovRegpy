
#     ________
#            /
#      \    /
#       \  /
#        \/

import numpy as np
from scipy.optimize import minimize

from CovRegpy_RPP import (cons_sum_weight, cons_long_only_weight, cons_short_limit_weight, cons_short_limit_sum_weight,
                          cons_long_limit_sum_weight, cons_long_limit_weight)

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


def sharpe_weights_short_and_long_restrict(cov, returns, risk_free=(0.00 / 365), short_limit=10.0, long_limit=10.0):
    """
    Maximum Sharpe ratio weight calculation with constraints on individual weight longing and shorting.
    Function created and designed to prevent exceedingly large weights in optimisation pursuit.

    Parameters
    ----------
    cov : real ndarray
        Covariance matrix of portfolio.

    a : float
        Individual long limit.

    b : float
        Individual short limit.

    Returns
    -------
    OptimizeResult : OptimizeResult
        Optimised result and optimised weights.

    Notes
    -----
    Maximum Sharpe ratio portfolio weighting with long and short restriction on individual weights.
    Constrained optimisation - 'SLSQP' won't change if variance is too low - must change 'ftol' to smaller value.

    Recommended above direct calculation via inverse and pseudo inverse as inverting large matrices is unstable.

    """
    w0 = np.ones((np.shape(cov)[0], 1)) / np.shape(cov)[0]
    cons = ({'type': 'eq', 'fun': cons_sum_weight},
            {'type': 'ineq', 'fun': cons_long_limit_weight, 'args': [long_limit]},
            {'type': 'ineq', 'fun': cons_short_limit_weight, 'args': [short_limit]})
    return minimize(sharpe_obj_fun, w0, args=(cov, returns, risk_free),
                    method='SLSQP', constraints=cons, options={'ftol': 1e-30})

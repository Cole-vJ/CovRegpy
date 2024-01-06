
#     ________
#            /
#      \    /
#       \  /
#        \/

# Main reference: Qian (2005)
# E. Qian. 2005. Risk Parity Portfolios: Efficient Portfolios Through True Diversification.
# White paper. Panagora Asset Management, Boston, Massachusetts, USA.

import numpy as np
from scipy.optimize import minimize


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


def risk_parity_obj_fun(x, p_cov, rb):
    """
    Risk Parity or Risk Premia Parity objective function.

    Parameters
    ----------
    x : real ndarray
        Weights of assets in portfolio.

    p_cov : real ndarray
        Covariance matrix of portfolio.

    rb : real ndarray
        The target weighting vector. i.e. if [1/n, ..., 1/n] then risk is equally allocated.

    Returns
    -------
    risk_budget_obj : float
        Risk budgeting objective function value.

    Notes
    -----

    """

    if not isinstance(x, (type(np.asarray([[1.0, 2.0], [3.0, 4.0]])))):
        raise TypeError('\'x\' must be of type np.ndarray.')
    if np.array(x).dtype != np.array([[1., 1., 1., 1.]]).dtype:
        raise TypeError('\'x\' must only contain floats.')
    if not isinstance(p_cov, (type(np.asarray([[1.0, 2.0], [3.0, 4.0]])))):
        raise TypeError('\'p_cov\' must be of type np.ndarray.')
    if np.array(p_cov).dtype != np.array([[1., 1., 1., 1.]]).dtype:
        raise TypeError('\'p_cov\' must only contain floats.')
    if not np.all(np.linalg.eigvals(p_cov) > 0):
        raise ValueError('\'p_cov\' must be PSD.')
    if len(x) != np.shape(p_cov)[0] or len(x) != np.shape(p_cov)[1]:
        raise ValueError('\'x\' and \'p_cov\' are incompatible lengths.')
    if not isinstance(rb, (type(np.asarray([[1.0, 2.0], [3.0, 4.0]])))):
        raise TypeError('\'rb\' must be of type np.ndarray.')
    if np.array(rb).dtype != np.array([[1., 1., 1., 1.]]).dtype:
        raise TypeError('\'rb\' must only contain floats.')
    if len(x) != len(rb):
        raise ValueError('\'x\' and \'rb\' are incompatible lengths.')

    risk_budget_obj = np.sum((x * np.dot(p_cov, x) / np.dot(x.transpose(), np.dot(p_cov, x)) - rb) ** 2)

    return risk_budget_obj


def equal_risk_parity_weights_long_restriction(cov):
    """
    Risk Parity or Risk Premia Parity minimizer with long restriction.

    Parameters
    ----------
    cov : real ndarray
        Covariance matrix of portfolio.

    Returns
    -------
    OptimizeResult : OptimizeResult
        Optimised result and optimised weights.

    Notes
    -----
    Risk Premia Parity weighting with long restriction.
    Constrained optimisation - 'SLSQP' won't change if variance is too low - must change 'ftol' to smaller value.

    """

    if not isinstance(cov, (type(np.asarray([[1.0, 2.0], [3.0, 4.0]])))):
        raise TypeError('\'cov\' must be of type np.ndarray.')
    if np.array(cov).dtype != np.array([[1., 1., 1., 1.]]).dtype:
        raise TypeError('\'cov\' must only contain floats.')
    if not np.all(np.linalg.eigvals(cov) > 0):
        raise ValueError('\'cov\' must be PSD.')

    w0 = np.ones(np.shape(cov)[0]) / np.shape(cov)[0]
    cons = ({'type': 'eq', 'fun': cons_sum_weight},
            {'type': 'ineq', 'fun': cons_long_only_weight})
    return minimize(risk_parity_obj_fun, w0, args=(cov, (1 / np.shape(cov)[0]) * np.ones_like(w0)),
                    method='SLSQP', constraints=cons, options={'ftol': 1e-9})


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

    return y


def cons_long_limit_weight(x, k):
    """
    Constraint function - weights must all be individually less than k.

    Parameters
    ----------
    x : real ndarray
        Weights of assets in portfolio.

    k : positive float
        Individual weights must each be less than k.

    Returns
    -------
    y : real ndarray
        Weights must all be less than k.

    Notes
    -----
    Inequality constraint: < k.
    Require a summation restriction as with a large number of assets this can grow unreasonably large.

    """
    y = x - k

    return y


def equal_risk_parity_weights_short_restriction(cov, short_limit=1.0):
    """
    Risk Parity or Risk Premia Parity minimizer with individual shorting restriction.

    Parameters
    ----------
    cov : real ndarray
        Covariance matrix of portfolio.

    short_limit : positive float
        Individual asset shorting restriction.

    Returns
    -------
    OptimizeResult : OptimizeResult
        Optimised result and optimised weights.

    Notes
    -----
    Risk Premia Parity weighting with individual shorting restriction.
    Constrained optimisation - 'SLSQP' won't change if variance is too low - must change 'ftol' to smaller value.

    """

    if not isinstance(cov, (type(np.asarray([[1.0, 2.0], [3.0, 4.0]])))):
        raise TypeError('\'cov\' must be of type np.ndarray.')
    if np.array(cov).dtype != np.array([[1., 1., 1., 1.]]).dtype:
        raise TypeError('\'cov\' must only contain floats.')
    if not np.all(np.linalg.eigvals(cov) > 0):
        raise ValueError('\'cov\' must be PSD.')
    if (not isinstance(short_limit, float)) or short_limit <= 0:
        raise ValueError('\'short_limit\' must be a positive float.')

    w0 = np.ones(np.shape(cov)[0]) / np.shape(cov)[0]
    cons = ({'type': 'eq', 'fun': cons_sum_weight},
            {'type': 'ineq', 'fun': cons_short_limit_weight, 'args': [short_limit]})
    return minimize(risk_parity_obj_fun, w0, args=(cov, (1 / np.shape(cov)[0]) * np.ones_like(w0)),
                    method='SLSQP', constraints=cons, options={'ftol': 1e-9})


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

    return y


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

    return y


def equal_risk_parity_weights_summation_restriction(cov, short_limit=0.3, long_limit=1.3):
    """
    Risk Parity or Risk Premia Parity minimizer with shorting summation restriction.
    Base example is the classic 130/30 strategy or 130/30 Long/Short Equity example.

    Parameters
    ----------
    cov : real ndarray
        Covariance matrix of portfolio.

    short_limit : positive float
        Negative weights summation asset shorting restriction.

    long_limit : positive float
        Positive weights summation asset long restriction.

    Returns
    -------
    OptimizeResult : OptimizeResult
        Optimised result and optimised weights.

    Notes
    -----
    Risk Premia Parity weighting with long and short summation restriction.
    Constrained optimisation - 'SLSQP' won't change if variance is too low - must change 'ftol' to smaller value.

    """

    if not isinstance(cov, (type(np.asarray([[1.0, 2.0], [3.0, 4.0]])))):
        raise TypeError('\'cov\' must be of type np.ndarray.')
    if np.array(cov).dtype != np.array([[1., 1., 1., 1.]]).dtype:
        raise TypeError('\'cov\' must only contain floats.')
    if not np.all(np.linalg.eigvals(cov) > 0):
        raise ValueError('\'cov\' must be PSD.')
    if (not isinstance(short_limit, float)) or short_limit <= 0:
        raise ValueError('\'short_limit\' must be a positive float.')
    if (not isinstance(long_limit, float)) or long_limit <= 0:
        raise ValueError('\'long_limit\' must be a positive float.')

    w0 = np.ones(np.shape(cov)[0]) / np.shape(cov)[0]
    cons = ({'type': 'eq', 'fun': cons_sum_weight},
            {'type': 'ineq', 'fun': cons_short_limit_sum_weight, 'args': [short_limit]},
            {'type': 'ineq', 'fun': cons_short_limit_sum_weight, 'args': [long_limit]})
    return minimize(risk_parity_obj_fun, w0, args=(cov, (1 / np.shape(cov)[0]) * np.ones_like(w0)),
                    method='SLSQP', constraints=cons, options={'ftol': 1e-9})


# The below functions are for the global minimum variance portfolios.


def global_obj_fun(x, p_cov):
    """
    Global minimum variance objective function.

    Parameters
    ----------
    x : real ndarray
        Weights of assets in portfolio.

    p_cov : real ndarray
        Covariance matrix of portfolio.

    Returns
    -------
    obj : float
        Global minimum variance objective function value.

    Notes
    -----

    """

    if not isinstance(x, (type(np.asarray([[1.0, 2.0], [3.0, 4.0]])))):
        raise TypeError('\'x\' must be of type np.ndarray.')
    if np.array(x).dtype != np.array([[1., 1., 1., 1.]]).dtype:
        raise TypeError('\'x\' must only contain floats.')
    if not isinstance(p_cov, (type(np.asarray([[1.0, 2.0], [3.0, 4.0]])))):
        raise TypeError('\'p_cov\' must be of type np.ndarray.')
    if np.array(p_cov).dtype != np.array([[1., 1., 1., 1.]]).dtype:
        raise TypeError('\'p_cov\' must only contain floats.')
    if not np.all(np.linalg.eigvals(p_cov) > 0):
        raise ValueError('\'p_cov\' must be PSD.')
    if len(x) != np.shape(p_cov)[0] or len(x) != np.shape(p_cov)[1]:
        raise ValueError('\'x\' and \'p_cov\' are incompatible lengths.')

    obj = np.sum(x * np.dot(p_cov, x))

    return obj


def global_weights(cov):
    """
    Global minimum variance weights direct calculation.

    Parameters
    ----------
    cov : real ndarray
        Covariance matrix of portfolio.

    Returns
    -------
    weights : real ndarray
        Global minimum variance weights.

    Notes
    -----

    """

    if not isinstance(cov, (type(np.asarray([[1.0, 2.0], [3.0, 4.0]])))):
        raise TypeError('\'cov\' must be of type np.ndarray.')
    if np.array(cov).dtype != np.array([[1., 1., 1., 1.]]).dtype:
        raise TypeError('\'cov\' must only contain floats.')
    if not np.all(np.linalg.eigvals(cov) > 0):
        raise ValueError('\'cov\' must be PSD.')

    try:
        weights = (np.matmul(np.linalg.inv(cov), np.ones(np.shape(cov)[1]).reshape(-1, 1)) /
                   np.matmul(np.ones(np.shape(cov)[1]).reshape(1, -1),
                             np.matmul(np.linalg.inv(cov), np.ones(np.shape(cov)[1]).flatten())))[:, 0]

        return weights
    except:
        weights = (np.matmul(np.linalg.pinv(cov), np.ones(np.shape(cov)[1]).reshape(-1, 1)) /
                   np.matmul(np.ones(np.shape(cov)[1]).reshape(1, -1),
                             np.matmul(np.linalg.pinv(cov), np.ones(np.shape(cov)[1]).flatten())))[:, 0]

        return weights


def global_weights_long(cov):
    """
    Global minimum variance weights constrained calculation.

    Parameters
    ----------
    cov : real ndarray
        Covariance matrix of portfolio.

    Returns
    -------
    OptimizeResult : OptimizeResult
        Optimised result and optimised weights.

    Notes
    -----
    Global minimum variance portfolio weighting with long restriction.
    Constrained optimisation - 'SLSQP' won't change if variance is too low - must change 'ftol' to smaller value.

    """

    if not isinstance(cov, (type(np.asarray([[1.0, 2.0], [3.0, 4.0]])))):
        raise TypeError('\'cov\' must be of type np.ndarray.')
    if np.array(cov).dtype != np.array([[1., 1., 1., 1.]]).dtype:
        raise TypeError('\'cov\' must only contain floats.')
    if not np.all(np.linalg.eigvals(cov) > 0):
        raise ValueError('\'cov\' must be PSD.')

    w0 = np.ones(np.shape(cov)[0]) / np.shape(cov)[0]
    cons = ({'type': 'eq', 'fun': cons_sum_weight},
            {'type': 'ineq', 'fun': cons_long_only_weight})
    return minimize(global_obj_fun, w0, args=(cov),
                    method='SLSQP', constraints=cons, options={'ftol': 1e-9})


def global_weights_short_and_long_restrict(cov, a, b):
    """
    Global minimum variance weight calculation with constraints on individual weight longing and shorting..

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
    Global minimum variance portfolio weighting with long and short restriction on individual weights.
    Constrained optimisation - 'SLSQP' won't change if variance is too low - must change 'ftol' to smaller value.

    Recommended above direct calculation via inverse and pseudo inverse as inverting large matrices is unstable.

    """

    if not isinstance(cov, (type(np.asarray([[1.0, 2.0], [3.0, 4.0]])))):
        raise TypeError('\'cov\' must be of type np.ndarray.')
    if np.array(cov).dtype != np.array([[1., 1., 1., 1.]]).dtype:
        raise TypeError('\'cov\' must only contain floats.')
    if not np.all(np.linalg.eigvals(cov) > 0):
        raise ValueError('\'cov\' must be PSD.')
    if (not isinstance(b, float)) or b <= 0:
        raise ValueError('\'b\' must be a positive float.')
    if (not isinstance(a, float)) or a <= 0:
        raise ValueError('\'a\' must be a positive float.')

    w0 = np.ones(np.shape(cov)[0]) / np.shape(cov)[0]
    cons = ({'type': 'eq', 'fun': cons_sum_weight},
            {'type': 'ineq', 'fun': cons_long_limit_weight, 'args': [a]},
            {'type': 'ineq', 'fun': cons_short_limit_weight, 'args': [b]})
    return minimize(global_obj_fun, w0, args=(cov),method='SLSQP', constraints=cons, options={'ftol': 1e-30})


def equal_risk_parity_weights_individual_restriction(cov, short_limit=1.0, long_limit=1.0):
    """
    Risk Parity or Risk Premia Parity minimizer with long and shorting summation restriction.
    Function created and designed to prevent exceedingly large weights in optimisation pursuit.

    Parameters
    ----------
    cov : real ndarray
        Covariance matrix of portfolio.

    short_limit : positive float
        Negative weights summation asset shorting restriction.

    long_limit : positive float
        Positive weights summation asset long restriction.

    Returns
    -------
    OptimizeResult : OptimizeResult
        Optimised result and optimised weights.

    Notes
    -----
    Risk Premia Parity weighting with long and short summation restriction.
    Constrained optimisation - 'SLSQP' won't change if variance is too low - must change 'ftol' to smaller value.

    """

    if not isinstance(cov, (type(np.asarray([[1.0, 2.0], [3.0, 4.0]])))):
        raise TypeError('\'cov\' must be of type np.ndarray.')
    if np.array(cov).dtype != np.array([[1., 1., 1., 1.]]).dtype:
        raise TypeError('\'cov\' must only contain floats.')
    if not np.all(np.linalg.eigvals(cov) > 0):
        raise ValueError('\'cov\' must be PSD.')
    if (not isinstance(short_limit, float)) or short_limit <= 0:
        raise ValueError('\'short_limit\' must be a positive float.')
    if (not isinstance(long_limit, float)) or long_limit <= 0:
        raise ValueError('\'long_limit\' must be a positive float.')

    w0 = np.ones(np.shape(cov)[0]) / np.shape(cov)[0]
    cons = ({'type': 'eq', 'fun': cons_sum_weight},
            {'type': 'ineq', 'fun': cons_long_limit_weight, 'args': [long_limit]},
            {'type': 'ineq', 'fun': cons_short_limit_weight, 'args': [short_limit]})
    return minimize(risk_parity_obj_fun, w0, args=(cov, (1 / np.shape(cov)[0]) * np.ones_like(w0)),
                    method='SLSQP', constraints=cons, options={'ftol': 1e-9})


#     ________
#            /
#      \    /
#       \  /
#        \/

# Main reference: Qian (2005)
# E. Qian. 2005. Risk Parity Portfolios: Efficient Portfolios Through True Diversification.
# White paper. Panagora Asset Management, Boston, Massachusetts, USA.

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
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
    w0 = np.ones((np.shape(cov)[0], 1)) / np.shape(cov)[0]
    cons = ({'type': 'eq', 'fun': cons_sum_weight},
            {'type': 'ineq', 'fun': cons_long_only_weight})
    return minimize(risk_parity_obj_fun, w0, args=(cov, 1 / np.shape(cov)[0]),
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


def equal_risk_parity_weights_short_restriction(cov, short_limit=1):
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
    w0 = np.ones((np.shape(cov)[0], 1)) / np.shape(cov)[0]
    cons = ({'type': 'eq', 'fun': cons_sum_weight},
            {'type': 'ineq', 'fun': cons_short_limit_weight, 'args': [short_limit]})
    return minimize(risk_parity_obj_fun, w0, args=(cov, 1 / np.shape(cov)[0]),
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
    w0 = np.ones((np.shape(cov)[0], 1)) / np.shape(cov)[0]
    cons = ({'type': 'eq', 'fun': cons_sum_weight},
            {'type': 'ineq', 'fun': cons_short_limit_sum_weight, 'args': [short_limit]},
            {'type': 'ineq', 'fun': cons_short_limit_sum_weight, 'args': [long_limit]})
    return minimize(risk_parity_obj_fun, w0, args=(cov, 1 / np.shape(cov)[0]),
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
    w0 = np.ones((np.shape(cov)[0], 1)) / np.shape(cov)[0]
    cons = ({'type': 'eq', 'fun': cons_sum_weight},
            {'type': 'ineq', 'fun': cons_long_only_weight})
    return minimize(global_obj_fun, w0, args=(cov),
                    method='SLSQP', constraints=cons, options={'ftol': 1e-9})


if __name__ == "__main__":

    variance = np.zeros((2, 2))
    variance[0, 0] = 1  # 1
    variance[1, 1] = 16  # 1/4
    variance[0, 1] = variance[1, 0] = 0  # -1.8? -1.9?
    weights = equal_risk_parity_weights_long_restriction(variance).x
    # print(weights)
    volatility_contribution = \
        weights * np.dot(variance, weights) / np.dot(weights.transpose(), np.dot(variance, weights))
    # print(volatility_contribution)

    if all(np.isclose(volatility_contribution, np.asarray([1/2, 1/2]), atol=1e-3)):
        print('EQUAL contributions to volatility.')
    else:
        print('NON-EQUAL contributions to volatility.')

    variance = np.zeros((3, 3))
    variance[0, 0] = 1  # 1
    variance[1, 1] = 1  # 1
    variance[2, 2] = 4  # 1/2
    variance[0, 1] = variance[1, 0] = 1  # -2? 2?
    variance[0, 2] = variance[2, 0] = 0
    variance[2, 1] = variance[1, 2] = 0
    weights = equal_risk_parity_weights_long_restriction(variance).x
    # print(weights)
    volatility_contribution = \
        weights * np.dot(variance, weights) / np.dot(weights.transpose(), np.dot(variance, weights))
    # print(volatility_contribution)

    if all(np.isclose(volatility_contribution, np.asarray([1 / 3, 1 / 3, 1 / 3]), atol=1e-3)):
        print('EQUAL contributions to volatility.')
    else:
        print('NON-EQUAL contributions to volatility.')

    # load some test data

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

    # no restrictions on global minimum variance
    weights = global_weights(realised_covariance)  # approx [0.25, 0.75]
    msft_weight = np.linspace(-1, 2, 3001)
    aapl_weight = 1 - msft_weight
    global_variance = np.zeros(3001)
    for i in range(3001):
        global_variance[i] = global_obj_fun(np.asarray([msft_weight[i], aapl_weight[i]]), realised_covariance)
    plt.title('Global Minimum Variance with No Restrictions')
    plt.plot(msft_weight, global_variance)
    plt.xlabel('MSFT Weights')
    plt.ylabel('Total Variance')
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.scatter(weights[0], global_obj_fun(weights, realised_covariance), c='r')
    plt.show()

    # only long weights global minimum variance
    # change correlation to negative for demonstration
    negative_covariance = realised_covariance.copy()
    negative_covariance[0, -1] = -negative_covariance[0, -1]
    negative_covariance[-1, 0] = -negative_covariance[-1, 0]
    weights = global_weights_long(negative_covariance).x  #
    msft_weight = np.linspace(0, 1, 1001)
    aapl_weight = 1 - msft_weight
    global_variance = np.zeros(1001)
    for i in range(1001):
        global_variance[i] = global_obj_fun(np.asarray([msft_weight[i], aapl_weight[i]]), negative_covariance)
    plt.title('Global Minimum Variance with Long Restriction')
    plt.plot(msft_weight, global_variance)
    plt.xlabel('MSFT Weights')
    plt.ylabel('Total Variance')
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.scatter(weights[0], global_obj_fun(weights, negative_covariance), c='r')
    plt.show()

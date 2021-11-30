
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# risk budgeting approach
def obj_fun(x, p_cov, rb):
    return np.sum((x * np.dot(p_cov, x) / np.dot(x.transpose(), np.dot(p_cov, x)) - rb) ** 2)


# risk budgeting approach
def global_obj_fun(x, p_cov):
    return np.sum(x * np.dot(p_cov, x))


# equality constraint: = 0
def cons_sum_weight(x):
   return np.sum(x) - 1


# inequality constraint: > 0
def cons_long_only_weight(x):
   return x


# risk budgeting weighting
def rb_p_weights(cov):
    # constrained optimisation - 'SLSQP' won't change if variance is too low - much change 'ftol' to smaller value
    w0 = np.ones((np.shape(cov)[0], 1)) / np.shape(cov)[0]
    cons = ({'type': 'eq', 'fun': cons_sum_weight}, {'type': 'ineq', 'fun': cons_long_only_weight})
    return minimize(obj_fun, w0, args=(cov, 1 / np.shape(cov)[0]),
                    method='SLSQP', constraints=cons, options={'ftol': 1e-9})


# global minimum weights
def global_weights(cov):
    try:
        return (np.matmul(np.linalg.inv(cov), np.ones(np.shape(cov)[1]).reshape(-1, 1)) /
                np.matmul(np.ones(np.shape(cov)[1]).reshape(1, -1),
                          np.matmul(np.linalg.inv(cov), np.ones(np.shape(cov)[1]).flatten())))[:, 0]
    except:
        return (np.matmul(np.linalg.pinv(cov), np.ones(np.shape(cov)[1]).reshape(-1, 1)) /
                np.matmul(np.ones(np.shape(cov)[1]).reshape(1, -1),
                          np.matmul(np.linalg.pinv(cov), np.ones(np.shape(cov)[1]).flatten())))[:, 0]


# risk budgeting weighting
def global_weights_long(cov):
    # constrained optimisation - 'SLSQP' won't change if variance is too low - much change 'ftol' to smaller value
    w0 = np.ones((np.shape(cov)[0], 1)) / np.shape(cov)[0]
    cons = ({'type': 'eq', 'fun': cons_sum_weight}, {'type': 'ineq', 'fun': cons_long_only_weight})
    return minimize(global_obj_fun, w0, args=(cov),
                    method='SLSQP', constraints=cons, options={'ftol': 1e-9})


if __name__ == "__main__":

    variance = np.zeros((2, 2))
    variance[0, 0] = 1  # 1
    variance[1, 1] = 16  # 1/4
    variance[0, 1] = variance[1, 0] = 0  # -1.8? -1.9?
    weights = rb_p_weights(variance).x
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
    weights = rb_p_weights(variance).x
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

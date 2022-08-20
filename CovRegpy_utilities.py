
# formatted

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from CovRegpy_PCA import pca_func
from CovRegpy_sharpe import sharpe_weights, sharpe_weights_individual_weight_restriction, \
    sharpe_weights_summation_weight_restriction
from CovRegpy_RPP import global_obj_fun, global_weights, global_weights_long


def efficient_frontier(global_minimum_weights, global_minimum_returns, global_minimum_sd,
                       maximum_sharpe_weights, maximum_sharpe_returns, maximum_sharpe_sd, variance, n=101):

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

    global_minimum_weights = global_weights(variance)
    global_minimum_sd = np.sqrt(global_obj_fun(global_minimum_weights, variance))
    global_minimum_returns = sum(global_minimum_weights * returns)

    return global_minimum_weights, global_minimum_sd, global_minimum_returns


def global_minimum_forward_applied_information(weight_calculating_variance, forward_looking_variance,
                                               forward_looking_returns):
    # Weights calculated on realised covariance, applied to monthly covariance and returns looking forward

    global_minimum_weights = global_weights(weight_calculating_variance)
    global_minimum_sd = np.sqrt(global_obj_fun(global_minimum_weights, forward_looking_variance))
    global_minimum_returns = sum(global_minimum_weights * forward_looking_returns)

    return global_minimum_weights, global_minimum_sd, global_minimum_returns


def global_minimum_forward_applied_information_long(weight_calculating_variance, forward_looking_variance,
                                                    forward_looking_returns):
    # Weights calculated on realised covariance, applied to monthly covariance and returns looking forward

    global_minimum_weights = global_weights_long(weight_calculating_variance).x
    global_minimum_sd = np.sqrt(global_obj_fun(global_minimum_weights, forward_looking_variance))
    global_minimum_returns = sum(global_minimum_weights * forward_looking_returns)

    return global_minimum_weights, global_minimum_sd, global_minimum_returns


def sharpe_information(variance, returns, risk_free, global_minimum_weights, global_minimum_returns):

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
                                       forward_looking_variance, forward_looking_returns, risk_free,
                                       global_minimum_weights, global_minimum_returns):
    # Weights calculated on realised covariance, applied to monthly covariance and returns looking forward

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
                                                              risk_free, global_minimum_weights,
                                                              global_minimum_returns, short_limit=3):
    # Weights calculated on realised covariance, applied to monthly covariance and returns looking forward

    sharpe_maximum_weights = sharpe_weights_individual_weight_restriction(weight_calculating_variance, weight_calculating_returns, risk_free,
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
                                                             risk_free, global_minimum_weights, global_minimum_returns,
                                                             short_limit=0.3, long_limit=1.3):
    # Weights calculated on realised covariance, applied to monthly covariance and returns looking forward

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

    pca_weights = pca_func(variance, factors)
    pca_weights = pca_weights.sum(axis=1)
    pca_sd = np.sqrt(global_obj_fun(pca_weights, variance))
    pca_returns = sum(pca_weights * returns)

    return pca_weights, pca_sd, pca_returns


def pca_forward_applied_information(weight_calculating_variance, forward_looking_variance,
                                    forward_looking_returns, factors=3):
    # Weights calculated on realised covariance, applied to monthly covariance and returns looking forward

    pca_weights = pca_func(weight_calculating_variance, factors)
    pca_weights = pca_weights.sum(axis=1)
    pca_sd = np.sqrt(global_obj_fun(pca_weights, forward_looking_variance))
    pca_returns = sum(pca_weights * forward_looking_returns)

    return pca_weights, pca_sd, pca_returns


if __name__ == "__main__":

    # pull all close data
    tickers_format = ['MSFT', 'AAPL', 'GOOGL', 'AMZN', 'TSLA']
    data = yf.download(tickers_format, start="2018-12-31", end="2021-12-01")
    close_data = data['Close']
    del data, tickers_format

    # create date range and interpolate
    date_index = pd.date_range(start='31/12/2018', end='01/12/2021')
    close_data = close_data.reindex(date_index).interpolate()
    close_data = close_data[::-1].interpolate()
    close_data = close_data[::-1]
    del date_index

    # calculate returns and realised covariance
    returns = (np.log(np.asarray(close_data)[1:, :]) -
               np.log(np.asarray(close_data)[:-1, :]))
    realised_covariance = np.cov(returns.T)
    risk_free = (0.02 / 365)

    # global minimum variance
    global_minimum_weights, global_minimum_sd, global_minimum_returns = \
        global_minimum_information(realised_covariance, returns[-1, :])

    # sharpe maximum ratio
    sharpe_maximum_weights, sharpe_maximum_sd, sharpe_maximum_returns = \
        sharpe_information(realised_covariance, returns[-1, :], risk_free,
                           global_minimum_weights, global_minimum_returns)

    # efficient frontier
    efficient_frontier_sd, efficient_frontier_returns = \
        efficient_frontier(global_minimum_weights, global_minimum_returns, global_minimum_sd,
                           sharpe_maximum_weights, sharpe_maximum_returns, sharpe_maximum_sd,
                           realised_covariance, n=101)

    # principle component analysis
    pca_weights, pca_sd, pca_returns = pca_information(realised_covariance, returns[-1, :], factors=3)

    # plots
    plt.title('Efficient Frontier and Principle Portfolio')
    plt.scatter(global_minimum_sd, global_minimum_returns, label='global minimum variance')
    plt.scatter(sharpe_maximum_sd, sharpe_maximum_returns, label='maximum sharpe ratio')
    plt.plot(efficient_frontier_sd, efficient_frontier_returns, label='efficient frontier')
    plt.scatter(pca_sd, pca_returns, label='principle portfolio')
    plt.ylabel('Returns')
    plt.xlabel('Variance')
    plt.legend(loc='lower right')
    plt.show()
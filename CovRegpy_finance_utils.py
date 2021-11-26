
import numpy as np
from PCA_functions import pca_func
from Maximum_Sharpe_ratio_portfolio import sharpe_weights
from Portfolio_weighting_functions import global_obj_fun, global_weights


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


def pca_information(variance, returns, factors=3):

    pca_weights = pca_func(variance, factors)
    pca_weights = pca_weights.sum(axis=1)
    pca_sd = np.sqrt(global_obj_fun(pca_weights, variance))
    pca_returns = sum(pca_weights * returns)

    return pca_weights, pca_sd, pca_returns
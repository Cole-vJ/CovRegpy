
#     ________
#            /
#      \    /
#       \  /
#        \/

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.decomposition import PCA


def pca_func(cov, n_components):
    """
    Calculate Principle portfolio weights.

    Parameters
    ----------
    cov : real ndarray
        Covariance matrix of returns.

    n_components : positive integer
        Necessarily less than number of assets.

    Returns
    -------
    pca_weights : real ndarray
        Weights of portfolio with relevant number of components.

    Notes
    -----
    Classic principle portfolio construction.

    """
    pca = PCA(n_components=n_components)
    pca.fit(np.asarray(cov))
    pca_components = pca.components_
    pca_singular_values = pca.singular_values_
    weights = pca_components.T / pca_components.sum(axis=1)
    pca_weights = weights * (pca_singular_values / pca_singular_values.sum())

    return pca_weights


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

    pca_weights = pca_func(cov=realised_covariance, n_components=3)
    pca_weights = pca_weights.sum(axis=1)

    print(pca_weights)

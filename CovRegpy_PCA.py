
#     ________
#            /
#      \    /
#       \  /
#        \/

import numpy as np
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

    # for comparison and verification of formula
    # print(np.matmul(pca_components.T / np.matmul(np.ones((1, 5)), pca_components.T), (
    #             pca_singular_values.reshape(3, 1) / np.matmul(np.ones((1, 3)), pca_singular_values.reshape(3, 1)))))
    # print(np.matmul(pca_weights, np.ones((3, 1))))

    return pca_weights

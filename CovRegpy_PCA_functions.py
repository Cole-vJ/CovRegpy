
import numpy as np
from sklearn.decomposition import PCA

# https://towardsdatascience.com/stock-market-analytics-with-pca-d1c2318e3f0e


def pca_func(cov, n_components):

    pca = PCA(n_components=n_components)
    pca.fit(np.asarray(cov))
    pca_components = pca.components_
    pca_singular_values = pca.singular_values_
    weights = pca_components.T / pca_components.sum(axis=1)
    pca_weights = weights * (pca_singular_values / pca_singular_values.sum())

    return pca_weights

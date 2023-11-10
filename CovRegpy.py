
#     ________
#            /
#      \    /
#       \  /
#        \/

# Main reference: Hoff and Niu (2012)
# Hoff, P. and Niu, X., A Covariance Regression Model.
# Statistica Sinica, Institute of Statistical Science, 2012, 22(2), 729–753.

import textwrap
import numpy as np
import group_lasso
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDRegressor, LassoLars, Lars
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(0)

sns.set(style='darkgrid')


def calc_B_Psi(m, v, x, y, basis, A_est, technique, alpha, l1_ratio_or_reg, group_reg, max_iter, groups,
               test_lasso=False):
    """
    This follows the calculation at the bottom of page 10 and top of page 11 in Hoff and Niu (2012).

    Parameters
    ----------
    m : real ndarray
        Column vector of shape (n x 1) of means in random effects model with 'n' being number of observations.

    v : real ndarray
        Column vector of shape (n x 1) of variances in random effects model with 'n' being number of observations.

    x : real ndarray
        Matrix of shape (m x n) of covariates with 'm' being number of covariates
        and 'n' being number of observations.

    y : real ndarray
        Matrix of shape (p x n) of dependent variables with 'p' being number of dependent variables
        and 'n' being number of observations.

    basis : real ndarray
        Basis matrix used to estimate local mean - (A_est^T * basis) approximates local mean of y matrix.

    A_est : real ndarray
        Coefficient matrix used to estimate local mean - (A_est^T * basis) approximates local mean of y matrix.

    technique : string
        'direct' : Direct calculation method used in Hoff and Niu (2012).
            beta = [(x_tilda^T * x_tilda)^(-1)] * (x_tilda^T * y)

        'lasso' : Least Absolute Shrinkage and Selection Operator (LASSO) Regression.
            Minimize: (1 / (2 * n)) * ||y_tilda - x_tilda * beta||^2_2 +
                      alpha * ||beta||_1

        'ridge' :
            Minimize: ||y_tilda - x_tilda * beta||^2_2 + alpha * ||beta||^2_2
            Equivalent to: beta = [(x_tilda^T * x_tilda + alpha * I)^(-1)] * (x_tilda^T * y)

        'elastic-net' :
            Minimize: (1 / (2 * n)) * ||y_tilda - x_tilda * beta||^2_2 +
                      alpha * l1_ratio * ||beta||_1 + 0.5 * alpha * (1 - l1_ratio) * ||beta||^2_2

            l1_ratio = 1 equivalent to 'lasso'
            l1_ratio = 0 and alpha = 2 equivalent to 'ridge'

        'group-lasso' :
            With G being the grouping of the covariates the objective function is given below.
            Minimize: ||∑g∈G[X_g * beta_g] - y||^2_2 + alpha * ||w||_1 + lambda_group * ∑g∈G||beta_g||_2

        'sub-gradient' :
            Minimize: ||beta||_1
            subject to: x_tilda * beta^T = y
            iterate by: B_{k+1} = B_k - alpha_k(I_p - X^T * (X * X^T)^{-1} * X * sign(B_k))

    alpha : float
        Constant used in chosen regression to multiply onto weights.

    l1_ratio_or_reg : float
        Least Absolute Shrinkage and Selection Operator (LASSO) ratio for elastic-net regression and
        LASSO regulator for group LASSO regression.

    group_reg : float
        Group LASSO regulator for group LASSO regression.

    max_iter : positive integer
        Maximum number of iterations to perform in chosen regression.

    groups : real ndarray (integer ndarray)
        Groups to be used in 'group-lasso' regression.

    test_lasso : bool
        If True, then given 'alpha' value is disregarded at each iteration and an optimal 'alpha' is calculated.

    Returns
    -------
    B_est : real ndarray
        Coefficients for covariates explaining attributable covariance.

    Psi_est : real ndarray
        Base unattributable covariance present in model.

    Notes
    -----
    Group LASSO regression and Subgradient optimisation are experimental and need to be improved to stop possible
    breaking of correlation structure or nonsensical results.

    """
    x_tilda = np.vstack([m * x.T, (v ** (1 / 2)) * x.T])
    y_tilda = np.vstack(((y.T - np.matmul(A_est.T, basis).T), np.zeros_like(y.T)))

    if technique == 'direct':

        try:
            B_est = \
                np.matmul(y_tilda.T,
                          np.matmul(x_tilda, np.linalg.inv(np.matmul(x_tilda.T, x_tilda).astype(np.float64))))
        except:
            B_est = \
                np.matmul(y_tilda.T,
                          np.matmul(x_tilda, np.linalg.pinv(np.matmul(x_tilda.T, x_tilda).astype(np.float64))))

    elif technique == 'lasso':

        if test_lasso:
            parameters = {'alpha': 10 ** (np.linspace(-12, 0, 121))}
            reg_lasso = linear_model.Lasso()
            clf = GridSearchCV(reg_lasso, parameters)
            clf.fit(x_tilda, y_tilda)
            alpha = np.asarray(clf.cv_results_['param_alpha'])[clf.best_index_]

        reg_lasso = linear_model.MultiTaskLasso(alpha=alpha, fit_intercept=False, max_iter=max_iter)
        reg_lasso.fit(x_tilda, y_tilda)
        B_est = reg_lasso.coef_

    elif technique == 'ridge':

        try:
            B_est = \
                np.matmul(np.matmul(y_tilda.T, x_tilda),
                          np.linalg.inv(np.matmul(x_tilda.T, x_tilda).astype(np.float64) +
                                        alpha * np.identity(np.shape(x_tilda)[1])))
        except:
            reg_ridge = linear_model.Ridge(alpha=alpha, fit_intercept=False, max_iter=max_iter)
            reg_ridge.fit(x_tilda, y_tilda)
            B_est = reg_ridge.coef_

    elif technique == 'elastic-net':

        reg_elas_net = linear_model.ElasticNet(alpha=alpha, fit_intercept=False, l1_ratio=l1_ratio_or_reg,
                                               max_iter=max_iter)
        reg_elas_net.fit(x_tilda, y_tilda)
        B_est = reg_elas_net.coef_

    elif technique == 'group-lasso':

        ################################################################
        # possibly breaks correlation structure when doing column-wise #
        ################################################################

        # https://group-lasso.readthedocs.io/en/latest/
        # https://group-lasso.readthedocs.io/en/latest/auto_examples/index.html
        B_est = np.zeros((np.shape(y_tilda)[1], np.shape(x_tilda)[1]))
        for covariate in range(np.shape(y_tilda)[1]):
            reg_group_lasso = group_lasso.GroupLasso(groups=groups, old_regularisation=True, supress_warning=True,
                                                     fit_intercept=False, group_reg=group_reg, l1_reg=l1_ratio_or_reg)
            reg_group_lasso.fit(x_tilda, y_tilda[:, covariate].reshape(-1, 1))
            B_est[covariate, :] = reg_group_lasso.coef_[:, 0]
            print(B_est[covariate, :])

    elif technique == 'sub-gradient':

        ################################################################
        # possibly breaks correlation structure when doing column-wise #
        ################################################################

        # B_est = np.zeros((np.shape(y_tilda)[1], np.shape(x_tilda)[1]))
        # for covariate in range(np.shape(y_tilda)[1]):
        #     # https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/linear_model/_sgd_fast.pyx
        #     reg_sgd = SGDRegressor()
        #     reg_sgd.fit(x_tilda, y_tilda[:, covariate])
        #     B_est[covariate, :] = reg_sgd.coef_
        #     print(B_est[covariate, :])

        B_est = subgrad_opt(x_tilda, y_tilda, alpha=alpha, max_iter=max_iter)

    C_est = np.vstack((A_est, B_est.T))

    x_tilda_extend = np.hstack((np.vstack((basis.T, np.zeros_like(basis.T))), x_tilda))
    y_tilda_extend = np.vstack((y.T, np.zeros_like(y.T)))

    const = (y_tilda_extend - np.matmul(x_tilda_extend, C_est))
    try:
        Psi_est = np.matmul(const.T, const) / np.shape(x)[1]
    except:
        Psi_est = np.matmul(const.T, const) / len(x)

    return B_est.astype(np.float64), Psi_est.astype(np.float64)


def gamma_v_m_error(errors, x, Psi, B):
    """
    Function to calculate variance and mean for random error formulation which follows calculation
    at the bottom of page 9 in Hoff and Niu (2012).

    Parameters
    ----------
    errors : real ndarray
        Errors (variance) about given mean of dependent variables matrix.

    x : real ndarray
        Independent variable matrix.

    Psi : real ndarray
        Base unattributable covariance present in model.

    B : real ndarray
        Coefficients for covariates explaining attributable covariance.

    Returns
    -------
    m : real ndarray
        Mean of random error formulation.

    v : real ndarray
        Variance of random error formulation.

    Notes
    -----

    """
    try:
        const = np.matmul(np.linalg.solve(Psi.astype(np.float64), B.T.astype(np.float64)), x)
    except:
        try:
            const = np.matmul(np.linalg.lstsq(Psi.astype(np.float64), B.T.astype(np.float64), rcond=None)[0], x)
        except:
            const = np.matmul(np.linalg.lstsq(Psi.astype(np.float64).dot(Psi.astype(np.float64).T),
                                              Psi.astype(np.float64).dot(B.T.astype(np.float64)), rcond=None)[0], x)

    v = np.abs((1 + (x * np.matmul(B, const)).sum(0)) ** (-1))
    m = v * sum(errors * const)

    return m.astype(np.float64), v.astype(np.float64)


# define covariance regression function with mean given


def cov_reg_given_mean(A_est, basis, x, y, iterations=10, technique='direct', alpha=1.0, l1_ratio_or_reg=0.1,
                       group_reg=1e-6, max_iter=10000, groups=np.arange(76), test_lasso=False, LARS=False,
                       true_coefficients=np.zeros((5, 15))):
    """
    Calculate Psi and B matrices of covariance regression as in Hoff and Niu (2012) except that A_est and basis
    are now given as inputs allowing for customisable definition of "mean" or "trend".

    Parameters
    ----------
    A_est : real ndarray
        Matrix of coefficients corresponding to 'basis' to estimate mean of dependent variables.

    basis : real ndarray
        Matrix of basis functions corresponding to 'A_est' to estimate mean of dependent variables.

    x : real ndarray
        Matrix of independent variables.

    y : real ndarray
        Matrix of dependent variables.

    iterations : positive integer
        Number of iterations of the Covariance Regression algorithm.

    technique : string
        'direct' : Direct calculation method used in Hoff and Niu (2012).
            beta = [(x_tild^T * x_tilda)^(-1)] * (x_tilda^T * y)

        'lasso' : Least Absolute Shrinkage and Selection Operator (LASSO) Regression.
            Minimize: (1 / (2 * n)) * ||y_tilda - x_tilda * beta||^2_2 +
                      alpha * ||beta||_1

        'ridge' :
            Minimize: ||y_tilda - x_tilda * beta||^2_2 + alpha * ||beta||^2_2
            Equivalent to: beta = [(x_tild^T * x_tilda + alpha * I)^(-1)] * (x_tilda^T * y)

        'elastic-net' :
            Minimize: (1 / (2 * n)) * ||y_tilda - x_tilda * beta||^2_2 +
                      alpha * l1_ratio * ||beta||_1 + 0.5 * alpha * (1 - l1_ratio) * ||beta||^2_2

            l1_ratio = 1 equivalent to 'lasso'
            l1_ratio = 0 and alpha = 2 equivalent to 'ridge'

        'group-lasso' :
            With G being the grouping of the covariates the objective function is given below.
            Minimize: ||∑g∈G[X_g * beta_g] - y||^2_2 + alpha * ||w||_1 + lambda_group * ∑g∈G||beta_g||_2

        'sub-gradient' :
            Minimize: ||beta||_1
            subject to: x_tilda * beta^T = y
            iterate by: B_{k+1} = B_k - alpha_k(I_p - X^T * (X * X^T)^{-1} * X * sign(B_k))

    alpha : float
        Lambda value used to weight coefficients.

    l1_ratio_or_reg : float
        Ratio of l1 normalisation in elastic-net regression.

    group_reg : float
        Lambda weighting for group lasso regression.

    max_iter : positive integer
        Maximum number of iterations in regularised regression.

    groups : real ndarray (consisting of negative integers)
        Vector of groups to be used in group LASSO regression.

    test_lasso : boolean
        Whether to optimise alpha value in LASSO regression.

    Returns
    -------
    B_est : real ndarray
        Coefficients for covariates explaining attributable covariance.

    Psi_est : real ndarray
        Base unattributable covariance present in model.

    Notes
    -----

    """
    if LARS:
        # https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_lars.html#sphx-glr-auto-examples-linear-model-plot-lasso-lars-py
        reg = LassoLars(normalize=False, alpha=1e-06)
        # reg = Lars(normalize=False)
        reg.fit(X=x.T, y=y.T)
        coef_paths = reg.coef_path_
        for path in range(len(coef_paths)):
            for row in range(np.shape(coef_paths[path])[0]):
                xx = np.sum(np.abs(coef_paths[path]), axis=0)
                xx /= xx[-1]
                plt.plot(xx[:len(coef_paths[path][row, :])], coef_paths[path][row, :],
                         label=f'Structure: {int(row + 1)} coef: {int(true_coefficients[path, row])}')
                if np.abs(true_coefficients[path, row]) > 0:
                    plt.plot(xx[:len(coef_paths[path][row, :])], coef_paths[path][row, :], ':', Linewidth=3)
            plt.vlines(xx[:len(coef_paths[path][row, :])], min(coef_paths[path][:, -1]),
                       max(coef_paths[path][:, -1]), linestyle="dashed")
            plt.legend(loc='upper left', fontsize=6)
            plt.show()

    m = (np.random.normal(0, 1, np.shape(y)[1])).reshape(-1, 1)  # initialise m
    v = np.ones_like(m)  # initialise v

    mean = np.matmul(A_est.T, basis)

    for iter in range(iterations):

        B_est, Psi_est = calc_B_Psi(m=m, v=v, x=x, y=y, basis=basis, A_est=A_est, technique=technique,
                                    l1_ratio_or_reg=l1_ratio_or_reg, group_reg=group_reg, alpha=alpha,
                                    max_iter=max_iter, groups=groups, test_lasso=test_lasso)

        m, v = gamma_v_m_error(errors=(y - mean), x=x, Psi=Psi_est, B=B_est.T)
        m = m.reshape(-1, 1)
        v = v.reshape(-1, 1)

    B_est = B_est.T

    return B_est.astype(np.float64), Psi_est.astype(np.float64)


# sub-gradient optimisation


def subgrad_opt(x_tilda, y_tilda, max_iter, alpha=1e-12):
    """
    Subgradient optimisation of coefficients.

    Parameters
    ----------
    x_tilda : real ndarray
        Matrix of independent variables.

    y_tilda : real ndarray
        Matrix of dependent variables.

    max_iter : positive integer
        Maximum number of integers.

    alpha : float
        Scale to be used in square summable, but not summable Polyak step size.

    Returns
    -------
    B_est : real ndarray
        Coefficients for covariates explaining attributable covariance.

    Notes
    -----
    Convergence results do not apply if applied to skinny matrices.
    Starting point of algorithm can be changed and optimised.

    """

    # will not necessarily converge if not satisfied
    # if np.shape(x_tilda)[0] > np.shape(x_tilda)[1]:
    #     raise ValueError('Matrix cannot be skinny/thin.')

    # reg_ridge = linear_model.Ridge(alpha=alpha, fit_intercept=False)
    # reg_ridge.fit(x_tilda, y_tilda)
    # B_sol = reg_ridge.coef_.T

    B_sol = np.matmul(y_tilda.T, np.matmul(x_tilda, np.linalg.pinv(np.matmul(x_tilda.T, x_tilda).astype(np.float64)))).T
    B_k = B_sol.copy()

    f_star = sum(np.abs(B_sol))
    f_best = 1e12 * np.ones_like(f_star)

    k = 1

    while k < int(max_iter + 1):

        B_k_1 = B_k - (alpha / k) * np.matmul((np.identity(np.shape(x_tilda)[1]) -
                                             np.matmul(np.matmul(x_tilda.T,
                                                                 np.linalg.pinv(np.matmul(x_tilda, x_tilda.T))),
                                                       x_tilda)), np.sign(B_k))
        f_potential = sum(np.abs(B_k_1))

        for i in range(len(f_potential)):
            if f_potential[i] <= f_best[i]:
                f_best[i] = f_potential[i]
                B_k[:, i] = B_k_1[:, i]

        print(k)
        k += 1

    B_est = B_k.copy().T

    return B_est

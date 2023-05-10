
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


def b(knots, time, degree):
    """
    Recursive method for building basis functions - concise and effective.

    Parameters
    ----------
    knots : real ndarray
        Entire knot vector or subset of knot vector depending on level of recursion.
        Number of knots provided depends on degree of basis function i.e. degree = 3 -> len(knots) = 5

    time : real ndarray
        Time over which basis functions will be defined.

    degree : positive integer
        Degree of basis spline to be constructed.

    Returns
    -------
    output : real ndarray
        Single basis spline of degree: "degree".

    Notes
    -----
    Continually subsets knot vector by one increment until base case is reached.

    """
    if degree == 0:

        output = ((knots[0] <= time) & (time < knots[1])) * 1.0

        return output

    else:

        c1 = (time - knots[0] * np.ones_like(time)) / \
             (knots[-2] * np.ones_like(time) - knots[0] * np.ones_like(time)) * b(knots[0:-1], time, degree - 1)

        c2 = (knots[-1] * np.ones_like(time) - time) / \
             (knots[-1] * np.ones_like(time) - knots[1] * np.ones_like(time)) * b(knots[1:], time, degree - 1)

        output = c1 + c2

    return output


def cubic_b_spline(knots, time):
    """
    Returns a (len(knots) - 4) x len(time) array. Each row is an individual cubic basis.
    Matrix is sparse. Each column contains at most 4 non-zero values (only four bases overlap at any point).

    Parameters
    ----------
    knots : real ndarray
        Knot points to be used (not necessarily evenly spaced).

    time : real ndarray
        Time over which basis matrix will be defined.

    Returns
    -------
    matrix_c : real ndarray
        Each row of matrix contains an individual cubic basis spline.

    Notes
    -----
    A vector 'c' can be calculated such that with output of this function being array 'B' and a time series being 's'
    the objective function ||(B^T)c - s||^2 is minimized to yield coefficient vector 'c'.

    """
    num_c = len(knots) - 4  # cubic basis-spline -> 4 fewer coefficients than knots

    matrix_c = np.zeros((num_c, len(time)))  # each row is a single basis function

    for tau in range(num_c):  # watch inequalities

        temp_knots = knots[tau:(tau + 5)]  # select 5 knots applicable to current cubic spline

        matrix_c[tau, :] = b(temp_knots, time, 3)  # calls func b above

    return matrix_c


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


if __name__ == "__main__":

    dt = 0.025
    x = np.linspace(0, 10, 401)
    y = np.linspace(0, 10, 401)
    Z, Y = np.meshgrid(x, y)
    X = (10 * np.ones(np.shape(Z)) - (Z / 2) + (Z / 2) * np.cos((1 / 2) * Z / (5 / (2 * np.pi)))
         - (Y / 2) + (Y / 2) * np.cos((1 / 2) * Y / (5 / (2 * np.pi))) + np.random.normal(0, 0.1, np.shape(Z)) + 5) / 1.5
    X_new = X.copy()
    X_new[np.asarray(np.sqrt((Z - 10) ** 2 + Y ** 2) <= 7)] = np.nan

    fig = plt.figure()
    fig.set_size_inches(8, 6)
    ax = plt.axes(projection='3d')
    ax.view_init(30, -70)
    ax.set_title(r'L$_2$ Norm Wells')
    # require vmin and vmax as np.nan causes error of colour bar
    cov_plot_1 = ax.plot_surface(Z, Y, X_new, rstride=1, cstride=1, cmap='gist_rainbow', edgecolor='none',
                                 antialiased=False, shade=True, alpha=1, zorder=2, vmin=2.8, vmax=10)

    for i in np.arange(3, 8, 1):
        Z_radius = Z * np.asarray(np.sqrt(Z ** 2 + (Y - 10) ** 2) <= (i + dt)) * np.asarray(np.sqrt(Z ** 2 + (Y - 10) ** 2) >= (i - dt))
        Z_radius[Z_radius == 0] = np.nan
        Y_radius = Y * np.asarray(np.sqrt(Z ** 2 + (Y - 10) ** 2) <= (i + dt)) * np.asarray(np.sqrt(Z ** 2 + (Y - 10) ** 2) >= (i - dt))
        Y_radius[Y_radius == 0] = np.nan
        X_radius = 0.1 + X * np.asarray(np.sqrt(Z ** 2 + (Y - 10) ** 2) <= (i + dt)) * np.asarray(np.sqrt(Z ** 2 + (Y - 10) ** 2) >= (i - dt))
        X_radius[X_radius == 0] = np.nan
        cov_plot_2 = ax.plot_surface(Z_radius, Y_radius, X_radius, rstride=1, cstride=1,
                                     edgecolor='black', antialiased=False, shade=True, alpha=1, zorder=1)
    plt.plot(x, y, -10 * np.ones_like(y), 'k-', label='Ridge regressions')
    ax.scatter(0.2, 7, 7.2, s=100, label='ridge minimum', zorder=10, c='grey')
    ax.scatter(2.6, 7, 6.2, s=100, zorder=10, c='grey')
    ax.scatter(3.6, 7, 4.6, s=100, zorder=10, c='grey')
    ax.scatter(4.5, 7, 3.2, s=100, zorder=10, c='grey')
    ax.scatter(5.0, 7, 2.2, s=100, zorder=10, c='grey')
    ax.set_xticks(ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    ax.set_xticklabels(labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], fontsize=8, ha="left", rotation_mode="anchor")
    ax.set_xlabel(r'$|\beta_1|^2$', fontsize=8)
    ax.set_yticks(ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    ax.set_yticklabels(labels=[10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0], rotation=0, fontsize=8)
    ax.set_ylabel(r'$|\beta_2|^2$', fontsize=8)
    ax.set_zticks(ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    ax.set_zticklabels([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], fontsize=8)
    ax.set_zlabel(r'$||f(\beta_1, \beta_2; t) - g(t)||^2_2$', fontsize=8)
    cbar = plt.colorbar(cov_plot_1)
    box_0 = ax.get_position()
    ax.set_zlim(0, 10)
    ax.set_position([box_0.x0 - 0.05, box_0.y0, box_0.width, box_0.height])
    ax.legend(loc='best', fontsize=8)
    plt.savefig('aas_figures/norm_well_example.png')
    plt.show()

    # load raw data
    raw_data = pd.read_csv('Peter_Hoff_Data/peter_hoff_data', header=0)
    raw_data = np.asarray(raw_data)

    # prepare data
    peter_hoff_data = np.zeros((654, 3))

    for row in range(654):

        if row < 309:
            peter_hoff_data[row, 0] = int(raw_data[row, 0][2])
        else:
            peter_hoff_data[row, 0] = int(raw_data[row, 0][1:3])

        if peter_hoff_data[row, 0] == 3:  # original paper groups those aged 3 into age 4
            peter_hoff_data[row, 0] = 4
        elif peter_hoff_data[row, 0] == 19:  # original paper groups those aged 19 into age 18
            peter_hoff_data[row, 0] = 18
        peter_hoff_data[row, 1] = float(raw_data[row, 0][4:10])  # fev values always 6 text values
        peter_hoff_data[row, 2] = float(raw_data[row, 0][11:15])  # height values always 4 text values

    peter_hoff_data = pd.DataFrame(peter_hoff_data, columns=['age', 'fev', 'height'])

    # knots and time used in original paper
    spline_basis = cubic_b_spline(knots=np.linspace(-17, 39, 9), time=np.linspace(4, 18, 15))
    spline_basis = np.vstack((spline_basis, np.linspace(4, 18, 15)))

    age_vector = np.asarray(peter_hoff_data['age'])
    spline_basis_transform = np.zeros((6, 654))
    for col in range(len(age_vector)):
        spline_basis_transform[:, col] = spline_basis[:, int(age_vector[col] - 4)]

    coef_fev = np.linalg.lstsq(spline_basis_transform.transpose(), np.asarray(peter_hoff_data['fev']), rcond=None)
    coef_fev = coef_fev[0]
    mean_fev = np.matmul(coef_fev, spline_basis)

    coef_height = np.linalg.lstsq(spline_basis_transform.transpose(), np.asarray(peter_hoff_data['height']), rcond=None)
    coef_height = coef_height[0]
    mean_height = np.matmul(coef_height, spline_basis)

    x_cov = np.vstack((np.ones((1, 654)), (age_vector ** (1 / 2)).reshape(1, 654), age_vector.reshape(1, 654)))
    y = np.vstack((np.asarray(peter_hoff_data['fev']).reshape(1, 654),
                   np.asarray(peter_hoff_data['height']).reshape(1, 654)))
    # mean = np.vstack((np.matmul(coef_fev, spline_basis_transform), np.matmul(coef_height, spline_basis_transform)))
    A_est = np.hstack((coef_fev.reshape(6, 1), coef_height.reshape(6, 1)))
    B_est, Psi_est = cov_reg_given_mean(A_est=A_est, basis=spline_basis_transform, x=x_cov, y=y, iterations=100)

    mod_x_cov = np.vstack((np.ones((1, 15)),
                           (np.linspace(4, 18, 15) ** (1 / 2)).reshape(1, 15),
                           np.linspace(4, 18, 15).reshape(1, 15)))

    # mean and covariance plots

    cov_3d = np.zeros((2, 2, 15))
    for depth in range(np.shape(cov_3d)[2]):
        cov_3d[:, :, depth] = Psi_est + np.matmul(np.matmul(B_est.T, mod_x_cov[:, depth]).reshape(2, -1),
                                                  np.matmul(mod_x_cov[:, depth].T, B_est).reshape(-1, 2))

    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    fig.suptitle('Rank 1 Figure 5 in Hoff and Niu (2012)')
    axs[0].scatter(peter_hoff_data['age'], peter_hoff_data['fev'], facecolor='none', edgecolor='black')
    axs[0].plot(np.linspace(4, 18, 15), mean_fev, linewidth=3, c='k')
    axs[0].plot(np.linspace(4, 18, 15), mean_fev + 2 * np.sqrt(cov_3d[0, 0, :]), c='grey')
    axs[0].plot(np.linspace(4, 18, 15), mean_fev - 2 * np.sqrt(cov_3d[0, 0, :]), c='grey')
    axs[0].set_xlabel('age')
    axs[0].set_ylabel('FEV')
    axs[0].set_xticks([4, 6, 8, 10, 12, 14, 16, 18])
    axs[0].set_yticks([1, 2, 3, 4, 5, 6])
    axs[1].scatter(peter_hoff_data['age'], peter_hoff_data['height'], facecolor='none', edgecolor='black')
    axs[1].plot(np.linspace(4, 18, 15), mean_height, linewidth=3, c='k')
    axs[1].plot(np.linspace(4, 18, 15), mean_height + 2 * np.sqrt(cov_3d[1, 1, :]), c='grey')
    axs[1].plot(np.linspace(4, 18, 15), mean_height - 2 * np.sqrt(cov_3d[1, 1, :]), c='grey')
    axs[1].set_xlabel('age')
    axs[1].set_ylabel('height')
    axs[1].set_xticks([4, 6, 8, 10, 12, 14, 16, 18])
    axs[1].set_yticks([45, 50, 55, 60, 65, 70, 75])
    plt.savefig('aas_figures/Hoff_Figure_5')
    plt.show()

    fig, axs = plt.subplots(1, 3, figsize=(10, 6))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)
    fig.suptitle('Rank 1 Figure 6 in Hoff and Niu (2012)')
    axs[0].plot(np.linspace(4, 18, 15), cov_3d[0, 0, :], c='grey')
    fev_var = np.zeros_like(np.linspace(4, 18, 15))
    for i, age in enumerate(range(4, 19)):
        fev_var[i] = np.var(np.asarray(peter_hoff_data['fev'])[np.asarray(peter_hoff_data['age']) == age])
    axs[0].scatter(np.linspace(4, 18, 15), fev_var, facecolor='none', edgecolor='black')
    axs[0].set_xlabel('age')
    axs[0].set_ylabel('Var(FEV)')
    axs[0].set_xticks([4, 6, 8, 10, 12, 14, 16, 18])
    axs[0].set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    axs[1].plot(np.linspace(4, 18, 15), cov_3d[1, 1, :], c='grey')
    height_var = np.zeros_like(np.linspace(4, 18, 15))
    for i, age in enumerate(range(4, 19)):
        height_var[i] = np.var(np.asarray(peter_hoff_data['height'])[np.asarray(peter_hoff_data['age']) == age])
    axs[1].scatter(np.linspace(4, 18, 15), height_var, facecolor='none', edgecolor='black')
    axs[1].set_xlabel('age')
    axs[1].set_ylabel('Var(height)')
    axs[1].set_xticks([4, 6, 8, 10, 12, 14, 16, 18])
    axs[1].set_yticks([4, 6, 8, 10, 12])
    axs[2].plot(np.linspace(4, 18, 15), cov_3d[0, 1, :] / (np.sqrt(cov_3d[0, 0, :]) * np.sqrt(cov_3d[1, 1, :])), c='grey')
    fev_height_cov = np.zeros_like(np.linspace(4, 18, 15))
    for i, age in enumerate(range(4, 19)):
        fev_height_cov[i] = np.corrcoef(np.asarray(peter_hoff_data['fev'])[np.asarray(peter_hoff_data['age']) == age],
                                        np.asarray(peter_hoff_data['height'])[
                                            np.asarray(peter_hoff_data['age']) == age])[0, 1]
    axs[2].scatter(np.linspace(4, 18, 15), fev_height_cov, facecolor='none', edgecolor='black')
    axs[2].set_xlabel('age')
    axs[2].set_ylabel('Cor(FEV,height)')
    axs[2].set_xticks([4, 6, 8, 10, 12, 14, 16, 18])
    axs[2].set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
    plt.savefig('aas_figures/Hoff_Figure_6')
    plt.show()

    # additions - ridge, lasso

    A_est = np.hstack((coef_fev.reshape(6, 1), coef_height.reshape(6, 1)))
    B_est_ridge, Psi_est_ridge = cov_reg_given_mean(A_est=A_est, basis=spline_basis_transform, x=x_cov, y=y,
                                                    iterations=100, technique='ridge')
    B_est_lasso, Psi_est_lasso = cov_reg_given_mean(A_est=A_est, basis=spline_basis_transform, x=x_cov, y=y,
                                                    iterations=100, technique='lasso', alpha=0.05)
    B_est_net, Psi_est_net = cov_reg_given_mean(A_est=A_est, basis=spline_basis_transform, x=x_cov, y=y,
                                                iterations=100, technique='elastic-net', alpha=0.01,
                                                l1_ratio_or_reg=0.1)
    B_est_sub, Psi_est_sub = cov_reg_given_mean(A_est=A_est, basis=spline_basis_transform, x=x_cov, y=y,
                                                iterations=10, technique='sub-gradient', alpha=0.01, max_iter=10)
    B_est_group, Psi_est_group = cov_reg_given_mean(A_est=A_est, basis=spline_basis_transform, x=x_cov, y=y,
                                                    iterations=100, technique='group-lasso', alpha=0.01, group_reg=1e-9,
                                                    l1_ratio_or_reg=1e-6, groups=np.asarray([0, 1, 1]).reshape(-1, 1))

    # mean and covariance plots

    cov_3d_ridge = np.zeros((2, 2, 15))
    cov_3d_lasso = np.zeros((2, 2, 15))
    cov_3d_net = np.zeros((2, 2, 15))
    cov_3d_sub = np.zeros((2, 2, 15))
    cov_3d_group = np.zeros((2, 2, 15))
    for depth in range(np.shape(cov_3d)[2]):
        cov_3d_ridge[:, :, depth] = \
            Psi_est_ridge + np.matmul(np.matmul(B_est_ridge.T, mod_x_cov[:, depth]).reshape(2, -1),
                                      np.matmul(mod_x_cov[:, depth].T, B_est_ridge).reshape(-1, 2))
        cov_3d_lasso[:, :, depth] = \
            Psi_est_lasso + np.matmul(np.matmul(B_est_lasso.T, mod_x_cov[:, depth]).reshape(2, -1),
                                      np.matmul(mod_x_cov[:, depth].T, B_est_lasso).reshape(-1, 2))
        cov_3d_net[:, :, depth] = \
            Psi_est_net + np.matmul(np.matmul(B_est_net.T, mod_x_cov[:, depth]).reshape(2, -1),
                                    np.matmul(mod_x_cov[:, depth].T, B_est_net).reshape(-1, 2))
        cov_3d_sub[:, :, depth] = \
            Psi_est_sub + np.matmul(np.matmul(B_est_sub.T, mod_x_cov[:, depth]).reshape(2, -1),
                                    np.matmul(mod_x_cov[:, depth].T, B_est_sub).reshape(-1, 2))
        cov_3d_group[:, :, depth] = \
            Psi_est_group + np.matmul(np.matmul(B_est_group.T, mod_x_cov[:, depth]).reshape(2, -1),
                                      np.matmul(mod_x_cov[:, depth].T, B_est_group).reshape(-1, 2))

    fig, axs = plt.subplots(1, 2, figsize=(8, 5))
    fig.suptitle('Rank 1 Figure 5 in Hoff and Niu (2012)')
    axs[0].scatter(peter_hoff_data['age'], peter_hoff_data['fev'], facecolor='none', edgecolor='black')
    axs[0].plot(np.linspace(4, 18, 15), mean_fev, linewidth=3, c='k')
    axs[0].plot(np.linspace(4, 18, 15), mean_fev + 2 * np.sqrt(cov_3d[0, 0, :]), c='grey')
    axs[0].plot(np.linspace(4, 18, 15), mean_fev - 2 * np.sqrt(cov_3d[0, 0, :]), c='grey')
    axs[0].plot(np.linspace(4, 18, 15), mean_fev + 2 * np.sqrt(cov_3d_ridge[0, 0, :]), c='red')
    axs[0].plot(np.linspace(4, 18, 15), mean_fev - 2 * np.sqrt(cov_3d_ridge[0, 0, :]), c='red')
    axs[0].plot(np.linspace(4, 18, 15), mean_fev + 2 * np.sqrt(cov_3d_lasso[0, 0, :]), c='green')
    axs[0].plot(np.linspace(4, 18, 15), mean_fev - 2 * np.sqrt(cov_3d_lasso[0, 0, :]), c='green')
    axs[0].plot(np.linspace(4, 18, 15), mean_fev + 2 * np.sqrt(cov_3d_net[0, 0, :]), c='blue')
    axs[0].plot(np.linspace(4, 18, 15), mean_fev - 2 * np.sqrt(cov_3d_net[0, 0, :]), c='blue')
    axs[0].plot(np.linspace(4, 18, 15), mean_fev + 2 * np.sqrt(cov_3d_sub[0, 0, :]), c='cyan')
    axs[0].plot(np.linspace(4, 18, 15), mean_fev - 2 * np.sqrt(cov_3d_sub[0, 0, :]), c='cyan')
    axs[0].plot(np.linspace(4, 18, 15), mean_fev + 2 * np.sqrt(cov_3d_group[0, 0, :]), c='magenta')
    axs[0].plot(np.linspace(4, 18, 15), mean_fev - 2 * np.sqrt(cov_3d_group[0, 0, :]), c='magenta')
    axs[0].set_xlabel('age')
    axs[0].set_ylabel('FEV')
    axs[0].set_xticks([4, 6, 8, 10, 12, 14, 16, 18])
    axs[0].set_yticks([1, 2, 3, 4, 5, 6])
    box_0 = axs[0].get_position()
    axs[0].set_position([box_0.x0 - 0.06, box_0.y0, box_0.width, box_0.height])
    axs[1].scatter(peter_hoff_data['age'], peter_hoff_data['height'], facecolor='none', edgecolor='black')
    axs[1].plot(np.linspace(4, 18, 15), mean_height, linewidth=3, c='k')
    axs[1].plot(np.linspace(4, 18, 15), mean_height + 2 * np.sqrt(cov_3d[1, 1, :]), c='grey',
                label=textwrap.fill('Direct estimation', 11))
    axs[1].plot(np.linspace(4, 18, 15), mean_height - 2 * np.sqrt(cov_3d[1, 1, :]), c='grey')
    axs[1].plot(np.linspace(4, 18, 15), mean_height + 2 * np.sqrt(cov_3d_ridge[1, 1, :]), c='red',
                label=textwrap.fill('Ridge regression', 11))
    axs[1].plot(np.linspace(4, 18, 15), mean_height - 2 * np.sqrt(cov_3d_ridge[1, 1, :]), c='red')
    axs[1].plot(np.linspace(4, 18, 15), mean_height + 2 * np.sqrt(cov_3d_lasso[1, 1, :]), c='green',
                label=textwrap.fill('LASSO regression', 11))
    axs[1].plot(np.linspace(4, 18, 15), mean_height - 2 * np.sqrt(cov_3d_lasso[1, 1, :]), c='green')
    axs[1].plot(np.linspace(4, 18, 15), mean_height + 2 * np.sqrt(cov_3d_net[1, 1, :]), c='blue',
                label=textwrap.fill('Elastic-net regression', 11))
    axs[1].plot(np.linspace(4, 18, 15), mean_height - 2 * np.sqrt(cov_3d_net[1, 1, :]), c='blue')
    axs[1].plot(np.linspace(4, 18, 15), mean_height + 2 * np.sqrt(cov_3d_sub[1, 1, :]), c='cyan',
                label=textwrap.fill('Subgradient optimization', 12))
    axs[1].plot(np.linspace(4, 18, 15), mean_height - 2 * np.sqrt(cov_3d_sub[1, 1, :]), c='cyan')
    axs[1].plot(np.linspace(4, 18, 15), mean_height + 2 * np.sqrt(cov_3d_group[1, 1, :]), c='magenta',
                label=textwrap.fill('Group LASSO regression', 11))
    axs[1].plot(np.linspace(4, 18, 15), mean_height - 2 * np.sqrt(cov_3d_group[1, 1, :]), c='magenta')
    axs[1].set_xlabel('age')
    axs[1].set_ylabel('height')
    axs[1].set_xticks([4, 6, 8, 10, 12, 14, 16, 18])
    axs[1].set_yticks([45, 50, 55, 60, 65, 70, 75])
    box_1 = axs[1].get_position()
    axs[1].set_position([box_1.x0 - 0.06, box_1.y0, box_1.width, box_1.height])
    axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    plt.savefig('aas_figures/Hoff_Figure_5_RCR')
    plt.show()

    fig, axs = plt.subplots(1, 3, figsize=(8, 5))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)
    fig.suptitle('Rank 1 Figure 6 in Hoff and Niu (2012)')
    axs[0].plot(np.linspace(4, 18, 15), cov_3d[0, 0, :], c='grey')
    axs[0].plot(np.linspace(4, 18, 15), cov_3d_ridge[0, 0, :], c='red')
    axs[0].plot(np.linspace(4, 18, 15), cov_3d_lasso[0, 0, :], c='green')
    axs[0].plot(np.linspace(4, 18, 15), cov_3d_net[0, 0, :], c='blue')
    axs[0].plot(np.linspace(4, 18, 15), cov_3d_sub[0, 0, :], c='cyan')
    axs[0].plot(np.linspace(4, 18, 15), cov_3d_group[0, 0, :], c='magenta')
    fev_var = np.zeros_like(np.linspace(4, 18, 15))
    for i, age in enumerate(range(4, 19)):
        fev_var[i] = np.var(np.asarray(peter_hoff_data['fev'])[np.asarray(peter_hoff_data['age']) == age])
    axs[0].scatter(np.linspace(4, 18, 15), fev_var, facecolor='none', edgecolor='black')
    axs[0].set_xlabel('age', fontsize=8)
    axs[0].set_ylabel('Var(FEV)', fontsize=8)
    axs[0].set_xticks([4, 6, 8, 10, 12, 14, 16, 18])
    axs[0].set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    plt.setp(axs[0].get_xticklabels(), fontsize=8)
    plt.setp(axs[0].get_yticklabels(), fontsize=8)
    box_0 = axs[0].get_position()
    axs[0].set_position([box_0.x0 - 0.051, box_0.y0, box_0.width, box_0.height])
    axs[1].plot(np.linspace(4, 18, 15), cov_3d[1, 1, :], c='grey')
    axs[1].plot(np.linspace(4, 18, 15), cov_3d_ridge[1, 1, :], c='red')
    axs[1].plot(np.linspace(4, 18, 15), cov_3d_lasso[1, 1, :], c='green')
    axs[1].plot(np.linspace(4, 18, 15), cov_3d_net[1, 1, :], c='blue')
    axs[1].plot(np.linspace(4, 18, 15), cov_3d_sub[1, 1, :], c='cyan')
    axs[1].plot(np.linspace(4, 18, 15), cov_3d_group[1, 1, :], c='magenta')
    height_var = np.zeros_like(np.linspace(4, 18, 15))
    for i, age in enumerate(range(4, 19)):
        height_var[i] = np.var(np.asarray(peter_hoff_data['height'])[np.asarray(peter_hoff_data['age']) == age])
    axs[1].scatter(np.linspace(4, 18, 15), height_var, facecolor='none', edgecolor='black')
    axs[1].set_xlabel('age', fontsize=8)
    axs[1].set_ylabel('Var(height)', fontsize=8)
    axs[1].set_xticks([4, 6, 8, 10, 12, 14, 16, 18])
    axs[1].set_yticks([4, 6, 8, 10, 12])
    plt.setp(axs[1].get_xticklabels(), fontsize=8)
    plt.setp(axs[1].get_yticklabels(), fontsize=8)
    box_1 = axs[1].get_position()
    axs[1].set_position([box_1.x0 - 0.051, box_1.y0, box_1.width, box_1.height])
    axs[2].plot(np.linspace(4, 18, 15), cov_3d[0, 1, :] / (np.sqrt(cov_3d[0, 0, :]) * np.sqrt(cov_3d[1, 1, :])),
                c='grey', label=textwrap.fill('Direct estimation', 11))
    axs[2].plot(np.linspace(4, 18, 15), cov_3d_ridge[0, 1, :] / (np.sqrt(cov_3d_ridge[0, 0, :]) * np.sqrt(cov_3d_ridge[1, 1, :])),
                c='red', label=textwrap.fill('Ridge regression', 11))
    axs[2].plot(np.linspace(4, 18, 15),
                cov_3d_lasso[0, 1, :] / (np.sqrt(cov_3d_lasso[0, 0, :]) * np.sqrt(cov_3d_lasso[1, 1, :])),
                c='green', label=textwrap.fill('LASSO regression', 11))
    axs[2].plot(np.linspace(4, 18, 15),
                cov_3d_net[0, 1, :] / (np.sqrt(cov_3d_net[0, 0, :]) * np.sqrt(cov_3d_net[1, 1, :])),
                c='blue', label=textwrap.fill('Elastic-net regression', 11))
    axs[2].plot(np.linspace(4, 18, 15),
                cov_3d_sub[0, 1, :] / (np.sqrt(cov_3d_sub[0, 0, :]) * np.sqrt(cov_3d_sub[1, 1, :])),
                c='cyan', label=textwrap.fill('Subgradient optimization', 12))
    axs[2].plot(np.linspace(4, 18, 15),
                cov_3d_group[0, 1, :] / (np.sqrt(cov_3d_group[0, 0, :]) * np.sqrt(cov_3d_group[1, 1, :])),
                c='magenta', label=textwrap.fill('Group LASSO regression', 11))
    fev_height_cov = np.zeros_like(np.linspace(4, 18, 15))
    for i, age in enumerate(range(4, 19)):
        fev_height_cov[i] = np.corrcoef(np.asarray(peter_hoff_data['fev'])[np.asarray(peter_hoff_data['age']) == age],
                                        np.asarray(peter_hoff_data['height'])[
                                            np.asarray(peter_hoff_data['age']) == age])[0, 1]
    axs[2].scatter(np.linspace(4, 18, 15), fev_height_cov, facecolor='none', edgecolor='black')
    axs[2].set_xlabel('age', fontsize=8)
    axs[2].set_ylabel('Cor(FEV,height)', fontsize=8)
    axs[2].set_xticks([4, 6, 8, 10, 12, 14, 16, 18])
    axs[2].set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
    plt.setp(axs[2].get_xticklabels(), fontsize=8)
    plt.setp(axs[2].get_yticklabels(), fontsize=8)
    box_2 = axs[2].get_position()
    axs[2].set_position([box_2.x0 - 0.051, box_2.y0, box_2.width, box_2.height])
    axs[2].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    plt.savefig('aas_figures/Hoff_Figure_6_RCR')
    plt.show()

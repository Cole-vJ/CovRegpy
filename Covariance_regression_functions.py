
import numpy as np
import group_lasso

from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from scipy.optimize import linprog
from sklearn.linear_model import SGDRegressor

# calculate B and Psi (base variance)


def calc_B_Psi(m, v, x, y, basis, A_est, technique, alpha, max_iter, groups):

    # follows calculation at the bottom of page 10 and top of page 11
    x_tilda = np.vstack([m * x.T, (v ** (1 / 2)) * x.T])
    y_tilda = np.vstack(((y.T - np.matmul(A_est.T, basis).T), np.zeros_like(y.T)))

    if technique == 'direct':
        B_est = np.matmul(y_tilda.T, np.matmul(x_tilda, np.linalg.inv(np.matmul(x_tilda.T, x_tilda).astype(np.float64))))
    elif technique == 'lasso':
        reg_lasso = linear_model.MultiTaskLasso(alpha=alpha, fit_intercept=False, max_iter=max_iter)
        reg_lasso.fit(x_tilda, y_tilda)
        B_est = reg_lasso.coef_
    elif technique == 'group-lasso':
        # need to fix and finalise
        # https://group-lasso.readthedocs.io/en/latest/
        # https://group-lasso.readthedocs.io/en/latest/auto_examples/index.html
        reg_group_lasso = group_lasso.GroupLasso(groups=groups)
        reg_group_lasso.fit(x_tilda, y_tilda)
        B_est = reg_group_lasso.coef_
    elif technique == 'ridge':
        reg_ridge = linear_model.Ridge(alpha=alpha, fit_intercept=False, max_iter=max_iter)
        reg_ridge.fit(x_tilda, y_tilda)
        B_est = reg_ridge.coef_
    elif technique == 'elastic-net':
        # l1_ratio = 0 --> l2 penalty only --> linear_model.Ridge(alpha=alpha, fit_intercept=False)
        # l1_ratio = 1 --> l1 penalty only --> linear_model.MultiTaskLasso(alpha=alpha, fit_intercept=False)
        reg_elas_net = linear_model.ElasticNet(alpha=alpha, fit_intercept=False, l1_ratio=0.5, max_iter=max_iter)
        reg_elas_net.fit(x_tilda, y_tilda)
        B_est = reg_elas_net.coef_
    elif technique == 'sub-gradient':
        # need to fix and finalise
        reg_sgd = SGDRegressor()
        reg_sgd.fit(x_tilda, y_tilda[:, 0])
        B_est = reg_sgd.coef_
        # B_est = subgrad_opt(x_tilda, y_tilda, alpha=alpha, max_iter=max_iter)

    C_est = np.vstack((A_est, B_est.T))

    x_tilda_extend = np.hstack((np.vstack((basis.T, np.zeros_like(basis.T))), x_tilda))
    y_tilda_extend = np.vstack((y.T, np.zeros_like(y.T)))

    const = (y_tilda_extend - np.matmul(x_tilda_extend, C_est))
    Psi_est = np.matmul(const.T, const) / np.shape(x)[1]

    return B_est.astype(np.float64), Psi_est.astype(np.float64)


# calculate variance and mean of gamma


def gamma_v_m_error(errors, x, Psi, B):

    # follows calculation at the bottom of page 9 of paper
    const = np.matmul(np.linalg.solve(Psi.astype(np.float64), B.T.astype(np.float64)), x)
    v = (1 + (x * np.matmul(B, const)).sum(0)) ** (-1)
    m = v * sum(errors * const)

    return m.astype(np.float64), v.astype(np.float64)


# define covariance regression function with mean given


def cov_reg_given_mean(A_est, basis, x, y, iterations=10, technique='direct', alpha=1, max_iter=10000,
                       groups=np.arange(76)):

    m = (np.random.normal(0, 1, np.shape(y)[1])).reshape(-1, 1)  # initialise m
    v = np.ones_like(m)  # initialise v

    mean = np.matmul(A_est.T, basis)

    for iter in range(iterations):
        print(iter + 1)

        B_est, Psi_est = calc_B_Psi(m=m, v=v, x=x, y=y, basis=basis, A_est=A_est, technique=technique,
                                    alpha=alpha, max_iter=max_iter, groups=groups)

        m, v = gamma_v_m_error(errors=(y - mean), x=x, Psi=Psi_est, B=B_est.T)
        m = m.reshape(-1, 1)
        v = v.reshape(-1, 1)

    B_est = B_est.T

    return B_est.astype(np.float64), Psi_est.astype(np.float64)


# sub-gradient optimisation


def subgrad_opt(x_tilda, y_tilda, alpha, max_iter):

    # will not converge otherwise
    if np.shape(x_tilda)[0] > np.shape(x_tilda)[1]:
        raise ValueError('Matrix cannot be skinny/thin.')

    # reg_ridge = linear_model.Ridge(alpha=alpha, fit_intercept=False)
    # reg_ridge.fit(x_tilda, y_tilda)
    # B_sol = reg_ridge.coef_.T
    B_sol = np.matmul(y_tilda.T, np.matmul(x_tilda, np.linalg.inv(np.matmul(x_tilda.T, x_tilda).astype(np.float64)))).T
    B_k = B_sol.copy()

    f_star = sum(np.abs(B_sol))
    f_best = 1e12 * np.ones_like(f_star)

    k = 1

    while k < int(max_iter + 1):

        B_k_1 = B_k - (1e-12 / k) * np.matmul((np.identity(np.shape(x_tilda)[1]) -
                                             np.matmul(np.matmul(x_tilda.T,
                                                                 np.linalg.inv(np.matmul(x_tilda, x_tilda.T))),
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

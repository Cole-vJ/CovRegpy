
import numpy as np
import group_lasso
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import SGDRegressor, LassoLars, Lars

sns.set(style='darkgrid')

# recursive basis construction


def b(knots, time, degree):

    if degree == 0:

        return ((knots[0] <= time) & (time < knots[1])) * 1.0

    else:

        c1 = (time - knots[0] * np.ones_like(time)) / \
             (knots[-2] * np.ones_like(time) - knots[0] * np.ones_like(time)) * b(knots[0:-1], time, degree - 1)

        c2 = (knots[-1] * np.ones_like(time) - time) / \
             (knots[-1] * np.ones_like(time) - knots[1] * np.ones_like(time)) * b(knots[1:], time, degree - 1)

    return c1 + c2


# cubic basis splines as function of knots points


def cubic_b_spline(knots, time):

    num_c = len(knots) - 4  # cubic basis-spline -> 4 fewer coefficients than knots

    matrix_c = np.zeros((num_c, len(time)))  # each row is a single basis function

    for tau in range(num_c):  # watch inequalities

        temp_knots = knots[tau:(tau + 5)]  # select 5 knots applicable to current cubic spline

        matrix_c[tau, :] = b(temp_knots, time, 3)

    return matrix_c


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
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html
        # Minimize:
        # 1 / (2 * n_samples) * ||y - Xw||^2_2 + alpha * l1_ratio * ||w||_1 + 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2
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
    try:
        const = np.matmul(np.linalg.solve(Psi.astype(np.float64), B.T.astype(np.float64)), x)
    except:
        const = np.matmul(np.linalg.lstsq(Psi.astype(np.float64), B.T.astype(np.float64), rcond=None)[0], x)

        # # pseudo-inverse solution approximation
        # const = np.matmul(np.linalg.solve(Psi.astype(np.float64).dot(Psi.astype(np.float64).T),
        #                                   Psi.astype(np.float64).dot(B.T.astype(np.float64))), x)

    v = (1 + (x * np.matmul(B, const)).sum(0)) ** (-1)
    m = v * sum(errors * const)

    return m.astype(np.float64), v.astype(np.float64)


# define covariance regression function with mean given


def cov_reg_given_mean(A_est, basis, x, y, iterations=10, technique='direct', alpha=1, max_iter=10000,
                       groups=np.arange(76), LARS=False, true_coefficients=np.zeros((5, 15))):

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


if __name__ == "__main__":

    # load raw data
    raw_data = pd.read_csv('data/peter_hoff_data', header=0)
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

    fig, axs = plt.subplots(1, 2, figsize=(8, 5))
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
    plt.show()

    fig, axs = plt.subplots(1, 3, figsize=(8, 5))
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
    plt.show()

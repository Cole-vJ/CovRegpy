
#     ________
#            /
#      \    /
#       \  /
#        \/

# Document Strings Publication

# Reference work:

# Bauwens, L., Laurent, S. and Rombouts, J.,
# Multivariate GARCH Models: A Survey. Journal of Applied
# Econometrics, 2006, 21, 79–109.

# Log-likelihood reference work - equations referenced throughout:

# Engle, R., Dynamic Conditional Correlation: A Simple Class of
# Multivariate Generalized Autoregressive Conditional Heteroskedasticity Models.
# Journal of Business & Economic Statistics, 2002, 20, 339–350.

# This package created as the following Python package did not work for our purposes:
# mgarch 0.2.0 - available at https://pypi.org/project/mgarch/

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.optimize import minimize


def covregpy_dcc(returns_matrix, p=3, q=3, days=10, print_correlation=False, rescale=False):
    """
    Dynamic Conditional Correlation - Multivariate Generalized Autoregressive Conditional Heteroskedasticity model.

    Parameters
    ----------
    returns_matrix : real ndarray
        Matrix of asset returns.

    p : positive integer
        GARCH(p, q) model 'p' value used in calculation of uni-variate variance used in model.

    q : positive integer
        GARCH(p, q) model 'q' value used in calculation of uni-variate variance used in model.

    days : positive integer
        Number of days ahead to forecast covariance.

    print_correlation : boolean
        Debugging to gauge if correlation structure is reasonable.

    rescale : boolean
        Automatically rescale data to help if encountering convergence problems - see:
        https://arch.readthedocs.io/en/latest/univariate/introduction.html

    Returns
    -------
    forecasted_covariance : real ndarray
        Forecasted covariance.

    Notes
    -----

    """
    if not isinstance(returns_matrix, (type(np.asarray([[1.0, 2.0], [3.0, 4.0]])),
                                       type(pd.DataFrame(np.asarray([[1.0, 2.0], [3.0, 4.0]]))))):
        raise TypeError('Returns must be of type np.ndarray and pd.Dataframe.')
    if pd.isnull(np.asarray(returns_matrix)).any():
        raise TypeError('Returns must not contain nans.')
    if np.array(returns_matrix).dtype != np.array([[1., 1.], [1., 1.]]).dtype:
        raise TypeError('Returns must only contain floats.')
    if (not isinstance(p, int)) or (p <= 0):
        raise ValueError('\'p\' must be a positive integer.')
    if (not isinstance(q, int)) or (q <= 0):
        raise ValueError('\'q\' must be a positive integer.')
    if (not isinstance(days, int)) or (days <= 0):
        raise ValueError('\'days\' must be a positive integer.')
    if not isinstance(print_correlation, bool):
        raise TypeError('\'print_correlation\' must be boolean.')
    if not isinstance(rescale, bool):
        raise TypeError('\'rescale\' must be boolean.')

    # convert Dataframe
    returns_matrix = np.asarray(returns_matrix)

    # flip matrix to be consistent
    if np.shape(returns_matrix)[0] < np.shape(returns_matrix)[1]:
        returns_matrix = returns_matrix.T

    # initialised modelled variance matrix
    modelled_variance = np.zeros_like(returns_matrix)

    # iteratively calculate modelled variance using univariate GARCH model
    for stock in range(np.shape(returns_matrix)[1]):
        model = arch_model(returns_matrix[:, stock], mean='Zero', vol='GARCH', p=p, q=q, rescale=rescale)
        model_fit = model.fit(disp="off")
        # if rescale:
        #     modelled_variance[:, stock] = model_fit.conditional_volatility / model.scale
        # else:
        #     modelled_variance[:, stock] = model_fit.conditional_volatility
        modelled_variance[:, stock] = model_fit.conditional_volatility

    # optimise alpha & beta parameters to be used in page 90 equation (40)
    params = minimize(dcc_loglike, (0.01, 0.94),
                      args=(returns_matrix, modelled_variance),
                      bounds=((1e-6, 1), (1e-6, 1)),
                      method='Nelder-Mead')

    # list of optimisation methods available

    # Nelder-Mead, Powell, CG, BFGS, Newton-CG,
    # L-BFGS-B, TNC, COBYLA, SLSQP, trust-constr,
    # dogleg, trust-ncg, trust-exact, trust-krylov

    a = params.x[0]
    b = params.x[1]

    t = np.shape(returns_matrix)[0]  # time interval
    q_bar = np.cov(returns_matrix.T)  # base (unconditional) covariance

    # setup matrices
    q_t = np.zeros((t, np.shape(q_bar)[0], np.shape(q_bar)[1]))  # page 90 - Equation (40)
    h_t = np.zeros((t, np.shape(q_bar)[0], np.shape(q_bar)[1]))  # page 89 - Equation (35)
    dts = np.zeros((t, np.shape(q_bar)[0], np.shape(q_bar)[1]))  # page 88 - Equation (32)
    dts_inv = np.zeros((t, np.shape(q_bar)[0], np.shape(q_bar)[1]))  # used in calculation of u_t
    u_t = np.zeros((np.shape(q_bar)[0], t))  # page 89 - defined between Equation (37) and (38)
    qts = np.zeros((t, np.shape(q_bar)[0], np.shape(q_bar)[1]))  # page 90 - defined within Equation (39)
    r_t = np.zeros((t, np.shape(q_bar)[0], np.shape(q_bar)[1]))  # page 90 - Equation (39)

    # initialise q_t
    # unsure of initialisation
    q_t[0] = np.zeros_like(np.matmul(returns_matrix[0, :].reshape(-1, 1) / 2, returns_matrix[0, :].reshape(1, -1) / 2))

    for var in range(t):
        # page 88 - Equation (32)
        dts[var] = np.diag(modelled_variance[int(var - 1), :])  # modelled variance - page 89 - Equation (34)
        # page 89 - defined between Equation (37) and (38)
        try:
            dts_inv[var] = np.linalg.inv(dts[var])
        except:
            dts_inv[var] = np.linalg.pinv(dts[var])
        u_t[:, var] = np.matmul(dts_inv[var], returns_matrix[var, :].reshape(-1, 1))[:, 0]

    for var in range(1, t):

        # page 90 - Equation (40)
        q_t[var] = (1 - a - b) * q_bar + \
                   a * np.matmul(u_t[:, int(var - 1)].reshape(-1, 1), u_t[:, int(var - 1)].reshape(1, -1)) + \
                   b * q_t[int(var - 1)]

        # page 90 - defined within Equation (39)
        try:
            qts[var] = np.linalg.inv(np.sqrt(np.diag(np.diag(q_t[var]))))
        except:
            qts[var] = np.linalg.pinv(np.sqrt(np.diag(np.diag(q_t[var]))))

        # page 90 - Equation (39)
        r_t[var] = np.matmul(qts[var], np.matmul(q_t[var], qts[var]))

        # page 89 - Equation (35)
        h_t[var] = np.matmul(dts[var], np.matmul(r_t[var], dts[var]))

    # Brownian uncertainty with  variance increasing linearly with square root of time
    forecasted_covariance = h_t[-1] * np.sqrt(days)

    if print_correlation:
        corr = np.zeros_like(forecasted_covariance)
        for i in range(np.shape(corr)[0]):
            for j in range(np.shape(corr)[1]):
                corr[i, j] = forecasted_covariance[i, j] / (np.sqrt(forecasted_covariance[i, i]) *
                                                            np.sqrt(forecasted_covariance[j, j]))
        print(corr)

    return forecasted_covariance


def dcc_loglike(params, returns_matrix, modelled_variance):
    """
    Dynamic Conditional Correlation loglikelihood function to optimise.

    Parameters
    ----------
    params : real ndarray
        Real array of shape (2, ) with two parameters to optimise.

    returns_matrix : real ndarray
        Matrix of asset returns.

    modelled_variance : real ndarray
        Uni-variate modelled variance to be used in optimisation.

    Returns
    -------
    loglike : float
        Loglikelihood function to be minimised.

    Notes
    -----

    """
    if pd.isnull(np.asarray(params)).any():
        raise ValueError('Parameters must not contain nans.')
    if np.array(params).dtype != np.array([[1., 1.], [1., 1.]]).dtype:
        raise ValueError('Parameters must only contain floats.')
    if pd.isnull(np.asarray(returns_matrix)).any():
        raise ValueError('Returns must not contain nans.')
    if np.array(returns_matrix).dtype != np.array([[1., 1.], [1., 1.]]).dtype:
        raise ValueError('Returns must only contain floats.')
    if pd.isnull(np.asarray(modelled_variance)).any():
        raise ValueError('Covariance must not contain nans.')
    if np.array(modelled_variance).dtype != np.array([[1., 1.], [1., 1.]]).dtype:
        raise ValueError('Covariance must only contain floats.')

    t = np.shape(returns_matrix)[0]  # time interval
    q_bar = np.cov(returns_matrix.T)  # base (unconditional) covariance

    # setup matrices
    q_t = np.zeros((t, np.shape(q_bar)[0], np.shape(q_bar)[1]))  # page 90 - Equation (40)
    h_t = np.zeros((t, np.shape(q_bar)[0], np.shape(q_bar)[1]))  # page 89 - Equation (35)
    dts = np.zeros((t, np.shape(q_bar)[0], np.shape(q_bar)[1]))  # page 88 - Equation (32)
    dts_inv = np.zeros((t, np.shape(q_bar)[0], np.shape(q_bar)[1]))  # used in calculation of u_t
    u_t = np.zeros((np.shape(q_bar)[0], t))  # page 89 - defined between Equation (37) and (38)
    qts = np.zeros((t, np.shape(q_bar)[0], np.shape(q_bar)[1]))  # page 90 - defined within Equation (39)
    r_t = np.zeros((t, np.shape(q_bar)[0], np.shape(q_bar)[1]))  # page 90 - Equation (39)

    # initialise q_t
    # unsure of initialisation
    q_t[0] = np.zeros_like(np.matmul(returns_matrix[0, :].reshape(-1, 1) / 2, returns_matrix[0, :].reshape(1, -1) / 2))

    for var in range(t):
        # page 88 - Equation (32)
        dts[var] = np.diag(modelled_variance[int(var - 1), :])  # modelled variance - page 89 - Equation (34)
        # page 89 - defined between Equation (37) and (38)
        try:
            dts_inv[var] = np.linalg.inv(dts[var])
        except:
            dts_inv[var] = np.linalg.pinv(dts[var])
        u_t[:, var] = np.matmul(dts_inv[var], returns_matrix[var, :].reshape(-1, 1))[:, 0]

    # initialise log-likehood value
    loglike = 0

    for var in range(1, t):

        # page 90 - Equation (40)
        q_t[var] = (1 - params[0] - params[1]) * q_bar + \
                   params[0] * np.matmul(u_t[:, int(var - 1)].reshape(-1, 1), u_t[:, int(var - 1)].reshape(1, -1)) + \
                   params[1] * q_t[int(var - 1)]

        # page 90 - defined within Equation (39)
        try:
            qts[var] = np.linalg.inv(np.sqrt(np.diag(np.diag(q_t[var]))))
        except:
            qts[var] = np.linalg.pinv(np.sqrt(np.diag(np.diag(q_t[var]))))

        # page 90 - Equation (39)
        r_t[var] = np.matmul(qts[var], np.matmul(q_t[var], qts[var]))

        # page 89 - Equation (35)
        h_t[var] = np.matmul(dts[var], np.matmul(r_t[var], dts[var]))

        # likelihood function from reference work page 11 - Equation 26
        try:
            loglike -= (np.shape(q_bar)[0] * np.log(2 * np.pi) +
                        2 * np.log(np.linalg.det(dts[var])) + np.log(np.linalg.det(r_t[var])) +
                        np.matmul(u_t[:, var].reshape(1, -1),
                                  np.matmul(np.linalg.inv(r_t[var]),
                                            u_t[:, var].reshape(-1, 1)))[0][0])
        except:
            loglike -= (np.shape(q_bar)[0] * np.log(2 * np.pi) +
                        2 * np.log(np.linalg.det(dts[var])) + np.log(np.linalg.det(r_t[var])) +
                        np.matmul(u_t[:, var].reshape(1, -1),
                                  np.matmul(np.linalg.pinv(r_t[var]),
                                            u_t[:, var].reshape(-1, 1)))[0][0])

    return loglike

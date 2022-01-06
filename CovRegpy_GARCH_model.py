
# Reference work:

# Bauwens, L., Laurent, S. and Rombouts, J.,
# Multivariate GARCH Models: A Survey. Journal of Applied
# Econometrics, 2006, 21, 79–109.

# Log-likelihood reference work:

# Engle, R., Dynamic Conditional Correlation: A Simple Class of
# Multivariate Generalized Autoregressive Conditional Heteroskedasticity Models.
# Journal of Business & Economic Statistics, 2002, 20, 339–350.

import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from AdvEMDpy import emd_basis

np.random.seed(3)

# pull all close data
tickers_format = ['MSFT', 'AAPL', 'GOOGL', 'AMZN', 'TSLA']
data = yf.download(tickers_format, start="2018-12-31", end="2022-01-01")
close_data = data['Close']
del data, tickers_format

# create date range and interpolate
date_index = pd.date_range(start='31/12/2018', end='01/01/2022')
close_data = close_data.reindex(date_index).interpolate()
close_data = close_data[::-1].interpolate()
close_data = close_data[::-1]
del date_index

# calculate returns and realised covariance
returns = (np.log(np.asarray(close_data)[1:, :]) -
           np.log(np.asarray(close_data)[:-1, :]))
realised_covariance = np.cov(returns.T)
risk_free = (0.02 / 365)

# calculate knots and returns subset
model_days = 731
knots = 80
forecast_days = 30
knots_vector = np.linspace(0, model_days - 1, int(knots - 6))
knots_vector = np.linspace(-knots_vector[3], 2 * (model_days - 1) - knots_vector[-4], knots)
returns_subset_forecast = returns[:model_days, :]

# calculate spline basis and mean forecast
spline_basis_transform = emd_basis.Basis(time_series=np.arange(model_days), time=np.arange(model_days))
spline_basis_transform = spline_basis_transform.cubic_b_spline(knots=knots_vector)
coef_forecast = np.linalg.lstsq(spline_basis_transform.T, returns_subset_forecast, rcond=None)[0]
mean_forecast = np.matmul(coef_forecast.T, spline_basis_transform).T

# # plot returns and means
# for i in range(5):
#     plt.plot(returns_subset_forecast[:, i])
#     plt.plot(mean_forecast[:, i])
#     plt.show()

# # not necessary, but helps to follow process
# rt = returns_subset_forecast
# import mgarch
# vol = mgarch.mgarch()
# vol.fit(rt)
# ndays = 10 # volatility of nth day
# cov_nextday = vol.predict(ndays)

returns_minus_mean = returns_subset_forecast - mean_forecast


def covregpy_mgarch(returns_matrix, p=3, q=3, days=10, print_correlation=False):

    # flip matrix to be consistent
    if np.shape(returns_matrix)[0] < np.shape(returns_matrix)[1]:
        returns_matrix = returns_matrix.T

    # initialised modelled variance matrix
    modelled_variance = np.zeros_like(returns_matrix)

    # iteratively calculate modelled variance using univariate GARCH model
    for stock in range(np.shape(returns_matrix)[1]):
        model = arch_model(returns_matrix[:, stock], mean='Zero', vol='GARCH', p=p, q=q)
        model_fit = model.fit()
        modelled_variance[:, stock] = model_fit.conditional_volatility

    # optimise alpha & beta parameters to be used in page 90 equation (40)
    params = minimize(custom_loglike, (0.01, 0.94),
                      args=(returns_matrix, modelled_variance),
                      bounds=((1e-6, 1), (1e-6, 1)),
                      method='Nelder-Mead')

    # list of optimisation methods available

    # Nelder-Mead, Powell, CG, BFGS, Newton-CG,
    # L-BFGS-B, TNC, COBYLA, SLSQP, trust-constr,
    # dogleg, trust-ncg, trust-exact, trust-krylov

    a = params.x[0]
    b = params.x[1]

    # debugging
    print(a)
    print(b)
    print(a + b)

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
        dts_inv[var] = np.linalg.inv(dts[var])
        u_t[:, var] = np.matmul(dts_inv[var], returns_matrix[var, :].reshape(-1, 1))[:, 0]

    for var in range(1, t):

        # page 90 - Equation (40)
        q_t[var] = (1 - a - b) * q_bar + \
                   a * np.matmul(u_t[:, int(var - 1)].reshape(-1, 1), u_t[:, int(var - 1)].reshape(1, -1)) + \
                   b * q_t[int(var - 1)]

        # page 90 - defined within Equation (39)
        qts[var] = np.linalg.inv(np.sqrt(np.diag(np.diag(q_t[var]))))

        # page 90 - Equation (39)
        r_t[var] = np.matmul(qts[var], np.matmul(q_t[var], qts[var]))

        # page 89 - Equation (35)
        h_t[var] = np.matmul(dts[var], np.matmul(r_t[var], dts[var]))

    forecasted_covariance = h_t[-1] * np.sqrt(days)

    if print_correlation:
        corr = np.zeros_like(forecasted_covariance)
        for i in range(np.shape(corr)[0]):
            for j in range(np.shape(corr)[1]):
                corr[i, j] = forecasted_covariance[i, j] / (np.sqrt(forecasted_covariance[i, i]) *
                                                            np.sqrt(forecasted_covariance[j, j]))
        print(corr)

    return forecasted_covariance


def custom_loglike(params, returns_matrix, modelled_variance):

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
        dts_inv[var] = np.linalg.inv(dts[var])
        u_t[:, var] = np.matmul(dts_inv[var], returns_matrix[var, :].reshape(-1, 1))[:, 0]

    # initialise log-likehood value
    loglike = 0

    for var in range(1, t):

        # page 90 - Equation (40)
        q_t[var] = (1 - params[0] - params[1]) * q_bar + \
                   params[0] * np.matmul(u_t[:, int(var - 1)].reshape(-1, 1), u_t[:, int(var - 1)].reshape(1, -1)) + \
                   params[1] * q_t[int(var - 1)]

        # page 90 - defined within Equation (39)
        qts[var] = np.linalg.inv(np.sqrt(np.diag(np.diag(q_t[var]))))

        # page 90 - Equation (39)
        r_t[var] = np.matmul(qts[var], np.matmul(q_t[var], qts[var]))

        # page 89 - Equation (35)
        h_t[var] = np.matmul(dts[var], np.matmul(r_t[var], dts[var]))

        # likelihood function from reference work page 11 - Equation 26
        loglike -= (np.shape(q_bar)[0] * np.log(2 * np.pi) +
                    2 * np.log(np.linalg.det(dts[var])) + np.log(np.linalg.det(r_t[var])) +
                    np.matmul(u_t[:, var].reshape(1, -1),
                              np.matmul(np.linalg.inv(r_t[var]),
                                        u_t[:, var].reshape(-1, 1)))[0][0])

    return loglike


temp = covregpy_mgarch(returns_minus_mean, p=3, q=3, days=10, print_correlation=True)

print(temp)

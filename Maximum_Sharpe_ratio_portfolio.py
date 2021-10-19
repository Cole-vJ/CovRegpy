
# included in covariance_regression GitHub package

import numpy as np
from scipy.optimize import minimize


# risk budgeting approach
def sharpe_obj_fun(x, p_cov, returns, risk_free):
    return -(sum(x * returns) - risk_free) / np.sqrt(np.sum(x * np.dot(p_cov, x)))


# equality constraint: = 0
def cons_sum_weight(x):
    return np.sum(x) - 1


# inequality constraint: > 0
def cons_long_only_weight(x):
    return x


# risk budgeting weighting
def sharpe_rb_p_weights(cov, returns, risk_free):
    w0 = np.ones((np.shape(cov)[0], 1)) / np.shape(cov)[0]
    cons = ({'type': 'eq', 'fun': cons_sum_weight}, {'type': 'ineq', 'fun': cons_long_only_weight})
    return minimize(sharpe_obj_fun, w0, args=(cov, returns, risk_free), method='SLSQP', constraints=cons)


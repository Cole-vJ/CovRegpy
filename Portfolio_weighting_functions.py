
import numpy as np
from scipy.optimize import minimize


# risk budgeting approach
def obj_fun(x, p_cov, rb):
    return np.sum((x * np.dot(p_cov, x) / np.dot(x.transpose(), np.dot(p_cov, x)) - rb) ** 2)


# risk budgeting approach
def global_obj_fun(x, p_cov):
    return np.sum(x * np.dot(p_cov, x))


# equality constraint: = 0
def cons_sum_weight(x):
   return np.sum(x) - 1


# inequality constraint: > 0
def cons_long_only_weight(x):
   return x


# risk budgeting weighting
def rb_p_weights(cov):
    w0 = np.ones((np.shape(cov)[0], 1)) / np.shape(cov)[0]
    cons = ({'type': 'eq', 'fun': cons_sum_weight}, {'type': 'ineq', 'fun': cons_long_only_weight})
    return minimize(obj_fun, w0, args=(cov, 1 / np.shape(cov)[0]), method='SLSQP', constraints=cons)


# global minimum weights
def global_weights(cov):
    return (np.matmul(np.linalg.inv(cov), np.ones(np.shape(cov)[1]).reshape(-1, 1)) /
            np.matmul(np.ones(np.shape(cov)[1]).reshape(1, -1), np.matmul(np.linalg.inv(cov),
                                                                          np.ones(np.shape(cov)[1]).flatten())))[:, 0]


if __name__ == "__main__":

    variance = np.zeros((2, 2))
    variance[0, 0] = 1  # 1
    variance[1, 1] = 16  # 1/4
    variance[0, 1] = variance[1, 0] = 0  # -1.8? -1.9?
    weights = rb_p_weights(variance).x
    # print(weights)
    volatility_contribution = \
        weights * np.dot(variance, weights) / np.dot(weights.transpose(), np.dot(variance, weights))
    # print(volatility_contribution)

    if all(np.isclose(volatility_contribution, np.asarray([1/2, 1/2]), atol=1e-3)):
        print('EQUAL contributions to volatility.')
    else:
        print('NON-EQUAL contributions to volatility.')

    variance = np.zeros((3, 3))
    variance[0, 0] = 1  # 1
    variance[1, 1] = 1  # 1
    variance[2, 2] = 4  # 1/2
    variance[0, 1] = variance[1, 0] = 1  # -2? 2?
    variance[0, 2] = variance[2, 0] = 0
    variance[2, 1] = variance[1, 2] = 0
    weights = rb_p_weights(variance).x
    # print(weights)
    volatility_contribution = \
        weights * np.dot(variance, weights) / np.dot(weights.transpose(), np.dot(variance, weights))
    # print(volatility_contribution)

    if all(np.isclose(volatility_contribution, np.asarray([1 / 3, 1 / 3, 1 / 3]), atol=1e-3)):
        print('EQUAL contributions to volatility.')
    else:
        print('NON-EQUAL contributions to volatility.')
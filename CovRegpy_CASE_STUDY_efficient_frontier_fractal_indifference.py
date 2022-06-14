
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.linalg import cholesky

from CovRegpy_utilities import efficient_frontier, global_minimum_forward_applied_information, \
    sharpe_forward_applied_information

from AdvEMDpy import emd_basis

sns.set(style='darkgrid')

# daily risk free rate
risk_free = (0.01 / 365)

increments = 10001
no_factors = 2
time = np.linspace(0, 50 * 2 * np.pi, increments)
factors = np.zeros((no_factors, increments))
factors[0, :] = np.sin(time)
factors[1, :] = np.sin(4 * time)
returns = np.random.normal(0, 1, (no_factors, increments))

corr = np.array(([1, -0.5], [-0.5, 1]))
L = np.linalg.cholesky(corr)
returns_corr = np.matmul(L, returns)

temp = 0

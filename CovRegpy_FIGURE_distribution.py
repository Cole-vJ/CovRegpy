
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
import textwrap
import matplotlib.pyplot as plt

sns.set(style='darkgrid')

def normal(mu, sigma, x):
    return (1/np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(-((mu - x)**2)/(sigma ** 2))

high_long_freq = np.asarray(pd.read_csv('Cumulative Returns/high_long_freq.csv', header=0))[:, 1]
high_res_freq = np.asarray(pd.read_csv('Cumulative Returns/high_res_freq.csv', header=0))[:, 1]
high_res_freq = np.append(high_res_freq[:198], high_res_freq[199:])
all_long_freq = np.asarray(pd.read_csv('Cumulative Returns/all_long_freq.csv', header=0))[:, 1]
all_res_freq = np.asarray(pd.read_csv('Cumulative Returns/all_res_freq.csv', header=0))[:, 1]
all_res_freq = np.append(all_res_freq[:30], all_res_freq[31:])
low_long_freq = np.asarray(pd.read_csv('Cumulative Returns/low_long_freq.csv', header=0))[:, 1]
low_res_freq = np.asarray(pd.read_csv('Cumulative Returns/low_res_freq.csv', header=0))[:, 1]

realised_variance_returns = np.log(1.4849082851150919)/4
dcc_returns = np.log(1.4668328786939422)/4
sp500_proxy_returns = np.log(1.6552954098543056)/4
minimum_var_returns = np.log(1.3587817911842188)/4
pca_returns = np.log(1.4897923114940013)/4

print('Skewness')
high_long_annual_returns = np.log(high_long_freq)/4
high_long_annual_mu = np.mean(high_long_annual_returns)
high_long_annual_sigma = np.std(high_long_annual_returns)
print(sp.stats.skew(high_long_annual_returns))
high_res_annual_returns = np.log(high_res_freq)/4
high_res_annual_mu = np.mean(high_res_annual_returns)
high_res_annual_sigma = np.std(high_res_annual_returns)
print(sp.stats.skew(high_res_annual_returns))

all_long_annual_returns = np.log(all_long_freq)/4
all_long_annual_mu = np.mean(all_long_annual_returns)
all_long_annual_sigma = np.std(all_long_annual_returns)
print(sp.stats.skew(all_long_annual_returns))
all_res_annual_returns = np.log(all_res_freq)/4
all_res_annual_mu = np.mean(all_res_annual_returns)
all_res_annual_sigma = np.std(all_res_annual_returns)
print(sp.stats.skew(all_res_annual_returns))

low_long_annual_returns = np.log(low_long_freq)/4
low_long_annual_mu = np.mean(low_long_annual_returns)
low_long_annual_sigma = np.std(low_long_annual_returns)
print(sp.stats.skew(low_long_annual_returns))
low_res_annual_returns = np.log(low_res_freq)/4
low_res_annual_mu = np.mean(low_res_annual_returns)
low_res_annual_sigma = np.std(low_res_annual_returns)
print(sp.stats.skew(low_res_annual_returns))

print('Means')
print(high_long_annual_mu)
print(high_res_annual_mu)
print(all_long_annual_mu)
print(all_res_annual_mu)
print(low_long_annual_mu)
print(low_res_annual_mu)

print('Variance')
print(high_long_annual_sigma ** 2)
print(high_res_annual_sigma ** 2)
print(all_long_annual_sigma ** 2)
print(all_res_annual_sigma ** 2)
print(low_long_annual_sigma ** 2)
print(low_res_annual_sigma ** 2)

x = np.linspace(0.05, 0.15, 100)

ax = plt.subplot(111)
plt.gcf().subplots_adjust(bottom=0.15)
plt.title(textwrap.fill('Annualised Returns of Benchmark Portfolios versus Risk Parity Portfolios', 45))
plt.plot(realised_variance_returns * np.ones(100), np.linspace(0, 25, 100), '--', label='Realised covariance', Linewidth=2)
plt.plot(dcc_returns * np.ones(100), np.linspace(0, 25, 100), '--', label='DDC-GARCH', Linewidth=2)
plt.plot(sp500_proxy_returns * np.ones(100), np.linspace(0, 25, 100), '--', label='S&P 500 Proxy', Linewidth=2)
plt.plot(minimum_var_returns * np.ones(100), np.linspace(0, 25, 100), '--', label='Minimum variance', Linewidth=2)
plt.plot(pca_returns * np.ones(100), np.linspace(0, 25, 100), '--', label='PCA', Linewidth=2)
plt.plot(x, normal(high_long_annual_mu, high_long_annual_sigma, x), label='High frequency long')
plt.plot(x, normal(high_res_annual_mu, high_res_annual_sigma, x), label=textwrap.fill('High frequency long/short', 15))
plt.plot(x, normal(all_long_annual_mu, all_long_annual_sigma, x), label='All frequencies long')
plt.plot(x, normal(all_res_annual_mu, all_res_annual_sigma, x), label=textwrap.fill('All frequencies long/short', 15))
plt.plot(x, normal(low_long_annual_mu, low_long_annual_sigma, x), label='Low frequency long')
plt.plot(x, normal(low_res_annual_mu, low_res_annual_sigma, x), label=textwrap.fill('Low frequency long/short', 15), c='k')
plt.xlabel('Annualised Returns')
plt.ylabel('Density')
box_0 = ax.get_position()
ax.set_position([box_0.x0 - 0.04, box_0.y0, box_0.width * 0.84, box_0.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
plt.savefig('figures/Annualised_returns.png')
plt.show()




import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

sns.set_style('darkgrid')

x = np.linspace(1, 10, 10)
y = np.linspace(1, 10, 10)
Z, Y = np.meshgrid(x, y)
X = np.random.uniform(0, 1, np.shape(Z))

temp_1 = np.random.normal(0, 1, (10, 10))
temp_2 = np.random.normal(0, 1, (10, 10))
temp_3 = np.random.normal(0, 1, (10, 10))
temp_1 = np.corrcoef(temp_1)
temp_2 = np.corrcoef(temp_2)
temp_3 = np.corrcoef(temp_3)

fig = plt.figure()
fig.set_size_inches(8, 10)
ax = plt.axes(projection='3d')
ax.set_title('Correlation Structure Through Time')
cov_plot = ax.plot_surface(Z, Y, temp_1, rstride=1, cstride=1, cmap='gist_rainbow', edgecolor='none')
ax.plot_surface(Z, Y, temp_2 + 40, rstride=1, cstride=1, cmap='gist_rainbow', edgecolor='none')
ax.plot_surface(Z, Y, temp_3 + 80, rstride=1, cstride=1, cmap='gist_rainbow', edgecolor='none')
# ax.set_xlabel('Asset')
ax.set_xticks(ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
ax.set_xticklabels(labels=['BTC', 'ETH', 'BNB', 'ADA', 'XRP', 'DOGE', 'DOT', 'UNI', 'LINK', 'SOL'], rotation=20,
                   fontsize=8, ha="left", rotation_mode="anchor")
# ax.set_zlim(0, 10)
# ax.set_ylabel('Asset')
ax.set_yticks(ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
ax.set_yticklabels(labels=['BTC', 'ETH', 'BNB', 'ADA', 'XRP', 'DOGE', 'DOT', 'UNI', 'LINK', 'SOL'], rotation=0,
                   fontsize=8)
ax.set_zticks(ticks=[0, 40, 80])
ax.set_zticklabels(['01-10-2020', '02-10-2020', '03-10-2020'], rotation=-60, fontsize=8)
cbar = plt.colorbar(cov_plot)
cbar.set_label("Covariance")
plt.savefig('figures/covariance_example.png')
plt.show()

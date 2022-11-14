
import textwrap
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
# fig.set_size_inches(8, 10)
ax = plt.axes(projection='3d')
ax.view_init(30, -120)
ax.set_xlabel(r'$x$', fontsize=8)
ax.set_xticks(ticks=np.arange(3))
ax.set_ylabel(r'$y$', fontsize=8)
ax.set_yticks(ticks=np.arange(-1, 2, 1))
ax.set_zlabel(r'$z$', fontsize=8)
ax.set_zticks(ticks=np.arange(3))
x = np.linspace(0, 2, 21)
y = np.linspace(0, 2, 21)
X, Y = np.meshgrid(x, y)
Z = np.sqrt(X*Y)
ax.set_title(textwrap.fill('Three-Dimensional Cone formed by Positive Semi-Definite Two-Dimensional Matrices', 45))
cov_plot = ax.plot_surface(X, Z, Y, rstride=1, cstride=1, cmap='gist_rainbow', edgecolor='black', antialiased=False,
                           shade=True, alpha=0.5, vmin=0, vmax=2)
cov_plot = ax.plot_surface(X, -Z, Y, rstride=1, cstride=1, cmap='gist_rainbow', edgecolor='black', antialiased=False,
                           shade=True, alpha=0.5, vmin=0, vmax=2)
ax.set_xlim(0, 2)
ax.set_zlim(0, 2)
ax.set_ylim(-2, 2)
plt.savefig('B and Psi Estimates/Cone.png')
plt.show()


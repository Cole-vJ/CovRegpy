
import textwrap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='darkgrid')
kl_distances = [0]
time = np.arange(-0.99, 1.00, 0.01)

for j in time:
    kl_distances.append((1 / 2) * (np.sum(np.diag(np.matmul(np.linalg.inv(np.asarray(([1, j], [j, 1]))),
                                                            np.asarray(([1, 0], [0, 1]))))) +
                                   np.log(np.linalg.det(np.asarray(([1, j], [j, 1]))) / np.linalg.det(np.asarray(([1, 0], [0, 1])))) - 2))

fig, axs = plt.subplots(1, 2)
plt.suptitle(textwrap.fill('Kullback–Leibler Divergence for 2D Standard Multivariate Normal Distributions', 45))
axs[0].plot(time, kl_distances[1:])
axs[0].set_xlabel(r'$\rho$')
axs[0].set_ylabel('Divergence')
axs[0].plot(np.linspace(-1.1, 1.1), -0.25 * np.ones(50), 'k--')
axs[0].plot(np.linspace(-1.1, 1.1), 3.25 * np.ones(50), 'k--')
axs[0].plot(-1.1 * np.ones(50), np.linspace(-0.25, 3.25), 'k--')
axs[0].plot(1.1 * np.ones(50), np.linspace(-0.25, 3.25), 'k--')
axs[1].plot(time, kl_distances[1:])
axs[1].set_xlabel(r'$\rho$')
axs[1].set_ylim(-0.25, 3.25)
axs[0].set_title('KL Divergence')
axs[1].set_title('Zoomed Region')
plt.subplots_adjust(wspace=0.1, top=0.8, bottom=0.16, left=0.16)
box_0 = axs[0].get_position()
axs[0].set_position([box_0.x0, box_0.y0, box_0.width * 0.8, box_0.height * 1.0])
box_1 = axs[1].get_position()
axs[1].set_position([box_1.x0, box_1.y0, box_1.width * 0.8, box_1.height * 1.0])
# axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
plt.savefig('B and Psi Estimates/KL_distance.png')
plt.show()

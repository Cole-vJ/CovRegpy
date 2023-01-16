
#     ________
#            /
#      \    /
#       \  /
#        \/

import numpy as np
import textwrap
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='darkgrid')

plt.title(textwrap.fill('Dominoes Instantaneous Effect', 50), fontsize=20)
for i in [0.1, 0.3, 0.5, 0.7, 0.9]:

    # plt.plot(i * np.ones(100), np.linspace(0.3, 0.7, 100), c='slategray')
    # plt.plot((i + 0.025) * np.ones(100), np.linspace(0.3, 0.7, 100), c='slategray')
    # plt.plot(np.linspace(i, (i + 0.025), 100), 0.3 * np.ones(100), c='slategray')
    # plt.plot(np.linspace(i, (i + 0.025), 100), 0.7 * np.ones(100), c='slategray')

    # plt.plot(np.linspace((i + 0.025), (i + 0.05), 100), np.linspace(0.7, 0.8, 100), c='darkslategray')
    # plt.plot(np.linspace((i + 0.025), (i + 0.05), 100), np.linspace(0.3, 0.4, 100), c='darkslategray')
    # plt.plot((i + 0.05) * np.ones(100), np.linspace(0.4, 0.8, 100), c='darkslategray')
    plt.fill(np.hstack((np.linspace((i + 0.025), (i + 0.05), 100), np.linspace((i + 0.025), (i + 0.05), 100)[::-1])),
             np.hstack((np.linspace(0.7, 0.8, 100), np.linspace(0.3, 0.4, 100)[::-1])), c='darkslategray')

    # plt.plot(np.linspace(i, (i + 0.025), 100), np.linspace(0.7, 0.8, 100), c='slategray')
    # plt.plot(np.linspace((i + 0.025), (i + 0.05), 100), 0.8 * np.ones(100), c='slategray')
    plt.fill(np.hstack((np.linspace(i, (i + 0.025), 100), np.linspace((i + 0.025), (i + 0.05), 100)[::-1])),
             np.hstack((np.linspace(0.7, 0.8, 100), np.linspace(0.7, 0.8, 100)[::-1])), c='gray')

    plt.fill(np.hstack((i * np.ones(100), (i + 0.025) * np.ones(100))),
             np.hstack((np.linspace(0.7, 0.3, 100), np.linspace(0.3, 0.7, 100))), c='slategray')
    plt.fill(np.hstack(((i + 0.005) * np.ones(100), (i + 0.02) * np.ones(100))),
             np.hstack((np.linspace(0.69, 0.67, 100), np.linspace(0.67, 0.69, 100))), c='silver')
    plt.fill(np.hstack(((i + 0.01) * np.ones(100), (i + 0.015) * np.ones(100))),
             np.hstack((np.linspace(0.685, 0.675, 100), np.linspace(0.675, 0.685, 100))), c='black')

plt.plot(np.linspace(0.1125, 0.9125, 100), 0.68 * np.ones(100), c='black', linewidth=2)
plt.plot(np.linspace(0.1375, 0.9375, 100), 0.78 * np.ones(100), c='black', linewidth=2, zorder=0)
plt.plot(np.linspace(0.025, 0.1, 100), 0.75 * np.ones(100), c='black', linewidth=2)
plt.plot(np.linspace(0.0875, 0.1, 100), np.linspace(0.76, 0.75, 100), c='black', linewidth=2)
plt.plot(np.linspace(0.0875, 0.1, 100), np.linspace(0.74, 0.75, 100), c='black', linewidth=2)
plt.grid(visible=None)
plt.yticks([0], '')
plt.xticks([0], '')
plt.xlim(0, 1.025)
plt.ylim(0, 1)
plt.show()

plt.title(textwrap.fill('Dominoes Lagged Effect', 50), fontsize=20)
for i in [0.1, 0.3, 0.5, 0.7, 0.9]:

    # plt.plot(i * np.ones(100), np.linspace(0.3, 0.7, 100), c='slategray')
    # plt.plot((i + 0.025) * np.ones(100), np.linspace(0.3, 0.7, 100), c='slategray')
    # plt.plot(np.linspace(i, (i + 0.025), 100), 0.3 * np.ones(100), c='slategray')
    # plt.plot(np.linspace(i, (i + 0.025), 100), 0.7 * np.ones(100), c='slategray')

    # plt.plot(np.linspace((i + 0.025), (i + 0.05), 100), np.linspace(0.7, 0.8, 100), c='darkslategray')
    # plt.plot(np.linspace((i + 0.025), (i + 0.05), 100), np.linspace(0.3, 0.4, 100), c='darkslategray')
    # plt.plot((i + 0.05) * np.ones(100), np.linspace(0.4, 0.8, 100), c='darkslategray')
    plt.fill(np.hstack((np.linspace((i + 0.025), (i + 0.05), 100), np.linspace((i + 0.025), (i + 0.05), 100)[::-1])),
             np.hstack((np.linspace(0.7, 0.8, 100), np.linspace(0.3, 0.4, 100)[::-1])), c='darkslategray')

    # plt.plot(np.linspace(i, (i + 0.025), 100), np.linspace(0.7, 0.8, 100), c='slategray')
    # plt.plot(np.linspace((i + 0.025), (i + 0.05), 100), 0.8 * np.ones(100), c='slategray')
    plt.fill(np.hstack((np.linspace(i, (i + 0.025), 100), np.linspace((i + 0.025), (i + 0.05), 100)[::-1])),
             np.hstack((np.linspace(0.7, 0.8, 100), np.linspace(0.7, 0.8, 100)[::-1])), c='gray')

    plt.fill(np.hstack((i * np.ones(100), (i + 0.025) * np.ones(100))),
             np.hstack((np.linspace(0.7, 0.3, 100), np.linspace(0.3, 0.7, 100))), c='slategray')
    # plt.fill(np.hstack(((i + 0.005) * np.ones(100), (i + 0.02) * np.ones(100))),
    #          np.hstack((np.linspace(0.69, 0.67, 100), np.linspace(0.67, 0.69, 100))), c='silver')
    # plt.fill(np.hstack(((i + 0.01) * np.ones(100), (i + 0.015) * np.ones(100))),
    #          np.hstack((np.linspace(0.685, 0.675, 100), np.linspace(0.675, 0.685, 100))), c='black')

# plt.plot(np.linspace(0.1125, 0.9125, 100), 0.68 * np.ones(100), c='black', linewidth=2)
plt.plot(np.linspace(0.025, 0.1, 100), 0.75 * np.ones(100), c='black', linewidth=2)
plt.plot(np.linspace(0.0875, 0.1, 100), np.linspace(0.76, 0.75, 100), c='black', linewidth=2)
plt.plot(np.linspace(0.0875, 0.1, 100), np.linspace(0.74, 0.75, 100), c='black', linewidth=2)
plt.grid(visible=None)
plt.yticks([0], '')
plt.xticks([0], '')
plt.xlim(0, 1.025)
plt.ylim(0, 1)
plt.show()

# frequency refresher

np.random.seed(0)

time = np.linspace(0, 5 * np.pi, 1001)
high_freq = np.sin(5 * time + np.random.uniform(0, 2 * np.pi))
low_freq = np.sin(time + np.random.uniform(0, 2 * np.pi))

plt.figure(figsize=[6.4, 3.8])
plt.title('High Frequency Structure', fontsize=16)
plt.plot(time, high_freq)
plt.yticks([-1, 0, 1], ['-1', '0', '1'])
plt.xticks([0, np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi, 5 * np.pi],
           ['0', r'$\pi$', r'$2\pi$', r'$3\pi$', r'$4\pi$', r'$5\pi$'])
plt.savefig('/home/cole/Desktop/Cole/Cole Documents/3rd Year PhD Presentation/High_freq')
plt.show()

plt.figure(figsize=[6.4, 3.8])
plt.title('Low Frequency Structure', fontsize=16)
plt.plot(time, low_freq)
plt.yticks([-1, 0, 1], ['-1', '0', '1'])
plt.xticks([0, np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi, 5 * np.pi],
           ['0', r'$\pi$', r'$2\pi$', r'$3\pi$', r'$4\pi$', r'$5\pi$'])
plt.savefig('/home/cole/Desktop/Cole/Cole Documents/3rd Year PhD Presentation/Low_freq')
plt.show()

plt.figure(figsize=[6.4, 3.8])
plt.title('High Amplitude Structure', fontsize=16)
plt.plot(time, 5 * low_freq)
plt.yticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
           ['-5', '-4', '-3', '-2', '-1', '0', '1', '2', '3', '4', '5'])
plt.xticks([0, np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi, 5 * np.pi],
           ['0', r'$\pi$', r'$2\pi$', r'$3\pi$', r'$4\pi$', r'$5\pi$'])
plt.savefig('/home/cole/Desktop/Cole/Cole Documents/3rd Year PhD Presentation/High_amp')
plt.show()

plt.figure(figsize=[6.4, 3.8])
plt.title('Low Amplitude Structure', fontsize=16)
plt.plot(time, low_freq)
plt.yticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
           ['-5', '-4', '-3', '-2', '-1', '0', '1', '2', '3', '4', '5'])
plt.xticks([0, np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi, 5 * np.pi],
           ['0', r'$\pi$', r'$2\pi$', r'$3\pi$', r'$4\pi$', r'$5\pi$'])
plt.ylim(-5.5, 5.5)
plt.savefig('/home/cole/Desktop/Cole/Cole Documents/3rd Year PhD Presentation/Low_amp')
plt.show()

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn import linear_model
import sys
from tqdm import tqdm
from cycler import cycler
import string
from itertools import cycle


def label_axes(fig, labels=None, loc=None, **kwargs):
    if labels is None:
        labels = string.ascii_lowercase
    labels = cycle(labels)
    if loc is None:
        loc = (-0.1, 1.1)
    axes = [ax for ax in fig.axes if ax.get_label() != '<colorbar>']
    for ax, lab in zip(axes, labels):
        ax.annotate('(' + lab + ')', size=16, xy=loc,
                    xycoords='axes fraction',
                    **kwargs)


plt.style.use(['science', 'grid'])
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
colors = [colors[3], colors[1], colors[0]]
colors_nipy1 = mpl.cm.nipy_spectral(np.linspace(0.1, 0.9, 6))
colors_nipy2 = mpl.cm.nipy_spectral(np.linspace(0.6, 0.9, 7))
colors_nipy = list(colors_nipy1[0:3]) + list(colors_nipy2[3:-2]) + list(colors_nipy1[-1:])
plt.rcParams['axes.prop_cycle'] = cycler(color=colors)

df_tddft = pd.read_csv('TDDFT_MOPSSAM_test_data.csv')
df_mopssam = pd.read_csv('xtb_tddft_calib_data.csv')

fig = plt.figure(num=2, figsize=[7, 4], dpi=300, clear=True)
ax = fig.add_subplot(1, 1, 1)
plt.plot(df_tddft['S1'], df_mopssam['aug-cc-TDDFT'], '.', color=colors_nipy[1])
x = np.linspace(0, 10, 100)
plt.plot(x, x, 'k--')
plt.xlim(2, 7)
plt.ylim(2, 7)
ax.set_axisbelow(True)
plt.grid(True)
ax.set_aspect('equal', adjustable='box')
# plt.legend(markerscale=6, fontsize=14)
plt.xlabel('Independent TD-DFT S$_1$ (eV)', fontsize=16)
plt.ylabel('MOPSSAM S$_1$ (eV)', fontsize=16)
plt.savefig('mopssam_S1_comp.png')

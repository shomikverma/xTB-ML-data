print('importing')
import os
import sys
import time
from typing import List
# import joblib
from tqdm import tqdm

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

import seaborn as sns
import seaborn.distributions as sd
import matplotlib as mpl
import matplotlib.pyplot as plt

import umap
import hdbscan
from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from rdkit.Chem import AllChem

import string
from itertools import cycle
from cycler import cycler

num_AL_adds = 6
shift_range = -1
plt.style.use(['science'])
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
colors = [colors[0], colors[2], colors[1]] + colors[3:]
colors_nipy = mpl.cm.nipy_spectral(np.linspace(0.1, 0.9, num_AL_adds))
# colors_nipy = mpl.cm.tab10(np.linspace(0, 1, num_AL_adds))
colors_plas = mpl.cm.plasma(np.linspace(0.1, 0.9, num_AL_adds))
colors_gist = mpl.cm.gist_rainbow(np.linspace(0.1, 0.9, num_AL_adds))
colors_past = mpl.cm.tab10(np.linspace(0.1, 0.9, num_AL_adds))
plt.rcParams['axes.prop_cycle'] = cycler(color=colors)


def label_axes(fig, labels=None, loc=None, **kwargs):
    if labels is None:
        labels = string.ascii_lowercase
    labels = cycle(labels)
    if loc is None:
        loc = (-0.1, 1.1)
    axes = [ax for ax in fig.axes if ax.get_label() != '<colorbar>']
    for ax, lab in zip(axes, labels):
        ax.annotate('(' + lab + ')', size=14, xy=loc,
                    xycoords='axes fraction',
                    **kwargs)


def label_axes_global(fig, labels=None, loc=None, **kwargs):
    if labels is None:
        labels = string.ascii_lowercase
    labels = cycle(labels)
    if loc is None:
        loc = (0.02, 0.95)
    axes = [ax for ax in fig.axes if ax.get_label() != '<colorbar>']
    for ax, lab in zip(axes, labels):
        ax.annotate('(' + lab + ')', size=20, xy=loc,
                    xycoords='axes fraction',
                    **kwargs)


def label_axes_cluster(fig, labels=None, loc=None, **kwargs):
    if labels is None:
        labels = string.ascii_lowercase
    labels = cycle(labels)
    if loc is None:
        loc = (-0.1, 1.05)
    axes = [ax for ax in fig.axes if ax.get_label() != '<colorbar>']
    for ax, lab in zip(axes, labels):
        ax.annotate('(' + lab + ')', size=14, xy=loc,
                    xycoords='axes fraction',
                    **kwargs)


def compute_ecfp_descriptors(smiles_list: List[str]):
    """ Computes ecfp descriptors """

    keep_idx = []
    descriptors = []
    for i, smiles in enumerate(tqdm(smiles_list)):
        ecfp = _compute_single_ecfp_descriptor(smiles)
        if ecfp is not None:
            keep_idx.append(i)
            descriptors.append(ecfp)

    return np.vstack(descriptors), keep_idx


def _compute_single_ecfp_descriptor(smiles: str):
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception as E:
        return None

    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        return np.array(fp)

    return None


def _bivariate_kdeplot(x, y, filled, fill_lowest,
                       kernel, bw, gridsize, cut, clip,
                       axlabel, cbar, cbar_ax, cbar_kws, ax, **kwargs):
    """Plot a joint KDE estimate as a bivariate contour plot."""
    # Determine the clipping
    if clip is None:
        clip = [(-np.inf, np.inf), (-np.inf, np.inf)]
    elif np.ndim(clip) == 1:
        clip = [clip, clip]

    # Calculate the KDE
    if sd._has_statsmodels:
        xx, yy, z = sd._statsmodels_bivariate_kde(x, y, bw, gridsize, cut, clip)
    else:
        xx, yy, z = sd._scipy_bivariate_kde(x, y, bw, gridsize, cut, clip)

    # Plot the contours
    n_levels = kwargs.pop("n_levels", 10)
    cmap = kwargs.get("cmap", "BuGn" if filled else "BuGn_d")
    if isinstance(cmap, string_types):
        if cmap.endswith("_d"):
            pal = ["#333333"]
            pal.extend(color_palette(cmap.replace("_d", "_r"), 2))
            cmap = blend_palette(pal, as_cmap=True)
        else:
            cmap = plt.cm.get_cmap(cmap)

    kwargs["cmap"] = cmap
    contour_func = ax.contourf if filled else ax.contour
    cset = contour_func(xx, yy, z, n_levels, **kwargs)
    if filled and not fill_lowest:
        cset.collections[0].set_alpha(0)
    kwargs["n_levels"] = n_levels

    if cbar:
        cbar_kws = {} if cbar_kws is None else cbar_kws
        ax.figure.colorbar(cset, cbar_ax, ax, **cbar_kws)

    # Label the axes
    if hasattr(x, "name") and axlabel:
        ax.set_xlabel(x.name)
    if hasattr(y, "name") and axlabel:
        ax.set_ylabel(y.name)

    return ax, cset


# monkey patching
sd._bivariate_kdeplot = _bivariate_kdeplot


def load_data():
    print('loading all data')
    pcqc = pd.read_csv('data/pcqc_allData.csv')
    TTA_SF_data = pd.read_csv('data/TTA_SF_allData.csv')
    return pcqc, TTA_SF_data


def plot_global_embedding(df_data: pd.DataFrame,
                          df_global: pd.DataFrame,
                          x_col: str,
                          y_col: str,
                          name: str,
                          title: str = "",
                          x_lim=None,
                          y_lim=None):
    plt.figure(num=1, figsize=(10, 8), dpi=300, clear=True)
    ax = sns.scatterplot(data=df_global,
                         x=x_col,
                         y=y_col,
                         color=(0.5, 0.5, 0.5),
                         s=10,
                         alpha=0.1,
                         edgecolor="none",
                         label='PCQC (global)')
    sns.scatterplot(data=df_data,
                    x=x_col,
                    y=y_col,
                    color=colors[3],
                    s=10,
                    alpha=0.1,
                    edgecolor="none",
                    label=name,
                    ax=ax)
    if x_lim:
        ax.set_xlim(x_lim)
    if y_lim:
        ax.set_ylim(y_lim)
    leg = plt.legend(markerscale=3, loc='lower right', fontsize=20)
    for lh in leg.legendHandles:
        lh.set_alpha(1)
    # plt.title(title, fontsize=20)
    plt.xlabel('UMAP-0', fontsize=14)
    plt.ylabel('UMAP-1', fontsize=14)
    plt.xticks()
    plt.yticks()
    plt.gca().tick_params(axis='x', label1On=False)
    plt.gca().tick_params(axis='y', label1On=False)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig('plots/umap_global_' + name + '.png')


def plot_global_embedding_multiple(df_data: List[pd.DataFrame],
                                   df_global: pd.DataFrame,
                                   x_col: str,
                                   y_col: str,
                                   name: str,
                                   names: List[str],
                                   title: str = "",
                                   x_lim=None,
                                   y_lim=None):
    plt.figure(num=1, clear=True, figsize=(10, 8), dpi=300)
    ax = sns.scatterplot(data=df_global,
                         x=x_col,
                         y=y_col,
                         color=(0.5, 0.5, 0.5),
                         s=10,
                         alpha=0.1,
                         edgecolor="none",
                         label='PCQC (global)')
    for i in tqdm(range(len(df_data))):
        ax = sns.scatterplot(data=df_data[i],
                             x=x_col,
                             y=y_col,
                             color=colors_nipy[i],
                             s=5,
                             alpha=0.1,
                             edgecolor="none",
                             label=name + ' ' + names[i])
    if x_lim:
        ax.set_xlim(x_lim)
    if y_lim:
        ax.set_ylim(y_lim)
    # leg = plt.legend(markerscale=3, bbox_to_anchor=(-.1, -0.05), loc='upper left', fontsize=16, ncol=3)
    leg = plt.legend(markerscale=3, loc='lower right', fontsize=16)
    # leg = plt.legend(markerscale=3, loc='upper left', fontsize=20, frameon=True)
    for lh in leg.legendHandles:
        lh.set_alpha(1)
    # plt.title(title, fontsize=20)
    plt.xlabel('UMAP-0', fontsize=14)
    plt.ylabel('UMAP-1', fontsize=14)
    plt.xticks()
    plt.yticks()
    plt.gca().tick_params(axis='x', label1On=False)
    plt.gca().tick_params(axis='y', label1On=False)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig('plots/umap_global_' + name + '.png')

    sd._bivariate_kdeplot = _bivariate_kdeplot

    plt.figure(num=2, clear=True, figsize=(10, 8), dpi=300)
    ax = sns.scatterplot(data=df_global,
                         x=x_col,
                         y=y_col,
                         color=(0.5, 0.5, 0.5),
                         s=10,
                         alpha=0.1,
                         edgecolor="none",
                         label='PCQC (global)')
    for i in tqdm(range(len(df_data))):
        ax = sns.kdeplot(data=df_data[i],
                         x=x_col,
                         y=y_col,
                         gridsize=100,
                         thresh=0.75,
                         levels=4,
                         color=colors_nipy[i],
                         label=name + ' ' + names[i])
    if x_lim:
        ax.set_xlim(x_lim)
    if y_lim:
        ax.set_ylim(y_lim)
    # leg = plt.legend(markerscale=3, bbox_to_anchor=(-.1, -0.05), loc='upper left', fontsize=16, ncol=3)
    leg = plt.legend(markerscale=3, loc='lower right', fontsize=16)
    for lh in leg.legendHandles:
        lh.set_alpha(1)

    # plt.title(title)
    # plt.title('Contour of ' + title, fontsize=20)
    plt.xlabel('UMAP-0', fontsize=14)
    plt.ylabel('UMAP-1', fontsize=14)
    plt.xticks()
    plt.yticks()
    plt.gca().tick_params(axis='x', label1On=False)
    plt.gca().tick_params(axis='y', label1On=False)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig('plots/umap_global_contour_' + name + '.png')


def plot_global_embedding_figure(df_data1: pd.DataFrame,
                                 df_data: List[pd.DataFrame],
                                 df_global: pd.DataFrame,
                                 x_col: str,
                                 y_col: str,
                                 name: str,
                                 names: List[str],
                                 title: str = "",
                                 x_lim=None,
                                 y_lim=None):
    df_data1[x_col] = df_data1[x_col] + 0.5
    fig = plt.figure(num=1, figsize=(14, 6.5), dpi=300, clear=True)
    ax = fig.add_subplot(1, 2, 1)
    sns.scatterplot(data=df_global,
                    x=x_col,
                    y=y_col,
                    color=(0.5, 0.5, 0.5),
                    s=10,
                    alpha=0.1,
                    edgecolor="none",
                    label='PCQC (global)')
    sns.scatterplot(data=df_data1,
                    x=x_col,
                    y=y_col,
                    color=colors[3],
                    s=10,
                    alpha=0.1,
                    edgecolor="none",
                    label=names[0],
                    ax=ax)
    if x_lim:
        ax.set_xlim(x_lim)
    if y_lim:
        ax.set_ylim(y_lim)
    leg = plt.legend(markerscale=3, loc='lower right', fontsize=20)
    for lh in leg.legendHandles:
        lh.set_alpha(1)
    # plt.title(title, fontsize=20)
    plt.xlabel('UMAP-0', fontsize=14)
    plt.ylabel('UMAP-1', fontsize=14)
    plt.xticks()
    plt.yticks()
    plt.gca().tick_params(axis='x', label1On=False)
    plt.gca().tick_params(axis='y', label1On=False)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()

    # plt.figure(num=1, clear=True, figsize=(10, 8), dpi=300)
    ax = fig.add_subplot(1, 2, 2)
    sns.scatterplot(data=df_global,
                    x=x_col,
                    y=y_col,
                    color=(0.5, 0.5, 0.5),
                    s=10,
                    alpha=0.1,
                    edgecolor="none",
                    label='PCQC (global)')
    for i in tqdm(range(len(df_data))):
        sns.scatterplot(data=df_data[i],
                        x=x_col,
                        y=y_col,
                        color=colors_nipy[i],
                        s=5,
                        alpha=0.1,
                        edgecolor="none",
                        label=names[i + 1])
    if x_lim:
        ax.set_xlim(x_lim)
    if y_lim:
        ax.set_ylim(y_lim)
    # leg = plt.legend(markerscale=3, bbox_to_anchor=(-.1, -0.05), loc='upper left', fontsize=16, ncol=3)
    leg = plt.legend(markerscale=3, loc='lower right', fontsize=16)
    # leg = plt.legend(markerscale=3, loc='upper left', fontsize=20, frameon=True)
    for lh in leg.legendHandles:
        lh.set_alpha(1)
    # plt.title(title, fontsize=20)
    plt.xlabel('UMAP-0', fontsize=14)
    plt.ylabel('', fontsize=14)
    plt.xticks()
    plt.yticks()
    plt.gca().tick_params(axis='x', label1On=False)
    plt.gca().tick_params(axis='y', label1On=False)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    label_axes_global(fig, ha='left')
    plt.savefig('plots/umap_global_' + name + '.png')


def cluster_error_analysis():
    plt.style.use(['science', 'grid'])
    pcqc = pd.read_csv('data/pcqc_xtb_error_notrain.csv')
    numBins = 101
    global_cluster = pcqc['global_cluster']
    hist, binEdge = np.histogram(global_cluster, bins=numBins, range=(0, max(global_cluster)))
    # print(hist, binEdge)
    inds = np.digitize(global_cluster, binEdge) - 1
    print(min(inds), max(inds))
    pcqc['global_cluster_nums'] = inds
    S1_MAEs = []
    T1_MAEs = []
    S1_RMSEs = []
    T1_RMSEs = []
    S1_MEs = []
    T1_MEs = []
    cluster_sizes = []
    for i in np.arange(numBins):
        df_temp = pcqc[pcqc['global_cluster_nums'] == i]
        S1err_temp = df_temp['S1err']
        T1err_temp = df_temp['T1err']
        S1_MAE = mean_absolute_error(np.zeros(len(S1err_temp)), S1err_temp)
        T1_MAE = mean_absolute_error(np.zeros(len(T1err_temp)), T1err_temp)
        S1_ME = np.mean(S1err_temp)
        T1_ME = np.mean(T1err_temp)
        S1_RMSE = mean_squared_error(np.zeros(len(S1err_temp)), S1err_temp, squared=False)
        T1_RMSE = mean_squared_error(np.zeros(len(T1err_temp)), T1err_temp, squared=False)
        S1_MAEs.append(S1_MAE)
        T1_MAEs.append(T1_MAE)
        S1_RMSEs.append(S1_RMSE)
        T1_RMSEs.append(T1_RMSE)
        S1_MEs.append(S1_ME)
        T1_MEs.append(T1_ME)
        cluster_sizes.append(len(df_temp))
    # print(S1_MAEs, T1_MAEs)

    my_cmap = plt.get_cmap("nipy_spectral")
    rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))

    fig = plt.figure(num=2, figsize=[8, 8], dpi=300, clear=True)
    ax = fig.add_subplot(2, 2, 1)
    ax.set_axisbelow(True)
    x = np.arange(numBins)
    plt.bar(x, S1_MEs, color=my_cmap(rescale(x)), width=1.01, alpha=1)
    plt.xlabel('Cluster num', fontsize=16)
    plt.ylabel('xTB error (eV)', fontsize=16)
    # plt.title('S$_1$ ME', fontsize=16)
    plt.annotate('S$_1$ ME', xy=(77, -0.34), bbox=dict(facecolor='white', alpha=1, edgecolor='white'), fontsize=16)
    ax.set_box_aspect(1)
    plt.tight_layout()
    # plt.savefig('plots/cluster_ME_S1.png')

    ax = fig.add_subplot(2, 2, 2)
    ax.set_axisbelow(True)
    x = np.arange(numBins)
    plt.bar(x, T1_MEs, color=my_cmap(rescale(x)), width=1.01, alpha=1)
    plt.xlabel('Cluster num', fontsize=16)
    plt.ylabel('xTB error (eV)', fontsize=16)
    # plt.title('T$_1$ ME', fontsize=16)
    plt.annotate('T$_1$ ME', xy=(77, -0.75), bbox=dict(facecolor='white', alpha=1, edgecolor='white'), fontsize=16)
    ax.set_box_aspect(1)
    plt.tight_layout()
    # plt.savefig('plots/cluster_ME_T1.png')

    ax = fig.add_subplot(2, 2, 3)
    ax.set_axisbelow(True)
    x = np.arange(numBins)
    plt.bar(x, S1_MAEs, color=my_cmap(rescale(x)), width=1.01, alpha=1)
    plt.xlabel('Cluster num', fontsize=16)
    plt.ylabel('xTB error (eV)', fontsize=16)
    # plt.title('S$_1$ MAE', fontsize=16)
    plt.annotate('S$_1$ MAE', xy=(73, 0.59), bbox=dict(facecolor='white', alpha=1, edgecolor='white'), fontsize=16)
    ax.set_box_aspect(1)
    plt.tight_layout()
    # plt.savefig('plots/cluster_MAE_S1.png')

    ax = fig.add_subplot(2, 2, 4)
    ax.set_axisbelow(True)
    x = np.arange(numBins)
    plt.bar(x, T1_MAEs, color=my_cmap(rescale(x)), width=1.01, alpha=1)
    plt.xlabel('Cluster num', fontsize=16)
    plt.ylabel('xTB error (eV)', fontsize=16)
    # plt.title('T$_1$ MAE', fontsize=16)
    plt.annotate('T$_1$ MAE', xy=(73, 0.9), bbox=dict(facecolor='white', alpha=1, edgecolor='white'), fontsize=16)
    ax.set_box_aspect(1)
    plt.tight_layout()
    # plt.savefig('plots/cluster_MAE_T1.png')

    label_axes_cluster(fig, ha='left')

    plt.savefig('plots/cluster_errors.pdf')

    fig = plt.figure(num=3, figsize=[4, 4], dpi=300, clear=True)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_axisbelow(True)
    x = np.arange(numBins)
    plt.bar(x, cluster_sizes, color=my_cmap(rescale(x)), width=1, alpha=0.9)
    plt.xlabel('Cluster num')
    plt.ylabel('Size')
    plt.title('Cluster Sizes')
    ax.set_box_aspect(1)
    plt.tight_layout()
    # plt.savefig('plots/cluster_sizes.png')


def load_data_multiple():
    print('loading all data')
    pcqc = pd.read_csv('data/pcqc_allData.csv')
    all_AL_adds = []
    for index, i in enumerate(['S1', 'T1']):
        all_AL_adds.append(pd.read_csv('data/TDDFT_SCOP_AL_exp_' + str(i) + '_allData.csv'))
    return pcqc, all_AL_adds


def load_data_with_color():
    print('loading all data')
    pcqc = pd.read_csv('data/pcqc_xtb_error_notrain.csv')
    return pcqc


def generate_umap(pcqc):
    print('generating umap')
    pcqc_ecfp_descriptors, pcqc_keep_idx = compute_ecfp_descriptors(pcqc["smiles"])
    np.save('pcqc_ecfp_descriptors.npy', pcqc_ecfp_descriptors)
    np.save('pcqc_keep_idx.npy', pcqc_keep_idx)
    pcqc_ecfp_descriptors = np.load('pcqc_ecfp_descriptors.npy')
    pcqc_keep_idx = np.load('pcqc_keep_idx.npy')
    pcqc = pcqc.iloc[pcqc_keep_idx]
    umap_model_global = umap.UMAP(metric="jaccard",
                                  n_neighbors=100,
                                  n_components=2,
                                  low_memory=False,
                                  min_dist=0.5)
    X_umap_global = umap_model_global.fit_transform(pcqc_ecfp_descriptors)
    # pcqc.to_csv('PCQC_350k.csv')
    np.save('X_umap_global.npy', X_umap_global)
    joblib.dump(umap_model_global, 'umap_model_global.sav')
    return pcqc


def do_clustering(pcqc):
    print('doing clustering')
    # umap_model_global = joblib.load('umap_model_global.sav')
    # pcqc_ecfp_descriptors = np.load('pcqc_ecfp_descriptors.npy')
    # X_umap_global = umap_model_global.transform(pcqc_ecfp_descriptors)
    X_umap_global = np.load('X_umap_global.npy')
    # X_umap_global = list(X_umap_global)
    # print(type(X_umap_global))

    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=10,
        min_samples=10,
        prediction_data=True,
        cluster_selection_method="leaf",
        memory='/Users/shomikverma/Documents/Imperial/Databases/SCOP_DB/pub_plots/tmp',
        core_dist_n_jobs=1)

    # hdbscan_model = hdbscan.HDBSCAN()

    pcqc["global_cluster"] = hdbscan_model.fit_predict(X_umap_global)

    return pcqc


def plot_global_embedding_with_data_multi(df: pd.DataFrame,
                                          x_col: str,
                                          y_col: str,
                                          name: str,
                                          title: str = "",
                                          df_global: pd.DataFrame = None,
                                          x_lim=None,
                                          y_lim=None):
    """ Plots data colored by dataset
    """
    fig = plt.figure(num=1, figsize=(8, 6), dpi=300, clear=True)
    ax = fig.add_subplot(2, 2, 1)
    data_col = 'S1err'
    plt.scatter(df[x_col], df[y_col], c=df[data_col], edgecolor='none',
                alpha=0.1, cmap='nipy_spectral', vmin=-0.5, vmax=0.5, s=5)
    if x_lim:
        ax.set_xlim(x_lim)
    if y_lim:
        ax.set_ylim(y_lim)
    ax.grid(False)
    ticks = np.linspace(-0.5, 0.5, 9).tolist()
    ticksstr = list(map(str, np.linspace(-0.5, 0.5, 9).tolist()))
    # ticksstr[0] = '$<$' + str(vmin)
    # ticksstr[-1] = '$>$' + str(vmax)
    cb = plt.colorbar(ticks=ticks, fraction=0.042, pad=0.04)
    cb.set_ticklabels(ticksstr)
    cb.set_label('S$_1$ xTB-sTDA Error', fontsize=16)
    cb.set_alpha(1)
    cb.draw_all()
    # ax.get_legend().remove()
    # plt.title(title)
    plt.xlabel('UMAP-0', fontsize=12)
    plt.ylabel('UMAP-1', fontsize=12)
    plt.xticks()
    plt.yticks()
    plt.gca().tick_params(axis='x', label1On=False)
    plt.gca().tick_params(axis='y', label1On=False)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()

    ax = fig.add_subplot(2, 2, 2)
    data_col = 'T1err'
    plt.scatter(df[x_col], df[y_col], c=df[data_col], edgecolor='none',
                alpha=0.1, cmap='nipy_spectral', vmin=-1.0, vmax=0.0, s=5)
    if x_lim:
        ax.set_xlim(x_lim)
    if y_lim:
        ax.set_ylim(y_lim)
    ax.grid(False)
    ticks = np.linspace(-1.0, 0.0, 9).tolist()
    ticksstr = list(map(str, np.linspace(-1.0, 0.0, 9).tolist()))
    # ticksstr[0] = '$<$' + str(vmin)
    # ticksstr[-1] = '$>$' + str(vmax)
    cb = plt.colorbar(ticks=ticks, fraction=0.042, pad=0.04)
    cb.set_ticklabels(ticksstr)
    cb.set_label('T$_1$ xTB-sTDA Error', fontsize=16)
    cb.set_alpha(1)
    cb.draw_all()
    # ax.get_legend().remove()
    # plt.title(title)
    plt.xlabel('UMAP-0', fontsize=12)
    plt.ylabel('UMAP-1', fontsize=12)
    plt.xticks()
    plt.yticks()
    plt.gca().tick_params(axis='x', label1On=False)
    plt.gca().tick_params(axis='y', label1On=False)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()

    ax = fig.add_subplot(2, 2, 3)
    data_col = 'global_cluster'
    clustered = df[data_col].values >= 0
    print(len(df.iloc[clustered]))
    plt.scatter(df.iloc[clustered][x_col], df.iloc[clustered][y_col], c=df.iloc[clustered][data_col],
                alpha=0.1, edgecolor='none', cmap='nipy_spectral', s=5)
    if x_lim:
        ax.set_xlim(x_lim)
    if y_lim:
        ax.set_ylim(y_lim)
    ax.grid(False)
    norm = mpl.colors.Normalize(vmin=0, vmax=100)
    sm = plt.cm.ScalarMappable(norm=norm, cmap="nipy_spectral")
    sm.set_array([])
    cb = ax.figure.colorbar(sm, fraction=0.042, pad=0.04)
    cb.set_label('Cluster', fontsize=16)
    cb.set_alpha(1)
    cb.draw_all()
    # ax.get_legend().remove()
    # plt.title(title)
    plt.xlabel('UMAP-0', fontsize=12)
    plt.ylabel('UMAP-1', fontsize=12)
    plt.xticks()
    plt.yticks()
    plt.gca().tick_params(axis='x', label1On=False)
    plt.gca().tick_params(axis='y', label1On=False)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()

    # plt.title(title)
    # plt.show()
    # plt.savefig('plots/data_' + name + '.png', pad_inches=0.2, bbox_inches='tight')

    numBins = 101
    global_cluster = pcqc['global_cluster']
    hist, binEdge = np.histogram(global_cluster, bins=numBins, range=(0, max(global_cluster)))
    # print(hist, binEdge)
    inds = np.digitize(global_cluster, binEdge) - 1
    print(min(inds), max(inds))
    pcqc['global_cluster_nums'] = inds
    cluster_sizes = []
    for i in np.arange(numBins):
        df_temp = pcqc[pcqc['global_cluster_nums'] == i]
        cluster_sizes.append(len(df_temp))
    my_cmap = plt.get_cmap("nipy_spectral")
    rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))

    ax = fig.add_subplot(2, 2, 4)
    ax.set_axisbelow(True)
    x = np.arange(numBins)
    plt.bar(x, cluster_sizes, color=my_cmap(rescale(x)), width=1, alpha=0.9)
    plt.xlabel('Cluster num', fontsize=14)
    plt.ylabel('Size (num. molecules)', fontsize=16)
    # plt.title('Cluster Sizes')
    ax.set_box_aspect(1)
    plt.tight_layout()

    label_axes_cluster(fig, ha='left')

    plt.savefig('plots/data_pcqc_xtb_error.pdf', bbox_inches='tight')


def save_data(pcqc):
    print('writing data to file')
    pcqc.to_csv('data/pcqc_xtb_error.csv', index=False)


pcqc, TTA_SF_data = load_data()
_, all_AL_adds = load_data_multiple()
# pcqc = generate_umap(pcqc)
# pcqc = do_clustering(pcqc)
# save_data(pcqc)


buffer_space = 0.5  # Extra space around the edges
x_lim = pcqc["UMAP_0"].min() - buffer_space, pcqc["UMAP_0"].max() + buffer_space
y_lim = pcqc["UMAP_1"].min() - buffer_space, pcqc["UMAP_1"].max() + buffer_space

# plot_global_embedding(TTA_SF_data,
#                       pcqc,
#                       x_col="UMAP_0",
#                       y_col="UMAP_1",
#                       x_lim=x_lim,
#                       y_lim=y_lim,
#                       name='SCOP-PCQC',
#                       title="Global Embedding of SCOP-PCQC Dataset")
#
# plot_global_embedding_multiple(all_AL_adds,
#                                pcqc,
#                                x_col="UMAP_0",
#                                y_col="UMAP_1",
#                                x_lim=x_lim,
#                                y_lim=y_lim,
#                                name='SCOP-AL-Exp',
#                                names=['S$_1$', 'T$_1$'],
#                                title="Global Embedding of SCOP-PCQC AL Expansions")

plot_global_embedding_figure(TTA_SF_data, all_AL_adds, pcqc,
                             x_col="UMAP_0",
                             y_col="UMAP_1",
                             x_lim=x_lim,
                             y_lim=y_lim,
                             name='SCOP-PCQC-AL-Exp',
                             names=['SCOP-PCQC', 'SCOP-AL-Exp S$_1$', 'SCOP-AL-Exp T$_1$'],
                             title="Global Embedding of Training Datasets"
                             )

pcqc = load_data_with_color()

plot_global_embedding_with_data_multi(pcqc,
                                      x_col='UMAP_0',
                                      y_col='UMAP_1',
                                      x_lim=x_lim,
                                      y_lim=y_lim,
                                      name='PCQC-xTB_Error')

cluster_error_analysis()

import os
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib as mpl
import string
from itertools import cycle
from cycler import cycler
import numpy as np
import time


def label_axes(fig, labels=None, loc=None, **kwargs):
    if labels is None:
        labels = string.ascii_lowercase
    labels = cycle(labels)
    if loc is None:
        loc = (-15, 200)
    for ax, lab in zip(fig.axes, labels):
        ax.annotate('(' + lab + ')', size=14, xy=loc,
                    xycoords='axes points',
                    **kwargs)


plt.style.use(['science', 'grid'])
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
colors = [colors[3], colors[1], colors[0], colors[2]]
colors_nipy1 = mpl.cm.nipy_spectral(np.linspace(0.1, 0.9, 6))
colors_nipy2 = mpl.cm.nipy_spectral(np.linspace(0.6, 0.9, 7))
colors_nipy = list(colors_nipy1[0:3]) + list(colors_nipy2[3:-2]) + list(colors_nipy1[-1:])
plt.rcParams['axes.prop_cycle'] = cycler(color=colors)


def load_data(filename):
    df_CC2 = pd.read_csv('TDDFT_' + filename + '.csv')
    CC2_col = 'S1-CC2'
    df_CC2 = df_CC2[0 < df_CC2[CC2_col]][df_CC2[CC2_col] < 10]
    df_xtb = pd.read_csv('xTB_' + filename + '.csv')
    xtb_col = 'xtbS1'
    df_xtb = df_xtb[0 < df_xtb[xtb_col]][df_xtb[xtb_col] < 10]
    df_all = df_xtb.merge(df_CC2, on='SMILES', how='inner')
    return df_all


def write_training_test_data(df, training_size):
    df['S1err'] = df['S1-CC2'] - df['xtbS1']
    df = df[['SMILES', 'S1err']]
    df_train = df.sample(int(training_size), random_state=1)
    df_test = df[~df.apply(tuple, 1).isin(df_train.apply(tuple, 1))]
    print(df_train.info(), df_test.info())
    df_train.to_csv('xTB_CC_train_' + training_size + '.csv', index=False)
    df_test.to_csv('xTB_CC_test_' + training_size + '.csv', index=False)


def do_ML(training_size):
    os.system('chemprop_train --data_path xTB_CC_train_' + training_size +
              '.csv --dataset_type regression --save_dir xTB_CC_ML_model_' + training_size +
              ' --split_type cv-no-test --save_smiles_splits --num_folds 20 --target_columns S1err')
    os.system('chemprop_predict --test_path xTB_CC_test_' + training_size +
              '.csv --checkpoint_dir xTB_CC_ML_model_' + training_size +
              ' --preds_path xTB_CC_preds_' + training_size + '.csv --drop_extra_columns')


def analyze_ML(df_all, training_size):
    df_preds = pd.read_csv('xTB_CC_preds_' + training_size + '.csv')
    df_preds['S1err'] = pd.to_numeric(df_preds['S1err'], errors='coerce')
    df_preds.dropna(subset=["S1err"], inplace=True)
    df_comp = df_all.merge(df_preds, how='inner')
    df_comp['xtb_ML'] = df_comp['xtbS1'] + df_comp['S1err']
    print(df_comp.info())

    CCS1s = df_comp['S1-CC2']
    xtbS1s = df_comp['xtbS1']
    xtbMLS1s = df_comp['xtb_ML']
    TD_PBE_SVP = df_comp['S1-PBE0-def2SVP-eV']
    TD_PBE_TZVP = df_comp['S1-PBE0-def2TZVP-eV']
    TD_CAM_TZVP = df_comp['S1-CAM-def2TZVP-eV']
    r2_orig = r2_score(CCS1s, xtbS1s)
    r2_ML = r2_score(CCS1s, xtbMLS1s)
    r2_PBE_SVP = r2_score(CCS1s, TD_PBE_SVP)
    r2_PBE_TZVP = r2_score(CCS1s, TD_PBE_TZVP)
    r2_CAM_TZVP = r2_score(CCS1s, TD_CAM_TZVP)
    MAE_orig = mean_absolute_error(CCS1s, xtbS1s)
    MAE_ML = mean_absolute_error(CCS1s, xtbMLS1s)
    MAE_PBE_SVP = mean_absolute_error(CCS1s, TD_PBE_SVP)
    MAE_PBE_TZVP = mean_absolute_error(CCS1s, TD_PBE_TZVP)
    MAE_CAM_TZVP = mean_absolute_error(CCS1s, TD_CAM_TZVP)

    print('S1')
    print('r2 orig:', r2_orig)
    print('r2 ML:', r2_ML)
    print('r2 PBE SVP:', r2_PBE_SVP)
    print('r2 PBE TZVP:', r2_PBE_TZVP)
    print('r2 CAM TZVP:', r2_CAM_TZVP)
    print('MAE orig:', MAE_orig)
    print('MAE ML:', MAE_ML)
    print('MAE PBE SVP:', MAE_PBE_SVP)
    print('MAE PBE TZVP:', MAE_PBE_TZVP)
    print('MAE CAM TZVP:', MAE_CAM_TZVP)

    # with open('comp_train_size.csv', 'a') as file:
    #     file.write(str(training_size) + ',' + str(MAE_ML) + ',' + str(MAE_PBE_TZVP) + ',' + str(MAE_CAM_TZVP) + '\n')

    fig = plt.figure(num=1, clear=True, figsize=[7, 7], dpi=300)
    ax = fig.add_subplot(2, 2, 1)
    ax.set_axisbelow(True)
    x = np.linspace(0, 10, 100)
    plt.plot(xtbS1s, CCS1s, '.', alpha=0.2, color=colors_nipy[-1], markeredgewidth=0, label='xTB')
    plt.plot(xtbMLS1s, CCS1s, '.', alpha=0.2, color=colors_nipy[2], markeredgewidth=0, label='xTB-CC-ML')
    plt.plot(x, x, 'k--')
    plt.grid(True)
    leg = plt.legend(markerscale=2, loc='upper left', fontsize=12)
    for lh in leg.legendHandles:
        lh.set_alpha(1)
    plt.xlabel('xTB-sTDA S$_1$ (eV)', fontsize=16)
    plt.ylabel('CC2 S$_1$ (eV)', fontsize=16)
    # plt.title('CC2 vs. xTB-sTDA S1')
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.annotate('R$^2$ xTB: %0.2f\n' % r2_orig +
                 'R$^2$ xTB-CC-ML: %0.2f\n' % r2_ML +
                 'MAE xTB: %0.2f\n' % MAE_orig +
                 'MAE xTB-CC-ML: %0.2f' % MAE_ML,
                 (9.5, 0.5),
                 bbox=dict(facecolor='white', alpha=0.5),
                 fontsize=12,
                 ha='right')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()

    ax = fig.add_subplot(2, 2, 2)
    ax.set_axisbelow(True)
    x = np.linspace(0, 10, 100)
    plt.plot(TD_PBE_TZVP, CCS1s, '.', color=colors_nipy[1], alpha=0.2, markeredgewidth=0, label='PBE0-TZVP')
    plt.plot(TD_CAM_TZVP, CCS1s, '.', color=colors_nipy[4], alpha=0.2, markeredgewidth=0, label='CAM-TZVP')
    plt.plot(x, x, 'k--')
    plt.grid(True)
    leg = plt.legend(markerscale=2, loc='upper left', fontsize=12)
    for lh in leg.legendHandles:
        lh.set_alpha(1)
    plt.xlabel('TDDFT S$_1$ (eV)', fontsize=16)
    plt.ylabel('CC2 S$_1$ (eV)', fontsize=16)
    # plt.title('CC2 vs. TD-DFT S1')
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.annotate('R$^2$ PBE0-TZVP: %0.2f\n' % r2_PBE_TZVP +
                 'R$^2$ CAM-TZVP: %0.2f\n' % r2_CAM_TZVP +
                 'MAE PBE0-TZVP: %0.2f\n' % MAE_PBE_TZVP +
                 'MAE CAM-TZVP: %0.2f' % MAE_CAM_TZVP,
                 (9.5, 0.5),
                 bbox=dict(facecolor='white', alpha=0.5),
                 fontsize=12,
                 ha='right')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()

    # plt.savefig('comp_xTB_CC_TDDFT_' + training_size + '.png')

    df = pd.read_csv('comp_train_size.csv')
    # fig = plt.figure(num=2)
    ax = fig.add_subplot(2, 1, 2)
    plt.plot(df['train_size'], df['MAE_ML'], '.-', color=colors_nipy[-1], label='xTB-CC-ML')
    plt.plot(df['train_size'], df['MAE_PBE'], '-', color=colors_nipy[2], label='PBE0/def2-TZVP')
    plt.plot(df['train_size'], df['MAE_CAM'], '-', color=colors_nipy[1], label='CAM-B3LYP/def2-TZVP')
    plt.legend(markerscale=2, fontsize=12)
    plt.xlabel('Training size', fontsize=16)
    plt.ylabel('MAE', fontsize=16)
    # plt.title('MAE vs. training size')
    ax.set_box_aspect(1)
    plt.tight_layout()
    label_axes(fig, ha='right')
    fig.subplots_adjust(hspace=0.3)
    plt.savefig('comp_xTB_CC_ML_' + training_size + '.pdf')

    fig = plt.figure(num=3, clear=True, figsize=[7, 4], dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_axisbelow(True)
    x = np.linspace(0, 10, 100)
    plt.plot(xtbS1s, CCS1s, '.', color=colors[0], alpha=0.2, markeredgewidth=0, label='xTB')
    plt.plot(TD_PBE_TZVP, CCS1s, '.', color=colors[2], alpha=0.2, markeredgewidth=0, label='PBE0-TZVP')
    plt.plot(TD_CAM_TZVP, CCS1s, '.', color=colors[1], alpha=0.2, markeredgewidth=0, label='CAM-TZVP')
    plt.plot(x, x, 'k--')
    plt.grid(True)
    leg = plt.legend(loc='upper left')
    for lh in leg.legendHandles:
        lh.set_alpha(1)
    plt.xlabel('TDDFT/sTDA S1 (eV)')
    plt.ylabel('CC2 S1 (eV)')
    plt.title('CC2 vs. TDDFT/sTDA S1')
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.annotate('R2 xTB-sTDA: %0.2f\n' % r2_orig +
                 'R2 PBE0-TZVP: %0.2f\n' % r2_PBE_TZVP +
                 'R2 CAM-TZVP: %0.2f\n' % r2_CAM_TZVP +
                 'MAE xTB-sTDA: %0.2f\n' % MAE_orig +
                 'MAE PBE0-TZVP: %0.2f\n' % MAE_PBE_TZVP +
                 'MAE CAM-TZVP: %0.2f' % MAE_CAM_TZVP,
                 (9.5, 0.5),
                 bbox=dict(facecolor='white', alpha=0.5),
                 ha='right')
    plt.gca().set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig('comp_xTB_TDDFT_CC2_' + training_size + '.png')


def plot_improvement():
    df = pd.read_csv('comp_train_size.csv')
    fig = plt.figure(num=2, clear=True, figsize=[4, 4], dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(df['train_size'], df['MAE_ML'], '.-', color=colors_nipy[-1], label='xTB-CC-ML')
    plt.plot(df['train_size'], df['MAE_PBE'], '-', color=colors_nipy[2], label='PBE0/def2-TZVP')
    plt.plot(df['train_size'], df['MAE_CAM'], '-', color=colors_nipy[1], label='CAM-B3LYP/def2-TZVP')
    plt.legend(markerscale=2, fontsize=12)
    plt.xlabel('Training size', fontsize=16)
    plt.ylabel('MAE', fontsize=16)
    # plt.title('MAE vs. training size')
    plt.tight_layout()
    plt.savefig('comp_ML_train_size.png')


def xtb_tddft_plot(df_all):
    pass


# write_training_test_data(df_all, 10_000)
# do_ML(10_000)
# analyze_ML(df_all, '2')
# with open('comp_train_size.csv', 'w') as file:
#     file.write('train_size,MAE_ML,MAE_PBE,MAE_CAM\n')
# training_sizes = [100, 500, 1000, 2500, 5000, 7500, 10000, 15000]
# training_sizes = [100]
training_sizes = [10000]
for training_size in training_sizes:
    df_all = load_data('QM8')
    analyze_ML(df_all, str(training_size))

# plot_improvement()

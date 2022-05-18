import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import json
import sys
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn import linear_model
from cycler import cycler
import string
from itertools import cycle
import pandas as pd


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


plt.style.use(['science', 'grid'])
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
colors = [colors[3], colors[1], colors[0]]
colors_nipy1 = mpl.cm.nipy_spectral(np.linspace(0.1, 0.9, 6))
colors_nipy2 = mpl.cm.nipy_spectral(np.linspace(0.6, 0.9, 7))
colors_nipy = list(colors_nipy1[0:3]) + list(colors_nipy2[3:-2]) + list(colors_nipy1[-1:])
plt.rcParams['axes.prop_cycle'] = cycler(color=colors)


def gen_train_data():
    df = pd.read_csv('combined_train_data.csv')
    df.to_csv('comb_train_data.csv', columns=['SMILES', 'TDDFT_S1', 'TDDFT_T1'], index=False)
    df.to_csv('comb_featu_data.csv', columns=['xTB_S1', 'xTB_T1'], index=False)


def train_ML():
    os.system('chemprop_train --data_path comb_train_data.csv --features_path comb_featu_data.csv '
              '--dataset_type regression --save_dir xTB_ML_model_comb '
              '--target_columns TDDFT_S1 TDDFT_T1')


def gen_test_data(testType):
    if os.path.isfile('xTB_' + testType + '.csv'):
        os.system('cp xTB_' + testType + '.csv ' + testType + '_xTB.csv')
    if not os.path.isdir('data'):
        os.mkdir('data')
    os.system('mv xTB_' + testType + '.csv data')
    with open(testType + '_xTB.csv', 'r') as file:
        dataXTB = pd.read_csv(file)
    dataXTB.rename({'xtbT1': 'xtb_T1', 'xtbS1': 'xtb_S1'}, axis='columns', inplace=True)
    dataXTB.to_csv(testType + '_SMILES.csv', columns=['SMILES'], index=False)
    dataXTB.to_csv(testType + '_features.csv', columns=['xtb_T1', 'xtb_S1'], index=False)


def predict_ML(testType):
    if os.path.isfile(testType + '_preds.csv'):
        return
    os.system('chemprop_predict --test_path ' + testType + '_SMILES.csv --features_path ' + testType +
              '_features.csv --checkpoint_dir xTB_ML_model_comb ' + ' --preds_path ' + testType + '_preds.csv --drop_extra_columns')


def evaluate_predictions(testType):
    df_TDDFT = pd.read_csv(testType + '_TDDFT.csv')
    df_TDDFT = df_TDDFT[['SMILES', 'S1', 'T1']]
    df_TDDFT.rename({'S1': 'TDDFT_S1', 'T1': 'TDDFT_T1'}, axis='columns', inplace=True)
    df_preds = pd.read_csv(testType + '_preds.csv')
    df_preds.rename({'TDDFT_S1': 'pred_S1', 'TDDFT_T1': 'pred_T1'}, axis='columns', inplace=True)
    df_xTB = pd.read_csv(testType + '_xTB.csv')
    df_xTB.rename({'xtbT1': 'xTB_T1', 'xtbS1': 'xTB_S1'}, axis='columns', inplace=True)
    df_all = df_preds.merge(df_TDDFT, on='SMILES').merge(df_xTB, on='SMILES')

    r2_S1 = r2_score(df_all['TDDFT_S1'], df_all['xTB_S1'])
    r2_T1 = r2_score(df_all['TDDFT_S1'], df_all['xTB_T1'])
    MAE_S1 = mean_absolute_error(df_all['TDDFT_S1'], df_all['xTB_S1'])
    MAE_T1 = mean_absolute_error(df_all['TDDFT_T1'], df_all['xTB_T1'])
    # r2_lin_S1 = r2_score(df_all['TDDFT_S1'], df_all['xTB_Lin_S1'])
    # r2_lin_T1 = r2_score(df_all['TDDFT_T1'], df_all['xTB_Lin_T1'])
    # MAE_lin_S1 = mean_absolute_error(df_all['TDDFT_S1'], df_all['xTB_Lin_S1'])
    # MAE_lin_T1 = mean_absolute_error(df_all['TDDFT_T1'], df_all['xTB_Lin_T1'])
    r2_ML_S1 = r2_score(df_all['TDDFT_S1'], df_all['pred_S1'])
    r2_ML_T1 = r2_score(df_all['TDDFT_T1'], df_all['pred_T1'])
    MAE_ML_S1 = mean_absolute_error(df_all['TDDFT_S1'], df_all['pred_S1'])
    MAE_ML_T1 = mean_absolute_error(df_all['TDDFT_T1'], df_all['pred_T1'])
    print(testType)
    print('stda')
    print(r2_S1, r2_T1)
    print(MAE_S1, MAE_T1)
    print('ML')
    print(r2_ML_S1, r2_ML_T1)
    print(MAE_ML_S1, MAE_ML_T1)

    fig = plt.figure(num=1, clear=True, figsize=[7, 4],  dpi=300)

    ax = fig.add_subplot(121)
    plt.plot(df_all['xTB_S1'], df_all['TDDFT_S1'], '.', color=colors[0], markersize=1, label='xTB-sTDA')
    plt.plot(df_all['pred_S1'], df_all['TDDFT_S1'], '.', color=colors[2], markersize=1, label='xTB-ML')
    x = np.linspace(0, 9, 100)
    plt.plot(x, x, 'k--')
    plt.xlim(0, 9)
    plt.ylim(0, 9)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('xTB-sTDA S1 (eV)')
    plt.ylabel('TDDFT S1 (eV)')
    plt.legend()
    plt.annotate('R2 orig: %0.2f\n' % r2_S1 +
                 #  'R2 lin: %0.2f\n' % r2_lin_S1 +
                 'R2 ML: %0.2f\n' % r2_ML_S1 +
                 'MAE orig: %0.2f\n' % MAE_S1 +
                 #  'MAE lin: %0.2f\n' % MAE_lin_S1 +
                 'MAE ML: %0.2f' % MAE_ML_S1,
                 (8.5, 0.5),
                 bbox=dict(facecolor='white', alpha=0.5),
                 ha='right')
    plt.tight_layout()

    ax = fig.add_subplot(122)
    plt.plot(df_all['xTB_T1'], df_all['TDDFT_T1'], '.', color=colors[0], markersize=1, label='xTB-sTDA')
    plt.plot(df_all['pred_T1'], df_all['TDDFT_T1'], '.', color=colors[2], markersize=1, label='xTB-ML')
    plt.plot(x, x, 'k--')
    plt.xlim(0, 9)
    plt.ylim(0, 9)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('xTB-sTDA T1 (eV)')
    plt.ylabel('TDDFT T1 (eV)')
    plt.legend()
    plt.annotate('R2 orig: %0.2f\n' % r2_T1 +
                 #  'R2 lin: %0.2f\n' % r2_lin_T1 +
                 'R2 ML: %0.2f\n' % r2_ML_T1 +
                 'MAE orig: %0.2f\n' % MAE_T1 +
                 #  'MAE lin: %0.2f\n' % MAE_lin_T1 +
                 'MAE ML: %0.2f' % MAE_ML_T1,
                 (8.5, 0.5),
                 bbox=dict(facecolor='white', alpha=0.5),
                 ha='right')
    plt.tight_layout()

    plt.savefig(testType+'_xTB_ML.png')


# gen_train_data()
# train_ML()
# gen_test_data('Photosensitizers_DA')
# predict_ML('Photosensitizers_DA')
# gen_test_data('Photosensitizers_DAD')
# predict_ML('Photosensitizers_DAD')
evaluate_predictions('Photosensitizers_DA')
evaluate_predictions('Photosensitizers_DAD')

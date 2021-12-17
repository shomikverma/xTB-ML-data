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


def analyze_preds():
    df = pd.read_csv('SCOP_PCQC_all_preds.csv')
    S1xtbs = df['xtb_S1']
    S1xtb_Lins = df['lin_S1_preds']
    S1xtb_MLs = df['C2_S1_preds']
    S1TDDFTs = df['TDDFT_S1']
    T1xtbs = df['xtb_T1']
    T1xtb_Lins = df['lin_T1_preds']
    T1xtb_MLs = df['C2_T1_preds']
    T1TDDFTs = df['TDDFT_T1']

    fig = plt.figure(num=2, figsize=[8, 8], dpi=300, clear=True)
    ax = fig.add_subplot(2, 2, 1)
    plt.plot(S1xtbs, S1TDDFTs, '.', markersize=2, color=colors_nipy[-1], label='orig')
    plt.plot(S1xtb_Lins, S1TDDFTs, '.', markersize=2, color=colors_nipy[2], label='lin calib')
    plt.plot(S1xtb_MLs, S1TDDFTs, '.', markersize=2, color=colors_nipy[1], label='ML calib')
    x = np.linspace(0, 10, 100)
    plt.plot(x, x, 'k--')
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    ax.set_axisbelow(True)
    plt.grid(True)
    ax.set_aspect('equal', adjustable='box')
    plt.legend(markerscale=6, fontsize=14)
    plt.xlabel('xTB-sTDA S$_1$ (eV)', fontsize=16)
    plt.ylabel('TDDFT S$_1$ (eV)', fontsize=16)
    # plt.title('TDDFT vs. xTB S1 (eV)')
    plt.annotate('R$^2$ orig: %0.2f\n' % r2_score(S1TDDFTs, S1xtbs) +
                 'R$^2$ lin: %0.2f\n' % r2_score(S1TDDFTs, S1xtb_Lins) +
                 'R$^2$ ML: %0.2f\n' % r2_score(S1TDDFTs, S1xtb_MLs) +
                 'MAE orig: %0.2f\n' % mean_absolute_error(S1TDDFTs, S1xtbs) +
                 'MAE lin: %0.2f\n' % mean_absolute_error(S1TDDFTs, S1xtb_Lins) +
                 'MAE ML: %0.2f' % mean_absolute_error(S1TDDFTs, S1xtb_MLs),
                 (9.5, 0.5),
                 bbox=dict(facecolor='white', alpha=0.5),
                 fontsize=14,
                 ha='right')
    plt.tight_layout()

    ax = fig.add_subplot(2, 2, 2)
    plt.plot(T1xtbs, T1TDDFTs, '.', markersize=2, color=colors_nipy[-1], label='orig')
    plt.plot(T1xtb_Lins, T1TDDFTs, '.', markersize=2, color=colors_nipy[2], label='lin calib')
    plt.plot(T1xtb_MLs, T1TDDFTs, '.', markersize=2, color=colors_nipy[1], label='ML calib')
    x = np.linspace(0, 10, 100)
    plt.plot(x, x, 'k--')
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    ax.set_axisbelow(True)
    plt.grid(True)
    ax.set_aspect('equal', adjustable='box')
    plt.legend(markerscale=6, fontsize=14)
    plt.xlabel('xTB-sTDA T$_1$ (eV)', fontsize=16)
    plt.ylabel('TDDFT T$_1$ (eV)', fontsize=16)
    # plt.title('TDDFT vs. xTB T1 (eV)')
    plt.annotate('R$^2$ orig: %0.2f\n' % r2_score(T1TDDFTs, T1xtbs) +
                 'R$^2$ lin: %0.2f\n' % r2_score(T1TDDFTs, T1xtb_Lins) +
                 'R$^2$ ML: %0.2f\n' % r2_score(T1TDDFTs, T1xtb_MLs) +
                 'MAE orig: %0.2f\n' % mean_absolute_error(T1TDDFTs, T1xtbs) +
                 'MAE lin: %0.2f\n' % mean_absolute_error(T1TDDFTs, T1xtb_Lins) +
                 'MAE ML: %0.2f' % mean_absolute_error(T1TDDFTs, T1xtb_MLs),
                 (9.5, 0.5),
                 bbox=dict(facecolor='white', alpha=0.5),
                 fontsize=14,
                 ha='right')
    plt.tight_layout()

    label_axes(fig, ha='left')
    plt.savefig('plots/SCOP_PCQC_ML_comp.png')


def analyze_preds_old():
    T1xtbs = []
    T1TDDFTs = []
    T1xtb_MLs = []
    T1xtb_Lins = []
    S1xtbs = []
    S1TDDFTs = []
    S1xtb_MLs = []
    S1xtb_Lins = []
    # 'SMILES,CID,xtb_T1,xtb_S1,TDDFT_T1,TDDFT_S1,T1err,S1err,xtb_S1_fL,TDDFT_S1_fL,S1fLerr,'
    # 'xtb_ML_T1,xtb_ML_S1,xtb_Lin_T1,xtb_Lin_S1\n'
    with open('SCOP_PCQC_xtb_calib.csv', 'r') as file:
        data = file.readlines()
        for line in data:
            if 'SMILES' in line or 'smiles' in line:
                continue
            line = line.replace('\n', '')
            lineData = line.split(',')
            xtbS1 = float(lineData[3])
            TDDFTS1 = float(lineData[5])
            xtbMLS1 = float(lineData[12])
            xtbLinS1 = float(lineData[14])
            S1xtbs.append(xtbS1)
            S1TDDFTs.append(TDDFTS1)
            S1xtb_MLs.append(xtbMLS1)
            S1xtb_Lins.append(xtbLinS1)
            # try:
            xtbT1 = float(lineData[2])
            TDDFTT1 = float(lineData[4])
            xtbMLT1 = float(lineData[11])
            xtbLinT1 = float(lineData[13])
            # except:
            # continue
            T1xtbs.append(xtbT1)
            T1TDDFTs.append(TDDFTT1)
            T1xtb_MLs.append(xtbMLT1)
            T1xtb_Lins.append(xtbLinT1)

    T1_origr2 = r2_score(T1TDDFTs, T1xtbs)
    T1_lin_fixedr2 = r2_score(T1TDDFTs, T1xtb_Lins)
    T1_fixedr2 = r2_score(T1TDDFTs, T1xtb_MLs)
    T1_origMAE = mean_absolute_error(T1TDDFTs, T1xtbs)
    T1_lin_fixedMAE = mean_absolute_error(T1TDDFTs, T1xtb_Lins)
    T1_fixedMAE = mean_absolute_error(T1TDDFTs, T1xtb_MLs)
    T1_origRMSE = mean_squared_error(T1TDDFTs, T1xtbs, squared=False)
    T1_lin_fixedRMSE = mean_squared_error(T1TDDFTs, T1xtb_Lins, squared=False)
    T1_fixedRMSE = mean_squared_error(T1TDDFTs, T1xtb_MLs, squared=False)
    S1_origr2 = r2_score(S1TDDFTs, S1xtbs)
    S1_lin_fixedr2 = r2_score(S1TDDFTs, S1xtb_Lins)
    S1_fixedr2 = r2_score(S1TDDFTs, S1xtb_MLs)
    S1_origMAE = mean_absolute_error(S1TDDFTs, S1xtbs)
    S1_lin_fixedMAE = mean_absolute_error(S1TDDFTs, S1xtb_Lins)
    S1_fixedMAE = mean_absolute_error(S1TDDFTs, S1xtb_MLs)
    S1_origRMSE = mean_squared_error(S1TDDFTs, S1xtbs, squared=False)
    S1_lin_fixedRMSE = mean_squared_error(S1TDDFTs, S1xtb_Lins, squared=False)
    S1_fixedRMSE = mean_squared_error(S1TDDFTs, S1xtb_MLs, squared=False)

    print('separate')
    print('T1 orig r2:\t' + str(T1_origr2))
    print('T1 lin fixed r2:\t' + str(T1_lin_fixedr2))
    print('T1 fixed r2:\t' + str(T1_fixedr2))
    print('T1 orig MAE:\t' + str(T1_origMAE))
    print('T1 lin fixed MAE:\t' + str(T1_lin_fixedMAE))
    print('T1 fixed MAE:\t' + str(T1_fixedMAE))
    print('T1 orig RMSE:\t' + str(T1_origRMSE))
    print('T1 lin fixed RMSE:\t' + str(T1_lin_fixedRMSE))
    print('T1 fixed RMSE:\t' + str(T1_fixedRMSE))
    print('S1 orig r2:\t' + str(S1_origr2))
    print('S1 lin fixed r2:\t' + str(S1_lin_fixedr2))
    print('S1 fixed r2:\t' + str(S1_fixedr2))
    print('S1 orig MAE:\t' + str(S1_origMAE))
    print('S1 lin fixed MAE:\t' + str(S1_lin_fixedMAE))
    print('S1 fixed MAE:\t' + str(S1_fixedMAE))
    print('S1 orig RMSE:\t' + str(S1_origRMSE))
    print('S1 lin fixed RMSE:\t' + str(S1_lin_fixedRMSE))
    print('S1 fixed RMSE:\t' + str(S1_fixedRMSE))
    with open('SCOP_PCQC_sep_xtb_calib.txt', 'w') as file:
        file.write('T1 orig r2:\t' + str(T1_origr2) + '\n')
        file.write('T1 lin fixed r2:\t' + str(T1_lin_fixedr2) + '\n')
        file.write('T1 fixed r2:\t' + str(T1_fixedr2) + '\n')
        file.write('T1 orig MAE:\t' + str(T1_origMAE) + '\n')
        file.write('T1 lin fixed MAE:\t' + str(T1_lin_fixedMAE) + '\n')
        file.write('T1 fixed MAE:\t' + str(T1_fixedMAE) + '\n')
        file.write('T1 orig RMSE:\t' + str(T1_origRMSE) + '\n')
        file.write('T1 lin fixed RMSE:\t' + str(T1_lin_fixedRMSE) + '\n')
        file.write('T1 fixed RMSE:\t' + str(T1_fixedRMSE) + '\n')
        file.write('S1 orig r2:\t' + str(S1_origr2) + '\n')
        file.write('S1 lin fixed r2:\t' + str(S1_lin_fixedr2) + '\n')
        file.write('S1 fixed r2:\t' + str(S1_fixedr2) + '\n')
        file.write('S1 orig MAE:\t' + str(S1_origMAE) + '\n')
        file.write('S1 lin fixed MAE:\t' + str(S1_lin_fixedMAE) + '\n')
        file.write('S1 fixed MAE:\t' + str(S1_fixedMAE) + '\n')
        file.write('S1 orig RMSE:\t' + str(S1_origRMSE) + '\n')
        file.write('S1 lin fixed RMSE:\t' + str(S1_lin_fixedRMSE) + '\n')
        file.write('S1 fixed RMSE:\t' + str(S1_fixedRMSE) + '\n')

    fig = plt.figure(num=1, figsize=[6, 4], dpi=300, clear=True)
    ax = fig.add_subplot(1, 2, 1)
    plt.plot(T1xtbs, T1TDDFTs, '.', markersize=2, label='orig')
    plt.plot(T1xtb_Lins, T1TDDFTs, '.', markersize=2, label='lin calib')
    plt.plot(T1xtb_MLs, T1TDDFTs, '.', markersize=2, label='ML calib')
    x = np.linspace(0, 10, 100)
    plt.plot(x, x, 'k--')
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    ax.set_axisbelow(True)
    plt.grid(True)
    ax.set_aspect('equal', adjustable='box')
    plt.legend()
    plt.xlabel('xTB T1 (eV)')
    plt.ylabel('TDDFT T1 (eV)')
    plt.title('TDDFT vs. xTB T1 (eV)')
    plt.tight_layout()
    plt.savefig('plots/T1_xtb_TDDFT.png')

    fig = plt.figure(num=1)
    ax = fig.add_subplot(1, 2, 2)
    plt.plot(S1xtbs, S1TDDFTs, '.', markersize=2, label='orig')
    plt.plot(S1xtb_Lins, S1TDDFTs, '.', markersize=2, label='lin calib')
    plt.plot(S1xtb_MLs, S1TDDFTs, '.', markersize=2, label='ML calib')
    x = np.linspace(0, 10, 100)
    plt.plot(x, x, 'k--')
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    ax.set_axisbelow(True)
    plt.grid(True)
    ax.set_aspect('equal', adjustable='box')
    plt.legend(loc='upper left')
    plt.xlabel('xTB S1 (eV)')
    plt.ylabel('TDDFT S1 (eV)')
    plt.title('TDDFT vs. xTB S1 (eV)')
    plt.tight_layout()
    plt.savefig('plots/S1_xtb_TDDFT.png')

    T1xtbs = []
    T1TDDFTs = []
    T1xtb_MLs = []
    T1xtb_Lins = []
    S1xtbs = []
    S1TDDFTs = []
    S1xtb_MLs = []
    S1xtb_Lins = []
    S1fLxtbs = []
    S1fLTDDFTs = []
    S1fLxtb_MLs = []
    S1fLxtb_Lins = []
    # 'SMILES,CID,xtb_T1,xtb_S1,TDDFT_T1,TDDFT_S1,T1err,S1err,xtb_S1_fL,TDDFT_S1_fL,S1fLerr,'
    # 'xtb_ML_T1,xtb_ML_S1,xtb_ML_S1fL,xtb_Lin_T1,xtb_Lin_S1,xtb_Lin_S1fL\n'
    with open('SCOP_PCQC_xtb_calib_multi.csv', 'r') as file:
        data = file.readlines()
        for line in data:
            if 'SMILES' in line or 'smiles' in line:
                continue
            line = line.replace('\n', '')
            lineData = line.split(',')
            xtbT1 = float(lineData[2])
            TDDFTT1 = float(lineData[4])
            xtbMLT1 = float(lineData[11])
            xtbLinT1 = float(lineData[14])
            T1xtbs.append(xtbT1)
            T1TDDFTs.append(TDDFTT1)
            T1xtb_MLs.append(xtbMLT1)
            T1xtb_Lins.append(xtbLinT1)
            xtbS1 = float(lineData[3])
            TDDFTS1 = float(lineData[5])
            xtbMLS1 = float(lineData[12])
            xtbLinS1 = float(lineData[15])
            S1xtbs.append(xtbS1)
            S1TDDFTs.append(TDDFTS1)
            S1xtb_MLs.append(xtbMLS1)
            S1xtb_Lins.append(xtbLinS1)
            xtbS1fL = float(lineData[8])
            TDDFTS1fL = float(lineData[9])
            xtbMLS1fL = float(lineData[13])
            xtbLinS1fL = float(lineData[16])
            S1fLxtbs.append(xtbS1fL)
            S1fLTDDFTs.append(TDDFTS1fL)
            S1fLxtb_MLs.append(xtbMLS1fL)
            S1fLxtb_Lins.append(xtbLinS1fL)

    T1_origr2 = r2_score(T1TDDFTs, T1xtbs)
    T1_lin_fixedr2 = r2_score(T1TDDFTs, T1xtb_Lins)
    T1_fixedr2 = r2_score(T1TDDFTs, T1xtb_MLs)
    T1_origMAE = mean_absolute_error(T1TDDFTs, T1xtbs)
    T1_lin_fixedMAE = mean_absolute_error(T1TDDFTs, T1xtb_Lins)
    T1_fixedMAE = mean_absolute_error(T1TDDFTs, T1xtb_MLs)
    T1_origRMSE = mean_squared_error(T1TDDFTs, T1xtbs, squared=False)
    T1_lin_fixedRMSE = mean_squared_error(T1TDDFTs, T1xtb_Lins, squared=False)
    T1_fixedRMSE = mean_squared_error(T1TDDFTs, T1xtb_MLs, squared=False)
    S1_origr2 = r2_score(S1TDDFTs, S1xtbs)
    S1_lin_fixedr2 = r2_score(S1TDDFTs, S1xtb_Lins)
    S1_fixedr2 = r2_score(S1TDDFTs, S1xtb_MLs)
    S1_origMAE = mean_absolute_error(S1TDDFTs, S1xtbs)
    S1_lin_fixedMAE = mean_absolute_error(S1TDDFTs, S1xtb_Lins)
    S1_fixedMAE = mean_absolute_error(S1TDDFTs, S1xtb_MLs)
    S1_origRMSE = mean_squared_error(S1TDDFTs, S1xtbs, squared=False)
    S1_lin_fixedRMSE = mean_squared_error(S1TDDFTs, S1xtb_Lins, squared=False)
    S1_fixedRMSE = mean_squared_error(S1TDDFTs, S1xtb_MLs, squared=False)
    S1fL_origr2 = r2_score(S1fLTDDFTs, S1fLxtbs)
    S1fL_lin_fixedr2 = r2_score(S1fLTDDFTs, S1fLxtb_Lins)
    S1fL_fixedr2 = r2_score(S1fLTDDFTs, S1fLxtb_MLs)
    S1fL_origMAE = mean_absolute_error(S1fLTDDFTs, S1fLxtbs)
    S1fL_lin_fixedMAE = mean_absolute_error(S1fLTDDFTs, S1fLxtb_Lins)
    S1fL_fixedMAE = mean_absolute_error(S1fLTDDFTs, S1fLxtb_MLs)
    S1fL_origRMSE = mean_squared_error(S1fLTDDFTs, S1fLxtbs, squared=False)
    S1fL_lin_fixedRMSE = mean_squared_error(S1fLTDDFTs, S1fLxtb_Lins, squared=False)
    S1fL_fixedRMSE = mean_squared_error(S1fLTDDFTs, S1fLxtb_MLs, squared=False)

    print('multi')
    print('T1 orig r2:\t' + str(T1_origr2))
    print('T1 lin fixed r2:\t' + str(T1_lin_fixedr2))
    print('T1 fixed r2:\t' + str(T1_fixedr2))
    print('T1 orig MAE:\t' + str(T1_origMAE))
    print('T1 lin fixed MAE:\t' + str(T1_lin_fixedMAE))
    print('T1 fixed MAE:\t' + str(T1_fixedMAE))
    print('T1 orig RMSE:\t' + str(T1_origRMSE))
    print('T1 lin fixed RMSE:\t' + str(T1_lin_fixedRMSE))
    print('T1 fixed RMSE:\t' + str(T1_fixedRMSE))
    print('S1 orig r2:\t' + str(S1_origr2))
    print('S1 lin fixed r2:\t' + str(S1_lin_fixedr2))
    print('S1 fixed r2:\t' + str(S1_fixedr2))
    print('S1 orig MAE:\t' + str(S1_origMAE))
    print('S1 lin fixed MAE:\t' + str(S1_lin_fixedMAE))
    print('S1 fixed MAE:\t' + str(S1_fixedMAE))
    print('S1 orig RMSE:\t' + str(S1_origRMSE))
    print('S1 lin fixed RMSE:\t' + str(S1_lin_fixedRMSE))
    print('S1 fixed RMSE:\t' + str(S1_fixedRMSE))
    print('S1fL orig r2:\t' + str(S1fL_origr2))
    print('S1fL lin fixed r2:\t' + str(S1fL_lin_fixedr2))
    print('S1fL fixed r2:\t' + str(S1fL_fixedr2))
    print('S1fL orig MAE:\t' + str(S1fL_origMAE))
    print('S1fL lin fixed MAE:\t' + str(S1fL_lin_fixedMAE))
    print('S1fL fixed MAE:\t' + str(S1fL_fixedMAE))
    print('S1fL orig RMSE:\t' + str(S1fL_origRMSE))
    print('S1fL lin fixed RMSE:\t' + str(S1fL_lin_fixedRMSE))
    print('S1fL fixed RMSE:\t' + str(S1fL_fixedRMSE))
    with open('SCOP_PCQC_xtb_calib_multi.txt', 'w') as file:
        file.write('T1 orig r2:\t' + str(T1_origr2) + '\n')
        file.write('T1 lin fixed r2:\t' + str(T1_lin_fixedr2) + '\n')
        file.write('T1 fixed r2:\t' + str(T1_fixedr2) + '\n')
        file.write('T1 orig MAE:\t' + str(T1_origMAE) + '\n')
        file.write('T1 lin fixed MAE:\t' + str(T1_lin_fixedMAE) + '\n')
        file.write('T1 fixed MAE:\t' + str(T1_fixedMAE) + '\n')
        file.write('T1 orig RMSE:\t' + str(T1_origRMSE) + '\n')
        file.write('T1 lin fixed RMSE:\t' + str(T1_lin_fixedRMSE) + '\n')
        file.write('T1 fixed RMSE:\t' + str(T1_fixedRMSE) + '\n')
        file.write('S1 orig r2:\t' + str(S1_origr2) + '\n')
        file.write('S1 lin fixed r2:\t' + str(S1_lin_fixedr2) + '\n')
        file.write('S1 fixed r2:\t' + str(S1_fixedr2) + '\n')
        file.write('S1 orig MAE:\t' + str(S1_origMAE) + '\n')
        file.write('S1 lin fixed MAE:\t' + str(S1_lin_fixedMAE) + '\n')
        file.write('S1 fixed MAE:\t' + str(S1_fixedMAE) + '\n')
        file.write('S1 orig RMSE:\t' + str(S1_origRMSE) + '\n')
        file.write('S1 lin fixed RMSE:\t' + str(S1_lin_fixedRMSE) + '\n')
        file.write('S1 fixed RMSE:\t' + str(S1_fixedRMSE) + '\n')
        file.write('S1fL orig r2:\t' + str(S1fL_origr2) + '\n')
        file.write('S1fL lin fixed r2:\t' + str(S1fL_lin_fixedr2) + '\n')
        file.write('S1fL fixed r2:\t' + str(S1fL_fixedr2) + '\n')
        file.write('S1fL orig MAE:\t' + str(S1fL_origMAE) + '\n')
        file.write('S1fL lin fixed MAE:\t' + str(S1fL_lin_fixedMAE) + '\n')
        file.write('S1fL fixed MAE:\t' + str(S1fL_fixedMAE) + '\n')
        file.write('S1fL orig RMSE:\t' + str(S1fL_origRMSE) + '\n')
        file.write('S1fL lin fixed RMSE:\t' + str(S1fL_lin_fixedRMSE) + '\n')
        file.write('S1fL fixed RMSE:\t' + str(S1fL_fixedRMSE) + '\n')

    fig = plt.figure(num=2, figsize=[8, 8], dpi=300, clear=True)

    ax = fig.add_subplot(2, 2, 1)
    plt.plot(S1xtbs, S1TDDFTs, '.', markersize=2, color=colors_nipy[-1], label='orig')
    plt.plot(S1xtb_Lins, S1TDDFTs, '.', markersize=2, color=colors_nipy[2], label='lin calib')
    plt.plot(S1xtb_MLs, S1TDDFTs, '.', markersize=2, color=colors_nipy[1], label='ML calib')
    x = np.linspace(0, 10, 100)
    plt.plot(x, x, 'k--')
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    ax.set_axisbelow(True)
    plt.grid(True)
    ax.set_aspect('equal', adjustable='box')
    plt.legend(markerscale=6, fontsize=14)
    plt.xlabel('xTB-sTDA S$_1$ (eV)', fontsize=16)
    plt.ylabel('TDDFT S$_1$ (eV)', fontsize=16)
    # plt.title('TDDFT vs. xTB S1 (eV)')
    plt.annotate('R$^2$ orig: %0.2f\n' % S1_origr2 +
                 'R$^2$ lin: %0.2f\n' % S1_lin_fixedr2 +
                 'R$^2$ ML: %0.2f\n' % S1_fixedr2 +
                 'MAE orig: %0.2f\n' % S1_origMAE +
                 'MAE lin: %0.2f\n' % S1_lin_fixedMAE +
                 'MAE ML: %0.2f' % S1_fixedMAE,
                 (9.5, 0.5),
                 bbox=dict(facecolor='white', alpha=0.5),
                 fontsize=14,
                 ha='right')
    plt.tight_layout()
    plt.savefig('plots/S1_xtb_TDDFT_multi.png')

    ax = fig.add_subplot(2, 2, 2)
    plt.plot(T1xtbs, T1TDDFTs, '.', markersize=2, color=colors_nipy[-1], label='orig')
    plt.plot(T1xtb_Lins, T1TDDFTs, '.', markersize=2, color=colors_nipy[2], label='lin calib')
    plt.plot(T1xtb_MLs, T1TDDFTs, '.', markersize=2, color=colors_nipy[1], label='ML calib')
    x = np.linspace(0, 10, 100)
    plt.plot(x, x, 'k--')
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    ax.set_axisbelow(True)
    plt.grid(True)
    ax.set_aspect('equal', adjustable='box')
    plt.legend(markerscale=6, fontsize=14)
    plt.xlabel('xTB-sTDA T$_1$ (eV)', fontsize=16)
    plt.ylabel('TDDFT T$_1$ (eV)', fontsize=16)
    # plt.title('TDDFT vs. xTB T1 (eV)')
    plt.annotate('R$^2$ orig: %0.2f\n' % T1_origr2 +
                 'R$^2$ lin: %0.2f\n' % T1_lin_fixedr2 +
                 'R$^2$ ML: %0.2f\n' % T1_fixedr2 +
                 'MAE orig: %0.2f\n' % T1_origMAE +
                 'MAE lin: %0.2f\n' % T1_lin_fixedMAE +
                 'MAE ML: %0.2f' % T1_fixedMAE,
                 (9.5, 0.5),
                 bbox=dict(facecolor='white', alpha=0.5),
                 fontsize=14,
                 ha='right')
    plt.tight_layout()
    plt.savefig('plots/T1_xtb_TDDFT_multi.png')

    label_axes(fig, ha='left')
    plt.savefig('plots/SCOP_PCQC_ML_comp.png')

    fig = plt.figure(num=3, clear=True)
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(S1fLxtbs, S1fLTDDFTs, 'r.', markersize=2, label='orig')
    plt.plot(S1fLxtb_Lins, S1fLTDDFTs, 'g.', markersize=2, label='lin calib')
    plt.plot(S1fLxtb_MLs, S1fLTDDFTs, 'b.', markersize=2, label='ML calib')
    x = np.linspace(0, 6, 100)
    plt.plot(x, x, 'k--')
    plt.xlim(0, 2)
    plt.ylim(0, 2)
    ax.set_axisbelow(True)
    plt.grid(True)
    ax.set_aspect('equal', adjustable='box')
    plt.legend()
    plt.xlabel('xTB S1fL')
    plt.ylabel('TDDFT S1fL')
    plt.title('TDDFT vs. xTB S1fL')
    plt.tight_layout()
    plt.savefig('plots/S1fL_xtb_TDDFT_multi.png')

    T1xtbs = []
    T1TDDFTs = []
    T1xtb_MLs = []
    S1xtbs = []
    S1TDDFTs = []
    S1xtb_MLs = []
    # 'SMILES,CID,xtb_T1,xtb_S1,TDDFT_T1,TDDFT_S1,T1err,S1err,xtb_S1_fL,TDDFT_S1_fL,S1fLerr,'
    # 'xtb_ML_T1,xtb_ML_S1\n'
    with open('SCOP_PCQC_xtb_calib_multi_hyperopt.csv', 'r') as file:
        data = file.readlines()
        for line in data:
            if 'SMILES' in line or 'smiles' in line:
                continue
            line = line.replace('\n', '')
            lineData = line.split(',')
            xtbS1 = float(lineData[3])
            TDDFTS1 = float(lineData[5])
            xtbMLS1 = float(lineData[12])
            S1xtbs.append(xtbS1)
            S1TDDFTs.append(TDDFTS1)
            S1xtb_MLs.append(xtbMLS1)
            # try:
            xtbT1 = float(lineData[2])
            TDDFTT1 = float(lineData[4])
            xtbMLT1 = float(lineData[11])
            # except:
            # continue
            T1xtbs.append(xtbT1)
            T1TDDFTs.append(TDDFTT1)
            T1xtb_MLs.append(xtbMLT1)

    T1_origr2 = r2_score(T1TDDFTs, T1xtbs)
    T1_fixedr2 = r2_score(T1TDDFTs, T1xtb_MLs)
    T1_origMAE = mean_absolute_error(T1TDDFTs, T1xtbs)
    T1_fixedMAE = mean_absolute_error(T1TDDFTs, T1xtb_MLs)
    T1_origRMSE = mean_squared_error(T1TDDFTs, T1xtbs, squared=False)
    T1_fixedRMSE = mean_squared_error(T1TDDFTs, T1xtb_MLs, squared=False)
    S1_origr2 = r2_score(S1TDDFTs, S1xtbs)
    S1_fixedr2 = r2_score(S1TDDFTs, S1xtb_MLs)
    S1_origMAE = mean_absolute_error(S1TDDFTs, S1xtbs)
    S1_fixedMAE = mean_absolute_error(S1TDDFTs, S1xtb_MLs)
    S1_origRMSE = mean_squared_error(S1TDDFTs, S1xtbs, squared=False)
    S1_fixedRMSE = mean_squared_error(S1TDDFTs, S1xtb_MLs, squared=False)

    print('multitask hyperopt')
    print('T1 orig r2:\t' + str(T1_origr2))
    print('T1 fixed r2:\t' + str(T1_fixedr2))
    print('T1 orig MAE:\t' + str(T1_origMAE))
    print('T1 fixed MAE:\t' + str(T1_fixedMAE))
    print('T1 orig RMSE:\t' + str(T1_origRMSE))
    print('T1 fixed RMSE:\t' + str(T1_fixedRMSE))
    print('S1 orig r2:\t' + str(S1_origr2))
    print('S1 fixed r2:\t' + str(S1_fixedr2))
    print('S1 orig MAE:\t' + str(S1_origMAE))
    print('S1 fixed MAE:\t' + str(S1_fixedMAE))
    print('S1 orig RMSE:\t' + str(S1_origRMSE))
    print('S1 fixed RMSE:\t' + str(S1_fixedRMSE))
    with open('SCOP_PCQC_multi_xtb_calib.txt', 'w') as file:
        file.write('T1 orig r2:\t' + str(T1_origr2) + '\n')
        file.write('T1 fixed r2:\t' + str(T1_fixedr2) + '\n')
        file.write('T1 orig MAE:\t' + str(T1_origMAE) + '\n')
        file.write('T1 fixed MAE:\t' + str(T1_fixedMAE) + '\n')
        file.write('T1 orig RMSE:\t' + str(T1_origRMSE) + '\n')
        file.write('T1 fixed RMSE:\t' + str(T1_fixedRMSE) + '\n')
        file.write('S1 orig r2:\t' + str(S1_origr2) + '\n')
        file.write('S1 fixed r2:\t' + str(S1_fixedr2) + '\n')
        file.write('S1 orig MAE:\t' + str(S1_origMAE) + '\n')
        file.write('S1 fixed MAE:\t' + str(S1_fixedMAE) + '\n')
        file.write('S1 orig RMSE:\t' + str(S1_origRMSE) + '\n')
        file.write('S1 fixed RMSE:\t' + str(S1_fixedRMSE) + '\n')


# T1_ML()
# S1_ML()
# multi_ML()
# lin_ML()
# compile_preds()
analyze_preds()

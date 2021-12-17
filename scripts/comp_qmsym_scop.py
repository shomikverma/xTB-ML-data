import os
import json
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn import linear_model
import string
from itertools import cycle
from cycler import cycler


def label_axes(fig, labels=None, loc=None, **kwargs):
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


plt.style.use(['science', 'grid'])
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
colors = [colors[3], colors[1], colors[0]]
colors_nipy1 = mpl.cm.nipy_spectral(np.linspace(0.1, 0.9, 6))
colors_nipy2 = mpl.cm.nipy_spectral(np.linspace(0.6, 0.9, 7))
colors_nipy = list(colors_nipy1[0:3]) + list(colors_nipy2[3:-2]) + list(colors_nipy1[-1:])
plt.rcParams['axes.prop_cycle'] = cycler(color=colors)


def gen_train_data():
    with open('../xtb_results_qmsym_10k/QMsym_10k_xtb.csv', 'r') as file:
        dataQMsym = file.readlines()

    with open('../xtb_results_scop_pcqc_files/SCOP_PCQC_xtb.csv', 'r') as file:
        dataSCOP = file.readlines()

    # allS1errData = {}

    with open('Qmsym_scop_train_data.csv', 'w') as file:
        file.write('SMILES,CID,xtb_T1,xtb_S1,TDDFT_T1,TDDFT_S1,T1err,S1err,xtb_S1_fL,TDDFT_S1_fL,S1fLerr\n')
        # for val in allS1errData:
        #     file.write(val + ',' + allS1errData[val] + '\n')
        for line in dataSCOP:
            if 'SMILES' in line:
                continue
            file.write(line)

        for line in dataQMsym:
            if 'SMILES' in line:
                continue
            file.write(line)


def gen_test_data():
    with open('xtb_tddft_calib_data.csv', 'r') as file:
        dataXTB = file.readlines()

    allS1errDataTest = {}
    for line in dataXTB:
        line = line.replace('\n', '')
        if ('Smiles' in line):
            continue
        data = line.split(',')
        smiles = data[1]
        TDDFT = float(data[2])
        xTB = float(data[13])
        S1err = TDDFT - xTB
        allS1errDataTest[smiles] = str(S1err)

    with open('MOPSSAM_test_data.csv', 'w') as file:
        file.write('smiles,S1errTrue\n')
        for val in allS1errDataTest:
            file.write(val + ',' + allS1errDataTest[val] + '\n')

    with open('xtb_calib_props.csv', 'r') as file:
        dataProps = file.readlines()

    allXtbProps = {}
    for index, line in enumerate(dataProps):
        if index == 0:
            continue
        line = line.replace('\n', '')
        data = line.split(',')
        xTB = data[3]
        xTB_Lin = data[10]
        smiles = data[6]
        allXtbProps[smiles] = {}
        allXtbProps[smiles]['xTB'] = xTB
        allXtbProps[smiles]['xTB_Lin'] = xTB_Lin

    with open('MOPSSAM_test_xtb_calib.csv', 'w') as file:
        file.write('smiles,xTB,xTB_Lin\n')
        for val in allXtbProps:
            file.write(val + ',' + allXtbProps[val]['xTB'] + ',' +
                       allXtbProps[val]['xTB_Lin'] + '\n')


def do_ML():
    os.system('chemprop_train --data_path Qmsym_scop_train_data.csv --dataset_type regression '
              '--save_dir Qmsym_scop_stda --split_type cv --save_smiles_splits '
              '--num_folds 10 --target_columns S1err T1err S1fLerr')
    os.system('chemprop_predict --test_path MOPSSAM_test_data.csv --checkpoint_dir Qmsym_scop_stda '
              '--preds_path MOPSSAM_preds_stda.csv --drop_extra_columns')
    os.system('chemprop_predict --test_path MOPSSAM_test_xtb_calib.csv --checkpoint_dir Qmsym_scop_stda '
              '--preds_path MOPSSAM_preds_stda_calib.csv --drop_extra_columns')


def do_direct_ML():
    os.system('chemprop_train --data_path Qmsym_scop_train_data.csv --dataset_type regression '
              '--save_dir Qmsym_scop_stdaS1_direct --split_type cv --save_smiles_splits '
              '--num_folds 10 --target_columns TDDFT_S1')
    os.system('chemprop_predict --test_path MOPSSAM_test_data.csv --checkpoint_dir Qmsym_scop_stdaS1_direct '
              '--preds_path MOPSSAM_preds_stdaS1_direct.csv --drop_extra_columns')
    os.system('chemprop_predict --test_path MOPSSAM_test_xtb_calib.csv --checkpoint_dir Qmsym_scop_stdaS1_direct '
              '--preds_path MOPSSAM_preds_stdaS1_calib_direct.csv --drop_extra_columns')


def analyze_ML(MLtype):
    with open('xtb_tddft_calib_data.csv', 'r') as file:
        dataCalib = file.readlines()

    with open('xtb_calib_props.csv', 'r') as file:
        dataProps = file.readlines()

    if MLtype == 'xtb':
        filename = 'MOPSSAM_preds_stdaS1.csv'
    elif MLtype == 'direct':
        filename = 'MOPSSAM_preds_stdaS1_direct.csv'
    else:
        filename = 'MOPSSAM_preds_stda.csv'
    with open(filename, 'r') as file:
        dataS1 = file.readlines()

    allData = {}
    ID2SMI = {}

    for index, line in enumerate(dataCalib):
        if index == 0:
            continue
        line = line.replace('\n', '')
        data = line.split(',')
        smiles = data[1]
        TDDFT = float(data[2])
        ID = float(data[10])
        xTB = float(data[13])
        ID2SMI[ID] = smiles
        allData[smiles] = {}
        allData[smiles]['xTB'] = xTB
        allData[smiles]['TDDFT'] = TDDFT

    for index, line in enumerate(dataProps):
        if index == 0:
            continue
        line = line.replace('\n', '')
        data = line.split(',')
        ID = float(data[0])
        xTB_Lin = float(data[10])
        try:
            smiles = ID2SMI[ID]
        except KeyError:
            continue
        allData[smiles]['xTB_Lin'] = xTB_Lin

    for index, line in enumerate(dataS1):
        if index == 0:
            continue
        line = line.replace('\n', '')
        data = line.split(',')
        smiles = data[0]
        S1err = float(data[1])
        if MLtype == 'xtb':
            xTB_ML = allData[smiles]['xTB'] + S1err
        elif MLtype == 'direct':
            xTB_ML = S1err
        else:
            xTB_ML = allData[smiles]['xTB'] + S1err
        allData[smiles]['xTB_ML'] = xTB_ML

    xTBs = []
    TDDFTs = []
    xTB_Lins = []
    xTB_MLs = []

    for smiles in allData:
        xTBs.append(allData[smiles]['xTB'])
        TDDFTs.append(allData[smiles]['TDDFT'])
        xTB_Lins.append(allData[smiles]['xTB_Lin'])
        xTB_MLs.append(allData[smiles]['xTB_ML'])

    orig_r2 = r2_score(TDDFTs, xTBs)
    lin_fixed_r2 = r2_score(TDDFTs, xTB_Lins)
    ML_fixed_r2 = r2_score(TDDFTs, xTB_MLs)
    orig_MAE = mean_absolute_error(TDDFTs, xTBs)
    lin_fixed_MAE = mean_absolute_error(TDDFTs, xTB_Lins)
    ML_fixed_MAE = mean_absolute_error(TDDFTs, xTB_MLs)
    orig_RMSE = mean_squared_error(TDDFTs, xTBs, squared=False)
    lin_fixed_RMSE = mean_squared_error(TDDFTs, xTB_Lins, squared=False)
    ML_fixed_RMSE = mean_squared_error(TDDFTs, xTB_MLs, squared=False)

    print('orig r2:\t', orig_r2)
    print('lin fixed r2:\t', lin_fixed_r2)
    print('ML fixed r2:\t', ML_fixed_r2)
    print('orig MAE:\t', orig_MAE)
    print('lin fixed MAE:\t', lin_fixed_MAE)
    print('ML fixed MAE:\t', ML_fixed_MAE)
    print('orig RMSE:\t', orig_RMSE)
    print('lin fixed RMSE:\t', lin_fixed_RMSE)
    print('ML fixed RMSE:\t', ML_fixed_RMSE)
    if MLtype == 'xtb':
        resultsFilename = 'MOPSSAM_calib_results.txt'
    elif MLtype == 'direct':
        resultsFilename = 'MOPSSAM_calib_results_direct.txt'
    else:
        resultsFilename = 'MOPSSAM_calib_results_multi.txt'
    with open(resultsFilename, 'w') as file:
        file.write('orig r2:\t' + str(orig_r2) + '\n')
        file.write('lin fixed r2:\t' + str(lin_fixed_r2) + '\n')
        file.write('ML fixed r2:\t' + str(ML_fixed_r2) + '\n')
        file.write('orig MAE:\t' + str(orig_MAE) + '\n')
        file.write('lin fixed MAE:\t' + str(lin_fixed_MAE) + '\n')
        file.write('ML fixed MAE:\t' + str(ML_fixed_MAE) + '\n')
        file.write('orig RMSE:\t' + str(orig_RMSE) + '\n')
        file.write('lin fixed RMSE:\t' + str(lin_fixed_RMSE) + '\n')
        file.write('ML fixed RMSE:\t' + str(ML_fixed_RMSE) + '\n')

    fig = plt.figure(num=1, clear=True)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_axisbelow(True)
    plt.plot(xTBs, TDDFTs, 'r.', label='orig')
    plt.plot(xTB_Lins, TDDFTs, 'g.', label='lin calib')
    if MLtype == 'xtb':
        plt.plot(xTB_MLs, TDDFTs, 'b.', label='ML calib')
    elif MLtype == 'direct':
        plt.plot(xTB_MLs, TDDFTs, 'b.', label='ML')
    else:
        plt.plot(xTB_MLs, TDDFTs, 'b.', label='ML calib')
    x = np.linspace(0, 9, 100)
    plt.plot(x, x, 'k--')
    plt.grid(True)
    plt.legend()
    if MLtype == 'xtb':
        plt.xlabel('stda S1 (eV)')
    elif MLtype == 'direct':
        plt.xlabel('predicted S1 (eV)')
    else:
        plt.xlabel('stda S1 (eV)')
    plt.ylabel('TDDFT S1 (eV)')
    if MLtype == 'xtb':
        plt.title('TD-DFT vs. stda S1 comparison')
    elif MLtype == 'direct':
        plt.title('TD-DFT vs. stda/ML S1 comparison')
    else:
        plt.title('TD-DFT vs. stda S1 comparison')
    plt.xlim(0, 9)
    plt.ylim(0, 9)
    plt.annotate('R2 orig: %0.2f\n' % orig_r2 +
                 'R2 lin: %0.2f\n' % lin_fixed_r2 +
                 'R2 ML: %0.2f\n' % ML_fixed_r2 +
                 'MAE orig: %0.2f\n' % orig_MAE +
                 'MAE lin: %0.2f\n' % lin_fixed_MAE +
                 'MAE ML: %0.2f' % ML_fixed_MAE,
                 (8.5, 0.5),
                 bbox=dict(facecolor='white', alpha=0.5),
                 ha='right')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    if MLtype == 'xtb':
        plt.savefig('mopssam_S1_calib.png')
    elif MLtype == 'direct':
        plt.savefig('mopssam_S1_calib_direct.png')


def analyze_ML_direct():
    with open('xtb_tddft_calib_data.csv', 'r') as file:
        dataCalib = file.readlines()

    with open('xtb_calib_props.csv', 'r') as file:
        dataProps = file.readlines()

    xtbfilename = 'MOPSSAM_preds_stdaS1.csv'
    directfilename = 'MOPSSAM_preds_stdaS1_direct.csv'
    with open(xtbfilename, 'r') as file:
        dataS1 = file.readlines()
    with open(directfilename, 'r') as file:
        dataS1Direct = file.readlines()

    allData = {}
    ID2SMI = {}

    for index, line in enumerate(dataCalib):
        if index == 0:
            continue
        line = line.replace('\n', '')
        data = line.split(',')
        smiles = data[1]
        TDDFT = float(data[2])
        ID = float(data[10])
        xTB = float(data[13])
        ID2SMI[ID] = smiles
        allData[smiles] = {}
        allData[smiles]['xTB'] = xTB
        allData[smiles]['TDDFT'] = TDDFT

    for index, line in enumerate(dataProps):
        if index == 0:
            continue
        line = line.replace('\n', '')
        data = line.split(',')
        ID = float(data[0])
        xTB_Lin = float(data[10])
        try:
            smiles = ID2SMI[ID]
        except KeyError:
            continue
        allData[smiles]['xTB_Lin'] = xTB_Lin

    for index, line in enumerate(dataS1):
        if index == 0:
            continue
        line = line.replace('\n', '')
        data = line.split(',')
        smiles = data[0]
        S1err = float(data[1])
        xTB_ML = allData[smiles]['xTB'] + S1err
        allData[smiles]['xTB_ML'] = xTB_ML

    for index, line in enumerate(dataS1Direct):
        if index == 0:
            continue
        line = line.replace('\n', '')
        data = line.split(',')
        smiles = data[0]
        S1err = float(data[1])
        xTB_ML = S1err
        allData[smiles]['direct_ML'] = xTB_ML

    xTBs = []
    TDDFTs = []
    xTB_Lins = []
    xTB_MLs = []
    xTB_MLs_direct = []

    for smiles in allData:
        xTBs.append(allData[smiles]['xTB'])
        TDDFTs.append(allData[smiles]['TDDFT'])
        xTB_Lins.append(allData[smiles]['xTB_Lin'])
        xTB_MLs.append(allData[smiles]['xTB_ML'])
        xTB_MLs_direct.append(allData[smiles]['direct_ML'])

    orig_r2 = r2_score(xTBs, TDDFTs)
    lin_fixed_r2 = r2_score(xTB_Lins, TDDFTs)
    ML_fixed_r2 = r2_score(xTB_MLs, TDDFTs)
    direct_ML_fixed_r2 = r2_score(xTB_MLs_direct, TDDFTs)
    orig_MAE = mean_absolute_error(xTBs, TDDFTs)
    lin_fixed_MAE = mean_absolute_error(xTB_Lins, TDDFTs)
    ML_fixed_MAE = mean_absolute_error(xTB_MLs, TDDFTs)
    direct_ML_fixed_MAE = mean_absolute_error(xTB_MLs_direct, TDDFTs)
    orig_RMSE = mean_squared_error(xTBs, TDDFTs, squared=False)
    lin_fixed_RMSE = mean_squared_error(xTB_Lins, TDDFTs, squared=False)
    ML_fixed_RMSE = mean_squared_error(xTB_MLs, TDDFTs, squared=False)
    direct_ML_fixed_RMSE = mean_squared_error(xTB_MLs_direct, TDDFTs, squared=False)

    fig = plt.figure(num=3, figsize=[4, 4], dpi=300, clear=True)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_axisbelow(True)
    plt.plot(xTBs, TDDFTs, '.', color=colors_nipy[-1], label='orig')
    plt.plot(xTB_MLs_direct, TDDFTs, '.', color=colors_nipy[2], label='direct ML')
    plt.plot(xTB_MLs, TDDFTs, '.', color=colors_nipy[1], label='xTB ML')
    x = np.linspace(0, 9, 100)
    plt.plot(x, x, 'k--')
    plt.grid(True)
    plt.legend(markerscale=2, fontsize=14)
    plt.xlabel('Predicted S$_1$ (eV)', fontsize=16)
    plt.ylabel('TDDFT S$_1$ (eV)', fontsize=16)
    # plt.title('TD-DFT vs. xTB/ML S1 comparison')
    plt.xlim(0, 9)
    plt.ylim(0, 9)
    plt.annotate('R$^2$ orig: %0.2f\n' % orig_r2 +
                 'R$^2$ direct ML: %0.2f\n' % direct_ML_fixed_r2 +
                 'R$^2$ xTB-ML: %0.2f\n' % ML_fixed_r2 +
                 'MAE orig: %0.2f\n' % orig_MAE +
                 'MAE ML: %0.2f\n' % direct_ML_fixed_MAE +
                 'MAE xTB-ML: %0.2f' % ML_fixed_MAE,
                 (8.5, 0.5),
                 bbox=dict(facecolor='white', alpha=0.5),
                 fontsize=12,
                 ha='right')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig('mopssam_S1_calib_direct.pdf')

    print('orig r2:\t', orig_r2)
    print('lin fixed r2:\t', lin_fixed_r2)
    print('ML fixed r2:\t', ML_fixed_r2)
    print('direct ML fixed r2:\t', direct_ML_fixed_r2)
    print('orig MAE:\t', orig_MAE)
    print('lin fixed MAE:\t', lin_fixed_MAE)
    print('ML fixed MAE:\t', ML_fixed_MAE)
    print('direct ML fixed MAE:\t', direct_ML_fixed_MAE)
    print('orig RMSE:\t', orig_RMSE)
    print('lin fixed RMSE:\t', lin_fixed_RMSE)
    print('ML fixed RMSE:\t', ML_fixed_RMSE)
    print('direct ML fixed RMSE:\t', direct_ML_fixed_RMSE)
    resultsFilename = 'MOPSSAM_calib_results_direct.txt'
    with open(resultsFilename, 'w') as file:
        file.write('orig r2:\t' + str(orig_r2) + '\n')
        file.write('lin fixed r2:\t' + str(lin_fixed_r2) + '\n')
        file.write('ML fixed r2:\t' + str(ML_fixed_r2) + '\n')
        file.write('direct ML fixed r2:\t' + str(direct_ML_fixed_r2) + '\n')
        file.write('orig MAE:\t' + str(orig_MAE) + '\n')
        file.write('lin fixed MAE:\t' + str(lin_fixed_MAE) + '\n')
        file.write('ML fixed MAE:\t' + str(ML_fixed_MAE) + '\n')
        file.write('direct ML fixed MAE:\t' + str(direct_ML_fixed_MAE) + '\n')
        file.write('orig RMSE:\t' + str(orig_RMSE) + '\n')
        file.write('lin fixed RMSE:\t' + str(lin_fixed_RMSE) + '\n')
        file.write('ML fixed RMSE:\t' + str(ML_fixed_RMSE) + '\n')
        file.write('direct ML fixed RMSE:\t' + str(direct_ML_fixed_RMSE) + '\n')


def plot_calib(MLtype):
    with open('MOPSSAM_test_xtb_calib.csv', 'r') as file:
        dataProps = file.readlines()

    if MLtype == 'xtb':
        filename = 'MOPSSAM_preds_stdaS1_calib.csv'
    elif MLtype == 'direct':
        filename = 'MOPSSAM_preds_stdaS1_calib_direct.csv'
    else:
        filename = 'MOPSSAM_preds_stda_calib.csv'
    with open(filename, 'r') as file:
        dataS1err = file.readlines()
    with open('xTB_ML_errs_xtb_calib_props.csv', 'r') as file:
        dataT1err = file.readlines()

    allXtbProps = {}
    for index, line in enumerate(dataProps):
        if index == 0:
            continue
        line = line.replace('\n', '')
        data = line.split(',')
        xTB = data[1]
        xTB_Lin = data[2]
        smiles = data[0]
        allXtbProps[smiles] = {}
        allXtbProps[smiles]['xTB'] = xTB
        allXtbProps[smiles]['xTB_Lin'] = xTB_Lin

    for index, line in enumerate(dataS1err):
        if index == 0:
            continue
        line = line.replace('\n', '')
        data = line.split(',')
        smiles = data[0]
        S1err = float(data[1])
        try:
            if MLtype == 'xtb':
                xTB = float(allXtbProps[smiles]['xTB'])
                xTB_ML = xTB + S1err
            elif MLtype == 'direct':
                xTB_ML = S1err
            else:
                xTB = float(allXtbProps[smiles]['xTB'])
                xTB_ML = xTB + S1err
        except KeyError:
            xTB_ML = None
        allXtbProps[smiles]['xTB_ML'] = xTB_ML

    for index, line in enumerate(dataT1err):
        if index == 0:
            continue
        line = line.replace('\n', '')
        data = line.split(',')
        smiles = data[0]
        S1err = float(data[3])
        T1err = float(data[4])
        xTB_S1 = float(data[1])
        xTB_T1 = float(data[2])
        xTB_ML_S1 = xTB_S1 + S1err
        xTB_ML_T1 = xTB_T1 + T1err
        allXtbProps[smiles]['xTB_T1'] = xTB_T1
        allXtbProps[smiles]['xTB_ML_T1'] = xTB_ML_T1
        allXtbProps[smiles]['xTB_S1'] = xTB_S1
        allXtbProps[smiles]['xTB_ML_S1'] = xTB_ML_S1

    xTBs = []
    xTB_T1s = []
    xTB_Lins = []
    xTB_MLs = []
    xTB_ML_S1s = []
    xTB_ML_T1s = []
    final_smiles = []
    FOM_sens_all = []
    FOM_emit_all = []
    sens_smiles = []
    emit_smiles = []

    for smiles in allXtbProps:
        xTB = float(allXtbProps[smiles]['xTB'])
        xTB_Lin = float(allXtbProps[smiles]['xTB_Lin'])
        try:
            xTB_ML = float(allXtbProps[smiles]['xTB_ML'])
        except:
            continue
        try:
            xTB_T1 = float(allXtbProps[smiles]['xTB_T1'])
        except:
            continue
        try:
            xTB_ML_T1 = float(allXtbProps[smiles]['xTB_ML_T1'])
        except:
            continue
        try:
            xTB_ML_S1 = float(allXtbProps[smiles]['xTB_ML_S1'])
        except:
            continue
        xTBs.append(xTB)
        xTB_Lins.append(xTB_Lin)
        xTB_MLs.append(xTB_ML)
        xTB_T1s.append(xTB_T1)
        xTB_ML_T1s.append(xTB_ML_T1)
        xTB_ML_S1s.append(xTB_ML_S1)
        final_smiles.append(smiles)
        FOM_sens = 0 if xTB_ML_T1 > xTB_ML_S1 else np.exp(-(1 - xTB_ML_T1 / xTB_ML_S1))
        # np.exp(-abs(1.1 - xTB_ML_S1))
        FOM_emit = 0 if xTB_ML_S1 > 2 * xTB_ML_T1 else np.exp(-(2 - xTB_ML_S1 / xTB_ML_T1))
        # np.exp(-abs(2 - xTB_ML_S1))
        FOM_sens_all.append(FOM_sens)
        FOM_emit_all.append(FOM_emit)
        if FOM_sens > 0.9:
            temp = {'smiles': smiles, 'FOM': FOM_sens, 'S1': xTB_ML_S1, 'T1': xTB_ML_T1}
            sens_smiles.append(temp)
            # sens_smiles[smiles] = {}
            # sens_smiles[smiles]['S1'] = xTB_ML_S1
            # sens_smiles[smiles]['T1'] = xTB_ML_T1
        if FOM_emit > 0.9:
            temp = {'smiles': smiles, 'FOM': FOM_emit, 'S1': xTB_ML_S1, 'T1': xTB_ML_T1}
            emit_smiles.append(temp)
            # emit_smiles[smiles] = {}
            # emit_smiles[smiles]['S1'] = xTB_ML_S1
            # emit_smiles[smiles]['T1'] = xTB_ML_T1
    sens_smiles = sorted(sens_smiles, key=lambda k: k['FOM'])
    emit_smiles = sorted(emit_smiles, key=lambda k: k['FOM'])
    with open('pot_sens_mopssam.csv', 'w') as file:
        file.write('smiles,xtbMLS1,xtbMLT1,FOM\n')
        for smiles in sens_smiles:
            file.write(smiles['smiles'] + ',' + str(smiles['S1']) + ',' + str(smiles['T1']) +
                       ',' + str(smiles['FOM']) + '\n')
            # file.write(smiles + ',' + str(sens_smiles[smiles]['S1']) + ',' + str(sens_smiles[smiles]['T1']) + '\n')
    with open('pot_emit_mopssam.csv', 'w') as file:
        file.write('smiles,xtbMLS1,xtbMLT1,FOM\n')
        for smiles in emit_smiles:
            file.write(smiles['smiles'] + ',' + str(smiles['S1']) + ',' + str(smiles['T1']) +
                       ',' + str(smiles['FOM']) + '\n')

    xTB1s = []
    xTB_indeps = []
    for smiles in allXtbProps:
        xTB = float(allXtbProps[smiles]['xTB'])
        try:
            xTB_indep = float(allXtbProps[smiles]['xTB_indep'])
        except:
            continue
        xTB1s.append(xTB)
        xTB_indeps.append(xTB_indep)

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    colors = [colors[2], colors[0], colors[1]]
    plt.rcParams['axes.prop_cycle'] = cycler(color=colors)

    fig = plt.figure(num=1, figsize=[8, 5], dpi=300, clear=True)
    ax = fig.add_subplot(1, 2, 1)
    ax.set_axisbelow(True)
    x = np.linspace(0, 9, 100)
    plt.plot(x, x, 'k--')
    plt.plot(xTBs, xTB_MLs, '.', markersize=2, color=colors_nipy[1], label='ML calib')
    plt.plot(xTBs, xTBs, '.', markersize=2, color=colors_nipy[-1], label='no calib')
    plt.plot(xTBs, xTB_Lins, '.', markersize=2, color=colors_nipy[3], label='lin calib')
    plt.grid(True)
    plt.legend(markerscale=6, fontsize=14)
    plt.xlabel('xTB-sTDA S$_1$ (eV)', fontsize=16)
    plt.ylabel('calib. xTB-sTDA S$_1$ (eV)', fontsize=16)
    # plt.title('stda S1 comparison')
    plt.xlim(0, 7)
    plt.ylim(0, 7)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    if MLtype == 'xtb':
        plt.savefig('mopssam_S1_calib_all.png')
    elif MLtype == 'direct':
        plt.savefig('mopssam_S1_calib_all_direct.png')
    else:
        plt.savefig('mopssam_S1_calib_all_multi.png')

    dfLinTDDFT = pd.read_csv('TDDFT_MOPSSAM_test_data.csv')
    dfLinXTB = pd.read_csv('xTB_xtb_tddft_calib_data.csv')
    dfLin = pd.merge(dfLinTDDFT, dfLinXTB)
    regrT1 = linear_model.LinearRegression()
    regrT1.fit(np.array(dfLin['xtbT1']).reshape(-1, 1),
               np.array(dfLin['T1']).reshape(-1, 1))
    xTB_Lins_T1 = [x[0] for x in regrT1.predict(np.array(xTB_T1s).reshape(-1, 1))]

    fig = plt.figure(num=1)
    ax = fig.add_subplot(1, 2, 2)
    ax.set_axisbelow(True)
    x = np.linspace(0, 9, 100)
    plt.plot(x, x, 'k--')
    plt.plot(xTB_T1s, xTB_ML_T1s, '.', color=colors_nipy[1], markersize=2, label='ML calib')
    plt.plot(xTB_T1s, xTB_T1s, '.', markersize=2, color=colors_nipy[-1], label='no calib')
    plt.plot(xTB_T1s, xTB_Lins_T1, '.', markersize=2, color=colors_nipy[3], label='lin calib')
    plt.grid(True)
    plt.legend(markerscale=6, fontsize=14)
    plt.xlabel('xTB-sTDA T$_1$ (eV)', fontsize=16)
    plt.ylabel('calib. xTB-sTDA T$_1$ (eV)', fontsize=16)
    # plt.title('stda T1 comparison')
    plt.xlim(0, 7)
    plt.ylim(0, 7)
    plt.gca().set_aspect('equal', adjustable='box')
    label_axes(fig, ha='left')
    plt.tight_layout()
    plt.savefig('mopssam_calib_all.png')

    fig = plt.figure(num=2, figsize=[8, 5], dpi=300)
    ax = fig.add_subplot(1, 2, 1)
    ax.set_axisbelow(True)
    x = np.linspace(0, 9, 100)
    plt.plot(x, x, 'k--')
    colors = FOM_sens_all
    plt.scatter(xTB_ML_T1s, xTB_ML_S1s, c=colors, s=0.1, cmap='nipy_spectral_r')
    plt.grid(True)
    # plt.legend()
    # cbar = plt.colorbar()
    plt.clim(0.3, 1)
    # cbar.set_label('Sensitizer FOM', rotation=90)
    plt.xlabel('xTB-ML T$_1$ (eV)', fontsize=16)
    plt.ylabel('xTB-ML S$_1$ (eV)', fontsize=16)
    plt.title('Sensitizers', fontsize=16)
    plt.xlim(0, 7)
    plt.ylim(0, 7)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    # plt.savefig('mopssam_calib_all_sens.png')

    fig = plt.figure(num=2)
    ax = fig.add_subplot(1, 2, 2)
    ax.set_axisbelow(True)
    x = np.linspace(0, 9, 100)
    plt.plot(x, x, 'k--')
    colors = FOM_emit_all
    im = plt.scatter(xTB_ML_T1s, xTB_ML_S1s, c=colors, s=0.1, cmap='nipy_spectral_r')
    plt.grid(True)
    # plt.legend()
    # cbar = plt.colorbar()
    plt.xlabel('xTB-ML T$_1$ (eV)', fontsize=16)
    # plt.ylabel('xTB-ML S$_1$ (eV)', fontsize=16)
    plt.title('Emitters', fontsize=16)
    plt.xlim(0, 7)
    plt.ylim(0, 7)
    plt.gca().set_aspect('equal', adjustable='box')
    label_axes(fig)
    plt.tight_layout()

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.21, 0.03, 0.58])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Sensitizer/Emitter FOM', rotation=90, fontsize=16)
    plt.clim(0.3, 1)

    plt.savefig('mopssam_calib_all_sens_emit.png')

    fig = plt.figure(num=1, clear=True)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_axisbelow(True)
    x = np.linspace(0, 9, 100)
    plt.plot(x, x, 'k--')
    plt.plot(xTB_indeps, xTB1s, 'b.', markersize=0.1)
    plt.grid(True)
    # plt.legend()
    plt.xlabel('xTB S1, independent (eV)')
    plt.ylabel('xTB S1, MOPSSAM (eV)')
    plt.title('S1 comparison vs. MOPSSAM')
    plt.xlim(0, 7)
    plt.ylim(0, 7)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig('mopssam_xtb_comp.png')


def plot_calib_new():
    df = pd.read_csv('all_data_scop_qm_AL_ex_xtb_calib_all.csv')

    fig = plt.figure(num=1, figsize=[8, 8], dpi=300, clear=True)
    ax = fig.add_subplot(2, 2, 1)
    ax.set_axisbelow(True)
    x = np.linspace(0, 9, 100)
    plt.plot(x, x, 'k--')
    plt.plot(df['xtbS1'], df['xTB_ML_S1'], '.', markersize=2, color=colors_nipy[1], label='ML calib')
    plt.plot(df['xtbS1'], df['xtbS1'], '.', markersize=2, color=colors_nipy[-1], label='no calib')
    plt.plot(df['xtbS1'], df['xTB_Lin_S1'], '.', markersize=2, color=colors_nipy[3], label='lin calib')
    plt.grid(True)
    plt.legend(markerscale=6, fontsize=14)
    plt.xlabel('xTB-sTDA S$_1$ (eV)', fontsize=16)
    plt.ylabel('calib. xTB-sTDA S$_1$ (eV)', fontsize=16)
    # plt.title('stda S1 comparison')
    plt.xlim(0, 7)
    plt.ylim(0, 7)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    # if MLtype == 'xtb':
    #     plt.savefig('mopssam_S1_calib_all.png')
    # elif MLtype == 'direct':
    #     plt.savefig('mopssam_S1_calib_all_direct.png')
    # else:
    #     plt.savefig('mopssam_S1_calib_all_multi.png')

    ax = fig.add_subplot(2, 2, 2)
    ax.set_axisbelow(True)
    x = np.linspace(0, 9, 100)
    plt.plot(x, x, 'k--')
    plt.plot(df['xtbT1'], df['xTB_ML_T1'], '.', color=colors_nipy[1], markersize=2, label='ML calib')
    plt.plot(df['xtbT1'], df['xtbT1'], '.', markersize=2, color=colors_nipy[-1], label='no calib')
    plt.plot(df['xtbT1'], df['xTB_Lin_T1'], '.', markersize=2, color=colors_nipy[3], label='lin calib')
    plt.grid(True)
    plt.legend(markerscale=6, fontsize=14)
    plt.xlabel('xTB-sTDA T$_1$ (eV)', fontsize=16)
    plt.ylabel('calib. xTB-sTDA T$_1$ (eV)', fontsize=16)
    # plt.title('stda T1 comparison')
    plt.xlim(0, 7)
    plt.ylim(0, 7)
    plt.gca().set_aspect('equal', adjustable='box')
    label_axes(fig, ha='left')
    plt.tight_layout()
    # plt.savefig('mopssam_calib_all.png')

    df['FOM_sens'] = np.exp(-np.abs(1 - df['xTB_ML_T1'] / df['xTB_ML_S1']))
    # FOM_sens = 0 if xTB_ML_T1 > xTB_ML_S1 else np.exp(-(1 - xTB_ML_T1 / xTB_ML_S1))
    # np.exp(-abs(1.1 - xTB_ML_S1))
    df['FOM_emit'] = np.exp(-np.abs(2 - df['xTB_ML_S1'] / df['xTB_ML_T1']))
    # FOM_emit = 0 if xTB_ML_S1 > 2 * xTB_ML_T1 else np.exp(-(2 - xTB_ML_S1 / xTB_ML_T1))
    # np.exp(-abs(2 - xTB_ML_S1))
    # FOM_sens_all.append(FOM_sens)
    # FOM_emit_all.append(FOM_emit)

    # fig = plt.figure(num=2, figsize=[8, 5], dpi=300)
    ax = fig.add_subplot(2, 2, 3)
    ax.set_axisbelow(True)
    x = np.linspace(0, 9, 100)
    plt.plot(x, x, 'k--')
    colors = df['FOM_sens']
    plt.scatter(df['xTB_ML_T1'], df['xTB_ML_S1'], c=colors, s=0.1, cmap='nipy_spectral_r')
    plt.grid(True)
    # plt.legend()
    # cbar = plt.colorbar()
    plt.clim(0.3, 1)
    # cbar.set_label('Sensitizer FOM', rotation=90)
    plt.xlabel('xTB-ML T$_1$ (eV)', fontsize=16)
    plt.ylabel('xTB-ML S$_1$ (eV)', fontsize=16)
    plt.title('Sensitizers', fontsize=16)
    plt.xlim(0, 7)
    plt.ylim(0, 7)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    # plt.savefig('mopssam_calib_all_sens.png')

    # fig = plt.figure(num=2)
    ax = fig.add_subplot(2, 2, 4)
    ax.set_axisbelow(True)
    x = np.linspace(0, 9, 100)
    plt.plot(x, x, '--', color='gray')
    plt.plot(x, 2 * x, 'k--')
    colors = df['FOM_emit']
    im = plt.scatter(df['xTB_ML_T1'], df['xTB_ML_S1'], c=colors, s=0.1, cmap='nipy_spectral_r')
    # colors = FOM_emit_all
    # im = plt.scatter(xTB_ML_T1s, xTB_ML_S1s, c=colors, s=0.1, cmap='nipy_spectral_r')
    plt.grid(True)
    # plt.legend()
    # cbar = plt.colorbar()
    plt.xlabel('xTB-ML T$_1$ (eV)', fontsize=16)
    # plt.ylabel('xTB-ML S$_1$ (eV)', fontsize=16)
    plt.title('Emitters', fontsize=16)
    plt.xlim(0, 7)
    plt.ylim(0, 7)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.colorbar(fraction=0.046, pad=0.04)
    fig.subplots_adjust(hspace=0.32)
    # fig.subplots_adjust(right=0.85)
    # cbar_ax = fig.add_axes([0.88, 0.21, 0.03, 0.58])
    # cbar = fig.colorbar(im, cax=cbar_ax)
    # cbar.set_label('Sensitizer/Emitter FOM', rotation=90, fontsize=16)
    # plt.clim(0.3, 1)

    label_axes(fig, ha='left')
    plt.savefig('mopssam_calib_all_sens_emit.pdf')
    pass


# gen_train_data()
# gen_test_data()
# do_ML()
# do_direct_ML()
# analyze_ML('xtb')
analyze_ML_direct()
# plot_calib('xtb')
plot_calib_new()

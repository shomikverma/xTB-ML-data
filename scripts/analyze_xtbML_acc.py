import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn import linear_model
import pandas as pd
import string
from itertools import cycle
from cycler import cycler


def label_axes(fig, labels=None, loc=None, **kwargs):
    if labels is None:
        labels = string.ascii_lowercase
    labels = cycle(labels)
    if loc is None:
        loc = (-0.1, 1.1)
    for ax, lab in zip(fig.axes, labels):
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


def analyze_xtbML(filename):
    with open('TDDFT_' + filename + '.csv', 'r') as file:
        dataGauss = file.readlines()

    with open('xTB_ML_errs_' + filename + '.csv', 'r') as file:
        dataXtb = file.readlines()

    with open('xTB_ML_calib_' + filename + '.csv', 'r') as file:
        dataXtbML = file.readlines()

    # with open('TDDFT_MOPSSAM_test_data.csv.csv', 'r') as file:
    #     dataLinTDDFT = file.readlines()
    #
    # with open('xTB_xtb_tddft_calib_data.csv', 'r') as file:
    #     dataLinXTB = file.readlines()

    idx = np.random.choice(np.arange(min(len(dataGauss), len(dataXtb), len(dataXtbML))),
                           min(len(dataGauss), len(dataXtb), len(dataXtbML), 1_000),
                           replace=False)
    dataGauss = np.array(dataGauss)[idx]
    dataXtb = np.array(dataXtb)[idx]
    dataXtbML = np.array(dataXtbML)[idx]

    S1comp = {}
    for line in dataGauss:
        if 'smiles' in line.lower():
            continue
        line = line.replace('\n', '')
        line = line.split(',')
        smiles = line[0]
        S1 = float(line[1])
        T1 = float(line[2])
        if 0 < S1 < 10 and 0 < T1 < 10:
            S1comp[smiles] = {}
            S1comp[smiles]['GaussS1'] = S1
            S1comp[smiles]['GaussT1'] = T1

    for line in dataXtb:
        if 'smiles' in line.lower():
            continue
        line = line.replace('\n', '')
        line = line.split(',')
        smiles = line[0]
        xtb_S1 = float(line[1])
        xtb_T1 = float(line[2])
        try:
            GaussS1 = S1comp[smiles]['GaussS1']
        except KeyError:
            continue
        if 0 < xtb_S1 < 10 and 0 < xtb_T1 < 10:
            S1comp[smiles]['xtb_S1'] = xtb_S1
            S1comp[smiles]['xtb_T1'] = xtb_T1

    for line in dataXtbML:
        if 'smiles' in line.lower():
            continue
        line = line.replace('\n', '')
        line = line.split(',')
        smiles = line[0]
        xtb_ML_S1 = float(line[1])
        xtb_ML_T1 = float(line[2])
        try:
            GaussS1 = S1comp[smiles]['GaussS1']
        except KeyError:
            continue
        if 0.25 < xtb_ML_S1 < 10 and 0 < xtb_ML_T1 < 10:
            S1comp[smiles]['xtb_ML_S1'] = xtb_ML_S1
            S1comp[smiles]['xtb_ML_T1'] = xtb_ML_T1

    dfLinTDDFT = pd.read_csv('TDDFT_MOPSSAM_test_data.csv')
    dfLinXTB = pd.read_csv('xTB_xtb_tddft_calib_data.csv')
    dfLin = pd.merge(dfLinTDDFT, dfLinXTB)
    regrS1 = linear_model.LinearRegression()
    regrS1.fit(np.array(dfLin['xtbS1']).reshape(-1, 1),
               np.array(dfLin['S1']).reshape(-1, 1))
    regrT1 = linear_model.LinearRegression()
    regrT1.fit(np.array(dfLin['xtbT1']).reshape(-1, 1),
               np.array(dfLin['T1']).reshape(-1, 1))

    GaussS1s = []
    GaussT1s = []
    xtbS1s = []
    xtbT1s = []
    xtbMLS1s = []
    xtbMLT1s = []
    for smiles in S1comp:
        allKeys = True
        testKeys = ['GaussS1', 'GaussT1', 'xtb_S1', 'xtb_T1', 'xtb_ML_S1', 'xtb_ML_T1']
        for key in testKeys:
            try:
                temp = S1comp[smiles][key]
            except:
                allKeys = False
                break
        if not allKeys:
            continue
        GaussS1s.append(S1comp[smiles]['GaussS1'])
        GaussT1s.append(S1comp[smiles]['GaussT1'])
        try:
            xtbS1s.append(S1comp[smiles]['xtb_S1'])
        except:
            print(smiles)
        xtbT1s.append(S1comp[smiles]['xtb_T1'])
        xtbMLS1s.append(S1comp[smiles]['xtb_ML_S1'])
        xtbMLT1s.append(S1comp[smiles]['xtb_ML_T1'])

    idx = np.random.choice(np.arange(len(xtbS1s)), int(0.8 * len(xtbS1s)), replace=False)
    # TrainXtbS1s = np.array(xtbS1s)[idx]
    # TrainGaussS1s = np.array(GaussS1s)[idx]
    # regrS1 = linear_model.LinearRegression()
    # regrS1.fit(np.array(TrainXtbS1s).reshape(-1, 1),
    #          np.array(TrainGaussS1s).reshape(-1, 1))
    xtbLinS1s = [x[0] for x in regrS1.predict(np.array(xtbS1s).reshape(-1, 1))]

    idx = np.random.choice(np.arange(len(xtbT1s)), int(0.8 * len(xtbT1s)), replace=False)
    # TrainXtbT1s = np.array(xtbT1s)[idx]
    # TrainGaussT1s = np.array(GaussT1s)[idx]
    # regrT1 = linear_model.LinearRegression()
    # regrT1.fit(np.array(TrainXtbT1s).reshape(-1, 1),
    #          np.array(TrainGaussT1s).reshape(-1, 1))
    xtbLinT1s = [x[0] for x in regrT1.predict(np.array(xtbT1s).reshape(-1, 1))]

    r2_orig = r2_score(GaussS1s, xtbS1s)
    MAE_orig = mean_absolute_error(GaussS1s, xtbS1s)
    r2_lin = r2_score(GaussS1s, xtbLinS1s)
    MAE_lin = mean_absolute_error(GaussS1s, xtbLinS1s)
    r2_ML = r2_score(GaussS1s, xtbMLS1s)
    MAE_ML = mean_absolute_error(GaussS1s, xtbMLS1s)

    print('S1')
    print(r2_orig)
    print(r2_lin)
    print(r2_ML)
    print(MAE_orig)
    print(MAE_lin)
    print(MAE_ML)

    fig = plt.figure(num=1, clear=True, figsize=[7, 4], dpi=300)
    ax = fig.add_subplot(1, 2, 1)
    ax.set_axisbelow(True)
    x = np.linspace(0, 9, 100)
    plt.plot(xtbS1s, GaussS1s, '.', color=colors_nipy[-1], label='orig')
    plt.plot(xtbLinS1s, GaussS1s, '.', color=colors_nipy[2], label='lin calib')
    plt.plot(xtbMLS1s, GaussS1s, '.', color=colors_nipy[1], label='ML calib')
    plt.plot(x, x, 'k--')
    plt.grid(True)
    plt.legend(markerscale=2, fontsize=14)
    plt.xlabel('xTB-sTDA S$_1$ (eV)', fontsize=16)
    plt.ylabel('TDDFT S$_1$ (eV)', fontsize=16)
    # plt.title('TDDFT vs. sTDA S1 comparison')
    plt.xlim(0, 9)
    plt.ylim(0, 9)
    plt.annotate('R$^2$ orig: %0.2f\n' % r2_orig +
                 'R$^2$ lin: %0.2f\n' % r2_lin +
                 'R$^2$ ML: %0.2f\n' % r2_ML +
                 'MAE orig: %0.2f\n' % MAE_orig +
                 'MAE lin: %0.2f\n' % MAE_lin +
                 'MAE ML: %0.2f' % MAE_ML,
                 (8.5, 0.5),
                 bbox=dict(facecolor='white', alpha=0.5),
                 fontsize=12,
                 ha='right')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    # plt.savefig('S1_calib_' + filename + '.png')

    r2_orig = r2_score(GaussT1s, xtbT1s)
    MAE_orig = mean_absolute_error(GaussT1s, xtbT1s)
    r2_lin = r2_score(GaussT1s, xtbLinT1s)
    MAE_lin = mean_absolute_error(GaussT1s, xtbLinT1s)
    r2_ML = r2_score(GaussT1s, xtbMLT1s)
    MAE_ML = mean_absolute_error(GaussT1s, xtbMLT1s)

    print('T1')
    print(r2_orig)
    print(r2_lin)
    print(r2_ML)
    print(MAE_orig)
    print(MAE_lin)
    print(MAE_ML)

    ax = fig.add_subplot(1, 2, 2)
    ax.set_axisbelow(True)
    x = np.linspace(0, 9, 100)
    plt.plot(xtbT1s, GaussT1s, '.', color=colors_nipy[-1], label='orig')
    plt.plot(xtbLinT1s, GaussT1s, '.', color=colors_nipy[2], label='lin calib')
    plt.plot(xtbMLT1s, GaussT1s, '.', color=colors_nipy[1], label='ML calib')
    plt.plot(x, x, 'k--')
    plt.grid(True)
    plt.legend(markerscale=2, fontsize=14)
    plt.xlabel('xTB-sTDA T$_1$ (eV)', fontsize=16)
    plt.ylabel('TDDFT T$_1$ (eV)', fontsize=16)
    # plt.title('TDDFT vs. sTDA T1 comparison')
    plt.xlim(0, 9)
    plt.ylim(0, 9)
    plt.annotate('R$^2$ orig: %0.2f\n' % r2_orig +
                 'R$^2$ lin: %0.2f\n' % r2_lin +
                 'R$^2$ ML: %0.2f\n' % r2_ML +
                 'MAE orig: %0.2f\n' % MAE_orig +
                 'MAE lin: %0.2f\n' % MAE_lin +
                 'MAE ML: %0.2f' % MAE_ML,
                 (8.5, 0.5),
                 bbox=dict(facecolor='white', alpha=0.5),
                 fontsize=12,
                 ha='right')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    label_axes(fig, ha='left')
    plt.savefig('calib_' + filename + '.pdf')


def analyze_xtbML_noT1(filename):
    df_tddft = pd.read_csv('TDDFT_' + filename + '.csv')
    tddft_col = 'S1-CAM-def2TZVP-eV'
    methodName = tddft_col.replace('S1-', '').replace('S1_', '').replace('-eV', '').replace('_eV', '')
    df_tddft = df_tddft[0 < df_tddft[tddft_col]][df_tddft[tddft_col] < 10]
    df_xtb = pd.read_csv('xTB_' + filename + '.csv')
    xtb_col = 'xtbS1'
    df_xtb = df_xtb[0 < df_xtb[xtb_col]][df_xtb[xtb_col] < 10]
    df_xtb_ML = pd.read_csv('xTB_ML_calib_' + filename + '.csv')
    xtb_ML_col = 'S1_xTB_ML'
    df_xtb_ML = df_xtb_ML[0 < df_xtb_ML[xtb_ML_col]][df_xtb_ML[xtb_ML_col] < 10]
    df_all = df_xtb.merge(df_tddft, on='SMILES', how='inner').merge(df_xtb_ML, on='SMILES', how='inner')

    GaussS1s = df_all[tddft_col]
    xtbS1s = df_all[xtb_col]
    xtbMLS1s = df_all[xtb_ML_col]
    r2_orig = r2_score(GaussS1s, xtbS1s)
    MAE_orig = mean_absolute_error(GaussS1s, xtbS1s)
    # r2_lin = r2_score(GaussS1s, xtbLinS1s)
    # MAE_lin = mean_absolute_error(GaussS1s, xtbLinS1s)
    r2_ML = r2_score(GaussS1s, xtbMLS1s)
    MAE_ML = mean_absolute_error(GaussS1s, xtbMLS1s)

    fig = plt.figure(num=1, clear=True, figsize=[3.5, 4], dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    x = np.linspace(0, 10, 100)
    plt.plot(xtbS1s, GaussS1s, '.', label='orig')
    # plt.plot(xtbLinS1s, GaussS1s, '.', label='lin calib')
    plt.plot(xtbMLS1s, GaussS1s, '.', label='ML calib')
    plt.plot(x, x, 'k--')
    plt.grid(True)
    plt.legend()
    plt.xlabel('sTDA S1 (eV)')
    plt.ylabel('' + methodName + ' S1 (eV)')
    plt.title('' + methodName + ' vs. sTDA S1 comparison')
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.annotate('R2 orig: %0.2f\n' % r2_orig +
                 # 'R2 lin: %0.2f\n' % r2_lin +
                 'R2 ML: %0.2f\n' % r2_ML +
                 'MAE orig: %0.2f\n' % MAE_orig +
                 # 'MAE lin: %0.2f\n' % MAE_lin +
                 'MAE ML: %0.2f' % MAE_ML,
                 (9.5, 0.5),
                 bbox=dict(facecolor='white', alpha=0.5),
                 ha='right')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig('calib_' + filename + '_' + methodName + '_noT1.png')


def run_xTB_ML(filename):
    # run chemprop
    os.system('chemprop_predict --test_path xTB_' + filename +
              '.csv --checkpoint_dir xTB_ML_model_ '
              '--preds_path xTB_ML_calib_scop_qm_' + filename + '.csv')
    os.system('chemprop_predict --test_path xTB_' + filename +
              '.csv --checkpoint_dir xTB_ML_model_scop_qm_ALS1T1_qmex '
              '--preds_path xTB_ML_calib_scop_qm_ALS1T1_qmex_' + filename + '.csv')
    # calibrate xTB with ML
    df_scqm = pd.read_csv('xTB_ML_calib_scop_qm_' + filename + '.csv')
    df_scqm_exp = pd.read_csv('xTB_ML_calib_scop_qm_ALS1T1_qmex_' + filename + '.csv')
    df_scqm['xtbMLS1'] = df_scqm['xtbS1'] + df_scqm['S1err']
    df_scqm['xtbMLT1'] = df_scqm['xtbT1'] + df_scqm['T1err']
    df_scqm_exp['xtbMLS1_exp'] = df_scqm_exp['xtbS1'] + df_scqm_exp['S1err']
    df_scqm_exp['xtbMLT1_exp'] = df_scqm_exp['xtbT1'] + df_scqm_exp['T1err']
    # calibrate xTB with Lin reg
    dfLinTDDFT = pd.read_csv('TDDFT_MOPSSAM_test_data.csv')
    dfLinXTB = pd.read_csv('xTB_xtb_tddft_calib_data.csv')
    dfLin = pd.merge(dfLinTDDFT, dfLinXTB)
    regrS1 = linear_model.LinearRegression()
    regrS1.fit(np.array(dfLin['xtbS1']).reshape(-1, 1),
               np.array(dfLin['S1']).reshape(-1, 1))
    regrT1 = linear_model.LinearRegression()
    regrT1.fit(np.array(dfLin['xtbT1']).reshape(-1, 1),
               np.array(dfLin['T1']).reshape(-1, 1))
    df_scqm['xtbLinS1'] = [x[0] for x in regrS1.predict(np.array(df_scqm['xtbS1']).reshape(-1, 1))]
    df_scqm['xtbLinT1'] = [x[0] for x in regrT1.predict(np.array(df_scqm['xtbT1']).reshape(-1, 1))]
    # export
    df_scqm.to_csv('xTB_ML_calib_scop_qm_' + filename + '.csv', index=False)
    df_scqm_exp.to_csv('xTB_ML_calib_scop_qm_ALS1T1_qmex_' + filename + '.csv', index=False)


def analyze_xTB_ML_comp(filename):
    df_scqm = pd.read_csv('xTB_ML_calib_scop_qm_' + filename + '.csv')
    df_scqm_exp = pd.read_csv('xTB_ML_calib_scop_qm_ALS1T1_qmex_' + filename + '.csv',
                              usecols=['SMILES', 'xtbMLS1_exp', 'xtbMLT1_exp'])
    df_tddft = pd.read_csv('TDDFT_' + filename + '.csv')
    df_tddft.columns = ['SMILES', 'S1', 'T1']
    df_tddft = df_tddft[0 < df_tddft['S1']][df_tddft['S1'] < 10]
    df_tddft = df_tddft[0 < df_tddft['T1']][df_tddft['T1'] < 10]
    df_scqm = df_scqm[0 < df_scqm['xtbMLS1']][df_scqm['xtbMLS1'] < 10]
    df_scqm = df_scqm[0 < df_scqm['xtbMLT1']][df_scqm['xtbMLT1'] < 10]
    df_scqm_exp = df_scqm_exp[0 < df_scqm_exp['xtbMLS1_exp']][df_scqm_exp['xtbMLS1_exp'] < 10]
    df_scqm_exp = df_scqm_exp[0 < df_scqm_exp['xtbMLT1_exp']][df_scqm_exp['xtbMLT1_exp'] < 10]
    df_all = df_tddft.merge(df_scqm, on='SMILES', how='inner').merge(df_scqm_exp, on='SMILES', how='inner')
    if len(df_all) > 1_000:
        df_all = df_all.sample(1_000, random_state=1)
    df_all.to_csv('xTB_ML_calib_comp_' + filename + '.csv', index=False)

    GaussS1s = df_all['S1']
    xtbS1s = df_all['xtbS1']
    xtbLinS1s = df_all['xtbLinS1']
    xtbMLS1s = df_all['xtbMLS1']
    xtbMLexpS1s = df_all['xtbMLS1_exp']

    GaussT1s = df_all['T1']
    xtbT1s = df_all['xtbT1']
    xtbLinT1s = df_all['xtbLinT1']
    xtbMLT1s = df_all['xtbMLT1']
    xtbMLexpT1s = df_all['xtbMLT1_exp']

    r2_orig = r2_score(GaussS1s, xtbS1s)
    MAE_orig = mean_absolute_error(GaussS1s, xtbS1s)
    r2_lin = r2_score(GaussS1s, xtbLinS1s)
    MAE_lin = mean_absolute_error(GaussS1s, xtbLinS1s)
    r2_ML = r2_score(GaussS1s, xtbMLS1s)
    MAE_ML = mean_absolute_error(GaussS1s, xtbMLS1s)
    r2_ML_exp = r2_score(GaussS1s, xtbMLexpS1s)
    MAE_ML_exp = mean_absolute_error(GaussS1s, xtbMLexpS1s)

    print('S1')
    print(r2_orig)
    print(r2_lin)
    print(r2_ML)
    print(r2_ML_exp)
    print(MAE_orig)
    print(MAE_lin)
    print(MAE_ML)
    print(MAE_ML_exp)

    fig = plt.figure(num=1, clear=True, figsize=[7, 4], dpi=300)
    ax = fig.add_subplot(1, 2, 1)
    ax.set_axisbelow(True)
    x = np.linspace(0, 9, 100)
    plt.plot(xtbS1s, GaussS1s, '.', label='orig')
    plt.plot(xtbLinS1s, GaussS1s, '.', label='lin calib')
    plt.plot(xtbMLS1s, GaussS1s, '.', label='ML calib')
    plt.plot(x, x, 'k--')
    plt.grid(True)
    plt.legend()
    plt.xlabel('sTDA S1 (eV)')
    plt.ylabel('TDDFT S1 (eV)')
    plt.title('TDDFT vs. sTDA S1 comparison: 20k')
    plt.xlim(0, 9)
    plt.ylim(0, 9)
    plt.annotate('R2 orig: %0.3f\n' % r2_orig +
                 'R2 lin: %0.3f\n' % r2_lin +
                 'R2 ML: %0.3f\n' % r2_ML +
                 'MAE orig: %0.3f\n' % MAE_orig +
                 'MAE lin: %0.3f\n' % MAE_lin +
                 'MAE ML: %0.3f' % MAE_ML,
                 (8.5, 0.5),
                 bbox=dict(facecolor='white', alpha=0.5),
                 ha='right')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()

    fig = plt.figure(num=2, clear=True, figsize=[7, 4], dpi=300)
    ax = fig.add_subplot(1, 2, 1)
    ax.set_axisbelow(True)
    x = np.linspace(0, 9, 100)
    plt.plot(xtbS1s, GaussS1s, '.', label='orig')
    plt.plot(xtbLinS1s, GaussS1s, '.', label='lin calib')
    plt.plot(xtbMLexpS1s, GaussS1s, '.', label='ML calib')
    plt.plot(x, x, 'k--')
    plt.grid(True)
    plt.legend()
    plt.xlabel('sTDA S1 (eV)')
    plt.ylabel('TDDFT S1 (eV)')
    plt.title('TDDFT vs. sTDA S1 comparison: 300k')
    plt.xlim(0, 9)
    plt.ylim(0, 9)
    plt.annotate('R2 orig: %0.3f\n' % r2_orig +
                 'R2 lin: %0.3f\n' % r2_lin +
                 'R2 ML: %0.3f\n' % r2_ML_exp +
                 'MAE orig: %0.3f\n' % MAE_orig +
                 'MAE lin: %0.3f\n' % MAE_lin +
                 'MAE ML: %0.3f' % MAE_ML_exp,
                 (8.5, 0.5),
                 bbox=dict(facecolor='white', alpha=0.5),
                 ha='right')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()

    # plt.savefig('S1_calib_' + filename + '.png')

    r2_orig = r2_score(GaussT1s, xtbT1s)
    MAE_orig = mean_absolute_error(GaussT1s, xtbT1s)
    r2_lin = r2_score(GaussT1s, xtbLinT1s)
    MAE_lin = mean_absolute_error(GaussT1s, xtbLinT1s)
    r2_ML = r2_score(GaussT1s, xtbMLT1s)
    MAE_ML = mean_absolute_error(GaussT1s, xtbMLT1s)
    r2_ML_exp = r2_score(GaussT1s, xtbMLexpT1s)
    MAE_ML_exp = mean_absolute_error(GaussT1s, xtbMLexpT1s)

    print('T1')
    print(r2_orig)
    print(r2_lin)
    print(r2_ML)
    print(r2_ML_exp)
    print(MAE_orig)
    print(MAE_lin)
    print(MAE_ML)
    print(MAE_ML_exp)

    fig = plt.figure(num=1)
    ax = fig.add_subplot(1, 2, 2)
    ax.set_axisbelow(True)
    x = np.linspace(0, 9, 100)
    plt.plot(xtbT1s, GaussT1s, '.', label='orig')
    plt.plot(xtbLinT1s, GaussT1s, '.', label='lin calib')
    plt.plot(xtbMLT1s, GaussT1s, '.', label='ML calib')
    plt.plot(x, x, 'k--')
    plt.grid(True)
    plt.legend()
    plt.xlabel('sTDA T1 (eV)')
    plt.ylabel('TDDFT T1 (eV)')
    plt.title('TDDFT vs. sTDA T1 comparison: 20k')
    plt.xlim(0, 9)
    plt.ylim(0, 9)
    plt.annotate('R2 orig: %0.3f\n' % r2_orig +
                 'R2 lin: %0.3f\n' % r2_lin +
                 'R2 ML: %0.3f\n' % r2_ML +
                 'MAE orig: %0.3f\n' % MAE_orig +
                 'MAE lin: %0.3f\n' % MAE_lin +
                 'MAE ML: %0.3f' % MAE_ML,
                 (8.5, 0.5),
                 bbox=dict(facecolor='white', alpha=0.5),
                 ha='right')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    label_axes(fig, ha='left')
    plt.savefig('calib_' + filename + '.png')

    fig = plt.figure(num=2)
    ax = fig.add_subplot(1, 2, 2)
    ax.set_axisbelow(True)
    x = np.linspace(0, 9, 100)
    plt.plot(xtbT1s, GaussT1s, '.', label='orig')
    plt.plot(xtbLinT1s, GaussT1s, '.', label='lin calib')
    plt.plot(xtbMLexpT1s, GaussT1s, '.', label='ML calib')
    plt.plot(x, x, 'k--')
    plt.grid(True)
    plt.legend()
    plt.xlabel('sTDA T1 (eV)')
    plt.ylabel('TDDFT T1 (eV)')
    plt.title('TDDFT vs. sTDA T1 comparison: 300k')
    plt.xlim(0, 9)
    plt.ylim(0, 9)
    plt.annotate('R2 orig: %0.3f\n' % r2_orig +
                 'R2 lin: %0.3f\n' % r2_lin +
                 'R2 ML: %0.3f\n' % r2_ML_exp +
                 'MAE orig: %0.3f\n' % MAE_orig +
                 'MAE lin: %0.3f\n' % MAE_lin +
                 'MAE ML: %0.3f' % MAE_ML_exp,
                 (8.5, 0.5),
                 bbox=dict(facecolor='white', alpha=0.5),
                 ha='right')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    label_axes(fig, ha='left')
    plt.savefig('calib_exp_' + filename + '.png')


def compare_20k_300k_ML(filename):
    TDDFT_df = pd.read_csv('../xtb_results_' + filename + '/' + filename + '_xtb.csv',
                           usecols=['SMILES', 'TDDFT_S1', 'TDDFT_T1', 'xtb_S1', 'xtb_T1'])
    xTB_ML_300k_df = pd.read_csv('xTB_ML_calib_scop_qm_300k_' + filename + '.csv')
    xTB_ML_20k_df = pd.read_csv('xTB_ML_calib_scop_qm_' + filename + '.csv')
    xTB_ML_20k_df.rename({'S1_xTB_ML': 'S1_xTB_ML_20k', 'T1_xTB_ML': 'T1_xTB_ML_20k'}, axis='columns', inplace=True)
    df_compiled = TDDFT_df.merge(xTB_ML_300k_df).merge(xTB_ML_20k_df)
    print(df_compiled.info())
    MAE_T1_orig = mean_absolute_error(df_compiled['TDDFT_T1'], df_compiled['xtb_T1'])
    MAE_S1_orig = mean_absolute_error(df_compiled['TDDFT_S1'], df_compiled['xtb_S1'])
    MAE_T1_300k = mean_absolute_error(df_compiled['TDDFT_T1'], df_compiled['T1_xTB_ML'])
    MAE_S1_300k = mean_absolute_error(df_compiled['TDDFT_S1'], df_compiled['S1_xTB_ML'])
    MAE_T1_20k = mean_absolute_error(df_compiled['TDDFT_T1'], df_compiled['T1_xTB_ML_20k'])
    MAE_S1_20k = mean_absolute_error(df_compiled['TDDFT_S1'], df_compiled['S1_xTB_ML_20k'])
    r2_T1_orig = r2_score(df_compiled['TDDFT_T1'], df_compiled['xtb_T1'])
    r2_S1_orig = r2_score(df_compiled['TDDFT_S1'], df_compiled['xtb_S1'])
    r2_T1_300k = r2_score(df_compiled['TDDFT_T1'], df_compiled['T1_xTB_ML'])
    r2_S1_300k = r2_score(df_compiled['TDDFT_S1'], df_compiled['S1_xTB_ML'])
    r2_T1_20k = r2_score(df_compiled['TDDFT_T1'], df_compiled['T1_xTB_ML_20k'])
    r2_S1_20k = r2_score(df_compiled['TDDFT_S1'], df_compiled['S1_xTB_ML_20k'])
    print(MAE_S1_orig, MAE_T1_orig)
    print(MAE_S1_20k, MAE_T1_20k)
    print(MAE_S1_300k, MAE_T1_300k)

    xtbS1s = df_compiled['xtb_S1']
    GaussS1s = df_compiled['TDDFT_S1']
    xtbMLS1s = df_compiled['S1_xTB_ML']
    xtbMLS1s_20k = df_compiled['S1_xTB_ML_20k']
    xtbT1s = df_compiled['xtb_T1']
    GaussT1s = df_compiled['TDDFT_T1']
    xtbMLT1s = df_compiled['T1_xTB_ML']
    xtbMLT1s_20k = df_compiled['T1_xTB_ML_20k']

    markersize = 1

    fig = plt.figure(num=1, clear=True, figsize=[7, 4], dpi=300)
    ax = fig.add_subplot(1, 2, 1)
    ax.set_axisbelow(True)
    x = np.linspace(0, 9, 100)
    plt.plot(xtbS1s, GaussS1s, '.', label='orig', markersize=markersize)
    plt.plot(xtbMLS1s_20k, GaussS1s, '.', label='20k calib', markersize=markersize)
    plt.plot(xtbMLS1s, GaussS1s, '.', label='300k calib', markersize=markersize)
    plt.plot(x, x, 'k--')
    plt.grid(True)
    plt.legend(markerscale=5 / markersize)
    plt.xlabel('sTDA S1 (eV)')
    plt.ylabel('TDDFT S1 (eV)')
    plt.title('TDDFT vs. sTDA S1 comparison')
    plt.xlim(0, 9)
    plt.ylim(0, 9)
    plt.annotate('R2 orig: %0.2f\n' % r2_S1_orig +
                 'R2 20k ML: %0.2f\n' % r2_S1_20k +
                 'R2 300k ML: %0.2f\n' % r2_S1_300k +
                 'MAE orig: %0.2f\n' % MAE_S1_orig +
                 'MAE 20k ML: %0.2f\n' % MAE_S1_20k +
                 'MAE 300k ML: %0.2f' % MAE_S1_300k,
                 (8.5, 0.5),
                 bbox=dict(facecolor='white', alpha=0.5),
                 ha='right')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    # plt.savefig('S1_calib_' + filename + '.png')

    ax = fig.add_subplot(1, 2, 2)
    ax.set_axisbelow(True)
    x = np.linspace(0, 9, 100)
    plt.plot(xtbT1s, GaussT1s, '.', label='orig', markersize=markersize)
    plt.plot(xtbMLT1s_20k, GaussT1s, '.', label='20k calib', markersize=markersize)
    plt.plot(xtbMLT1s, GaussT1s, '.', label='300k calib', markersize=markersize)
    plt.plot(x, x, 'k--')
    plt.grid(True)
    plt.legend(markerscale=5 / markersize, loc='upper left')
    plt.xlabel('sTDA T1 (eV)')
    plt.ylabel('TDDFT T1 (eV)')
    plt.title('TDDFT vs. sTDA T1 comparison')
    plt.xlim(0, 9)
    plt.ylim(0, 9)
    plt.annotate('R2 orig: %0.2f\n' % r2_T1_orig +
                 'R2 20k ML: %0.2f\n' % r2_T1_20k +
                 'R2 300k ML: %0.2f\n' % r2_T1_300k +
                 'MAE orig: %0.2f\n' % MAE_T1_orig +
                 'MAE 20k ML: %0.2f\n' % MAE_T1_20k +
                 'MAE 300k ML: %0.2f' % MAE_T1_300k,
                 (8.5, 0.5),
                 bbox=dict(facecolor='white', alpha=0.5),
                 ha='right')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    label_axes(fig, ha='left')
    plt.savefig('comp_20k_300k_' + filename + '.png')


analyze_xtbML('MOPSSAM_143')
# analyze_xtbML('MOPSSAM_1k')
# analyze_xtbML('INDT_smiles')
# analyze_xtbML('scop_qm_MOPSSAM_1k')
# analyze_xtbML_noT1('QM8')
# run_xTB_ML('MOPSSAM_1k')
# analyze_xTB_ML_comp('MOPSSAM_1k')
# run_xTB_ML('INDT_smiles')
# analyze_xTB_ML_comp('INDT_smiles')
# compare_20k_300k_ML('AL_PCQC_T1_xTB_test')
# compare_20k_300k_ML('verde_smiles')

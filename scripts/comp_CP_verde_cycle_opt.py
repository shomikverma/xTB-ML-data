import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from sklearn.metrics import r2_score, mean_absolute_error
import sys
# from SecretColors import Palette
# from SecretColors.cmaps import ColorMap
from cycler import cycler
import string
from itertools import cycle

plt.style.use(['science', 'grid'])
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
colors = [colors[0], colors[2], colors[1], colors[3]] + colors[4:]
colors_nipy1 = mpl.cm.nipy_spectral(np.linspace(0.1, 0.9, 6))
colors_nipy2 = mpl.cm.nipy_spectral(np.linspace(0.6, 0.9, 7))
colors_nipy = list(colors_nipy1[0:3]) + list(colors_nipy2[3:-2]) + list(colors_nipy1[-1:])
plt.rcParams['axes.prop_cycle'] = cycler(color=colors)


def label_axes(fig, labels=None, loc=None, **kwargs):
    if labels is None:
        labels = string.ascii_lowercase
    labels = cycle(labels)
    if loc is None:
        loc = (-10, 250)
    for ax, lab in zip(fig.axes, labels):
        ax.annotate('(' + lab + ')', size=16, xy=loc,
                    xycoords='axes points',
                    **kwargs)


with open('verde_CP_results/verde_ML.csv', 'r') as file:
    origData = file.readlines()
with open('verde_CP_results/verde_ML_stda.csv', 'r') as file:
    origDataSTDA = file.readlines()
with open('verde_CP_results/verde_ML_stddft.csv', 'r') as file:
    origDataSTDDFT = file.readlines()
with open('verde_CP_results/verde_ML_stda_S1.csv', 'r') as file:
    origDataSTDA_S1 = file.readlines()
with open('verde_CP_results/verde_ML_stddft_S1.csv', 'r') as file:
    origDataSTDDFT_S1 = file.readlines()

labelOrig = {}
allSmiles = []
allSmiles_S1 = []
for index, line in enumerate(origData):
    if (index == 0):
        continue
    line = line.replace('\n', '')
    label = float(line.split(',')[1])
    xtbT1 = float(line.split(',')[2])
    gaussT1 = float(line.split(',')[3])
    smiles = line.split(',')[0]
    allSmiles.append(smiles)
    labelOrig[smiles] = {}
    labelOrig[smiles]['label'] = label
    labelOrig[smiles]['xtbT1'] = xtbT1
    labelOrig[smiles]['gaussT1'] = gaussT1

for index, line in enumerate(origDataSTDA):
    if (index == 0):
        continue
    line = line.replace('\n', '')
    label = float(line.split(',')[1])
    stdaT1 = float(line.split(',')[2])
    smiles = line.split(',')[0]
    # labelOrig[smiles] = {}
    labelOrig[smiles]['label_stda'] = label
    labelOrig[smiles]['stdaT1'] = stdaT1

for index, line in enumerate(origDataSTDDFT):
    if (index == 0):
        continue
    line = line.replace('\n', '')
    label = float(line.split(',')[1])
    stddftT1 = float(line.split(',')[2])
    smiles = line.split(',')[0]
    # labelOrig[smiles] = {}
    labelOrig[smiles]['label_stddft'] = label
    labelOrig[smiles]['stddftT1'] = stddftT1

for index, line in enumerate(origDataSTDA_S1):
    if (index == 0):
        continue
    line = line.replace('\n', '')
    label_S1 = float(line.split(',')[1])
    stdaS1 = float(line.split(',')[2])
    gaussS1 = float(line.split(',')[3])
    smiles = line.split(',')[0]
    allSmiles_S1.append(smiles)
    # labelOrig[smiles] = {}
    try:
        T1err = labelOrig[smiles]['label_stda']
    except KeyError:
        labelOrig[smiles] = {}
    labelOrig[smiles]['label_stda_S1'] = label_S1
    labelOrig[smiles]['stdaS1'] = stdaS1
    labelOrig[smiles]['gaussS1'] = gaussS1

for index, line in enumerate(origDataSTDDFT_S1):
    if (index == 0):
        continue
    line = line.replace('\n', '')
    label_S1 = float(line.split(',')[1])
    stddftS1 = float(line.split(',')[2])
    smiles = line.split(',')[0]
    # labelOrig[smiles] = {}
    labelOrig[smiles]['label_stddft_S1'] = label_S1
    labelOrig[smiles]['stddftS1'] = stddftS1


def write_test_files(labelOrig):
    randomSize = len(allSmiles)
    randomSize_S1 = len(allSmiles_S1)
    testSize = int(randomSize / 10)
    testSize_S1 = int(randomSize_S1 / 10)

    print(randomSize, testSize, randomSize_S1, testSize_S1)

    np.random.seed(3)
    a = np.arange(randomSize)
    b = np.arange(randomSize_S1)
    np.random.shuffle(a)
    np.random.shuffle(b)
    # print(a[:95])
    try:
        os.mkdir('verde_cp_cycle')
    except:
        pass
    for i in range(10):
        try:
            os.mkdir('verde_cp_cycle/cycle' + str(i))
        except:
            pass
        stdlist = ['', 'stda', 'stddft', 'stda_S1', 'stddft_S1']
        for val in stdlist:
            if 'S1' in val:
                eng = 'S1'
                arr = b
                valSize = testSize_S1
                randSize = randomSize_S1
                allSmi = allSmiles_S1
            else:
                eng = 'T1'
                arr = a
                valSize = testSize
                randSize = randomSize
                allSmi = allSmiles
            if len(val) == 0:
                underscore = ''
            else:
                underscore = '_'
            testSmiles = {}
            validSmiles = {}
            with open('verde_cp_cycle/cycle' + str(i) + '/testSmiles' + underscore + val + '.csv', 'w') as file:
                file.write('smiles,' + eng + 'err\n')
                for j in range(valSize):
                    num = i * valSize + j
                    index = arr[num % randSize]
                    # print(num, index)
                    smiles = allSmi[index]
                    testSmiles[smiles] = 1
                    file.write(smiles + ',' +
                               str(labelOrig[smiles]['label' + underscore + val]) + '\n')
            with open('verde_cp_cycle/cycle' + str(i) + '/validSmiles' + underscore + val + '.csv', 'w') as file:
                file.write('smiles,' + eng + 'err\n')
                for j in range(valSize):
                    num = (i + 1) * valSize + j
                    index = arr[num % randSize]
                    # print(num, index)
                    smiles = allSmi[index]
                    validSmiles[smiles] = 1
                    file.write(smiles + ',' +
                               str(labelOrig[smiles]['label' + underscore + val]) + '\n')
            with open('verde_cp_cycle/cycle' + str(i) + '/trainSmiles' + underscore + val + '.csv', 'w') as file:
                file.write('smiles,' + eng + 'err\n')
                for smiles in allSmi:
                    test = 0
                    try:
                        test = testSmiles[smiles]
                    except KeyError:
                        pass
                    try:
                        test = validSmiles[smiles]
                    except KeyError:
                        pass
                    if test == 0:
                        file.write(smiles + ',' +
                                   str(labelOrig[smiles]['label' + underscore + val]) + '\n')

            # with open('verde_cp_cycle/cycle' + str(i) + '/trainSmiles' + underscore + val + '.csv', 'r') as file:
            #     data = file.readlines()
            #     print(len(data))


def check_test_files():
    checkSmiles = []
    for i in range(10):
        with open('verde_cp_cycle/cycle' + str(i) + '/testSmiles.csv', 'r') as file:
            data = file.readlines()
        for line in data:
            if 'smiles' in line:
                continue
            smiles = line.split(',')[0]
            if (smiles not in checkSmiles):
                checkSmiles.append(smiles)
    print(len(checkSmiles))

    checkSmiles = []
    for i in range(10):
        with open('verde_cp_cycle/cycle' + str(i) + '/testSmiles_stda.csv', 'r') as file:
            data = file.readlines()
        for line in data:
            if 'smiles' in line:
                continue
            smiles = line.split(',')[0]
            if (smiles not in checkSmiles):
                checkSmiles.append(smiles)
    print(len(checkSmiles))

    checkSmiles = []
    for i in range(10):
        with open('verde_cp_cycle/cycle' + str(i) + '/testSmiles_stddft.csv', 'r') as file:
            data = file.readlines()
        for line in data:
            if 'smiles' in line:
                continue
            smiles = line.split(',')[0]
            if (smiles not in checkSmiles):
                checkSmiles.append(smiles)
    print(len(checkSmiles))

    checkSmiles = []
    for i in range(10):
        with open('verde_cp_cycle/cycle' + str(i) + '/testSmiles_stda_S1.csv', 'r') as file:
            data = file.readlines()
        for line in data:
            if 'smiles' in line:
                continue
            smiles = line.split(',')[0]
            if (smiles not in checkSmiles):
                checkSmiles.append(smiles)
    print(len(checkSmiles))

    checkSmiles = []
    for i in range(10):
        with open('verde_cp_cycle/cycle' + str(i) + '/testSmiles_stddft_S1.csv', 'r') as file:
            data = file.readlines()
        for line in data:
            if 'smiles' in line:
                continue
            smiles = line.split(',')[0]
            if (smiles not in checkSmiles):
                checkSmiles.append(smiles)
    print(len(checkSmiles))


# sys.exit()


def do_ML_cp():
    # xtb T1

    for i in range(10):
        os.system('chemprop_train --data_path verde_cp_cycle/cycle' + str(
            i) + '/trainSmiles.csv --dataset_type regression --save_dir verde_cp_cycle/cycle' + str(i) +
                  ' --metric r2 --separate_val_path verde_cp_cycle/cycle' + str(
            i) + '/validSmiles.csv --separate_test_path verde_cp_cycle/cycle' + str(
            i) + '/testSmiles.csv --target_columns T1err')
        os.system('chemprop_predict --test_path verde_ML.csv --checkpoint_dir verde_cp_cycle/cycle' +
                  str(i) + '/fold_0 --preds_path verde_cp_cycle/cycle' + str(
            i) + '/verde_preds.csv --drop_extra_columns')

    # stda T1

    for i in range(10):
        os.system('chemprop_train --data_path verde_cp_cycle/cycle' + str(
            i) + '/trainSmiles_stda.csv --dataset_type regression --save_dir verde_cp_cycle/cycle' + str(i) +
                  '/stda --metric r2 --separate_val_path verde_cp_cycle/cycle' + str(
            i) + '/validSmiles_stda.csv --separate_test_path verde_cp_cycle/cycle' + str(
            i) + '/testSmiles_stda.csv --target_columns T1err')
        os.system('chemprop_predict --test_path verde_ML_stda.csv --checkpoint_dir verde_cp_cycle/cycle' +
                  str(i) + '/stda --preds_path verde_cp_cycle/cycle' + str(
            i) + '/verde_stda_preds.csv --drop_extra_columns')

    # stddft T1

    for i in range(10):
        os.system('chemprop_train --data_path verde_cp_cycle/cycle' + str(
            i) + '/trainSmiles_stddft.csv --dataset_type regression --save_dir verde_cp_cycle/cycle' + str(i) +
                  '/stddft --metric r2 --separate_val_path verde_cp_cycle/cycle' + str(
            i) + '/validSmiles_stddft.csv --separate_test_path verde_cp_cycle/cycle' + str(
            i) + '/testSmiles_stddft.csv --target_columns T1err')
        os.system('chemprop_predict --test_path verde_ML_stddft.csv --checkpoint_dir verde_cp_cycle/cycle' +
                  str(i) + '/stddft --preds_path verde_cp_cycle/cycle' + str(
            i) + '/verde_stddft_preds.csv --drop_extra_columns')

    # stda S1

    for i in range(10):
        os.system('chemprop_train --data_path verde_cp_cycle/cycle' + str(
            i) + '/trainSmiles_stda_S1.csv --dataset_type regression --save_dir verde_cp_cycle/cycle' + str(i) +
                  '/stda_S1 --metric r2 --separate_val_path verde_cp_cycle/cycle' + str(
            i) + '/validSmiles_stda_S1.csv --separate_test_path verde_cp_cycle/cycle' + str(
            i) + '/testSmiles_stda_S1.csv --target_columns S1err')
        os.system('chemprop_predict --test_path verde_ML_stda_S1.csv --checkpoint_dir verde_cp_cycle/cycle' +
                  str(i) + '/stda_S1 --preds_path verde_cp_cycle/cycle' + str(
            i) + '/verde_stda_S1_preds.csv --drop_extra_columns')

    # stddft T1

    for i in range(10):
        os.system('chemprop_train --data_path verde_cp_cycle/cycle' + str(
            i) + '/trainSmiles_stddft_S1.csv --dataset_type regression --save_dir verde_cp_cycle/cycle' + str(i) +
                  '/stddft_S1 --metric r2 --separate_val_path verde_cp_cycle/cycle' + str(
            i) + '/validSmiles_stddft_S1.csv --separate_test_path verde_cp_cycle/cycle' + str(
            i) + '/testSmiles_stddft_S1.csv --target_columns S1err')
        os.system('chemprop_predict --test_path verde_ML_stddft_S1.csv --checkpoint_dir verde_cp_cycle/cycle' +
                  str(i) + '/stddft_S1 --preds_path verde_cp_cycle/cycle' + str(
            i) + '/verde_stddft_S1_preds.csv --drop_extra_columns')


# sys.exit()


def do_ML_cp_opt():
    # xtb T1

    for i in range(10):
        os.system('chemprop_hyperopt --data_path verde_cp_cycle/cycle' + str(i) +
                  '/trainSmiles.csv --dataset_type regression --num_iters 20 --config_save_path verde_cp_cycle/cycle' + str(
            i) + '/hyperopt')
        os.system('chemprop_train --data_path verde_cp_cycle/cycle' + str(
            i) + '/trainSmiles.csv --dataset_type regression --save_dir verde_cp_cycle/cycle' + str(i) +
                  '/xtb_hyperopt --metric r2 --separate_val_path verde_cp_cycle/cycle' + str(
            i) + '/validSmiles.csv --separate_test_path verde_cp_cycle/cycle' + str(i) +
                  '/testSmiles.csv --target_columns T1err --config_path verde_cp_cycle/cycle' + str(i) + '/hyperopt')
        os.system('chemprop_predict --test_path verde_ML.csv --checkpoint_dir verde_cp_cycle/cycle' +
                  str(i) + '/xtb_hyperopt --preds_path verde_cp_cycle/cycle' + str(
            i) + '/verde_preds_hyperopt.csv --drop_extra_columns')

    stdlist = ['stda', 'stddft', 'stda_S1', 'stddft_S1']
    for val in stdlist:
        if 'S1' in val:
            labelErr = 'S1err'
        else:
            labelErr = 'T1err'
        for i in range(10):
            os.system('chemprop_hyperopt --data_path verde_cp_cycle/cycle' + str(i) +
                      '/trainSmiles_' + val + '.csv --dataset_type regression --num_iters 20 --config_save_path verde_cp_cycle/cycle' + str(
                i) + '/' + val + '/hyperopt')
            os.system('chemprop_train --data_path verde_cp_cycle/cycle' + str(
                i) + '/trainSmiles_' + val + '.csv --dataset_type regression --save_dir verde_cp_cycle/cycle' + str(i) +
                      '/' + val + '_hyperopt --metric r2 --separate_val_path verde_cp_cycle/cycle' + str(i) +
                      '/validSmiles_' + val + '.csv --separate_test_path verde_cp_cycle/cycle' + str(i) +
                      '/testSmiles_' + val + '.csv --target_columns ' + labelErr + ' --config_path verde_cp_cycle/cycle' + str(
                i) + '/' + val + '/hyperopt')
            os.system('chemprop_predict --test_path verde_ML_' + val + '.csv --checkpoint_dir verde_cp_cycle/cycle' +
                      str(i) + '/' + val + '_hyperopt --preds_path verde_cp_cycle/cycle' + str(
                i) + '/verde_' + val + '_preds_hyperopt.csv --drop_extra_columns')


def do_ML_cp_rdkit():
    # xtb T1

    for i in range(10):
        os.system('chemprop_train --data_path verde_cp_cycle/cycle' + str(
            i) + '/trainSmiles.csv --dataset_type regression --save_dir verde_cp_cycle/cycle' + str(i) +
                  '/xtb_rdkit --metric r2 --separate_val_path verde_cp_cycle/cycle' + str(
            i) + '/validSmiles.csv --separate_test_path verde_cp_cycle/cycle' + str(i) +
                  '/testSmiles.csv --target_columns T1err --features_generator rdkit_2d_normalized --no_features_scaling')
        os.system('chemprop_predict --test_path verde_ML.csv --checkpoint_dir verde_cp_cycle/cycle' +
                  str(i) + '/xtb_rdkit --preds_path verde_cp_cycle/cycle' + str(
            i) + '/verde_preds_rdkit.csv --drop_extra_columns --features_generator rdkit_2d_normalized --no_features_scaling')

    stdlist = ['stda', 'stddft', 'stda_S1', 'stddft_S1']
    for val in stdlist:
        if 'S1' in val:
            labelErr = 'S1err'
        else:
            labelErr = 'T1err'
        for i in range(10):
            os.system('chemprop_train --data_path verde_cp_cycle/cycle' + str(
                i) + '/trainSmiles_' + val + '.csv --dataset_type regression --save_dir verde_cp_cycle/cycle' + str(i) +
                      '/' + val + '_rdkit --metric r2 --separate_val_path verde_cp_cycle/cycle' + str(i) +
                      '/validSmiles_' + val + '.csv --separate_test_path verde_cp_cycle/cycle' + str(i) +
                      '/testSmiles_' + val + '.csv --target_columns ' + labelErr + ' --features_generator rdkit_2d_normalized --no_features_scaling')
            os.system('chemprop_predict --test_path verde_ML_' + val + '.csv --checkpoint_dir verde_cp_cycle/cycle' +
                      str(i) + '/' + val + '_rdkit --preds_path verde_cp_cycle/cycle' + str(
                i) + '/verde_' + val + '_preds_rdkit.csv --drop_extra_columns --features_generator rdkit_2d_normalized --no_features_scaling')


def do_ML_cp_morgan():
    # xtb T1

    for i in range(10):
        os.system('chemprop_train --data_path verde_cp_cycle/cycle' + str(
            i) + '/trainSmiles.csv --dataset_type regression --save_dir verde_cp_cycle/cycle' + str(i) +
                  '/xtb_morgan --metric r2 --separate_val_path verde_cp_cycle/cycle' + str(
            i) + '/validSmiles.csv --separate_test_path verde_cp_cycle/cycle' + str(i) +
                  '/testSmiles.csv --target_columns T1err --features_generator morgan')
        os.system('chemprop_predict --test_path verde_ML.csv --checkpoint_dir verde_cp_cycle/cycle' +
                  str(i) + '/xtb_morgan --preds_path verde_cp_cycle/cycle' + str(
            i) + '/verde_preds_morgan.csv --drop_extra_columns --features_generator morgan')

    stdlist = ['stda', 'stddft', 'stda_S1', 'stddft_S1']
    for val in stdlist:
        if 'S1' in val:
            labelErr = 'S1err'
        else:
            labelErr = 'T1err'
        for i in range(10):
            os.system('chemprop_train --data_path verde_cp_cycle/cycle' + str(
                i) + '/trainSmiles_' + val + '.csv --dataset_type regression --save_dir verde_cp_cycle/cycle' + str(i) +
                      '/' + val + '_morgan --metric r2 --separate_val_path verde_cp_cycle/cycle' + str(i) +
                      '/validSmiles_' + val + '.csv --separate_test_path verde_cp_cycle/cycle' + str(i) +
                      '/testSmiles_' + val + '.csv --target_columns ' + labelErr + ' --features_generator morgan')
            os.system('chemprop_predict --test_path verde_ML_' + val + '.csv --checkpoint_dir verde_cp_cycle/cycle' +
                      str(i) + '/' + val + '_morgan --preds_path verde_cp_cycle/cycle' + str(
                i) + '/verde_' + val + '_preds_morgan.csv --drop_extra_columns --features_generator morgan')


def do_ML_cp_morganc():
    # xtb T1

    for i in range(10):
        os.system('chemprop_train --data_path verde_cp_cycle/cycle' + str(
            i) + '/trainSmiles.csv --dataset_type regression --save_dir verde_cp_cycle/cycle' + str(i) +
                  '/xtb_morganc --metric r2 --separate_val_path verde_cp_cycle/cycle' + str(
            i) + '/validSmiles.csv --separate_test_path verde_cp_cycle/cycle' + str(i) +
                  '/testSmiles.csv --target_columns T1err --features_generator morgan_count')
        os.system('chemprop_predict --test_path verde_ML.csv --checkpoint_dir verde_cp_cycle/cycle' +
                  str(i) + '/xtb_morganc --preds_path verde_cp_cycle/cycle' + str(
            i) + '/verde_preds_morganc.csv --drop_extra_columns --features_generator morgan_count')

    stdlist = ['stda', 'stddft', 'stda_S1', 'stddft_S1']
    for val in stdlist:
        if 'S1' in val:
            labelErr = 'S1err'
        else:
            labelErr = 'T1err'
        for i in range(10):
            os.system('chemprop_train --data_path verde_cp_cycle/cycle' + str(
                i) + '/trainSmiles_' + val + '.csv --dataset_type regression --save_dir verde_cp_cycle/cycle' + str(i) +
                      '/' + val + '_morganc --metric r2 --separate_val_path verde_cp_cycle/cycle' + str(i) +
                      '/validSmiles_' + val + '.csv --separate_test_path verde_cp_cycle/cycle' + str(i) +
                      '/testSmiles_' + val + '.csv --target_columns ' + labelErr + ' --features_generator morgan_count')
            os.system('chemprop_predict --test_path verde_ML_' + val + '.csv --checkpoint_dir verde_cp_cycle/cycle' +
                      str(i) + '/' + val + '_morganc --preds_path verde_cp_cycle/cycle' + str(
                i) + '/verde_' + val + '_preds_morganc.csv --drop_extra_columns --features_generator morgan_count')


def do_ML_cp_100ep():
    # xtb T1

    for i in range(10):
        os.system('chemprop_train --data_path verde_cp_cycle/cycle' + str(
            i) + '/trainSmiles.csv --dataset_type regression --save_dir verde_cp_cycle/cycle' + str(i) +
                  '/xtb_100ep --metric r2 --separate_val_path verde_cp_cycle/cycle' + str(
            i) + '/validSmiles.csv --separate_test_path verde_cp_cycle/cycle' + str(i) +
                  '/testSmiles.csv --target_columns T1err --epochs 100')
        os.system('chemprop_predict --test_path verde_ML.csv --checkpoint_dir verde_cp_cycle/cycle' +
                  str(i) + '/xtb_100ep --preds_path verde_cp_cycle/cycle' + str(
            i) + '/verde_preds_100ep.csv --drop_extra_columns')

    stdlist = ['stda', 'stddft', 'stda_S1', 'stddft_S1']
    for val in stdlist:
        if 'S1' in val:
            labelErr = 'S1err'
        else:
            labelErr = 'T1err'
        for i in range(10):
            os.system('chemprop_train --data_path verde_cp_cycle/cycle' + str(
                i) + '/trainSmiles_' + val + '.csv --dataset_type regression --save_dir verde_cp_cycle/cycle' + str(i) +
                      '/' + val + '_100ep --metric r2 --separate_val_path verde_cp_cycle/cycle' + str(i) +
                      '/validSmiles_' + val + '.csv --separate_test_path verde_cp_cycle/cycle' + str(i) +
                      '/testSmiles_' + val + '.csv --target_columns ' + labelErr + ' --epochs 100')
            os.system('chemprop_predict --test_path verde_ML_' + val + '.csv --checkpoint_dir verde_cp_cycle/cycle' +
                      str(i) + '/' + val + '_100ep --preds_path verde_cp_cycle/cycle' + str(
                i) + '/verde_' + val + '_preds_100ep.csv --drop_extra_columns')


def do_xtb_analysis(labelOrig, runType):
    # for i in range(10):
    #     os.system('chemprop_predict --test_path verde_ML.csv --checkpoint_dir verde_cp_cycle/cycle' +
    #               str(i) + '/fold_0 --preds_path verde_cp_cycle/cycle' + str(i) + '/verde_preds.csv --drop_extra_columns')

    if len(runType) == 0:
        runType = ''
    elif runType[0] != '_':
        runType = '_' + runType

    xtbT1sOrig_all = []
    xtbT1sFixed_all = []
    gaussT1s_all = []

    for i in range(10):

        with open('verde_cp_cycle/cycle' + str(i) + '/testSmiles.csv', 'r') as file:
            testData = file.readlines()

        with open('verde_cp_cycle/cycle' + str(i) + '/verde_preds' + runType + '.csv', 'r') as file:
            predData = file.readlines()

        labelTest = {}
        testSmiles = []
        for index, line in enumerate(testData):
            if (index == 0):
                continue
            smiles = line.split(',')[0]
            label = float(line.split(',')[1])
            testSmiles.append(smiles)
            labelTest[smiles] = label

        labelPred = {}
        for index, line in enumerate(predData):
            if (index == 0):
                continue
            line = line.replace('\n', '')
            label = float(line.split(',')[1])
            smiles = line.split(',')[0]
            if smiles in testSmiles:
                labelPred[smiles] = label

        preds = []
        origs = []
        xtbT1s = []
        gaussT1s = []
        errs = 0
        for smiles in labelPred:
            try:
                xtbT1s.append(labelOrig[smiles]['xtbT1'])
                gaussT1s.append(labelOrig[smiles]['gaussT1'])
                xtbT1sOrig_all.append(labelOrig[smiles]['xtbT1'])
                gaussT1s_all.append(labelOrig[smiles]['gaussT1'])
                preds.append(labelPred[smiles])
                origs.append(labelTest[smiles])
            except KeyError:
                errs += 1
                continue
        print(errs)
        xtbT1sfixed = np.array(xtbT1s) + np.array(preds)
        for xtbT1 in xtbT1sfixed:
            xtbT1sFixed_all.append(xtbT1)
        test_mae = mean_absolute_error(origs, preds)
        test_r2 = r2_score(origs, preds)
        orig_r2 = r2_score(xtbT1s, gaussT1s)
        fixed_r2 = r2_score(xtbT1sfixed, np.array(gaussT1s))
        orig_MAE = mean_absolute_error(xtbT1s, gaussT1s)
        fixed_MAE = mean_absolute_error(xtbT1sfixed, np.array(gaussT1s))

        print('cycle', i)
        print('cp mpnn model mae', test_mae)
        print('cp mpnn model r2', test_r2)
        print('cp mpnn xtborig r2', orig_r2)
        print('cp mpnn xtbfixed r2', fixed_r2)
        print('cp mpnn xtborig MAE', orig_MAE)
        print('cp mpnn xtbfixed MAE', fixed_MAE)

    orig_r2 = r2_score(xtbT1sOrig_all, gaussT1s_all)
    fixed_r2 = r2_score(xtbT1sFixed_all, gaussT1s_all)
    orig_MAE = mean_absolute_error(xtbT1sOrig_all, gaussT1s_all)
    fixed_MAE = mean_absolute_error(xtbT1sFixed_all, gaussT1s_all)

    print('overall')
    print('cp mpnn xtborig r2', orig_r2)
    print('cp mpnn xtbfixed r2', fixed_r2)
    print('cp mpnn xtborig MAE', orig_MAE)
    print('cp mpnn xtbfixed MAE', fixed_MAE)

    # plt.figure(num=1, clear=True)
    # plt.plot(origs, preds, 'b.')
    # x = np.linspace(min(origs), max(origs), 100)
    # plt.plot(x, x, 'k--')
    # plt.grid()
    # plt.xlabel('Actual Error (eV)')
    # plt.ylabel('Predicted Error (eV)')
    # plt.title('Predicted vs. Actual Error, CP MPNN')
    # plt.savefig('CPMPNN_err_comp.png')
    #
    plt.figure(num=1, clear=True)
    plt.plot(xtbT1sOrig_all, gaussT1s_all, 'r.', label='orig')
    plt.plot(xtbT1sFixed_all, gaussT1s_all, 'b.', label='fixed')
    x = np.linspace(min(xtbT1sOrig_all), max(xtbT1sOrig_all), 100)
    plt.plot(x, x, 'k--')
    plt.grid()
    plt.legend()
    plt.xlabel('xtb T1 (eV)')
    plt.ylabel('gauss T1 (eV)')
    plt.title('Fixed vs. original xTB T1 energy, CP MPNN')
    plt.savefig('CPMPNN_xtb_comp_cycle' + runType + '.png')

    with open('verde_CP_xtb_T1' + runType + '.txt', 'w') as file:
        file.write('xTB\n')
        file.write('overall\n')
        file.write('cp mpnn xtborig r2 ' + str(orig_r2) + '\n')
        file.write('cp mpnn xtbfixed r2 ' + str(fixed_r2) + '\n')
        file.write('cp mpnn xtborig MAE ' + str(orig_MAE) + '\n')
        file.write('cp mpnn xtbfixed MAE ' + str(fixed_MAE) + '\n')

    xtbT1sOrig_filt = []
    xtbT1sFixed_filt = []
    gaussT1s_filt = []
    for index, val in enumerate(xtbT1sOrig_all):
        if (abs(gaussT1s_all[index] - val) > 1.5 or val < -0.1 or gaussT1s_all[index] < -0.1):
            continue
        else:
            xtbT1sOrig_filt.append(xtbT1sOrig_all[index])
            xtbT1sFixed_filt.append(xtbT1sFixed_all[index])
            gaussT1s_filt.append(gaussT1s_all[index])

    orig_r2 = r2_score(xtbT1sOrig_filt, gaussT1s_filt)
    fixed_r2 = r2_score(xtbT1sFixed_filt, gaussT1s_filt)
    orig_MAE = mean_absolute_error(xtbT1sOrig_filt, gaussT1s_filt)
    fixed_MAE = mean_absolute_error(xtbT1sFixed_filt, gaussT1s_filt)

    print('filtered')
    print('cp mpnn xtborig r2', orig_r2)
    print('cp mpnn xtbfixed r2', fixed_r2)
    print('cp mpnn xtborig MAE', orig_MAE)
    print('cp mpnn xtbfixed MAE', fixed_MAE)

    plt.figure(num=1, clear=True)
    plt.plot(xtbT1sOrig_filt, gaussT1s_filt, 'r.', label='orig')
    plt.plot(xtbT1sFixed_filt, gaussT1s_filt, 'b.', label='fixed')
    x = np.linspace(min(xtbT1sOrig_filt), max(xtbT1sOrig_filt), 100)
    plt.plot(x, x, 'k--')
    plt.grid()
    plt.legend()
    # plt.xlim(0, 3.5)
    # plt.ylim(0, 3.5)
    plt.xlabel('xtb T1 (eV)')
    plt.ylabel('gauss T1 (eV)')
    plt.title('Fixed vs. original xTB T1 energy, CP MPNN')
    plt.savefig('CPMPNN_xtb_comp_cycle_2' + runType + '.png')

    with open('verde_CP_xtb_T1' + runType + '.txt', 'a') as file:
        file.write('filtered\n')
        file.write('cp mpnn xtborig r2 ' + str(orig_r2) + '\n')
        file.write('cp mpnn xtbfixed r2 ' + str(fixed_r2) + '\n')
        file.write('cp mpnn xtborig MAE ' + str(orig_MAE) + '\n')
        file.write('cp mpnn xtbfixed MAE ' + str(fixed_MAE) + '\n')


def do_stda_analysis(labelOrig, stdstr, engstr, runType, plotNum):
    if plotNum == 1:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        colors = [colors[3], colors[0]]
        plt.rcParams['axes.prop_cycle'] = cycler(color=colors)
    if len(runType) == 0:
        runType = ''
    elif runType[0] != '_':
        runType = '_' + runType

    stdaT1sOrig_all = []
    stdaT1sFixed_all = []
    gaussT1s_all = []
    print(stdstr, engstr, runType)
    for i in range(10):

        with open('verde_cp_cycle/cycle' + str(i) + '/' + 'testSmiles_' + stdstr + '.csv', 'r') as file:
            testData = file.readlines()

        with open('verde_cp_cycle/cycle' + str(i) + '/verde_' + stdstr + '_preds' + runType + '.csv', 'r') as file:
            predData = file.readlines()

        labelTest = {}
        testSmiles = []
        for index, line in enumerate(testData):
            if (index == 0):
                continue
            smiles = line.split(',')[0]
            label = float(line.split(',')[1])
            testSmiles.append(smiles)
            labelTest[smiles] = label

        labelPred = {}
        for index, line in enumerate(predData):
            if (index == 0):
                continue
            line = line.replace('\n', '')
            label = float(line.split(',')[1])
            smiles = line.split(',')[0]
            if smiles in testSmiles:
                labelPred[smiles] = label

        preds = []
        origs = []
        stdaT1s = []
        gaussT1s = []
        errs = 0
        stdstr2 = stdstr.split("_")[0]
        for smiles in labelPred:
            try:
                stdaT1s.append(labelOrig[smiles][stdstr2 + engstr])
                gaussT1s.append(labelOrig[smiles]['gauss' + engstr])
                stdaT1sOrig_all.append(labelOrig[smiles][stdstr2 + engstr])
                gaussT1s_all.append(labelOrig[smiles]['gauss' + engstr])
                preds.append(labelPred[smiles])
                origs.append(labelTest[smiles])
            except KeyError:
                errs += 1
                continue
        print(errs)
        stdaT1sfixed = np.array(stdaT1s) + np.array(preds)
        for stdaT1 in stdaT1sfixed:
            stdaT1sFixed_all.append(stdaT1)
        test_mae = mean_absolute_error(origs, preds)
        test_r2 = r2_score(origs, preds)
        orig_r2 = r2_score(stdaT1s, gaussT1s)
        fixed_r2 = r2_score(stdaT1sfixed, np.array(gaussT1s))
        orig_MAE = mean_absolute_error(stdaT1s, gaussT1s)
        fixed_MAE = mean_absolute_error(stdaT1sfixed, np.array(gaussT1s))

        print('cycle', i)
        print('cp mpnn model mae', test_mae)
        print('cp mpnn model r2', test_r2)
        print('cp mpnn ' + stdstr2 + 'orig r2', orig_r2)
        print('cp mpnn ' + stdstr2 + 'fixed r2', fixed_r2)
        print('cp mpnn ' + stdstr2 + 'orig MAE', orig_MAE)
        print('cp mpnn ' + stdstr2 + 'fixed MAE', fixed_MAE)

    orig_r2 = r2_score(stdaT1sOrig_all, gaussT1s_all)
    fixed_r2 = r2_score(stdaT1sFixed_all, gaussT1s_all)
    orig_MAE = mean_absolute_error(stdaT1sOrig_all, gaussT1s_all)
    fixed_MAE = mean_absolute_error(stdaT1sFixed_all, gaussT1s_all)

    print('overall')
    print('cp mpnn ' + stdstr2 + 'orig r2', orig_r2)
    print('cp mpnn ' + stdstr2 + 'fixed r2', fixed_r2)
    print('cp mpnn ' + stdstr2 + 'orig MAE', orig_MAE)
    print('cp mpnn ' + stdstr2 + 'fixed MAE', fixed_MAE)

    # plt.figure(num=1, clear=True)
    # plt.plot(origs, preds, 'b.')
    # x = np.linspace(min(origs), max(origs), 100)
    # plt.plot(x, x, 'k--')
    # plt.grid()
    # plt.xlabel('Actual Error (eV)')
    # plt.ylabel('Predicted Error (eV)')
    # plt.title('Predicted vs. Actual Error, CP MPNN')
    # plt.savefig('CPMPNN_err_comp.png')
    #
    if plotNum == 1:
        clearPlot = True
    else:
        clearPlot = False
    fig = plt.figure(num=1, clear=clearPlot)
    fig.add_subplot(1, 2, plotNum)
    plt.plot(stdaT1sOrig_all, gaussT1s_all, '.', label='orig')
    plt.plot(stdaT1sFixed_all, gaussT1s_all, '.', label='fixed')
    x = np.linspace(min(stdaT1sOrig_all), max(stdaT1sOrig_all), 100)
    plt.plot(x, x, 'k--')
    plt.grid(True)
    plt.legend()
    plt.xlabel(stdstr2 + ' ' + engstr + ' (eV)')
    plt.ylabel('gauss ' + engstr + ' (eV)')
    plt.title('Fixed vs. original ' + stdstr2 +
              ' ' + engstr + ' energy, CP MPNN')
    plt.tight_layout()
    plt.savefig('CPMPNN_' + stdstr2 + '_' + engstr +
                '_comp_cycle' + runType + '.png')

    with open('verde_CP_xtb_' + engstr + runType + '.txt', 'a') as file:
        file.write(stdstr2 + ' ' + engstr + '\n')
        file.write('overall\n')
        file.write('cp mpnn ' + stdstr2 + 'orig r2 ' + str(orig_r2) + '\n')
        file.write('cp mpnn ' + stdstr2 + 'fixed r2 ' + str(fixed_r2) + '\n')
        file.write('cp mpnn ' + stdstr2 + 'orig MAE ' + str(orig_MAE) + '\n')
        file.write('cp mpnn ' + stdstr2 + 'fixed MAE ' + str(fixed_MAE) + '\n')

    stdaT1sOrig_filt = []
    stdaT1sFixed_filt = []
    gaussT1s_filt = []
    for index, val in enumerate(stdaT1sOrig_all):
        if (abs(gaussT1s_all[index] - val) > 1.5 or val < -0.1 or gaussT1s_all[index] < -0.1):
            continue
        else:
            stdaT1sOrig_filt.append(stdaT1sOrig_all[index])
            stdaT1sFixed_filt.append(stdaT1sFixed_all[index])
            gaussT1s_filt.append(gaussT1s_all[index])

    orig_r2 = r2_score(stdaT1sOrig_filt, gaussT1s_filt)
    fixed_r2 = r2_score(stdaT1sFixed_filt, gaussT1s_filt)
    orig_MAE = mean_absolute_error(stdaT1sOrig_filt, gaussT1s_filt)
    fixed_MAE = mean_absolute_error(stdaT1sFixed_filt, gaussT1s_filt)

    print('filtered')
    print('cp mpnn ' + stdstr2 + 'orig r2', orig_r2)
    print('cp mpnn ' + stdstr2 + 'fixed r2', fixed_r2)
    print('cp mpnn ' + stdstr2 + 'orig MAE', orig_MAE)
    print('cp mpnn ' + stdstr2 + 'fixed MAE', fixed_MAE)

    if plotNum == 1:
        clearPlot = True
    else:
        clearPlot = False
    fig = plt.figure(num=2, figsize=[8, 5], dpi=300, clear=clearPlot)
    fig.add_subplot(1, 2, plotNum)
    plt.plot(stdaT1sOrig_filt, gaussT1s_filt, '.', color=colors_nipy[5], label='orig')
    plt.plot(stdaT1sFixed_filt, gaussT1s_filt, '.', color=colors_nipy[1], label='ML calib')
    x = np.linspace(0, 5, 100)
    plt.plot(x, x, 'k--')
    plt.grid(True)
    plt.legend(fontsize=16)
    plt.xlim(0, 5)
    plt.ylim(0, 5)
    stdstrplot = 'xTB-sTDA'
    engstrplot = engstr.replace('1', '$_1$')
    plt.xlabel(stdstrplot + ' ' + engstrplot + ' (eV)', fontsize=16)
    plt.ylabel('TDDFT ' + engstrplot + ' (eV)', fontsize=16)
    # plt.title('CP MPNN fixed ' + stdstr2 +
    # ' ' + engstr + ' energy')
    plt.annotate('R$^2$ orig: %0.2f\n' % orig_r2 +
                 'R$^2$ ML: %0.2f\n' % fixed_r2 +
                 'MAE orig: %0.2f\n' % orig_MAE +
                 'MAE ML: %0.2f' % fixed_MAE,
                 (4.75, 0.25),
                 bbox=dict(facecolor='white', alpha=0.5),
                 fontsize=14,
                 ha='right')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    label_axes(fig, ha='left')
    plt.savefig('CPMPNN_' + stdstr2 + '_' + 'S1' +
                '_comp_cycle_2' + runType + '.png')

    with open('verde_CP_xtb_' + engstr + runType + '.txt', 'a') as file:
        file.write('filtered\n')
        file.write('cp mpnn ' + stdstr2 + 'orig r2 ' + str(orig_r2) + '\n')
        file.write('cp mpnn ' + stdstr2 + 'fixed r2 ' + str(fixed_r2) + '\n')
        file.write('cp mpnn ' + stdstr2 + 'orig MAE ' + str(orig_MAE) + '\n')
        file.write('cp mpnn ' + stdstr2 + 'fixed MAE ' + str(fixed_MAE) + '\n')


def compare_all_ML_settings(settings):
    # ibm = Palette('ibm')
    # print(ibm.green())
    # c = ColorMap(mpl, ibm)
    # mpl.rcParams['axes.prop_cycle'] = cycler(color=['bgrcmyk'])
    # mpl.rcParams['axes.prop_cycle'] = cycler(color=['#7F3C8D', '#11A579', '#3969AC', '#F2B701',
    #                                                 '#E73F74', '#80BA5A', '#E68310', '#008695',
    #                                                 '#CF1C90', '#f97b72', '#4b4b8f', '#A5AA99'])
    # mpl.rcParams['axes.prop_cycle'] = cycler(color=['#5F4690', '#0F8554', '#E17C05',
    #                                                 '#1D6996', '#73AF48', '#CC503E',
    #                                                 '#38A6A5', '#EDAD08', '#94346E'])
    # mpl.rcParams['axes.prop_cycle'] = cycler(color=[ibm.red(), ibm.blue(), ibm.green(),
    #                                                 ibm.yellow(), ibm.magenta(),
    #                                                 ibm.cyan()])
    N = 6
    # plt.rcParams["axes.prop_cycle"] = plt.cycler(
    #     "color", plt.cm.viridis(np.linspace(0, 1, N)))
    # plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.Set1.colors)

    xtbsT1 = ['stda', 'stddft']
    xtbsS1 = ['stda', 'stddft']
    allsetsT1 = {}
    allsetsS1 = {}
    allsets = {}
    for setting in settings:

        MLtype = setting[0]
        set = setting[1]

        allsetsT1[MLtype + set] = []
        allsetsS1[MLtype + set] = []
        allsets[MLtype + set] = []

        if (set == 'default'):
            setname = ''
        else:
            setname = set
        if (len(setname) != 0):
            setname = '_' + setname

        with open('verde_CP_results/verde_CP_' + MLtype + '_S1' + setname + '.txt', 'r') as file:
            data = file.readlines()
        for line in data:
            if ('cp mpnn stdafixed r2' in line):
                stdafixedr2 = float(line.split()[4])
            if ('cp mpnn stddftfixed r2' in line):
                stddftfixedr2 = float(line.split()[4])
        allsetsS1[MLtype + set].append(stdafixedr2)
        allsetsS1[MLtype + set].append(stddftfixedr2)
        allsets[MLtype + set].append(stdafixedr2)

        with open('verde_CP_results/verde_CP_' + MLtype + '_T1' + setname + '.txt', 'r') as file:
            data = file.readlines()
        for line in data:
            if ('cp mpnn stdafixed r2' in line):
                stdafixedr2 = float(line.split()[4])
            if ('cp mpnn stddftfixed r2' in line):
                stddftfixedr2 = float(line.split()[4])
        allsetsT1[MLtype + set].append(stdafixedr2)
        allsetsT1[MLtype + set].append(stddftfixedr2)
        allsets[MLtype + set].append(stdafixedr2)

    x = np.arange(len(allsetsT1['xtbdefault']))  # the label locations
    width = 1 / (len(allsetsT1) + 1)  # the width of the bars
    plt.figure(num=1, clear=True)
    fig, ax = plt.subplots()
    for y in np.arange(len(allsetsT1)):
        ax.bar(x + (y - 0.5 * (len(allsetsT1) - 1)) * width,
               allsetsT1[list(allsetsT1.keys())[y]],
               width,
               label=list(allsetsT1.keys())[y],
               zorder=3)

    ax.set_ylabel('r2 score')
    ax.set_title('T1 r2 score by ML settings')
    ax.set_xticks(x)
    ax.set_xticklabels(xtbsT1)
    ax.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
    # ax.legend(loc='upper center',
    #           bbox_to_anchor=(0.5, -0.05),
    #           ncol=len(allsets))
    ax.set_ylim(0.3, 1)
    ax.grid(True, zorder=0)
    plt.tight_layout()
    plt.savefig('CPMPNN_all_comp_T1.png')
    plt.close()

    x = np.arange(len(allsetsS1['xtbdefault']))  # the label locations
    width = 1 / (len(allsetsS1) + 1)  # the width of the bars
    plt.figure(num=1, clear=True)
    fig, ax = plt.subplots()
    for y in np.arange(len(allsetsS1)):
        ax.bar(x + (y - 0.5 * (len(allsetsS1) - 1)) * width,
               allsetsS1[list(allsetsS1.keys())[y]],
               width,
               label=list(allsetsS1.keys())[y],
               zorder=3)

    ax.set_ylabel('r2 score')
    ax.set_title('S1 r2 score by ML settings')
    ax.set_xticks(x)
    ax.set_xticklabels(xtbsS1)
    ax.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
    # ax.legend(loc='upper center',
    #           bbox_to_anchor=(0.5, -0.05),
    #           ncol=len(allsets))
    ax.set_ylim(0.3, 1)
    ax.grid(True, zorder=0)
    plt.tight_layout()
    plt.savefig('CPMPNN_all_comp_S1.png')
    plt.close()

    x = np.arange(len(allsets['xtbdefault']))  # the label locations
    width = 1 / (len(allsets) + 2)  # the width of the bars
    fig = plt.figure(num=2, figsize=[6, 3], dpi=300, clear=True)
    ax = fig.add_subplot(1, 1, 1)
    labels = ['default', '100 epoch', 'hyperopt', 'multi, default', 'multi, 100 epoch', 'multi, hyperopt']
    for y in np.arange(len(allsets)):
        ax.bar(x + (y - 0.5 * (len(allsets) - 1)) * width,
               allsets[list(allsets.keys())[y]],
               width,
               label=labels[y],
               color=colors_nipy[y],
               zorder=3)

    ax.set_ylabel('R$^2$ score', fontsize=20)
    # ax.set_title('r2 score by ML settings')
    ax.set_xticks(x)

    ax.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', fontsize=16)
    # ax.legend(loc='upper center',
    #           bbox_to_anchor=(0.5, -0.05),
    #           ncol=len(allsets))
    ax.set_xticklabels(['S$_1$', 'T$_1$'], fontsize=20)
    ax.set_ylim(0.6, 1)
    ylabels = ax.get_yticks()
    ylabels = ['%0.2f' % s for s in ylabels]
    ax.set_yticklabels(ylabels, fontsize=16)
    # ax.set_yticklabels(np.linspace(0, 1, 11), fontsize=16)
    ax.grid(True, zorder=0)
    plt.tight_layout()
    plt.savefig('CPMPNN_all_comp_T1S1.png')
    plt.close()


def comp_ML_figure_plot():
    with open('ML_comp_DC_CP.txt', 'r') as file:
        data = file.readlines()
    dcGcnR2 = []
    dcGcnR2epoch = []
    dcMpnnR2 = []
    dcMpnnR2epoch = []
    cpMpnnR2 = []
    origR2 = []
    linR2 = []
    dcGcnXtbR2 = []
    dcGcnXtbR2epoch = []
    dcMpnnXtbR2 = []
    dcMpnnXtbR2epoch = []
    cpMpnnXtbR2 = []
    for line in data:
        line = line.replace('\n', '')
        if (len(line) == 0):
            continue
        r2 = float(line.split()[4])
        if 'dc gcn model r2' in line:
            if ('epoch' in line):
                dcGcnR2epoch.append(r2)
            else:
                dcGcnR2.append(r2)
        if 'dc mpnn model r2' in line:
            if ('epoch' in line):
                dcMpnnR2epoch.append(r2)
            else:
                dcMpnnR2.append(r2)
        if 'cp mpnn model r2' in line:
            cpMpnnR2.append(r2)
        if ('dc gcn xtborig r2' in line):
            origR2.append(r2)
        if ('dc gcn xtbfixed r2' in line):
            if 'epoch' in line:
                dcGcnXtbR2epoch.append(r2)
            else:
                dcGcnXtbR2.append(r2)
        if ('dc mpnn xtbfixed r2' in line):
            if 'epoch' in line:
                dcMpnnXtbR2epoch.append(r2)
            else:
                dcMpnnXtbR2.append(r2)
        if ('cp mpnn xtbfixed r2' in line):
            cpMpnnXtbR2.append(r2)
        if 'lin calib model r2' in line:
            linR2.append(r2)
    with open('ML_comp_DC_CP_S1.txt', 'r') as file:
        dataS1 = file.readlines()
    dcGcnR2_S1 = []
    dcMpnnR2_S1 = []
    cpMpnnR2_S1 = []
    origR2_S1 = []
    linR2_S1 = []
    dcGcnXtbR2_S1 = []
    dcMpnnXtbR2_S1 = []
    cpMpnnXtbR2_S1 = []
    for line in dataS1:
        line = line.replace('\n', '')
        if (len(line) == 0):
            continue
        r2 = float(line.split()[4])
        if 'dc gcn model r2' in line:
            if ('epoch' in line):
                pass
            else:
                dcGcnR2_S1.append(r2)
        if 'dc mpnn model r2' in line:
            if ('epoch' in line):
                pass
            else:
                dcMpnnR2_S1.append(r2)
        if 'cp mpnn model r2' in line:
            cpMpnnR2_S1.append(r2)
        if ('dc gcn xtborig r2' in line):
            origR2_S1.append(r2)
        if ('dc gcn xtbfixed r2' in line):
            if 'epoch' in line:
                pass
            else:
                dcGcnXtbR2_S1.append(r2)
        if ('dc mpnn xtbfixed r2' in line):
            if 'epoch' in line:
                pass
            else:
                dcMpnnXtbR2_S1.append(r2)
        if ('cp mpnn xtbfixed r2' in line):
            cpMpnnXtbR2_S1.append(r2)
        if 'lin calib model r2' in line:
            linR2_S1.append(r2)
    labels = ['orig', 'lin calib', 'DC GCN', 'DC MPNN', 'CP MPNN']
    conv_means = [np.mean(origR2), np.mean(linR2), np.mean(dcGcnXtbR2),
                  np.mean(dcMpnnXtbR2), np.mean(cpMpnnXtbR2)]
    conv_errbars = [[min(np.mean(origR2) - np.min(origR2), np.std(origR2)),
                     min(np.mean(linR2) - np.min(linR2), np.std(linR2)),
                     min(np.mean(dcGcnXtbR2) - np.min(dcGcnXtbR2), np.std(dcGcnXtbR2)),
                     min(np.mean(dcMpnnXtbR2) - np.min(dcMpnnXtbR2), np.std(dcMpnnXtbR2)),
                     min(np.mean(cpMpnnXtbR2) - np.min(cpMpnnXtbR2), np.std(cpMpnnXtbR2))],
                    [np.max(origR2) - np.mean(origR2), np.max(linR2) - np.mean(linR2),
                     np.max(dcGcnXtbR2) - np.mean(dcGcnXtbR2),
                     np.max(dcMpnnXtbR2) - np.mean(dcMpnnXtbR2),
                     np.max(cpMpnnXtbR2) - np.mean(cpMpnnXtbR2)]]
    conv_means_S1 = [np.mean(origR2_S1), np.mean(linR2_S1), np.mean(dcGcnXtbR2_S1),
                     np.mean(dcMpnnXtbR2_S1), np.mean(cpMpnnXtbR2_S1)]
    conv_errbars_S1 = [[min(np.mean(origR2_S1) - np.min(origR2_S1), np.std(origR2_S1)),
                        min(np.mean(linR2_S1) - np.min(linR2_S1), np.std(linR2_S1)),
                        min(np.mean(dcGcnXtbR2_S1) - np.min(dcGcnXtbR2_S1), np.std(dcGcnXtbR2_S1)),
                        min(np.mean(dcMpnnXtbR2_S1) - np.min(dcMpnnXtbR2_S1), np.std(dcMpnnXtbR2_S1)),
                        min(np.mean(cpMpnnXtbR2_S1) - np.min(cpMpnnXtbR2_S1), np.std(cpMpnnXtbR2_S1))],
                       [np.max(origR2_S1) - np.mean(origR2_S1), np.max(linR2_S1) - np.mean(linR2_S1),
                        np.max(dcGcnXtbR2_S1) - np.mean(dcGcnXtbR2_S1),
                        np.max(dcMpnnXtbR2_S1) - np.mean(dcMpnnXtbR2_S1),
                        np.max(cpMpnnXtbR2_S1) - np.mean(cpMpnnXtbR2_S1)]]
    x = np.arange(len(labels)) + 1  # the label locations
    width = 0.35  # the width of the bars
    fig = plt.figure(num=2, figsize=[8, 12], dpi=300, clear=True)
    ax = fig.add_subplot(3, 1, 1)
    rects2 = ax.bar(x - width / 2, conv_means_S1, width, color=colors_nipy[0],
                    label='S$_1$', yerr=conv_errbars_S1, ecolor='k', capsize=3)
    rects1 = ax.bar(x + width / 2, conv_means, width, color=colors_nipy[1],
                    label='T$_1$', yerr=conv_errbars, ecolor='k', capsize=3)
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('R$^2$ Scores', fontsize=16)
    # ax.set_title('Scores for xTB-ML by ML model')
    ax.set_xticks(x)
    ax.set_yticklabels([-0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=14)
    ax.set_xticklabels(labels, fontsize=16)
    ax.legend(loc='upper left', fontsize=16)
    plt.ylim(-0.25, 1)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    plt.tight_layout()
    # plt.savefig('ML_comp_DC_CP_T1.eps')
    # plt.savefig('ML_comp_DC_CP_T1S1.png')

    stdstr = 'stda_S1'
    engstr = 'S1'
    runType = ''
    plotNum = 1
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    colors = [colors[3], colors[0]]
    plt.rcParams['axes.prop_cycle'] = cycler(color=colors)
    stdaT1sOrig_all = []
    stdaT1sFixed_all = []
    gaussT1s_all = []
    for i in range(10):
        with open('verde_cp_cycle/cycle' + str(i) + '/' + 'testSmiles_' + stdstr + '.csv', 'r') as file:
            testData = file.readlines()
        with open('verde_cp_cycle/cycle' + str(i) + '/verde_' + stdstr + '_preds' + runType + '.csv', 'r') as file:
            predData = file.readlines()
        labelTest = {}
        testSmiles = []
        for index, line in enumerate(testData):
            if (index == 0):
                continue
            smiles = line.split(',')[0]
            label = float(line.split(',')[1])
            testSmiles.append(smiles)
            labelTest[smiles] = label
        labelPred = {}
        for index, line in enumerate(predData):
            if (index == 0):
                continue
            line = line.replace('\n', '')
            label = float(line.split(',')[1])
            smiles = line.split(',')[0]
            if smiles in testSmiles:
                labelPred[smiles] = label
        preds = []
        origs = []
        stdaT1s = []
        gaussT1s = []
        errs = 0
        stdstr2 = stdstr.split("_")[0]
        for smiles in labelPred:
            try:
                stdaT1s.append(labelOrig[smiles][stdstr2 + engstr])
                gaussT1s.append(labelOrig[smiles]['gauss' + engstr])
                stdaT1sOrig_all.append(labelOrig[smiles][stdstr2 + engstr])
                gaussT1s_all.append(labelOrig[smiles]['gauss' + engstr])
                preds.append(labelPred[smiles])
                origs.append(labelTest[smiles])
            except KeyError:
                errs += 1
                continue
        stdaT1sfixed = np.array(stdaT1s) + np.array(preds)
        for stdaT1 in stdaT1sfixed:
            stdaT1sFixed_all.append(stdaT1)
    stdaT1sOrig_filt = []
    stdaT1sFixed_filt = []
    gaussT1s_filt = []
    for index, val in enumerate(stdaT1sOrig_all):
        if (abs(gaussT1s_all[index] - val) > 1.5 or val < -0.1 or gaussT1s_all[index] < -0.1):
            continue
        else:
            stdaT1sOrig_filt.append(stdaT1sOrig_all[index])
            stdaT1sFixed_filt.append(stdaT1sFixed_all[index])
            gaussT1s_filt.append(gaussT1s_all[index])
    orig_r2 = r2_score(stdaT1sOrig_filt, gaussT1s_filt)
    fixed_r2 = r2_score(stdaT1sFixed_filt, gaussT1s_filt)
    orig_MAE = mean_absolute_error(stdaT1sOrig_filt, gaussT1s_filt)
    fixed_MAE = mean_absolute_error(stdaT1sFixed_filt, gaussT1s_filt)
    clearPlot = False
    fig = plt.figure(num=2, figsize=[8, 5], dpi=300, clear=clearPlot)
    fig.add_subplot(3, 2, 3)
    plt.plot(stdaT1sOrig_filt, gaussT1s_filt, '.', color=colors_nipy[5], label='orig')
    plt.plot(stdaT1sFixed_filt, gaussT1s_filt, '.', color=colors_nipy[1], label='ML calib')
    x = np.linspace(0, 5, 100)
    plt.plot(x, x, 'k--')
    plt.grid(True)
    plt.legend(fontsize=16)
    plt.xlim(0, 5)
    plt.ylim(0, 5)
    stdstrplot = 'xTB-sTDA'
    engstrplot = engstr.replace('1', '$_1$')
    plt.xlabel(stdstrplot + ' ' + engstrplot + ' (eV)', fontsize=16)
    plt.ylabel('TDDFT ' + engstrplot + ' (eV)', fontsize=16)
    plt.annotate('R$^2$ orig: %0.2f\n' % orig_r2 +
                 'R$^2$ ML: %0.2f\n' % fixed_r2 +
                 'MAE orig: %0.2f\n' % orig_MAE +
                 'MAE ML: %0.2f' % fixed_MAE,
                 (4.75, 0.25),
                 bbox=dict(facecolor='white', alpha=0.5),
                 fontsize=14,
                 ha='right')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    # label_axes(fig, ha='left')
    # plt.savefig('CPMPNN_' + stdstr2 + '_' + 'S1' +
    #             '_comp_cycle_2' + runType + '.png')

    stdstr = 'stda'
    engstr = 'T1'
    runType = ''
    plotNum = 2
    stdaT1sOrig_all = []
    stdaT1sFixed_all = []
    gaussT1s_all = []
    for i in range(10):
        with open('verde_cp_cycle/cycle' + str(i) + '/' + 'testSmiles_' + stdstr + '.csv', 'r') as file:
            testData = file.readlines()
        with open('verde_cp_cycle/cycle' + str(i) + '/verde_' + stdstr + '_preds' + runType + '.csv', 'r') as file:
            predData = file.readlines()
        labelTest = {}
        testSmiles = []
        for index, line in enumerate(testData):
            if (index == 0):
                continue
            smiles = line.split(',')[0]
            label = float(line.split(',')[1])
            testSmiles.append(smiles)
            labelTest[smiles] = label
        labelPred = {}
        for index, line in enumerate(predData):
            if (index == 0):
                continue
            line = line.replace('\n', '')
            label = float(line.split(',')[1])
            smiles = line.split(',')[0]
            if smiles in testSmiles:
                labelPred[smiles] = label
        preds = []
        origs = []
        stdaT1s = []
        gaussT1s = []
        errs = 0
        stdstr2 = stdstr.split("_")[0]
        for smiles in labelPred:
            try:
                stdaT1s.append(labelOrig[smiles][stdstr2 + engstr])
                gaussT1s.append(labelOrig[smiles]['gauss' + engstr])
                stdaT1sOrig_all.append(labelOrig[smiles][stdstr2 + engstr])
                gaussT1s_all.append(labelOrig[smiles]['gauss' + engstr])
                preds.append(labelPred[smiles])
                origs.append(labelTest[smiles])
            except KeyError:
                errs += 1
                continue
        stdaT1sfixed = np.array(stdaT1s) + np.array(preds)
        for stdaT1 in stdaT1sfixed:
            stdaT1sFixed_all.append(stdaT1)
    stdaT1sOrig_filt = []
    stdaT1sFixed_filt = []
    gaussT1s_filt = []
    for index, val in enumerate(stdaT1sOrig_all):
        if (abs(gaussT1s_all[index] - val) > 1.5 or val < -0.1 or gaussT1s_all[index] < -0.1):
            continue
        else:
            stdaT1sOrig_filt.append(stdaT1sOrig_all[index])
            stdaT1sFixed_filt.append(stdaT1sFixed_all[index])
            gaussT1s_filt.append(gaussT1s_all[index])
    orig_r2 = r2_score(stdaT1sOrig_filt, gaussT1s_filt)
    fixed_r2 = r2_score(stdaT1sFixed_filt, gaussT1s_filt)
    orig_MAE = mean_absolute_error(stdaT1sOrig_filt, gaussT1s_filt)
    fixed_MAE = mean_absolute_error(stdaT1sFixed_filt, gaussT1s_filt)
    clearPlot = False
    fig = plt.figure(num=2, figsize=[8, 5], dpi=300, clear=clearPlot)
    fig.add_subplot(3, 2, 4)
    plt.plot(stdaT1sOrig_filt, gaussT1s_filt, '.', color=colors_nipy[5], label='orig')
    plt.plot(stdaT1sFixed_filt, gaussT1s_filt, '.', color=colors_nipy[1], label='ML calib')
    x = np.linspace(0, 5, 100)
    plt.plot(x, x, 'k--')
    plt.grid(True)
    plt.legend(fontsize=16)
    plt.xlim(0, 5)
    plt.ylim(0, 5)
    stdstrplot = 'xTB-sTDA'
    engstrplot = engstr.replace('1', '$_1$')
    plt.xlabel(stdstrplot + ' ' + engstrplot + ' (eV)', fontsize=16)
    plt.ylabel('TDDFT ' + engstrplot + ' (eV)', fontsize=16)
    plt.annotate('R$^2$ orig: %0.2f\n' % orig_r2 +
                 'R$^2$ ML: %0.2f\n' % fixed_r2 +
                 'MAE orig: %0.2f\n' % orig_MAE +
                 'MAE ML: %0.2f' % fixed_MAE,
                 (4.75, 0.25),
                 bbox=dict(facecolor='white', alpha=0.5),
                 fontsize=14,
                 ha='right')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()

    settings = [['xtb', 'default'], ['xtb', '100ep'], ['xtb', 'hyperopt'],
                ['multi', 'default'], ['multi', '100ep'], ['multi', 'hyperopt']]
    xtbsT1 = ['stda', 'stddft']
    xtbsS1 = ['stda', 'stddft']
    allsetsT1 = {}
    allsetsS1 = {}
    allsets = {}
    for setting in settings:
        MLtype = setting[0]
        set = setting[1]
        allsetsT1[MLtype + set] = []
        allsetsS1[MLtype + set] = []
        allsets[MLtype + set] = []
        if (set == 'default'):
            setname = ''
        else:
            setname = set
        if (len(setname) != 0):
            setname = '_' + setname
        with open('verde_CP_results/verde_CP_' + MLtype + '_S1' + setname + '.txt', 'r') as file:
            data = file.readlines()
        for line in data:
            if ('cp mpnn stdafixed r2' in line):
                stdafixedr2 = float(line.split()[4])
            if ('cp mpnn stddftfixed r2' in line):
                stddftfixedr2 = float(line.split()[4])
        allsetsS1[MLtype + set].append(stdafixedr2)
        allsetsS1[MLtype + set].append(stddftfixedr2)
        allsets[MLtype + set].append(stdafixedr2)
        with open('verde_CP_results/verde_CP_' + MLtype + '_T1' + setname + '.txt', 'r') as file:
            data = file.readlines()
        for line in data:
            if ('cp mpnn stdafixed r2' in line):
                stdafixedr2 = float(line.split()[4])
            if ('cp mpnn stddftfixed r2' in line):
                stddftfixedr2 = float(line.split()[4])
        allsetsT1[MLtype + set].append(stdafixedr2)
        allsetsT1[MLtype + set].append(stddftfixedr2)
        allsets[MLtype + set].append(stdafixedr2)
    x = np.arange(len(allsets['xtbdefault']))  # the label locations
    width = 1 / (len(allsets) + 2)  # the width of the bars
    # fig = plt.figure(num=2, figsize=[6, 3], dpi=300, clear=False)
    ax = fig.add_subplot(3, 2, 5)
    labels = ['default', '100 epoch', 'hyperopt', 'multi, default', 'multi, 100 epoch', 'multi, hyperopt']
    for y in np.arange(len(allsets)):
        ax.bar(x + (y - 0.5 * (len(allsets) - 1)) * width,
               allsets[list(allsets.keys())[y]],
               width,
               label=labels[y],
               color=colors_nipy[y],
               zorder=3)
    ax.set_ylabel('R$^2$ Scores', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(['S$_1$', 'T$_1$'], fontsize=20)
    ax.set_ylim(0.7, 1)
    ylabels = ax.get_yticks()
    ylabels = ['%0.2f' % s for s in ylabels]
    ax.set_yticklabels(ylabels, fontsize=14)
    ax.grid(True, zorder=0)
    plt.tight_layout()
    ax.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', fontsize=16)
    plt.subplots_adjust(hspace=0.3)
    # plt.savefig('CPMPNN_all_comp_T1S1.png')
    # plt.close()

    label_axes(fig, ha='right')
    plt.savefig('ML_comp_plots.pdf')
    # plt.savefig('CPMPNN_' + stdstr2 + '_' + 'S1' +
    #             '_comp_cycle_2' + runType + '.png')


# write_test_files(labelOrig)
# check_test_files()
#
# do_ML_cp()
# do_xtb_analysis(labelOrig, '')
# stdlist = ['stda', 'stddft', 'stda_S1', 'stddft_S1']
# stdlist = ['stda_S1', 'stda']
# for i, val in enumerate(stdlist):
#     if 'S1' in val:
#         eng = 'S1'
#     else:
#         eng = 'T1'
#     do_stda_analysis(labelOrig, val, eng, '', i + 1)
comp_ML_figure_plot()
# do_ML_cp_opt()
# do_xtb_analysis(labelOrig, 'hyperopt')
# stdlist = ['stda', 'stddft', 'stda_S1', 'stddft_S1']
# for val in stdlist:
#     if 'S1' in val:
#         eng = 'S1'
#     else:
#         eng = 'T1'
#     do_stda_analysis(labelOrig, val, eng, 'hyperopt')
#
# do_ML_cp_rdkit()
# do_xtb_analysis(labelOrig, 'rdkit')
# stdlist = ['stda', 'stddft', 'stda_S1', 'stddft_S1']
# for val in stdlist:
#     if 'S1' in val:
#         eng = 'S1'
#     else:
#         eng = 'T1'
#     do_stda_analysis(labelOrig, val, eng, 'rdkit')


# do_ML_cp_morgan()
# do_xtb_analysis(labelOrig, 'morgan')
# stdlist = ['stda', 'stddft', 'stda_S1', 'stddft_S1']
# for val in stdlist:
#     if 'S1' in val:
#         eng = 'S1'
#     else:
#         eng = 'T1'
#     do_stda_analysis(labelOrig, val, eng, 'morgan')


# do_ML_cp_morganc()
# do_xtb_analysis(labelOrig, 'morganc')
# stdlist = ['stda', 'stddft', 'stda_S1', 'stddft_S1']
# for val in stdlist:
#     if 'S1' in val:
#         eng = 'S1'
#     else:
#         eng = 'T1'
#     do_stda_analysis(labelOrig, val, eng, 'morganc')

# do_ML_cp_100ep()
# do_xtb_analysis(labelOrig, '100ep')
# stdlist = ['stda', 'stddft', 'stda_S1', 'stddft_S1']
# for val in stdlist:
#     if 'S1' in val:
#         eng = 'S1'
#     else:
#         eng = 'T1'
#     do_stda_analysis(labelOrig, val, eng, '100ep')

# compare_all_ML_settings([['xtb', 'default'], ['xtb', '100ep'], ['xtb', 'hyperopt'],
#                          ['xtb', 'rdkit'], ['multi', 'default'], ['multi', '100ep'], ['multi', 'hyperopt']])

# compare_all_ML_settings([['xtb', 'default'], ['xtb', '100ep'], ['xtb', 'hyperopt'],
#                          ['multi', 'default'], ['multi', '100ep'], ['multi', 'hyperopt']])

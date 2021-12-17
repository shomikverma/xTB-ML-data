from molz import ZScorer
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Scaffolds import MurckoScaffold
import svg_stack as ss
import os
import sys
import subprocess
from tqdm import tqdm
import numpy as np
import pandas as pd

data_file = 'PCQC_exData_GDrive_iso_headers.csv'
data_file_name = os.path.splitext(data_file)[0]


def shell(cmd, shell=False):
    if shell:
        p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             executable='/bin/bash')
    else:
        cmd = cmd.split()
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             executable='/bin/bash')

    output, err = p.communicate()
    return output


def divide_test_set():
    try:
        os.mkdir('split_data')
    except:
        pass
    shell('split_filter () { { head -n 1 ' + data_file +
          '; cat; } > "$FILE"; }; export -f split_filter; '
          'tail -n +2 ' + data_file +
          ' | split --lines=100000 -d --filter=split_filter - split_data/split_', shell=True)


def CP_error_preds():
    try:
        os.mkdir('preds_data')
    except:
        pass
    for file in tqdm(sorted(os.listdir('split_data'))):
        # os.system('chemprop_predict --test_path split_data/' + file +
        #           ' --checkpoint_dir /home/shomik/SCOP_DB/mopssam_data/Qmsym_scop_stda '
        #           '--smiles_columns SMILES --preds_path preds_data/' + file + '_preds.csv --drop_extra_columns')
        os.system('chemprop_predict --test_path split_data/' + file +
                  ' --checkpoint_dir /home/shomik/SCOP_DB/mopssam_data/xTB_ML_model_scop_qm_ALS1T1_qmex '
                  '--smiles_columns SMILES --preds_path preds_data/' + file + '_preds.csv --drop_extra_columns')


def compile_preds():
    os.system("awk '(NR == 1) || (FNR > 1)' preds_data/split* > " + data_file_name + "_preds.csv")


def clean_up():
    os.system('rm -r split_data')
    os.system('rm -r preds_data')


# divide_test_set()
# CP_error_preds()
# compile_preds()
# clean_up()
# sys.exit()

errorFile = "data/pcqc_xtb_error_notrain.csv"
errorFileSan = "data/pcqc_xtb_error_san.csv"
num_frags = 12
mols_per_row = 4


def compile_data():
    errorFile = "data/pcqc_xtb_error.csv"
    trainFile = "data/all_training_data_scop_qm_ALS1T1_qmex.csv"
    errorData = pd.read_csv(errorFile, usecols=['CID', 'smiles', 'S1', 'UMAP_0', 'UMAP_1',
                                                'S1err', 'T1err', 'global_cluster'])
    errorData.rename({'smiles': 'SMILES'}, axis='columns', inplace=True)
    trainData = pd.read_csv(trainFile, usecols=['SMILES'])
    cols = list(errorData.columns)
    errorData = errorData[['SMILES'] + [cols[0]] + cols[2:]]
    print(errorData.info())
    print(trainData.info())
    df_all = errorData.merge(trainData.drop_duplicates(), on=['SMILES'],
                             how='left', indicator=True)
    df_final = df_all[df_all['_merge'] == 'left_only']
    df_final.rename({'SMILES': 'smiles'}, axis='columns', inplace=True)
    print(df_final.info())
    df_final.to_csv('data/pcqc_xtb_error_notrain.csv', index=False)
    # sys.exit()

    with open(errorFile, 'r') as file:
        data = file.readlines()
    S1LowErrMols = []
    S1UnderEstMols = []
    S1OverEstMols = []
    T1LowErrMols = []
    T1UnderEstMols = []
    T1OverEstMols = []
    np.random.seed(3)
    f = open(errorFileSan, 'w')
    f.write(data[0])
    data = np.random.choice(data, 10_000, replace=False)
    # numAdded = 0
    # maxAdded = 100_000
    smiInd = list(df_final.columns).index('smiles')
    S1errInd = list(df_final.columns).index('S1err')
    T1errInd = list(df_final.columns).index('T1err')
    for line in tqdm(data):
        if 'smiles' in line.lower():
            continue
        origLine = line
        line = line.replace('\n', '')
        line = line.split(',')
        smiles = line[smiInd]
        m = Chem.MolFromSmiles(smiles, sanitize=False)
        if m is None:
            print('invalid SMILES')
            continue
        else:
            try:
                Chem.SanitizeMol(m)
            except:
                print('invalid chemistry')
                continue
        f.write(origLine)
        S1err = float(line[S1errInd])
        T1err = float(line[T1errInd])
        if abs(S1err) < 0.05:
            S1LowErrMols.append(smiles)
        if S1err > 0.5:
            S1UnderEstMols.append(smiles)
        if S1err < -0.5:
            S1OverEstMols.append(smiles)
        if abs(T1err) < 0.05:
            T1LowErrMols.append(smiles)
        if T1err > 0:
            T1UnderEstMols.append(smiles)
        if T1err < -1.0:
            T1OverEstMols.append(smiles)
        # numAdded += 1
        # if numAdded >= maxAdded:
        #     break
    f.close()

    print('S1 low error', len(S1LowErrMols), '%0.2f' % (len(S1LowErrMols) / len(data) * 100))
    print('S1 heavily underestimate', len(S1UnderEstMols), '%0.2f' % (len(S1UnderEstMols) / len(data) * 100))
    print('S1 heavily overestimate', len(S1OverEstMols), '%0.2f' % (len(S1OverEstMols) / len(data) * 100))
    print('T1 low error', len(T1LowErrMols), '%0.2f' % (len(T1LowErrMols) / len(data) * 100))
    print('T1 heavily underestimate', len(T1UnderEstMols), '%0.2f' % (len(T1UnderEstMols) / len(data) * 100))
    print('T1 heavily overestimate', len(T1OverEstMols), '%0.2f' % (len(T1OverEstMols) / len(data) * 100))


def plot_frags(energy, eMin, eMax, filename):
    print('initializing ZScorer')
    scorer = ZScorer(errorFileSan)
    print('calculating fragment scores')
    scorer.score_fragments(energy, [eMin, eMax])
    print('plotting')
    # scorer.plot(k=num_frags, save_to='zscores_molZ_xtb.png',
    #             top_only=True, figsize=[4, 4])
    frags, scores = scorer._get_k_min_max_zscores(num_frags)
    for frag in frags[-num_frags:]:
        # try:
        svgData = scorer.draw_fragment(int(frag))
        # except:
        #     print(frag)
        #     continue
        with open('molZ/molZ_frag' + frag + '.svg', 'w') as file:
            file.write(svgData)

    doc2 = ss.Document()
    layout2 = ss.VBoxLayout()
    for i in range(int(len(frags[-num_frags:]) / mols_per_row)):
        doc = ss.Document()
        layout1 = ss.HBoxLayout()
        for j in range(mols_per_row):
            layout1.addSVG('molZ/molZ_frag' + frags[(-1 - j) - mols_per_row * i] + '.svg', alignment=ss.AlignTop)
        doc.setLayout(layout1)
        doc.save('molZ/qt_api_test' + str(i) + '.svg')
        layout2.addSVG('molZ/qt_api_test' + str(i) + '.svg', alignment=ss.AlignLeft)
    doc2.setLayout(layout2)
    doc2.save(filename)


def analyze_scaff(all_smiles, filename):
    all_scaffolds = {}
    for index, val in tqdm(enumerate(all_smiles), total=len(all_smiles)):
        # print(val)
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(val))
        # print(smiles)
        m = MurckoScaffold.MurckoScaffoldSmilesFromSmiles(smiles)
        try:
            m = Chem.MolToSmiles(Chem.MolFromSmiles(m))
        except:
            continue
        # print(m)
        if len(m) == 0:
            continue
        try:
            count = all_scaffolds[m]
            all_scaffolds[m] = count + 1
        except:
            all_scaffolds[m] = 1
        # if len(all_scaffolds) >= 10:
        #     break

    all_scaffolds = dict(sorted(all_scaffolds.items(), key=lambda item: item[1], reverse=True))

    common_scaffs = []
    legends = []
    for val in tqdm(all_scaffolds):
        # if all_scaffolds[val] >= 10:
        m = Chem.MolFromSmiles(val)
        if m.GetNumHeavyAtoms() > 10:
            common_scaffs.append(m)
            legends.append(str(all_scaffolds[val]))
        if len(common_scaffs) >= 16:
            break

    dopts = rdMolDraw2D.MolDrawOptions()
    dopts.legendFontSize = 50
    dopts.fontFile = '/home/shomik/miniconda3/envs/chemprop/share/RDKit/Data/Fonts/lmroman10-regular.otf'
    img = Draw.MolsToGridImage(common_scaffs, molsPerRow=8, legends=legends, drawOptions=dopts)
    img.save(filename + '_scaffs.png')
    pass


def create_figure():
    doc2 = ss.Document()
    layout2 = ss.VBoxLayout()
    layout2.setSpacing(50)
    plots = [['Low_S1_Error', 'Low_T1_Error'], ['High_S1_Overest', 'High_T1_Overest'],
             ['High_S1_Underest', 'High_T1_Underest']]
    for i in range(3):
        doc = ss.Document()
        layout1 = ss.HBoxLayout()
        layout1.setSpacing(25)
        for j in range(2):
            layout1.addSVG('molz_plots/' + plots[i][j] + '_Map_all.svg', alignment=ss.AlignTop)
        doc.setLayout(layout1)
        doc.save('molz_plots/fig_' + str(i) + '.svg')
        layout2.addSVG('molz_plots/fig_' + str(i) + '.svg', alignment=ss.AlignLeft)
    doc2.setLayout(layout2)
    doc2.save('molz_plots/molz_fig.svg')


# compile_data()
# plot_frags('S1err', -0.05, 0.05, 'molz_plots/Low_S1_Error_Map_all.svg')
# plot_frags('S1err', 0.5, 10, 'molz_plots/High_S1_Underest_Map_all.svg')
# plot_frags('S1err', -10, -0.5, 'molz_plots/High_S1_Overest_Map_all.svg')
# plot_frags('T1err', -0.05, 0.05, 'molz_plots/Low_T1_Error_Map_all.svg')
# plot_frags('T1err', 0.05, 10, 'molz_plots/High_T1_Underest_Map_all.svg')
# plot_frags('T1err', -10, -1, 'molz_plots/High_T1_Overest_Map_all.svg')
create_figure()

# analyze_scaff(S1LowErrMols, 'S1LowErr')
# analyze_scaff(T1LowErrMols, 'T1LowErr')
# analyze_scaff(S1OverEstMols, 'S1OverEst')
# analyze_scaff(T1OverEstMols, 'T1OverEst')
# analyze_scaff(S1UnderEstMols, 'S1UnderEst')
# analyze_scaff(T1UnderEstMols, 'T1UnderEst')

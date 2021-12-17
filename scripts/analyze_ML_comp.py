import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
from cycler import cycler

plt.style.use(['science', 'grid'])
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
colors = [colors[0], colors[2], colors[1], colors[3]] + colors[4:]
colors_nipy = mpl.cm.nipy_spectral(np.linspace(0.1, 0.9, 6))
plt.rcParams['axes.prop_cycle'] = cycler(color=colors)

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

print('model r2')
print('dc gcn', np.mean(dcGcnR2), '+/-', np.std(dcGcnR2))
print('dc gcn epoch', np.mean(dcGcnR2epoch), '+/-', np.std(dcGcnR2epoch))
print('dc mpnn', np.mean(dcMpnnR2), '+/-', np.std(dcMpnnR2))
print('dc mpnn epoch', np.mean(dcMpnnR2epoch), '+/-', np.std(dcMpnnR2epoch))
print('cp mpnn', np.mean(cpMpnnR2), '+/-', np.std(cpMpnnR2))
print()
print('xtb fixed r2')
print('original', np.mean(origR2), '+/-', np.std(origR2))
print('lin calib', np.mean(linR2), '+/-', np.std(linR2))
print('dc gcn', np.mean(dcGcnXtbR2), '+/-', np.std(dcGcnXtbR2))
print('dc gcn epoch', np.mean(dcGcnXtbR2epoch),
      '+/-', np.std(dcGcnXtbR2epoch))
print('dc mpnn', np.mean(dcMpnnXtbR2), '+/-', np.std(dcMpnnXtbR2))
print('dc mpnn epoch', np.mean(dcMpnnXtbR2epoch),
      '+/-', np.std(dcMpnnXtbR2epoch))
print('cp mpnn', np.mean(cpMpnnXtbR2), '+/-', np.std(cpMpnnXtbR2))
print()
print('xtb fixed improvements')
dcGcnImp = np.array(dcGcnXtbR2) - np.array(origR2)
dcGcnEpochImp = np.array(dcGcnXtbR2epoch) - np.array(origR2)
try:
    dcMpnnImp = np.array(dcMpnnXtbR2) - np.array(dcGcnXtbR2)
    dcMpnnEpochImp = np.array(dcMpnnXtbR2epoch) - np.array(dcGcnXtbR2epoch)
except:
    origR2cut = origR2[:-1]
    dcMpnnImp = np.array(dcMpnnXtbR2) - np.array(origR2cut)
    dcMpnnEpochImp = np.array(dcMpnnXtbR2epoch) - np.array(origR2cut)
# try:
#     cpMpnnImp = np.array(cpMpnnXtbR2) - np.array(dcMpnnXtbR2epoch)
# except:
#     origR2cut = origR2[:-1]
#     cpMpnnImp = np.array(cpMpnnXtbR2) - np.array(origR2cut)
# print('dc gcn', np.mean(dcGcnImp), '+/-', np.std(dcGcnImp))
# print('dc gcn epoch', np.mean(dcGcnEpochImp), '+/-', np.std(dcGcnEpochImp))
# print('dc mpnn', np.mean(dcMpnnImp), '+/-', np.std(dcMpnnImp))
# print('dc mpnn epoch', np.mean(dcMpnnEpochImp),
#       '+/-', np.std(dcMpnnEpochImp))
# print('cp mpnn', np.mean(cpMpnnImp), '+/-', np.std(cpMpnnImp))

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

print('model r2')
print('dc gcn', np.mean(dcGcnR2_S1), '+/-', np.std(dcGcnR2_S1))
print('dc mpnn', np.mean(dcMpnnR2_S1), '+/-', np.std(dcMpnnR2_S1))
print('cp mpnn', np.mean(cpMpnnR2_S1), '+/-', np.std(cpMpnnR2_S1))
print()
print('xtb fixed r2')
print('original', np.mean(origR2_S1), '+/-', np.std(origR2_S1))
print('lin calib', np.mean(linR2_S1), '+/-', np.std(linR2_S1))
print('dc gcn', np.mean(dcGcnXtbR2_S1), '+/-', np.std(dcGcnXtbR2_S1))
print('dc mpnn', np.mean(dcMpnnXtbR2_S1), '+/-', np.std(dcMpnnXtbR2_S1))
print('cp mpnn', np.mean(cpMpnnXtbR2_S1), '+/-', np.std(cpMpnnXtbR2_S1))
print()

# mpl.rcParams['axes.prop_cycle'] = cycler(color=['#3969AC', '#11A579', '#E68310', '#7F3C8D',
#                                                 '#F2B701', '#E73F74', '#80BA5A', '#008695',
#                                                 '#CF1C90', '#f97b72', '#4b4b8f', '#A5AA99'])
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
fig = plt.figure(num=2, figsize=[6, 4], dpi=300, clear=True)
ax = fig.add_subplot(1, 1, 1)
rects2 = ax.bar(x - width / 2, conv_means_S1, width, color=colors_nipy[0],
                label='S$_1$', yerr=conv_errbars_S1, ecolor='k', capsize=3)
rects1 = ax.bar(x + width / 2, conv_means, width, color=colors_nipy[1],
                label='T$_1$', yerr=conv_errbars, ecolor='k', capsize=3)
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('R$^2$ Scores', fontsize=16)
# ax.set_title('Scores for xTB-ML by ML model')
ax.set_xticks(x)
ax.set_yticklabels([-0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=14)
ax.set_xticklabels(labels, fontsize=14)
ax.legend(loc='upper left', fontsize=16)
plt.ylim(-0.25, 1)
ax.yaxis.grid(True)
ax.set_axisbelow(True)
plt.tight_layout()
# plt.savefig('ML_comp_DC_CP_T1.eps')
plt.savefig('ML_comp_DC_CP_T1S1.png')

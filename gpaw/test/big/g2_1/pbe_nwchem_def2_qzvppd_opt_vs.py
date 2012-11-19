import csv

import numpy as np

from ase.units import kcal, mol

from ase.data.g2_1 import molecule_names

from ase.data.g2_1_ref import atomization_vasp, diatomic

from gpaw.test.big.g2_1.pbe_nwchem_def2_qzvppd_opt_analyse import tag

csvreader = csv.reader(open(tag + '_ea.csv', 'rb'))
d1 = {}
for r in csvreader:
    d1[r[0]] = float(r[1])
csvreader = csv.reader(open(tag + '_distance.csv', 'rb'))
d2 = {}
for r in csvreader:
    d2[r[0]] = float(r[1])

csvwriter1 = csv.writer(open(tag + '_ea_vs.csv', 'wb'))
csvwriter2 = csv.writer(open(tag + '_distance_vs.csv', 'wb'))
csvwriter1.writerow(['#molecule', 'Ea', 'error', '% error'])
csvwriter2.writerow(['#molecule', 'R', 'error', '% error'])
derr1 = []
dname1 = []
derr2 = []
dname2 = []
for m in molecule_names:
    r1 = [m]
    r2 = [m]
    v1 = d1[m]
    if m in atomization_vasp:
        ref1 = atomization_vasp[m][2] * kcal / mol
        err1 = v1 - ref1
        derr1.append(err1)
        dname1.append(m)
        perr1 = err1 / ref1 * 100
        r1.extend(["%.3f" % v1, "%.3f" % err1, "%.2f" % perr1])
    else:
        ref1 = 'N/A'
        err1 = 'N/A'
        perr1 = 'N/A'
        r1.extend([v1, err1, perr1])
    v2 = d2[m]
    if m in diatomic:
        ref2 = diatomic[m][2]
        err2 = v2 - ref2
        derr2.append(err2)
        dname2.append(m)
        perr2 = err2 / ref2 * 100
        r2.extend(["%.3f" % v2, "%.3f" % err2, "%.2f" % perr2])
    else:
        ref2 = 'N/A'
        err2 = 'N/A'
        perr2 = 'N/A'
        r2.extend([v2, err2, perr2])
    if 'N/A' not in r1:
        csvwriter1.writerow(r1)
    if 'N/A' not in r2:
        csvwriter2.writerow(r2)

# stats

absd1 = zip(np.abs(derr1), dname1)
absd1.sort()
maxd1 = "%.3f:(%s)" % (absd1[-1][0], absd1[-1][1])
mind1 = "%.3f:(%s)" % (absd1[0][0], absd1[0][1])
csvwriter1.writerow(['#np.mean', "%.3f" % np.mean(derr1), 'N/A', 'N/A'])
csvwriter1.writerow(['#np.std', "%.3f" % np.std(derr1), 'N/A', 'N/A'])
csvwriter1.writerow(['#np.max', maxd1, 'N/A', 'N/A'])
csvwriter1.writerow(['#np.min', mind1, 'N/A', 'N/A'])

absd2 = zip(np.abs(derr2), dname2)
absd2.sort()
maxd2 = "%.3f:(%s)" % (absd2[-1][0], absd2[-1][1])
mind2 = "%.3f:(%s)" % (absd2[0][0], absd2[0][1])
csvwriter2.writerow(['#np.mean', "%.3f" % np.mean(derr2), 'N/A', 'N/A'])
csvwriter2.writerow(['#np.std', "%.3f" % np.std(derr2), 'N/A', 'N/A'])
csvwriter2.writerow(['#np.max', maxd2, 'N/A', 'N/A'])
csvwriter2.writerow(['#np.min', mind2, 'N/A', 'N/A'])

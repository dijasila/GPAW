import csv

import numpy as np

from ase.data.g2_1 import molecule_names

from gpaw.test.big.g2_1.pbe_gpaw_nrel_analyse import tag

from gpaw.test.big.g2_1.pbe_nwchem_def2_qzvppd_analyse import ref

csvreader = csv.reader(open(tag + '_ea.csv', 'rb'))
d1 = {}
for r in csvreader:
    d1[r[0]] = float(r[1])

csvwriter1 = csv.writer(open(tag + '_ea_vs.csv', 'wb'))
csvwriter1.writerow(['#molecule', 'Ea', 'error', '% error'])
derr1 = []
dname1 = []
for m in molecule_names:
    r1 = [m]
    v1 = d1[m]
    if m in ref['ea']:
        ref1 = ref['ea'][m]
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
    if 'N/A' not in r1:
        csvwriter1.writerow(r1)

# stats

absd1 = zip(np.abs(derr1), dname1)
absd1.sort()
maxd1 = "%.3f:(%s)" % (absd1[-1][0], absd1[-1][1])
mind1 = "%.3f:(%s)" % (absd1[0][0], absd1[0][1])
csvwriter1.writerow(['#np.mean', "%.3f" % np.mean(derr1), 'N/A', 'N/A'])
csvwriter1.writerow(['#np.std', "%.3f" % np.std(derr1), 'N/A', 'N/A'])
csvwriter1.writerow(['#np.max', maxd1, 'N/A', 'N/A'])
csvwriter1.writerow(['#np.min', mind1, 'N/A', 'N/A'])

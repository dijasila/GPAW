# web-page: extrapolate.png, N2-data.csv
import numpy as np
import matplotlib.pyplot as plt
from gpaw.utilities.extrapolate import extrapolate
from pathlib import Path

a = np.loadtxt('rpa_N2.dat')
ext, A, B, sigma = extrapolate(a[:, 0], a[:, 1], reg=3, plot=False)
plt.plot(a[:, 0]**(-1.5), a[:, 1], 'o', label='Calculated points')
es = np.array([e for e in a[:, 0]] + [10000])
plt.plot(es**(-1.5), A + B * es**(-1.5), '--', label='Linear regression')

t = [int(a[i, 0]) for i in range(len(a))]
plt.xticks(a[:, 0]**(-1.5), t, fontsize=12)
plt.axis([0., 150**(-1.5), None, -4.])
plt.xlabel('Cutoff energy [eV]', fontsize=18)
plt.ylabel('RPA correlation energy [eV]', fontsize=18)
plt.legend(loc='lower right')
plt.savefig('extrapolate.png')

pbe, hf = (-float(line.split()[1])
           for line in Path('PBE_HF.dat').read_text().splitlines())
rpa = -A
Path('N2-data.csv').write_text(
    'PBE, HF, RPA, HF+RPA, Experimental\n'
    f'{pbe:.2f}, {hf:.2f}, {rpa:.2f}, {hf + rpa:.2f}, 9.89\n')

assert abs(pbe - 10.600) < 0.01
assert abs(hf - 4.840) < 0.01
assert abs(rpa - 4.912) < 0.01

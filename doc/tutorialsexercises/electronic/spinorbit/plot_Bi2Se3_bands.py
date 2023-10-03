# web-page: Bi2Se3_bands.png
import matplotlib.pyplot as plt
import numpy as np
from ase.dft.kpoints import BandPath
from gpaw import GPAW
from gpaw.spinorbit import soc_eigenstates

calc = GPAW('Bi2Se3_bands.gpw', txt=None)
bandpath = BandPath.read('bandpath.json')
x, X, labels = bandpath.get_linear_kpoint_axis()

# No spin-orbit

ef = calc.get_fermi_level()
e_kn = np.array([calc.get_eigenvalues(kpt=k)
                 for k in range(len(calc.get_ibz_k_points()))])
e_nk = e_kn.T
e_nk -= ef

for e_k in e_nk:
    plt.plot(x, e_k, '--', c='0.5')

# Spin-orbit calculation
soc = soc_eigenstates(calc, scale=1.0)
e_nk = soc.eigenvalues().T - soc.fermi_level

plt.xticks(X, [r'$\Gamma$' if label == 'G' else label for label in labels],
           size=24)
plt.yticks(size=20)
for i in range(len(X))[1:-1]:
    plt.plot(2 * [X[i]], [1.1 * np.min(e_nk), 1.1 * np.max(e_nk)],
             c='0.5', linewidth=0.5)
for e_k in e_nk:
    plt.plot(x, e_k, c='b')

plt.ylabel(r'$\varepsilon_n(k)$ [eV]', size=24)
plt.axis([0, x[-1], -1.7, 1.7])
plt.tight_layout()
# plt.show()
plt.savefig('Bi2Se3_bands.png')

import numpy as np
import pylab as pl
from ase.dft.kpoints import get_bandpath, ibz_points
from gpaw import GPAW
from gpaw.spinorbit import get_spinorbit_eigenvalues
pl.rc('text', usetex=True)

calc = GPAW('Pt_bands.gpw', txt=None)
ef = GPAW('Pt_gs.gpw').get_fermi_level()

points = ibz_points['fcc']
G = points['Gamma']
X= points['X']
W = points['W']
L = points['L']
K = points['K']
kpts, x, X = get_bandpath([G,X,W,L,G,K,X], calc.atoms.cell, npoints=200)

e_kn = np.array([calc.get_eigenvalues(kpt=k)[:20] 
                 for k in range(len(calc.get_ibz_k_points()))])
e_nk = e_kn.T
e_nk -= ef

for e_k in e_nk:
    pl.plot(x, e_k, '--', c='0.5')

e_mk = get_spinorbit_eigenvalues(calc)
e_mk -= ef

pl.xticks(X, [r'$\Gamma$', 'X', 'W', 'L', r'$\Gamma$', 'K', 'X'], size=20)
pl.yticks(size=20)
for i in range(len(X))[1:-1]:
    pl.plot(2 * [X[i]], [-11, 13],
            c='0.5', linewidth=0.5)

for e_k in e_mk[::2]:
    pl.plot(x, e_k, c='b', lw=2)
pl.plot([0.0, x[-1]], 2*[0.0], c='0.5')

pl.ylabel(r'$\varepsilon_n(k)$ [eV]', size=24)
pl.axis([0, x[-1], -11, 13])
#pl.show()
pl.savefig('Pt_bands.png')

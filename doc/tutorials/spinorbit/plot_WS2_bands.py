import numpy as np
import pylab as pl
from ase.dft.kpoints import get_bandpath
from gpaw import GPAW
from gpaw.spinorbit import get_spinorbit_eigenvalues
pl.rc('text', usetex=True)

calc = GPAW('../bands.gpw', txt=None)

#kpts, x, X = get_bandpath([[0,0,0],[0.5,0,0],[1/3.,1/3.,0],[0,0,0]],
#                          calc.get_atoms().cell, npoints=1000)
x = np.loadtxt('../kpath.dat')
X = np.loadtxt('../highsym.dat')
e_kn = np.array([calc.get_eigenvalues(kpt=k)[:20] 
                 for k in range(len(calc.get_ibz_k_points()))])
e_nk = e_kn.T
e_nk -= calc.get_fermi_level()
for e_k in e_nk:
    pl.plot(x, e_k, '--', c='0.5')

e_nk, s_nk =  get_spinorbit_eigenvalues(calc, return_spin=True, bands=range(20))
#e_nk =  get_spinorbit_correction(calc, bands=range(20))
e_nk -= calc.get_fermi_level()

iK = 167
print calc.get_ibz_k_points()[167]
#print e_nk[17,167] - e_nk[16,167]

pl.xticks(X, ['M', 'K', r'$\Gamma$', '-K', '-M'], size=20)
pl.yticks(size=20)
for i in range(len(X))[1:-1]:
    pl.plot(2 * [X[i]], [1.1*np.min(e_nk), 1.1*np.max(e_nk)],
            c='0.5', linewidth=0.5)

pl.scatter(np.tile(x, len(e_nk)), e_nk.reshape(-1),
           c=s_nk.reshape(-1),
           edgecolor=pl.get_cmap('jet')(s_nk.reshape(-1)),
           s=2,
           marker='+')

pl.ylabel(r'$\varepsilon_n(k)$ [eV]', size=24)
pl.axis([0, x[-1], -4.5, 4.5])
#pl.show()
pl.savefig('WS2_bands.png')

import numpy as np
import pylab as pl
from gpaw import GPAW
from gpaw.spinorbit import get_spinorbit_eigenvalues
pl.rc('text', usetex=True)

calc = GPAW('WS2_bands.gpw', txt=None)

x = np.loadtxt('WS2_kpath.dat')
X = np.loadtxt('WS2_highsym.dat')
e_kn = np.array([calc.get_eigenvalues(kpt=k)[:20]
                 for k in range(len(calc.get_ibz_k_points()))])
e_nk = e_kn.T
e_nk -= calc.get_fermi_level()
for e_k in e_nk:
    pl.plot(x, e_k, '--', c='0.5')

e_nk, s_nk = get_spinorbit_eigenvalues(calc, return_spin=True, bands=range(20))
e_nk -= calc.get_fermi_level()
s_nk = (s_nk + 1.0) / 2.0

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
# pl.show()
pl.savefig('WS2_bands.png')

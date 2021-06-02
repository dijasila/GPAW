# web-page: WS2_bands.png
import numpy as np
import matplotlib.pyplot as plt
from gpaw import GPAW
from gpaw.spinorbit import soc_eigenstates


calc = GPAW('WS2_bands.gpw', txt=None)
ef = calc.get_fermi_level()

x = np.loadtxt('WS2_kpath.dat')
X = np.loadtxt('WS2_highsym.dat')
e_kn = np.array([calc.get_eigenvalues(kpt=k)
                 for k in range(len(calc.get_ibz_k_points()))])
e_nk = e_kn.T
e_nk -= ef
for e_k in e_nk:
    plt.plot(x, e_k, '--', c='0.5')

soc = soc_eigenstates(calc)
e_kn = soc.eigenvalues()
s_knv = soc.spin_projections()
e_kn -= ef
s_nk = s_knv[:, :, 2].T

plt.xticks(X, [r'$\mathrm{M}$', r'$\mathrm{K}$', r'$\Gamma$',
               r'$\mathrm{-K}$', r'$\mathrm{-M}$'], size=20)
plt.yticks(size=20)
for i in range(len(X))[1:-1]:
    plt.plot(2 * [X[i]], [1.1 * np.min(e_kn), 1.1 * np.max(e_kn)],
             c='0.5', linewidth=0.5)

things = plt.scatter(np.tile(x, len(e_kn.T)),
                     e_kn.T.reshape(-1),
                     c=s_nk.reshape(-1),
                     s=2)
plt.colorbar(things)
plt.ylabel(r'$\varepsilon_n(k)$ [eV]', size=24)
plt.axis([0, x[-1], -4.5, 4.5])
# plt.show()
plt.savefig('WS2_bands.png')

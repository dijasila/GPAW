import numpy as np
import matplotlib.pyplot as plt

indata = np.loadtxt('model_potentials.dat')

z = indata[:, 0]
dV = indata[:, 1]
Vmod = indata[:, 2]
Vdiff = indata[:, 3]
plt.plot(z, dV, '-', label='$\Delta V(z)$')
plt.plot(z, Vmod, '-', label='$V(z)$')
plt.plot(z, Vdiff, '-', label=
         '$[V^{V_\mathrm{Ga}^{-3}}_\mathrm{el}(z) - V^{0}_\mathrm{el}(z) ]$')

plt.axhline(-0.138, ls='dashed')
plt.axhline(0.0, ls='-',color='grey')
plt.xlabel('$z$ (A))', fontsize=18)
plt.ylabel('Planar averages (eV)', fontsize=18)
plt.legend(loc='upper right')
plt.xlim((z[0],z[-1]))
plt.savefig('planaraverages.png')

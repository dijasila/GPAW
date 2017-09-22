import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

data = np.load('electrostatic_data_444.npz')

z = data['z']
dV = data['D_V']
V_model = data['V_model']
V_diff = data['V_X'] - data['V_0']
plt.plot(z, dV, '-', label='$\Delta V(z)$')
plt.plot(z, V_model, '-', label='$V(z)$')
plt.plot(z, V_diff, '-',
         label=(r'$[V^{V_\mathrm{Ga}^{-3}}_\mathrm{el}(z) -'
                'V^{0}_\mathrm{el}(z) ]$'))

middle = len(dV) // 2
restricted = dV[middle - len(dV) // 8: middle + len(dV) // 8]
constant = restricted.mean().real
print(constant)
plt.axhline(constant, ls='dashed')
plt.axhline(0.0, ls='-', color='grey')
plt.xlabel(r'$z\enspace (\mathrm{\AA})$', fontsize=18)
plt.ylabel('Planar averages (eV)', fontsize=18)
plt.legend(loc='upper right')
plt.xlim((z[0], z[-1]))
plt.savefig('planaraverages.png')

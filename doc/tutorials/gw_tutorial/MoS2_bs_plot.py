import numpy as np

from ase.dft.kpoints import ibz_points, get_bandpath
from ase.units import Bohr,Hartree

from gpaw.response.gw_bands import GWBands

import matplotlib.pyplot as plt
from matplotlib import rc

# Initializing bands object
points = ibz_points['hexagonal']
K = np.array([1/3.,1/3.,0])
G = points['Gamma']
kpoints = np.array([G, K, G])

GW = GWBands(calc = 'MoS2_fulldiag.gpw',
              gw_file='MoS2_g0w0_80_results.pckl',
              kpoints = kpoints)

results = GW.get_gw_bands(SO=False, interpolate=True, vac=True) #without spin-orbit

x_x = results['x_k']
X = results['X']
eGW_kn = results['e_kn']
ef = results['ef']


# Plotting Bands
rc('text', usetex=True)
labels_K = [r'$\Gamma$', r'$K$', r'$\Gamma$']

f = plt.figure()
plt.plot(x_x, eGW_kn, '-r')

plt.axhline(ef,color='k',linestyle='--')

for p in X:
    plt.axvline(p,color='k')

rc('xtick', labelsize=12) 
rc('ytick', labelsize=12) 

plt.xlim(0, x_x[-1])
plt.xticks(X,labels_K, fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel('E - E$_{vac}$ (eV)', fontsize=24)
plt.legend(loc='upper right')
plt.savefig('MoS2_bs.png')
plt.show()


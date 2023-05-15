import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from ase.units import Hartree, Bohr
from ase.utils import seterr


a = 2.5
c = 3.22
df_tetra = np.loadtxt('df_tetra.csv', delimiter=',')
df_point = np.loadtxt('df_point.csv', delimiter=',')

# convolve with gaussian to smooth the curve
sigma = 7
df2_wimag_result = gaussian_filter1d(df_point[:, 4], sigma)

plt.figure(figsize=(6, 6))
plt.plot(df_tetra[:, 0], df_tetra[:, 4] * 2, label='Img Tetrahedron')
plt.plot(df_point[:, 0], df_point[:, 4] * 2, label='Img Point sampling')
plt.plot(df_point[:, 0], df2_wimag_result * 2, 'magenta', label='Inter Im')

plt.xlabel('Frequency (eV)')
plt.ylabel('$\\mathrm{Im}\\varepsilon$')
plt.xlim(0, 6)
plt.ylim(0, 50)
plt.legend()
plt.tight_layout()
plt.savefig('graphene_eps.png', dpi=600)

from gpaw.test import findpeak
print(df_point[200:,0])

w1, I1 = findpeak(df_point[200:, 0], df_point[200:, 4])
w2, I2 = findpeak(df_tetra[200:, 0], df_tetra[200:, 4])
w3, I3 = findpeak(df_point[200:, 0], df2_wimag_result[200:])
print(w1, I1)
print(w2, I2)
print(w3, I3)


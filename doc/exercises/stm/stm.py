from sys import argv
import matplotlib.pyplot as plt

from ase.dft.stm import STM
from gpaw import restart

filename = argv[1]
z0 = 8
bias = 1.0

atoms, calc = restart(filename, txt=None)

stm = STM(atoms, symmetries=[0, 1, 2])
c = stm.get_averaged_current(bias, z0)

print(f'Average current at z={z0:f}: {c:f}')

# Get 2d array of constant current heights:
x, y, h = stm.scan(bias, c)

print(f'Min: {h.min():.2f} Ang, Max: {h.max():.2f} Ang')

plt.contourf(x, y, h, 40)
plt.hot()
plt.colorbar()
plt.show()

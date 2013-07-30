from ase.dft.stm import STM
from gpaw import GPAW
calc = GPAW('al100.gpw')
atoms = calc.get_atoms()
stm = STM(atoms, symmetries=[0, 1, 2])
z = 8.0
bias = 1.0
c = stm.get_averaged_current(bias, z)
h = stm.scan(bias, c)
import matplotlib.pyplot as plt
import numpy as np
h = np.tile(h, (3, 3))
plt.contourf(h, 40)
plt.hot()
plt.colorbar()
plt.savefig('2d.png')
plt.figure()
x, y = stm.linescan(bias, c, [0, 0], [2.5, 2.5])
plt.plot(x, y)
plt.savefig('line.png')

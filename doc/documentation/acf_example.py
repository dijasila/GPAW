# creates: acf_example.png
import numpy as np
import matplotlib.pyplot as plt
from gpaw.core import UniformGrid


alpha = 4.0
rcut = 2.0
l = 0
gauss = (l, rcut, lambda r: (4 * np.pi)**0.5 * np.exp(-alpha * r**2))
grid = UniformGrid(cell=[4.0, 2.0, 2.0], size=[40, 20, 20])
pos = [[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]]
acf = grid.atom_centered_functions([[gauss], [gauss]], pos)
coefs = acf.empty(dims=(2,))
coefs[0] = [[1], [-1]]
coefs[1] = [[2], [1]]
print(coefs.data, coefs[0])
f = grid.zeros(2)
acf.add_to(f, coefs)
x = grid.xyz()[:, 10, 10, 0]
y1, y2 = f.data[:, :, 10, 10]
ax = plt.subplot(1, 1, 1)
ax.plot(x, y1, 'o-')
ax.plot(x, y2, 'x-')
# plt.show()
plt.savefig('acf_example.png')

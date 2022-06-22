# web-page: VCl2.png
from gpaw.new.ase_interface import GPAW
import matplotlib.pyplot as plt

calc = GPAW('VCl2_gs.gpw')
n = calc.calculation.densities().pseudo_densities()
x, y = n.desc.xyz()[:, :, 30, :2].transpose((2, 0, 1))
u, v = n.data[1:3, :, :, 30]
plt.quiver(x, y, u, v)
# plt.show()
plt.savefig('VCl2.png')

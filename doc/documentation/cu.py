import numpy as np
import matplotlib.pyplot as plt
from ase.build import bulk
from ase.units import Ry
from ase.io import read
from gpaw import GPAW, PW

cu = bulk('Cu', 'fcc', a=3.6)

if 1:
    # for k in range(6, 14):
    for k in range(14, 21):
        cu.calc = GPAW(
            mode=PW(400),
            kpts=(k, k, k),
            # occupations={'name': 'tetrahedron-method'},
            # occupations={'name': 'marzari-vanderbilt', 'width': 0.2},
            occupations={'name': 'fermi-dirac', 'width': 0.05},
            txt=f'Cu-fd005-{k}.txt')
        e = cu.get_potential_energy()
        cu.calc.write(f'Cu-fd005-{k}.gpw')

if 1:
    data = []
    for k in range(6, 21):
        e1 = read(f'Cu-tm-{k}.txt').get_potential_energy()
        e2 = read(f'Cu-itm-{k}.txt').get_potential_energy()
        e3 = read(f'Cu-mv02-{k}.txt').get_potential_energy()
        e4 = read(f'Cu-fd005-{k}.txt').get_potential_energy()
        data.append((k, e1, e2, e3, e4))
    data = np.array(data)
    x, y, z, e3, e4 = data.T
    print(data[-1, 1:] / Ry * 1000)
    data[:, 1:] -= data[-1, 2]
    plt.plot(x**-2, y / Ry * 1000, label='TM')
    plt.plot(x**-2, z / Ry * 1000, label='iTM')
    plt.plot(x**-2, e3 / Ry * 1000, label='MV-0.2')
    plt.plot(x**-2, e4 / Ry * 1000, label='FD-0.05')
    plt.legend()
    plt.show()

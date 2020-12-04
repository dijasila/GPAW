import matplotlib.pyplot as plt

from gpaw import GPAW
from gpaw.zerro_field_splitting import zfs, convert_tensor


distance = []
coupling = []
for n in range(3, 13, 2):
    name = f'C{n}H{2 * n}'
    calc = GPAW(name + '.gpw')
    d = calc.atoms.get_distance(0, -3)
    D_vv = zfs(calc)
    D, E, axis, D_vv = convert_tensor(D_vv, 'Mhz')
    print(d, D, E, axis)
    print(D_vv)
    distance.append(d)
    coupling.append(D)


plt.plot(distance, coupling)
plt.show()

import numpy as np
from ase import Atoms
from gpaw import GPAW


def cnh2n(n):
    assert n % 2 == 1
    d = 1.54
    h = 1.1
    dx = d * (2 / 3)**0.5
    dz = d / 3**0.5
    positions = []
    for i in range(n):
        x = i * dx
        z = dz * (i % 2)
        positions.append((x, 0, z))
        if i == 0:
            positions.append((x, 0, -h))
            positions.append((x - dx, 0, dz))
        elif i == n - 1:
            positions.append((x, 0, -h))
            positions.append((x + dx, 0, dz))
        elif i % 2 == 0:
            positions.append((x, -h * (2 / 3)**0.5, -h / 3**0.5))
            positions.append((x, h * (2 / 3)**0.5, -h / 3**0.5))
        else:
            positions.append((x, -h * (2 / 3)**0.5, z + h / 3**0.5))
            positions.append((x, h * (2 / 3)**0.5, z + h / 3**0.5))

    atoms = Atoms(f'(CH2){n}', positions)
    atoms.set_distance(0, 2, h, 0)
    atoms.set_distance(-3, -1, h, 0)
    magmoms = np.zeros(3 * n)
    magmoms[0] = 1.0
    magmoms[-3] = 1.0
    atoms.set_initial_magnetic_moments(magmoms)
    return atoms


if __name__ == '__main__':
    n = 5
    a = cnh2n(n)
    # a.edit()
    a.center(vacuum=4.0)
    a.calc = GPAW(mode='pw', xc='PBE', txt=f'c{n}h{2 * n}.txt')
    a.get_potential_energy()
    a.calc.write(f'c{n}h{2 * n}.gpw', 'all')
    
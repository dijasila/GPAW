import numpy as np
from ase import Atoms
from gpaw import GPAW, PW

d = 1.54
dx = d * (2 / 3)**0.5
dz = d / 3**0.5
h = 1.1


def cnh2n(n):
    assert n % 2 == 1
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
    for n in range(3, 13, 2):
        n = 5
        a = cnh2n(n)
        a.center(vacuum=4.0)
        a.calc = GPAW(mode=PW(600),
                      xc='PBE', txt=f'c{n}h{2 * n}.txt')
        a.get_potential_energy()
        a.calc.write(f'c{n}h{2 * n}.600.gpw', 'all')
        break
    from gpaw.zero_field_splitting import main
    import matplotlib.pyplot as plt
    r = []
    D = []
    for n in range(3, 13, 2):
        gpw = f'c{n}h{2 * n}.gpw'
        d = main([gpw])[0, 0]
        r.append((n - 1) * dx)
        D.append(d)

    plt.plot(r, D)
    plt.show()
    
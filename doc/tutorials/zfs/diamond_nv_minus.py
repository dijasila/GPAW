import numpy as np
from ase.build import bulk
from gpaw import GPAW, PW


def nv(n):
    atoms = bulk('C', 'diamond') * n
    atoms.numbers[0] = 7
    del atoms[1]
    magmoms = np.zeros(len(atoms))
    magmoms[0] = 2.0
    magmoms[:] = 1.0
    atoms.set_initial_magnetic_moments(magmoms)
    name = f'c{2 * n**3 - 2}n'
    atoms.calc = GPAW(xc='PBE',
                      mode=PW(ecut=500),
                      charge=-1,
                      txt=name + '.txt')
    atoms.get_forces()
    atoms.calc.write(name + '.gpw', 'all')


if __name__ == '__main__':
    for n in range(2, 5):
        nv(n)

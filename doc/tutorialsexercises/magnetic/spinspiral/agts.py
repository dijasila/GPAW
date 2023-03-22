import numpy as np
from gpaw.new.ase_interface import GPAW
from myqueue.workflow import run


def workflow():
    with run(script='fe_sgs.py', cores=40, tmax='4h'):
        run(script='plot.py')
        run(function=check)


def check():
    energies = []
    magmoms = []
    for i in [0, 30]:
        atoms = GPAW(f'gs-{i:02}.gpw').get_atoms()
        energy = atoms.get_potential_energy()
        magmom = np.linalg.norm(atoms.calc.calculation.magmoms()[0])
        energies.append(energy)
        magmoms.append(magmom)

    print(energies, magmoms)

    assert abs(energies[0] - energies[1] - 0.0108) < 0.001
    assert abs(magmoms[0]) < 0.02
    assert abs(magmoms[1] - 0.83) < 0.02


if __name__ == '__main__':
    check()

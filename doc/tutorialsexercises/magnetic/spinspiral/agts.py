from gpaw import GPAW
from myqueue.workflow import run


def workflow():
    with run(script='fe_sgs.py', cores=40, tmax='4h'):
        run(script='plot.py')
        run(function=check)


def check():
    energies = []
    magmoms = []
    for i in range(31):
        atoms = GPAW(f'gs-{i:02}.gpw').get_atoms()
        energy = atoms.get_potential_energy()
        magmom = atoms.get_magnetic_moment()
        energies.append(energy)
        magmoms.append(magmom)

    assert abs(energies[0] - energies[30] - 0.1) < 0.001
    assert abs(magmoms[0] - 2.3) < 0.1
    assert abs(magmoms[30] - 2.3) < 0.1


if __name__ == '__main__':
    check()

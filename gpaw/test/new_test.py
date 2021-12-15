from gpaw.calculator import GPAW as OldGPAW
from gpaw.new.ase_interface import GPAW as NewGPAW
from ase import Atoms


def new():
    params = {'mode': {'name': 'pw'}}
    R = []
    for GPAW in [NewGPAW, OldGPAW]:
        atoms = Atoms('H2', cell=[2, 2, 3], pbc=True)
        atoms.positions[1, 2] = 0.8
        atoms.calc = GPAW(**params,
                          txt=f'{len(R)}.txt')
        f = atoms.get_forces()
        e = atoms.get_potential_energy()
        if 1:
            atoms.positions[1, 2] = 0.75
            f2 = atoms.get_forces()
            e2 = atoms.get_potential_energy()
        R.append((e, f, e2, f2))

    for e, f, e2, f2 in R:
        print(e, e2)
        print(f)
        print(f2)


if __name__ == '__main__':
    new()

from gpaw import GPAW
from ase.build import bulk


def getcalc():
    return GPAW(mode='pw',
                kpts=[[0.1, 0.2, 0.4]],
                #kpts=[2, 2, 2],
                txt='gpaw.txt')


def getatoms():
    atoms = bulk('Si')
    atoms.cell[0, 0] += 1e-3
    atoms.calc = getcalc()
    return atoms


def calculate(atoms):
    energy = atoms.get_potential_energy()
    return energy, atoms.calc.scf.niter


def adjust_cell(atoms):
    atoms.cell[0, 0] += 1e-1


def test_reuse_wfs():
    # Ordinary calculation
    atoms = getatoms()
    e1, niter1 = calculate(atoms)
    print(e1, niter1)


    # Recalculation after adjusting cell.  The goal is to require
    # few SCF steps for this.
    adjust_cell(atoms)
    e2, niter2 = calculate(atoms)
    print(e2, niter2)

    # Control calculation -- GS calculated for the adjusted cell,
    # but from scratch.  This must be equal to the preceding calculation,
    # but with approximately as many SCF steps as the initial calculation.
    atoms3 = getatoms()
    adjust_cell(atoms3)
    e3, niter3 = calculate(atoms3)
    print(e3, niter3)

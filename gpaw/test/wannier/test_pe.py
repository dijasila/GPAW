from ase import Atoms
from gpaw import GPAW
from gpaw.wannier import calculate_overlaps


def test_pe():
    d = 1.54
    h = 1.1
    x = d * (2 / 3)**0.5
    z = d / 3**0.5
    pe = Atoms('C2H4',
               positions=[[0, 0, 0],
                          [x, 0, z],
                          [0, -h * (2 / 3)**0.5, -h / 3**0.5],
                          [0, h * (2 / 3)**0.5, -h / 3**0.5],
                          [x, -h * (2 / 3)**0.5, z + h / 3**0.5],
                          [x, h * (2 / 3)**0.5, z + h / 3**0.5]],
               cell=[2 * x, 0, 0],
               pbc=(1, 0, 0))
    pe.center(vacuum=2.0, axis=(1, 2))
    pe.calc = GPAW(mode='pw',
                   kpts=(3, 1, 1),
                   symmetry='off',
                   txt='3k.txt')
    pe.get_potential_energy()
    pe.calc.write('3k.gpw', mode='all')
    o = calculate_overlaps(pe.calc, n2=6)
    pe3 = pe.repeat((3, 1, 1))
    pe3.calc = GPAW(mode='pw', txt='3.txt')
    pe3.get_potential_energy()
    pe3.calc.write('3.gpw', mode='all')
    o = calculate_overlaps(pe3.calc, n2=3 * 6)
    w = o.localize()
    w.centers_as_atoms().edit()


test_pe()

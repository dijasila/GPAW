'Test ase.dft.wannier module with k-points.'

from ase.structure import bulk
from ase.dft.wannier import Wannier

from gpaw import GPAW

if 1:
    si = bulk('Si', 'diamond', cubic=True, a=5.43)
    si.calc = GPAW(kpts=(2, 2, 2), nbands=24, h = 0.25, txt='Si-ibz.txt')
    e1 = si.get_potential_energy()
    si.calc.write('Si-ibz', mode='all')
    si.calc.set(usesymm=None, txt='Si-bz.txt')
    e2 = si.get_potential_energy()
    si.calc.write('Si-bz', mode='all')
    print e1, e2

def wan(calc):
    w = Wannier(16, calc)
    w.localize()
    x = w.get_functional_value()
    centers = (w.get_centers(1) * 2) % 1
    c = (centers + 0.125) * 4
    assert abs(c.round() - c).max() < 0.0003
    assert w.get_radii().ptp() < 0.0005
    assert [1, 1, 1] in (centers * 8).round().astype(int).tolist()
    if 0:
        from ase.visualize import view
        from ase import Atoms
        watoms = calc.atoms + Atoms(symbols='X16',
                                    scaled_positions=centers,
                                    cell=calc.atoms.cell)
        view(watoms)
    return x

calc1 = GPAW('Si-ibz', txt=None)
calc1.wfs.ibz2bz(calc1.atoms)
x1 = wan(calc1)
calc2 = GPAW('Si-bz', txt=None, parallel={'domain': calc1.wfs.world.size})
x2 = wan(calc2)
print x1, x2
assert abs(x1 - x2) < 1e-4

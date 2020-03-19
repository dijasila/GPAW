from ase import Atoms
from ase.build import mx2
from gpaw import GPAW

d = 3.0
a12 = mx2()
a12 += Atoms('H2', positions=[(0, 0, d), (0, 0, d + 0.74)])
a12.center(vacuum=3.0, axis=2)

a1 = a12[:3]
a2 = a12[3:]

bp = a12.cell.bandpath('GKM', npoints=20)

a1.calc = GPAW(mode='lcao',
               basis='sz(dzp)',
               kpts=(3, 3, 1),
               txt='1.txt')
a1.get_potential_energy()
a1.calc.write('1.gpw', mode='all')

a1.calc.set(fixdensity=True, kpts=bp, symmetry='off')
a1.get_potential_energy()
a1.calc.write('1bs.gpw', mode='all')
bs = a1.calc.band_structure()
bs.write('1bs.json')

a2.calc = GPAW(mode='lcao',
               basis='sz(dzp)',
               kpts=(3, 3, 1),
               txt='2.txt')
a2.get_potential_energy()
a2.calc.write('2.gpw', mode='all')

a2.calc.set(fixdensity=True, kpts=bp, symmetry='off')
a2.get_potential_energy()
a2.calc.write('2bs.gpw', mode='all')
bs = a2.calc.band_structure()
bs.write('2bs.json')

a12.calc = GPAW(mode='lcao',
                basis='sz(dzp)',
                kpts=(3, 3, 1),
                txt='12.txt')
a12.get_potential_energy()
a12.calc.write('12.gpw')

a12.calc.set(fixdensity=True, kpts=bp, symmetry='off')
a12.get_potential_energy()
a12.calc.write('12bs.gpw', mode='all')
bs = a12.calc.band_structure()
bs.write('12bs.json')

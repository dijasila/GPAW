from ase import Atoms
from ase.build import mx2
from gpaw import GPAW
from gpaw.lcao.scissors import Scissors

d = 3.0
a12 = mx2()
a12 += Atoms('H2', positions=[(0, 0, d), (0, 0, d + 0.74)])
a12.center(vacuum=3.0, axis=2)

a1 = a12[:3]
a2 = a12[3:]

a1.calc = GPAW(mode='lcao',
               basis='sz(dzp)',
               kpts=(3, 3, 1),
               txt='1.txt')
a1.get_potential_energy()
a1.calc.write('1.gpw', mode='all')

a2.calc = GPAW(mode='lcao',
               basis='sz(dzp)',
               kpts=(3, 3, 1),
               txt='2.txt')
a2.get_potential_energy()
a2.calc.write('2.gpw', mode='all')

a12.calc = GPAW(mode='lcao',
                basis='sz(dzp)',
                kpts=(3, 3, 1),
                txt='12.txt')
a12.get_potential_energy()
a12.calc.write('12.gpw')

a12.calc = GPAW(mode='lcao',
                basis='sz(dzp)',
                kpts=(3, 3, 1),
                eigensolver=Scissors([(-0.5, 0.5, a1.calc),
                                      (1.0, -1.0, a2.calc)]),
                txt='12s.txt')
a12.get_potential_energy()
a12.calc.write('12s.gpw')

from gpaw import GPAW
from gpaw.lcao.scissors import Scissors

c1 = GPAW('1.gpw', txt=None)
c2 = GPAW('2.gpw', txt=None)
a12 = GPAW('12.gpw', txt=None).atoms
bp = a12.cell.bandpath('GKM', npoints=20)
a12.calc = GPAW(mode='lcao',
                basis='sz(dzp)',
                kpts=(3, 3, 1),
                eigensolver=Scissors([(-0.5, 0.5, c1),
                                      (1.0, -1.0, c2)]),
                txt='12_s.txt')
a12.get_potential_energy()
a12.calc.write('12_s.gpw')

c1 = GPAW('1bs.gpw', txt=None)
c2 = GPAW('2bs.gpw', txt=None)
a12.calc.set(fixdensity=True, kpts=bp, symmetry='off',
             eigensolver=Scissors([(-0.5, 0.5, c1),
                                   (1.0, -1.0, c2)]))
a12.get_potential_energy()
a12.calc.write('12bs_s.gpw', mode='all')
bs = a12.calc.band_structure()
bs.write('12bs_s.json')

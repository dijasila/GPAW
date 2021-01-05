from gpaw import GPAW
from gpaw.lcao.scissors import Scissors

k = 6
c1 = GPAW('1.gpw')
c2 = GPAW('2.gpw')
a12 = GPAW('12.gpw').atoms
bp = a12.cell.bandpath('GMKG', npoints=80)
a12.calc = GPAW(mode='lcao',
                basis='dzp',
                kpts=(k, k, 1),
                eigensolver=Scissors([(-0.5, 0.5, c1),
                                      (-0.3, 0.3, c2)]),
                txt='12_s.txt')
a12.get_potential_energy()
a12.calc.write('12_s.gpw')

c1 = GPAW('1bs.gpw')
c2 = GPAW('2bs.gpw')
a12.calc.set(fixdensity=True, kpts=bp, symmetry='off',
             eigensolver=Scissors([(-0.5, 0.5, c1),
                                   (-0.3, 0.3, c2)]))
a12.get_potential_energy()
a12.calc.write('12bs_s.gpw', mode='all')
bs = a12.calc.band_structure()
bs.write('12bs_s.json')

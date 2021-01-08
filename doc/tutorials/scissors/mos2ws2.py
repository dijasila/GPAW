from ase.build import mx2
from gpaw import GPAW

k = 6
a12 = mx2(formula='MoS2', kind='2H', a=3.184, thickness=3.13,
          size=(1, 1, 1))
a12 += mx2(formula='WS2', kind='2H', a=3.184, thickness=3.15,
           size=(1, 1, 1))
a12.positions[3:, 2] += 3.6 + 3.184
a12.center(vacuum=3.0, axis=2)

a1 = a12[:3]
a2 = a12[3:]

bp = a12.cell.bandpath('GMKG', npoints=80)

a1.calc = GPAW(mode='lcao',
               basis='dzp',
               nbands='nao',
               kpts=(k, k, 1),
               txt='1.txt')
a1.get_potential_energy()
a1.calc.write('1.gpw', mode='all')

c1 = a1.calc.fixed_density(kpts=bp, symmetry='off')
c1.write('1bs.gpw', mode='all')
bs = c1.band_structure()
bs.write('1bs.json')

a2.calc = GPAW(mode='lcao',
               basis='dzp',
               nbands='nao',
               kpts=(k, k, 1),
               txt='2.txt')
a2.get_potential_energy()
a2.calc.write('2.gpw', mode='all')

c2 = a2.calc.fixed_density(kpts=bp, symmetry='off')
c2.write('2bs.gpw', mode='all')
bs = c2.band_structure()
bs.write('2bs.json')

a12.calc = GPAW(mode='lcao',
                basis='dzp',
                kpts=(k, k, 1),
                txt='12.txt')
a12.get_potential_energy()
a12.calc.write('12.gpw')

c12 = a12.calc.fixed_density(kpts=bp, symmetry='off')
c12.write('12bs.gpw', mode='all')
bs = c12.band_structure()
bs.write('12bs.json')

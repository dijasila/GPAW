from gpaw.lcaotddft import LCAOTDDFT
from gpaw.lcaotddft.dipolemomentwriter import DipoleMomentWriter
from gpaw.lcaotddft.magneticmomentwriter import MagneticMomentWriter

td_calc = LCAOTDDFT('gs.gpw', txt='td.out')
DipoleMomentWriter(td_calc, 'dm.dat')
MagneticMomentWriter(td_calc, 'mm.dat')
td_calc.absorption_kick([1e-5, 0.0, 0.0])
td_calc.propagate(10, 1000)
td_calc.write('td.gpw', mode='all')

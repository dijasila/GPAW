from gpaw.lcaotddft import LCAOTDDFT
from gpaw.lcaotddft.dipolemomentwriter import DipoleMomentWriter

# Parallelzation settings
parallel = {'sl_auto': True, 'domain': 2, 'augment_grids': True}

# Time propagation
td_calc = LCAOTDDFT('gs.gpw', parallel=parallel, txt='td.out')
DipoleMomentWriter(td_calc, 'dm.dat')
td_calc.absorption_kick([1e-5, 0.0, 0.0])
td_calc.propagate(10, 3000)

from gpaw.lcaotddft import LCAOTDDFT
from gpaw.lcaotddft.dipolemomentwriter import DipoleMomentWriter
from gpaw import setup_paths

# Insert the path to the created basis set
setup_paths.insert(0, '.')

# Parallelzation settings
parallel = {'sl_auto': True, 'domain': 2, 'augment_grids': True}

# Time propagation
td_calc = LCAOTDDFT('gs.gpw', parallel=parallel, txt='td.out')
DipoleMomentWriter(td_calc, 'dm.dat')
td_calc.absorption_kick([1e-5, 0.0, 0.0])
td_calc.propagate(10, 3000)

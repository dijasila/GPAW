from gpaw.lcaotddft import LCAOTDDFT
from gpaw.lcaotddft.dipolemomentwriter import DipoleMomentWriter

# Read the ground-state file
td_calc = LCAOTDDFT('gs.gpw', propagator=dict(name='wfw.ulm', update='all'))

# Attach analysis tools
DipoleMomentWriter(td_calc, 'dm_replayed.dat')

# Propagate according to the saved trajectory
td_calc.autopropagate()

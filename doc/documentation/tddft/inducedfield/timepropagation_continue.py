from gpaw.tddft import TDDFT, DipoleMomentWriter, RestartFileWriter
from gpaw.inducedfield.inducedfield_tddft import TDDFTInducedField

# Load TDDFT object
td_calc = TDDFT('na2_td.gpw')
DipoleMomentWriter(td_calc, 'na2_td_dm.dat')
RestartFileWriter(td_calc, 'na2_td.gpw')

# Load and attach InducedField object
ind = TDDFTInducedField(filename='na2_td.ind',
                        paw=td_calc,
                        restart_file='na2_td.ind')

# Continue propagation as usual
time_step = 20.0
iterations = 250
td_calc.propagate(time_step, iterations)

# Save TDDFT and InducedField objects
td_calc.write('na2_td.gpw', mode='all')
ind.write('na2_td.ind')

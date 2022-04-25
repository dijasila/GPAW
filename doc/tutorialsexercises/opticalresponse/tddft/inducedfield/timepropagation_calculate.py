from ase import Atoms
from gpaw import GPAW
from gpaw.tddft import TDDFT, DipoleMomentWriter, RestartFileWriter
from gpaw.inducedfield.inducedfield_tddft import TDDFTInducedField

# Na2 cluster
atoms = Atoms(symbols='Na2',
              positions=[(0, 0, 0), (3.0, 0, 0)],
              pbc=False)
atoms.center(vacuum=6.0)

# Standard ground state calculation
calc = GPAW(nbands=2, h=0.4, setups={'Na': '1'})
atoms.calc = calc
energy = atoms.get_potential_energy()
calc.write('na2_gs.gpw', mode='all')

# Standard time-propagation initialization
time_step = 10.0
iterations = 3000
kick_strength = [1.0e-3, 0.0, 0.0]
td_calc = TDDFT('na2_gs.gpw')
DipoleMomentWriter(td_calc, 'na2_td_dm.dat')
RestartFileWriter(td_calc, 'na2_td.gpw')

# Create and attach InducedField object
frequencies = [1.0, 2.08]     # Frequencies of interest in eV
folding = 'Gauss'             # Folding function
width = 0.1                   # Line width for folding in eV
ind = TDDFTInducedField(paw=td_calc,
                        frequencies=frequencies,
                        folding=folding,
                        width=width,
                        restart_file='na2_td.ind')

# Propagate as usual
td_calc.absorption_kick(kick_strength=kick_strength)
td_calc.propagate(time_step, iterations)

# Save TDDFT and InducedField objects
td_calc.write('na2_td.gpw', mode='all')
ind.write('na2_td.ind')

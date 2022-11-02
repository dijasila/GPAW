from ase.build import bulk
from gpaw import GPAW
from gpaw.response.df import DielectricFunction

# Part 1: Ground state calculation
atoms = bulk('Si', 'diamond', a=5.431)
calc = GPAW(mode='pw', kpts=(4, 4, 4))

atoms.calc = calc
atoms.get_potential_energy()  # ground state calculation is performed
calc.write('si.gpw', 'all')  # use 'all' option to write wavefunction

# Part 2 : Spectrum calculation
# DF: dielectric function object
# Ground state gpw file (with wavefunction) as input
df = DielectricFunction(
    calc='si.gpw',
    frequencies={'type': 'nonlinear',
                 'domega0': 0.05},  # using nonlinear frequency grid
    rate='eta')
# By default, a file called 'df.csv' is generated
df.get_dielectric_function()

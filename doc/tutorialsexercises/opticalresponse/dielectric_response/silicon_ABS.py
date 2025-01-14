# web-page: mac_eps.csv
# literalinclude0
# Refer to G. Kresse, Phys. Rev. B 73, 045112 (2006)
# for comparison of macroscopic and microscopic dielectric constant
# and absorption peaks.
from pathlib import Path

from ase.build import bulk
from ase.parallel import paropen, world

from gpaw import GPAW, FermiDirac
from gpaw.response.df import DielectricFunction

# Ground state calculation
a = 5.431
atoms = bulk('Si', 'diamond', a=a)

calc = GPAW(mode='pw',
            kpts={'density': 5.0, 'gamma': True},
            parallel={'band': 1, 'domain': 1},
            xc='LDA',
            occupations=FermiDirac(0.001))  # use small FD smearing

atoms.calc = calc
atoms.get_potential_energy()  # get ground state density

# Restart Calculation with fixed density and dense kpoint sampling
calc = calc.fixed_density(
    kpts={'density': 15.0, 'gamma': False})  # dense kpoint sampling

calc.diagonalize_full_hamiltonian(nbands=70)  # diagonalize Hamiltonian
calc.write('si_large.gpw', 'all')  # write wavefunctions
# literalinclude1
# Getting absorption spectrum
df = DielectricFunction(calc='si_large.gpw',
                        eta=0.05,
                        frequencies={'type': 'nonlinear', 'domega0': 0.02},
                        ecut=150)
df.get_dielectric_function(filename='si_abs.csv')
# literalinclude2
# Getting macroscopic constant
df = DielectricFunction(calc='si_large.gpw',
                        frequencies=[0.0],
                        hilbert=False,
                        eta=0.0001,
                        ecut=150,
                        )

epsNLF, epsLF = df.get_macroscopic_dielectric_constant()

# Make table
epsrefNLF = 14.08  # from [1] in top
epsrefLF = 12.66  # from [1] in top

with paropen('mac_eps.csv', 'w') as f:
    print(' , Without LFE, With LFE', file=f)
    print(f"{'GPAW-linear response'}, {epsNLF:.6f}, {epsLF:.6f}", file=f)
    print(f"{'[1]'}, {epsrefNLF:.6f}, {epsrefLF:.6f}", file=f)
    print(f"{'Exp.'}, {11.9:.6f}, {11.9:.6f}", file=f)

if world.rank == 0:
    Path('si_large.gpw').unlink()

from ase import Atoms
from gpaw import GPAW, FermiDirac, Mixer
from gpaw import PW

# This calculation is too heavy to run as an exercise!!

# Plane wave cutoff
pwcutoff = 400.0


# box length for isolated atom
L = 8.0


# Do the isolated calculation


isolated_silicon = Atoms(
    symbols=['Si'],
    positions=[[0.5 * L, 0.5 * L, 0.5 * L]],
    cell=[L + 0.1, L, L - 0.1],
    pbc=(1, 1, 1))

isolated_calc = GPAW(
    mode=PW(pwcutoff, force_complex_dtype=True),
    parallel={'domain': 1},
    xc='PBE',
    txt='si_isolated_rpa.init_pbe.txt',
    occupations=FermiDirac(0.01),  # fixmagmom=True),
    spinpol=True,
    hund=True,
    # convergence={'density': 1.e-6},
    mixer=Mixer(beta=0.05, nmaxold=5, weight=50.0))

isolated_silicon.calc = isolated_calc

isolated_silicon.get_potential_energy()
isolated_calc.diagonalize_full_hamiltonian()  # ouch
isolated_calc.write('si.rpa.isolated.gpw', mode='all')

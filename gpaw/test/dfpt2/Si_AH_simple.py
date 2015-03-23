from ase.lattice import bulk
import ase.units as units

from gpaw import GPAW, FermiDirac
from gpaw.dfpt2.phononcalculator import PhononCalculator
from gpaw.test import equal

k = 3
kT = 0
h = 0.2

a = 5.4

atoms = bulk('Si', 'diamond', a=a)
calc = GPAW(kpts=(k, k, k),
            setups='ah',
            symmetry={'point_group': False, 'time_reversal': False},
            occupations=FermiDirac(kT),
            h=h)

atoms.set_calculator(calc)
atoms.get_potential_energy()
calc.write('Si_AH.gpw', mode='all')

# Gamma phonon frequencies
name = 'Si_AH.gpw'

# Create phonon calculator
ph = PhononCalculator(name,
                      dispersion=False,
                      symmetry=False,
                      q_c =[0., 0., 0.]
                      )

# Run the self-consistent calculation
energies = ph.get_phonons()
energies *= 1000


reference = [-5.21660788e-07, 3.08577085e-07, 7.39654337e-07,
              4.49543403e+01, 4.49543901e+01, 4.49898951e+01]


equal(energies, reference, 1e-4)

# reference energies for Gamma
# [[ -5.21660788e-07   3.08577085e-07   7.39654337e-07   4.49543403e+01
#    4.49543901e+01   4.49898951e+01]
# reference energies for X
# [  1.91104820e+01   2.23700901e+01   3.07891199e+01   3.14599239e+01
#    3.26732813e+01   3.31591576e+01]]

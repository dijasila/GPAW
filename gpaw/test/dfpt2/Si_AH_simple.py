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
energies *= 1000  # Conversion to meV

# This was without acoustic sum rule

reference = [10.026842, 10.027247, 10.032217, 46.059036, 46.059075, 46.094857]
equal(energies, reference, 1e-4)

# Now check with acoustic sum rule

dynmat = ph.dyn.D_nn
dynmat_new = ph.dyn.acoustic_sum_rule(dynmat)
energies = ph.diagonalize_dynamicalmatrix(dynmat_new, modes=False)

energies *= 1000

print energies
reference = [0.000000, 0.000000, 0.000000, 44.954340, 44.954390, 44.989895]
equal(energies, reference, 1e-4)


# reference energies with acoustic sum rule

# reference energies for Gamma
# [0.000000, 0.000000, 0.000000, 44.954340, 44.954390, 44.989895]
# reference energies for (1/3, 0, 1/3)
# [16.859100, 16.860130, 24.181116, 36.081396, 36.108222, 39.722383]
# reference energies for X
# [19.110482, 22.370090, 30.789119, 31.459923, 32.673281, 33.159157]

# reference energies without acoustic sum rule

# reference energies for Gamma
# [10.026842, 10.027247, 10.032217, 46.059036, 46.059075, 46.094857]
# reference energies for (1/3, 0, 1/3)
# [18.838915, 18.841651, 25.601615, 37.047843, 37.074885, 40.602762]
# reference energies for X
# [20.877886, 23.899050, 31.917282, 32.563820, 33.738102, 34.208868]

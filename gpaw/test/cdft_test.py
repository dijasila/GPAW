from ase import *
from gpaw import *
import numpy as np
from ase import Atoms
from gpaw import *
from gpaw.cdft.cdft import CDFT
from gpaw import Mixer
from ase.io import *
from ase.optimize import *
from gpaw.cdft.cdft_coupling import *
import matplotlib.pyplot as plt
from ase.units import Bohr

sys = Atoms('He2', positions = ([0.,0.,0.],[0.,0.,distance]))
sys.center(2)
sys.set_initial_magnetic_moments([0.5,0.5])

calc_a = GPAW(convergence={'eigenstates':1.0e-4,'density':1.0e-3, 'energy':1e-3})


cdft_a = CDFT(calc = calc_a,
            atoms=sys,
            charge_regions = [[0]],
            charges = [1],
            charge_coefs = [2.7], 
            method = 'CG',
            txt = 'He2+_init.cdft',
            minimizer_options={'gtol':0.15, 'ftol': 1e-1, 'xtol':1e-1,
                 'max_trust_radius':5,'initial_trust_radius':0.1},
            forces = 'fd')

sys.set_calculator(cdft_a)
sys.get_potential_energy()
f = sys.get_forces()

sys.get_potential_energy()

calc = cdft_a.calc

sys = Atoms('He2', positions = ([0.,0.,0.],[0.,0.,distance]))
sys.center(2)
sys.set_initial_magnetic_moments([0.5,0.5])

calc_b = GPAW(convergence={'eigenstates':1.0e-4,'density':1.0e-3, 'energy':1e-3})

cdft_b = CDFT(calc_b,
            atoms=sys,
            charge_regions = [[1]],
            charges = [1],
            charge_coefs = [2.7], #in eV
            method = 'CG',
            txt = 'He2+_final.cdft',
            minimizer_options={'gtol':0.1, 'ftol': 1e-1, 'xtol':1e-1,
                 'max_trust_radius':5,'initial_trust_radius':0.1},
            forces = 'fd')

sys.set_calculator(cdft_b)
sys.get_potential_energy()

marcus = Marcus_parameters(cdft_a,cdft_b)
ct, H = marcus.get_coupling_term_from_lowdin()
print '#############################'
print 'COUPLING AND HAMILTONIAN', ct, H
print '#############################'
      


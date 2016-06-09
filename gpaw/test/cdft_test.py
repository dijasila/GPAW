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

for distance in [2.5]:
      sys = Atoms('He2', positions = ([0.,0.,0.],[0.,0.,distance]))
      sys.center(2)
      sys.set_initial_magnetic_moments([0.5,0.5])

      calc_a = GPAW(h = 0.2,
                  mode='fd',
                  basis='dzp',
                  charge=1,
                  xc='PBE',
                  symmetry = 'off',
                  occupations = FermiDirac(0.1, fixmagmom = True),
                  spinpol = True,
                  mixer = Mixer(beta=0.25, nmaxold=3, weight=100.0),
                  txt='He2+_init_%3.2f.txt'%distance,
                  convergence={'eigenstates':1.0e-1,'density':1.0e-1, 'energy':1e-1})


      cdft_a = CDFT(calc = calc_a,
                  atoms=sys,
                  charge_regions = [[0]],
                  charges = [1],
                  charge_coefs = [2.7], 
                  method = 'CG',
                  txt = 'He2+_init_%3.2f.cdft'%distance,
                  minimizer_options={'gtol':0.15, 'ftol': 1e-1, 'xtol':1e-1,
                       'max_trust_radius':5,'initial_trust_radius':0.1},
                  forces = 'fd')

      sys.set_calculator(cdft_a)
      sys.get_potential_energy()
      f = sys.get_forces()
      print('from python')
      print(f)

      sys.get_potential_energy()
      wA = cdft_a.get_weight()

      wA.dump('wA.p')

      calc = cdft_a.calc
      calc.write('initial.gpw', mode='all')

      sys = Atoms('He2', positions = ([0.,0.,0.],[0.,0.,distance]))
      sys.center(2)
      sys.set_initial_magnetic_moments([0.5,0.5])

      calc_b = GPAW(h=0.2,
                  mode='fd',
                  basis='dzp',
                  spinpol = True,
                  charge=1,
                  xc='PBE',
                  occupations = FermiDirac(0.1, fixmagmom = True),
                  mixer = Mixer(beta=0.25, nmaxold=3, weight=100.0),
                  txt='He2+_final.txt',
                  convergence={'eigenstates':1.0e-1,'density':1.0e-1, 'energy':1e-1})

      cdft_b = CDFT(calc_b,
                  atoms=sys,
                  charge_regions = [[1]],
                  charges = [1],
                  charge_coefs = [2.7], #in eV
                  method = 'CG',
                  txt = 'He2+_final%3.2f.cdft'%distance,
                  minimizer_options={'gtol':0.1, 'ftol': 1e-1, 'xtol':1e-1,
                       'max_trust_radius':5,'initial_trust_radius':0.1},
                  forces = 'fd')

      sys.set_calculator(cdft_b)
      sys.get_potential_energy()

      cdft_b.calc.write('final.gpw', mode='all')

      wB = cdft_b.get_weight()
      wB.dump('wB.p')

      marcus = Marcus_parameters(cdft_a,cdft_b)
      ct, H = marcus.get_coupling_term_from_lowdin()
      print '#############################'
      print 'COUPLING AND HAMILTONIAN', ct, H
      print '#############################'
      


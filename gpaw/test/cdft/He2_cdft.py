from ase import *
import numpy as np
from gpaw import *
from ase.io import *
from ase.optimize import *
import pickle
from gpaw.atom import * 
from gpaw.cdft import ConstrainedDFT

distance = 2.5
sys = Atoms('He2', positions = ([0.,0.,0.],[0.,0.,distance]))
sys.center(4)

calc = GPAW(h = 0.2,
            mode='fd',
            basis='dzp',
            charge=1,
            xc='PBE',
            spinpol = True,
            mixer = Mixer(beta=0.25, nmaxold=3, weight=100.0),
            txt='initial.txt',
            convergence={'eigenstates':1.0e-2,'density':1.0e-2, 'energy':1e-2})


cdft_a = CDFT([1], calc, sys, [[0]],alpha_max = 1, alpha_min = 1e-5,C = 1e-1,iterations = 30, 
        V_update='newton',step_size_control='factor_step',
        force_calculation = 'analytical', outfile = 'initial.out')

sys.set_calculator(cdft_a)
sys.get_potential_energy()

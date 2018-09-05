import numpy as np

from ase.lattice.hexagonal import Graphene

from gpaw import GPAW, PW, FermiDirac
from gpaw.mpi import world
from gpaw.response.df import DielectricFunction

import os

system = Graphene(symbol='C',
                  latticeconstant={'a': 2.467710, 'c': 1.0},
                  size=(1, 1, 1))
system.pbc = (1, 1, 0)
system.center(axis=2, vacuum=4.0)

kpt_refine = {"center": [1. / 3, 1. / 3, 0.],
              "size": [[5, 5, 1],
                       [5, 5, 1],
                       [5, 5, 1]],
              "reduce_symmetry": False,
              'q': [1. / (15 * 5), 0, 0]}

if not os.path.exists('graphene.gpw'):
    calc = GPAW(mode=PW(ecut=400),
                xc='PBE',
                kpts={"size": [15, 15, 1], "gamma": True},
                experimental={'kpt_refine': kpt_refine},
                occupations=FermiDirac(0.026))
    system.set_calculator(calc)
    system.get_potential_energy()
    calc.write('graphene.gpw', 'all')

df = DielectricFunction('graphene.gpw')

alpha0x_qw = []
alphax_qw = []

q_q = np.array(range(1, 2)) / (15 * 5)

for q in q_q:
    alpha0x_w, alphax_w = df.get_polarizability(q_c=[q, 0, 0])
    alpha0x_qw.append(alpha0x_w)
    alphax_qw.append(alphax_w)

frequencies = df.get_frequencies()
data = {'alpha0x_qw': np.array(alpha0x_qw),
        'alphax_qw': np.array(alphax_qw),
        'frequencies': frequencies}

filename = 'graphene_pol_data.npz'

if world.rank == 0:
    np.savez_compressed(filename, **data)

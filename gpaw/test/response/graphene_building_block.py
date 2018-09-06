import os

import numpy as np

from ase.lattice.hexagonal import Graphene

from gpaw import GPAW, PW, FermiDirac
from gpaw.response.df import DielectricFunction
from ase.units import Hartree
from gpaw.response.qeh import BuildingBlock


system = Graphene(symbol='C',
                  latticeconstant={'a': 2.467710, 'c': 1.0},
                  size=(1, 1, 1))
system.pbc = (1, 1, 0)
system.center(axis=2, vacuum=4.0)

nk = 6
nkrefine = 13
q_qc = [[[k / (nk * nkrefine), 0, 0] for k in range(1, nkrefine)],
        [[k / nk, 0, 0] for k in range(1, int(nk / 2))]]
q_qc = np.concatenate(q_qc)
kpt_refine = {"center": [1. / 3, 1. / 3, 0.],
              "size": [nkrefine, nkrefine, 1],
              "reduce_symmetry": False,
              'q': q_qc}

if not os.path.exists('graphene_bb.gpw'):
    calc = GPAW(mode=PW(ecut=400),
                xc='PBE',
                kpts={"size": [nk, nk, 1], "gamma": True},
                experimental={'kpt_refine': kpt_refine},
                occupations=FermiDirac(0.026))
    system.set_calculator(calc)
    system.get_potential_energy()
    calc.write('graphene_bb.gpw', 'all')

pbc = system.pbc

df = DielectricFunction(calc='graphene_bb.gpw',
                        eta=0.001,
                        domega0=0.05,
                        omega2=10.0,
                        # nblocks=8,
                        # ecut=150,
                        name='graphene_response_',
                        truncation='2D')

buildingblock = BuildingBlock('CC', df, nq_inf=0, isotropic_q=False,
                              q_qc=q_qc)
buildingblock.calculate_building_block()

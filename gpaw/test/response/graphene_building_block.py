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

nk = 12
nkrefine = 13
nqinf = 10
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

df = DielectricFunction(calc='graphene_bb.gpw',
                        eta=0.001,
                        domega0=0.05,
                        omega2=10.0,
                        gate_voltage=0.1,
                        # nblocks=8,
                        # ecut=150,
                        name='graphene_response_',
                        truncation='2D')

buildingblock = BuildingBlock('CC', df, nq_inf=nqinf, isotropic_q=False,
                              q_qc=q_qc)
buildingblock.calculate_building_block()


d_graphene = 3.32
n_MoSSe = 3

import matplotlib.pyplot as plt
from gpaw.response.qeh import Heterostructure

d = []

HS = Heterostructure(structure=['CC'],
                     d=d,
                     qmax=0.1,
                     wmax=1,
                     d0=d_graphene)

q, w, loss = HS.get_eels()
nw = 0
nq = 0
w = w[nw:]
q = q[nq:]
loss = loss[nq:, nw:]
qv, wv = np.meshgrid(q, w, sparse=False, indexing='ij')

plt.contourf(qv, wv, loss, 900, cmap='seismic', vmax=100)  # , vmax=40
plt.colorbar()
plt.title('Loss function', fontsize=20)
plt.xlabel(r'q', fontsize=15)
plt.ylabel(r'w', fontsize=15)
plt.savefig('graphene_3MoSSe_graphene.svg')
plt.show()

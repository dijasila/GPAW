import os
import numpy as np
from ase import Atoms
from ase.lattice.hexagonal import Hexagonal
from gpaw import GPAW, FermiDirac
from gpaw.test import findpeak, equal
from gpaw.response.g0w0 import G0W0
from gpaw.mpi import rank

if 1:
    calc = GPAW(mode='pw',
                xc='PBE',
                nbands=20,
                setups={'Mo': '6'},
                parallel={'band': 1, 'domain': 1},
                occupations=FermiDirac(0.001),
                convergence={'bands': -5},
                kpts={'size': (6,6,1), 'gamma': True})
    
    a = 3.1604
    c = 10.0
    
    cell = Hexagonal(symbol='Mo', latticeconstant={'a':a, 'c':c}).get_cell()
    layer = Atoms(symbols='MoS2', cell=cell, pbc=(1,1,1),
                  scaled_positions=[(0, 0, 0),
                                    (2/3., 1/3., 0.3),
                                    (2/3., 1/3., -0.3)])
    
    pos = layer.get_positions()
    pos[1][2] = pos[0][2] + 3.172/2
    pos[2][2] = pos[0][2] - 3.172/2
    layer.set_positions(pos)
    layer.set_calculator(calc)
    layer.get_potential_energy()
    calc.write('MoS2.gpw', mode='all')

gw = G0W0('MoS2.gpw',
          'gw-test',
          nbands=15,
          ecut=10,
          domega0=0.1,
          eta=0.2,
          truncation='2D',
          kpts=[((1/3., 1/3., 0))],
          bands=(8, 10),
          )
e_qp = gw.calculate()['qp'][0, 0]

ev = 2.701
ec = 6.779

equal(e_qp[0], ev, 0.01) 
equal(e_qp[1], ec, 0.01)

from __future__ import division
import numpy as np
from ase import Atoms
from ase.lattice.hexagonal import Hexagonal
from gpaw import GPAW, FermiDirac
from gpaw.test import findpeak, equal
from gpaw.response.bse import BSE

if 1:
    calc = GPAW(mode='pw',
                xc='PBE',
                nbands=20,
                setups={'Mo': '6'},
                parallel={'band': 1, 'domain': 1},
                occupations=FermiDirac(0.001),
                convergence={'bands': -5},
                kpts=(9, 9, 1))
    
    a = 3.1604
    c = 10.0
    
    cell = Hexagonal(symbol='Mo', latticeconstant={'a': a, 'c': c}).get_cell()
    layer = Atoms(symbols='MoS2', cell=cell, pbc=(1, 1, 1),
                  scaled_positions=[(0, 0, 0),
                                    (2 / 3, 1 / 3, 0.3),
                                    (2 / 3, 1 / 3, -0.3)])
    
    pos = layer.get_positions()
    pos[1][2] = pos[0][2] + 3.172 / 2
    pos[2][2] = pos[0][2] - 3.172 / 2
    layer.set_positions(pos)
    layer.set_calculator(calc)
    layer.get_potential_energy()
    calc.write('MoS2.gpw', mode='all')

bse = BSE('MoS2.gpw',
          spinors=True,
          ecut=10,
          valence_bands=[8],
          conduction_bands=[9],
          eshift=0.8,
          nbands=15,
          write_h=False,
          write_v=False,
          wfile=None,
          mode='BSE',
          truncation='2D')

w_w, alpha_w = bse.get_polarizability(filename=None,
                                      pbc=[True, True, False],
                                      write_eig=None,
                                      eta=0.02,
                                      w_w=np.linspace(0., 5., 5001))

w0_ = 1.01
I0_ = 12.56
w1_ = 1.17
I1_ = 13.29
w0, I0 = findpeak(w_w[:1100], alpha_w.imag[:1100])
w1, I1 = findpeak(w_w[1100:1300], alpha_w.imag[1100:1300])
w1 += 1.1
equal(w0, w0_, 0.01)
equal(I0, I0_, 0.1)
equal(w1, w1_, 0.01)
equal(I1, I1_, 0.1)

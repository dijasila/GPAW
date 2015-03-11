import os
import time
import pickle as pckl

import numpy as np

from ase import Atoms
from ase.io import read
from ase.lattice.compounds import Rocksalt

from gpaw import GPAW, FermiDirac, Mixer
from gpaw.eigensolvers import Davidson
from gpaw.wavefunctions.pw import PW
from gpaw.occupations import FermiDirac
from gpaw.response.df import DielectricFunction

# This test assures that some things that
# should be equal, are.

a = 2.5
c = 3.22

GR = Atoms(symbols='C2',
           positions=[(0.5*a, 0.2 + -np.sqrt(3) / 6 * a, 0.0),
                      (0.5*a, 0.2 + np.sqrt(3) / 6 * a, 0.0)],
           cell=[(0.5*a,-0.5*3**0.5*a,0),
                 (0.5*a,+0.5*3**0.5*a,0),
                 (0.0,0.0,c*2.0)])
GR.set_pbc((True,True,True))
atoms = GR
GSsettings = [{'symmetry': 'off', 'kpts': {'density': 2.5, 'gamma': False}},
              {'symmetry': {}, 'kpts': {'density': 2.5, 'gamma': False}},
              {'symmetry': 'off', 'kpts': {'density': 2.5, 'gamma': True}},
              {'symmetry': {}, 'kpts': {'density': 2.5, 'gamma': True}}]

DFsettings = [{'disable_point_group': True,
               'disable_time_reversal': True,
               'use_more_memory': 0},
              {'disable_point_group': False,
               'disable_time_reversal': True,
               'use_more_memory': 0},
              {'disable_point_group': True,
               'disable_time_reversal': False,
               'use_more_memory': 0},
              {'disable_point_group': False,
               'disable_time_reversal': False,
               'use_more_memory': 0},
              {'disable_point_group': False,
               'disable_time_reversal': False,
               'use_more_memory': 1}]

for GSkwargs in GSsettings:
    calc = GPAW(h=0.18,
                mode=PW(600),
                occupations=FermiDirac(0.2),
                **GSkwargs)
 
    atoms.set_calculator(calc)
    atoms.get_potential_energy()
    calc.write('gr.gpw', 'all')

    dfs = []
    for kwargs in DFsettings:
        DF = DielectricFunction(calc='gr.gpw',
                                domega0=0.01,
                                eta=0.2,
                                ecut=40.0,
                                **kwargs)
        df1, df2 = DF.get_dielectric_function()
        dfs.append(df2)

    while len(dfs):
        df = dfs.pop()
        for df2 in dfs:
            assert np.allclose(df, df2)

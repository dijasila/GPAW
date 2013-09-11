import os
import sys
import traceback

import numpy as np
from ase.data import chemical_symbols
from ase.utils import opencew
from ase import Atoms

from gpaw.test.big.setups.structures import fcc, rocksalt
from gpaw import GPAW, PW, setup_paths
from gpaw.eigensolvers.rmm_diis_new import RMM_DIIS_new


sc = (chemical_symbols[3:5] + chemical_symbols[11:13] +
      chemical_symbols[19:31] + chemical_symbols[37:49] +
      chemical_symbols[55:81])

setup_paths[:] = ['../r3', '../r']

def run(symbol):
    if not os.path.isdir(symbol):
        os.mkdir(symbol)
    os.chdir(symbol)
    setups = ['std']
    if symbol in sc:
        setups.append('sc')
    for h in [0.16, 0.18, 0.20]:
        for setup in setups:
            name = 'egg-%s-%.2f' % (setup, h)
            fd = opencew(name + '.dat')
            if fd is None:
                continue
            a = 24 * h
            atoms = Atoms(symbol, cell=[a, a, a], pbc=True)
            atoms.calc = GPAW(txt=name + '.txt',
                              xc='PBE',
                              setups={'std': 'paw'}.get(setup, setup),
                              eigensolver='cg',
                              width=0.1)
            for x in np.linspace(0, h / 2, 15):
                atoms.positions[:] = x
                try:
                    e = atoms.get_potential_energy()
                    f = atoms.get_forces()[0, 0] * 3**0.5
                except:
                    traceback.print_exc(file=open(name + '.error', 'a'))
                    break
                fd.write('%f %f\n' % (e, f))


if len(sys.argv) == 2:
    run(sys.argv[1])
else:
    print ('for symbol in %s; do gpaw-qsub -q verylong -l nodes=1:ppn=1:xeon8 egg.py $symbol; done' %
            ' '.join(chemical_symbols[1:87]))

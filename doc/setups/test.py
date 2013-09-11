import os
import sys
import traceback

import numpy as np
from ase import Atoms
from ase.data import chemical_symbols
from ase.utils import opencew

from gpaw.test.big.setups.structures import fcc, rocksalt
from gpaw import GPAW, PW, setup_paths
from gpaw.eigensolvers.rmm_diis_new import RMM_DIIS_new


sc = (chemical_symbols[3:5] + chemical_symbols[11:13] +
      chemical_symbols[19:31] + chemical_symbols[37:49] +
      chemical_symbols[55:81])

fcc = fcc()
rocksalt = rocksalt()


def energy(atoms, fd, name):
    try:
        e = atoms.get_potential_energy()
    except:
        traceback.print_exc(file=open(name + '.error', 'a'))
        e = np.nan
    fd.write(repr(e) + '\n')
    fd.flush()


def run(symbol):
    if not os.path.isdir(symbol):
        os.mkdir(symbol)
    os.chdir(symbol)
    setups = ['std']
    if symbol in sc:
        setups.append('sc')
    for mode in ['lcao', 'pw']:
        if mode == 'lcao':
            m = 'lcao'
            kwargs = {'basis': 'dzp', 'h': 0.15, 'realspace': False}
        else:
            m = PW(1000)
            kwargs = {'eigensolver': 'cg'}
        for type in ['r', 'nr']:
            if type == 'nr' and mode == 'lcao':
                continue
            setup_paths[:] = ['../' + type]
            for setup in setups:
                if mode == 'lcao' and setup == 'sc':
                    kwargs['basis'] = {symbol: 'sc.dzp', 'O': 'dzp'}
                else:
                    kwargs['basis'] = 'dzp'
                for x in ['fcc', 'rocksalt']:
                    name = 'vol-%s-%s-%s-%s' % (mode, type, setup, x)
                    fd = opencew(name + '.dat')
                    if fd is None:
                        continue
                    s = 'paw'
                    if x == 'fcc':
                        atoms = fcc[symbol]
                        if setup == 'sc':
                            s = 'sc'
                    else:
                        atoms = rocksalt[symbol]
                        if setup == 'sc':
                            s = {symbol: 'sc'}
                    atoms.calc = GPAW(txt=name + '.txt',
                                      kpts=3.0,
                                      xc='PBE',
                                      setups=s,
                                      mode=m,
                                      **kwargs)
                    cell0 = atoms.cell
                    for f in np.linspace(0.80, 1.1, 7):
                        atoms.set_cell(cell0 * f, scale_atoms=True)
                        energy(atoms, fd, name)

    atoms = fcc[symbol]
    atom = Atoms(symbol,cell=[5,5,5],pbc=1)
    setup_paths[:] = ['../nr']
    for mode in ['lcao', 'fd', 'pw']:
        if mode != 'pw':
            continue
        for setup in setups:
            name = 'conv-%s-%s' % (mode, setup)
            fd = opencew(name + '.dat')
            if fd is None:
                continue
            for e in np.linspace(300, 600, 7):
                atoms.calc = GPAW(txt=name + '.txt',
                                  kpts=3.0,
                                  xc='PBE',
                                  setups={'std': 'paw'}.get(setup, setup),
                                  mode=PW(e),
                                  eigensolver='cg')
                energy(atoms, fd, name)
        for setup in setups:
            name = 'conv-atom-%s-%s' % (mode, setup)
            fd = opencew(name + '.dat')
            if fd is None:
                continue
            for e in np.linspace(300, 600, 7):
                atom.calc = GPAW(txt=name + '.txt',
                                  width=0.1,
                                  xc='PBE',
                                  setups={'std': 'paw'}.get(setup, setup),
                                  mode=PW(e),
                                  eigensolver='cg')
                energy(atom, fd, name)

    for es, type in  [('rmm-diis', 'nr'), ('rmm-diis4', 'nr'),
                      ('rmm-diis', 's09')]:
        setup_paths[:] = ['../' + type]
        for setup in setups:
            if type == 's09' and setup == 'sc':
                continue
            name = 'es-%s-%s-%s' % (es, type, setup)
            fd = opencew(name + '.dat')
            if fd is None:
                continue
            if es == 'rmm-diis4':
                eigs = RMM_DIIS_new(trial_step=0.1)
            else:
                eigs = es
            atoms.calc = GPAW(txt=name + '.txt',
                              kpts=3.0,
                              xc='PBE',
                              setups={'std': 'paw'}.get(setup, setup),
                              mode=PW(500),
                              eigensolver=eigs)
            energy(atoms, fd, name)


if len(sys.argv) == 2:
    run(sys.argv[1])
else:
    print ('for symbol in %s; do gpaw-qsub -q verylong -l nodes=1:ppn=1:xeon8 test.py $symbol; done' %
            ' '.join(chemical_symbols[1:87]))

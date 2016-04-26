import time

import ase.db
import ase.optimize
from ase.collection import g2

from gpaw import GPAW


optimizers = ['BFGS', 'BFGSLineSearch', 'FIRE', 'GoodOldQuasiNewton',
              'LBFGS', 'LBFGSLineSearch']

con = ase.db.connect('g2_dzp.db')
for name, atoms in zip(g2.names, g2):
    atoms.center(vacuum=3.5)
    for optimizer in optimizers:
        id = con.reserve(name=name, optimizer=optimizer)
        if id is None:
            continue
        mol = atoms.copy()
        mol.calc = GPAW(mode='lcao',
                        basis='dzp',
                        txt='{0}-{1}.txt'.format(name, optimizer))
        Optimizer = getattr(ase.optimize, optimizer)
        opt = Optimizer(mol)
        t = time.time()
        opt.run(fmax=0.05, steps=25)
        con.write(mol, name=name, optimizer=optimizer,
                  steps=opt.nsteps, time=time.time() - t)
        del con[id]

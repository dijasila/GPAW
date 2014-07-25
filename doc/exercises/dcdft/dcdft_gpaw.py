import os
import time

import numpy as np

import ase.db
from ase.utils import opencew
from ase.calculators.calculator import kpts2mp
from ase.io.trajectory import PickleTrajectory
from ase.test.tasks.dcdft import DeltaCodesDFTCollection as Collection
from gpaw import GPAW, PW

c = ase.db.connect('exercise_dcdft.db')

ecut = 340

kptdensity = 3.5
width = 0.10

pwcell = False

linspace = (0.98, 1.02, 5)  # eos numpy's linspace
linspacestr = ''.join([str(t) + 'x' for t in linspace])[:-1]

code = 'gpaw' + '-' + str(ecut) + '_e' + linspacestr
code = code + '_k' + str(kptdensity) + '_w' + str(width) + '_c' + str(pwcell)

collection = Collection()

#for name in collection.names:
for name in ['K', 'Ca', 'Ti']:
    # save all steps in one traj file in addition to the database
    # we should only used the database c.reserve, but here
    # traj file is used as another lock ...
    fd = opencew(name + '_' + code + '.traj')
    if fd is None:
        continue
    traj = PickleTrajectory(name + '_' + code + '.traj', 'w')
    atoms = collection[name]
    cell = atoms.get_cell()
    kpts = tuple(kpts2mp(atoms, kptdensity, even=True))
    kwargs = {}
    if pwcell:
        kwargs.update({'mode': PW(ecut, cell=cell)})
    else:
        kwargs.update({'mode': PW(ecut)})
    if name in ['Li', 'Na']:
        # https://listserv.fysik.dtu.dk/pipermail/gpaw-developers/2012-May/002870.html
        kwargs.update({'h': 0.10})
    # loop over EOS linspace
    for n, x in enumerate(np.linspace(linspace[0], linspace[1], linspace[2])):
        id = c.reserve(name=name, ecut=ecut, linspacestr=linspacestr,
                       kptdensity=kptdensity, width=width, pwcell=pwcell,
                       x=x)
        if id is None:
            continue
        # perform EOS step
        atoms.set_cell(cell * x, scale_atoms=True)
        # set calculator
        atoms.calc = GPAW(
            txt=name + '_' + code + '_' + str(n) + '.txt',
            xc='PBE',
            kpts=kpts,
            width=width,
            parallel={'band': 1},
            idiotproof=False)
        atoms.calc.set(**kwargs)  # remaining calc keywords
        t = time.time()
        atoms.get_potential_energy()
        c.write(atoms,
                name=name, ecut=ecut, linspacestr=linspacestr,
                kptdensity=kptdensity, width=width, pwcell=pwcell,
                x=x,
                time=time.time()-t,
                iter=atoms.calc.get_number_of_iterations())
        traj.write(atoms)
        del c[id]

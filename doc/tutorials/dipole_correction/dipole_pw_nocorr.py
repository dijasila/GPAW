#!/usr/bin/env python
from gpaw import GPAW,PW
from ase import Atoms
from ase.units import Hartree
from ase.parallel import world

atoms=Atoms('HCl',[[0,0,0],[0,0,2]])
atoms.set_cell((5,5,10), scale_atoms=False)
atoms.center()
atoms.set_pbc((1,1,1))

calc = GPAW(h=0.3,
            xc='LDA',
            mode=PW())

atoms.set_calculator(calc)
E1=atoms.get_potential_energy()

v=calc.hamiltonian.pd3.ifft(calc.hamiltonian.vHt_q)*Hartree
vz=v.mean(0).mean(0)

import pylab as pl
pl.figure()
pl.plot(vz)
pl.show()

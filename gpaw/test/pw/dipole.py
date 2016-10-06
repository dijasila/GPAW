#!/usr/bin/env python
"""
Similar to the test in fd mode, two electrostatic potentials are calculated, but with a slightly changed system:

Test the dipole correction code in pw mode by comparing this system:

z1  HCl  z2

(where z1 and z2 denote points where the potential is probed)

Expected potential:

      -----
     /
    /
----

to this system:

HCl  z1   HCl  z2  


Expected potential:

       -------
      /       \
     /         \
-----           ------

The height of the two potentials are tested to be the same.

Enable if-statement in the bottom for nice plots
"""

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
            mode=PW(dipole_corr_dir=2))

atoms.set_calculator(calc)
E1=atoms.get_potential_energy()

atoms2=atoms.copy()+atoms.copy()
atoms2.positions[2][2]=-atoms.positions[0][2]
atoms2.positions[3][2]=-atoms.positions[1][2]
atoms2.center(vacuum=4,axis=2)

calc2 = GPAW(h=0.3,
            xc='LDA',
            mode=PW())
atoms2.calc=calc2

E2=atoms2.get_potential_energy()

if world.rank == 0:
	v=calc.hamiltonian.pd3.ifft(calc.hamiltonian.vHt_q)*Hartree
	vz=v.mean(0).mean(0)
	v2=calc2.hamiltonian.pd3.ifft(calc2.hamiltonian.vHt_q)*Hartree
	vz2=v2.mean(0).mean(0)

	#Correction to the change in reference energy
	vzmin,vzmax=vz.min(),vz.max()
	vz2min,vz2max=vz2.min(),vz2.max()


	refcorrmin= vzmin-vz2min 
	refcorrmax= vzmax-vz2max 
	refcorr=(refcorrmin+refcorrmax)/2

	if 0:
        	import pylab as pl
        	pl.figure()
        	pl.plot(range(len(vz2)/2,len(vz2)/2+len(vz)),vz-refcorr)
        	pl.plot(vz2)
        	pl.show()

	assert abs(vz[-6]-refcorr - vz2[-6]) < 0.01
	assert abs(vz[6]-refcorr - vz2[len(vz2)/2]) < 0.01

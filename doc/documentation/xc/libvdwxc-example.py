from ase.build import molecule
from gpaw import GPAW

atoms = molecule('H2O')
atoms.center(vacuum=3.0)
# There are these functionals: vdW-DF, vdW-DF2, vdW-DF-cx, optPBE-vdW,
# optB88-vdW, C09-vdW, BEEF-vdW, and mBEEF-vdW.
# There are three modes: serial, mpi, and pfft. Default is auto.
calc = GPAW(xc={'name': 'BEEF-vdW', 'backend': 'libvdwxc', 'mode': 'mpi'})
atoms.calc = calc
atoms.get_potential_energy()

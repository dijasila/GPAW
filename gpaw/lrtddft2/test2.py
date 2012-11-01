import os
from gpaw import GPAW
from ase.all import *
from ase.units import Hartree
from lrigpaw.lrtddft2 import *
from gpaw.mpi import world, size, rank
import pickle
import numpy as np


# Ground state calculation
if not os.path.exists('CH4.gpw'):
    atoms = molecule('CH4')
    atoms.center(5.0)
    calc = GPAW(h=0.2, eigensolver='cg', nbands=-30)
    atoms.set_calculator(calc)
    e = atoms.get_potential_energy()
    calc.write('CH4.gpw', mode='all')

eh_size = 4
domain_size = size // eh_size
assert eh_size * domain_size == size

dd_comm, eh_comm = lr_communicators(world, domain_size, eh_size)
calc = GPAW('CH4.gpw', communicator=dd_comm, txt=None,
            parallel={'sl_lrtddft': (2,2,2)} )

# Big calculation
istart = 1
iend = 10
pstart = 0
pend = 20
ediff = 100.
nanal = 6
lr = LrTDDFTindexed( 'CH4_lri',
                     calc=calc,
                     xc = 'PBE',
                     min_occ=istart,
                     max_occ=iend,
                     min_unocc=pstart,
                     max_unocc=pend,
                     max_energy_diff=ediff,
                     eh_communicator=eh_comm
                   )
lr.calculate_excitations()
print >> lr.txt, lr.get_excitation_energy(0)

# Decrease
istart = 3
iend = 10
pstart = 0
pend = 15
lr = LrTDDFTindexed( 'CH4_lri',
                     calc=calc,
                     xc = 'PBE',
                     min_occ=istart,
                     max_occ=iend,
                     min_unocc=pstart,
                     max_unocc=pend,
                     max_energy_diff=ediff,
                     eh_communicator=eh_comm,
                     # recalculate='eigen'
                   )
lr.calculate_excitations()
print >> lr.txt, lr.get_excitation_energy(0)


# Big calculation
istart = 1
iend = 10
jstart = 0
jend = 20
ediff = 100.
nanal = 6
dd_comm, eh_comm = lr_communicators(world, domain_size//2, eh_size*2)
calc = GPAW('CH4.gpw', communicator=dd_comm, txt=None,
            parallel={'sl_lrtddft': (2,2,2)} )
lr = LrTDDFTindexed( 'CH4_lri',
                     calc=calc,
                     xc = 'PBE',
                     min_occ=istart,
                     max_occ=iend,
                     min_unocc=jstart,
                     max_unocc=jend,
                     max_energy_diff=ediff,
                     eh_communicator=eh_comm
                   )
lr.calculate_excitations()
print >> lr.txt, lr.get_excitation_energy(0)


# Decrease
istart = 3
iend = 10
pstart = 0
pend = 15
lr = LrTDDFTindexed( 'CH4_lri',
                     calc=calc,
                     xc = 'PBE',
                     min_occ=istart,
                     max_occ=iend,
                     min_unocc=pstart,
                     max_unocc=pend,
                     max_energy_diff=ediff,
                     eh_communicator=eh_comm,
                     # recalculate='eigen'
                   )
lr.calculate_excitations()
print >> lr.txt, lr.get_excitation_energy(0)

lr.get_spectrum(filename='spectrum.dat')

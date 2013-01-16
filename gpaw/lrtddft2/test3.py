import os
from gpaw import *
from ase import *
from ase.io import *
from gpaw.lrtddft2 import *
from gpaw.mpi import world, size, rank
import numpy as np

# Ground state calculation
if not os.path.exists('test.gpw'):
    #atoms = Atoms('Be', [(0.,0.,0.)])
    atoms = Atoms('Na2', [(0.,0.,0.),(0.,0.,3.1)]) #molecule('CH4')
    #d = 3.1
    #D = 5.
    #atoms = Atoms('Na4', [(0.,0.,0*d),(0.,0.,1*d),(0.,0.,1*d+D),(0.,0.,2*d+D)])
    atoms.center(5.0)
    calc = GPAW(h=0.25, nbands=7, convergence={ 'bands': 5 }, width=1.0 )
    atoms.set_calculator(calc)
    e = atoms.get_potential_energy()
    calc.write('test.gpw', mode='all')


domain_size = 4
eh_size = size//4
assert eh_size * domain_size == size
dd_comm, eh_comm = lr_communicators(world, domain_size, eh_size)
calc = GPAW('test.gpw', communicator=dd_comm, txt=None)

# Big calculation
istart = 0
iend = 1
pstart = 0
pend = 4
lr = LrTDDFTindexed( 'test_lri',
                     calc=calc,
                     xc = 'PBE',
                     min_occ=istart,
                     max_occ=iend,
                     min_unocc=pstart,
                     max_unocc=pend,
                     recalculate=None,
                     eh_communicator=eh_comm,
                     txt='-')
lr.calculate_excitations()
lr.get_transitions('trans.dat')
lr.get_spectrum('spectrum.dat', 0, 10.)

for i in range(500):
    omega = 0.01*i/27.211
    eta = .1/27.211
    (C_re,C_im) = lr.calculate_response_wavefunction(omega,eta,[1.,1.,1.])
    dm_re = 0.
    dm_im = 0.
    for (k,kss_ip) in enumerate(lr.kss_list):
        dm_re += 2 * omega * C_re[k] * kss_ip.dip_mom_r * kss_ip.pop_diff/27.211
        dm_im += 2 * omega * C_im[k] * kss_ip.dip_mom_r * kss_ip.pop_diff/27.211
    print "%12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  " % (omega*27.211, dm_re[0], dm_re[1], dm_re[2], dm_im[0], dm_im[1], dm_im[2])

#drhot_g = lr.get_transition_density(C_im)

# dm = 2 w int d3r drhot_g mu
#    = 2 w C_im dm n
# n_el ~= int dw dm =  2 w C_im eta dm n
# 2 * (2.2/27.211) * 5.30519545e+02 * (0.1/27.211) *  3.177897120359 * 2.0 ~= 2

# TODO SPECTRUM TEST

#write('test.cube', atoms, data=drhot_g)

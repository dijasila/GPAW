import numpy as np
import pickle
from ase.units import Hartree
from gpaw import GPAW
from gpaw.xc.hybridk import HybridXC
from gpaw.xc.tools import vxc
from gpaw.mpi import serial_comm, world, rank

# set the energy cutoff for calculating the exact exchange (in Hartree):
ecut = 100./Hartree

# read in the groundstate calculation:
file='Si_groundstate.gpw'
calc = GPAW(
            file,
            communicator=serial_comm, # necessary, when run in parallel
            parallel={'domain':1},    # necessary, when run in parallel
            txt=None
           )

# define which matrix elements should be calculated:

# here, we want the spectrum for the two highest valence and the two lowest conduction bands
# since Si has 8 valence electrons in the unit cell, the corresponding indices are 2,3,4,5
# (taking into account double occupation and the fact that the numbering in GPAW starts at 0)

# and all k-points of the irreducible Brillouin zone
# given by their indices in the full Brillouin zone
# (these can be found by using e.g. the calculator's k-point descriptor)

gwbands_n = np.array([2,3,4,5]) # band indices for which matrix elements should be calculated
gwkpt_k = calc.wfs.kd.ibz2bz_k  # k-point indices for which matrix elements should be calculated

# calculate DFT exchange-correlation contributions:
v_xc = vxc(calc)

# calculate exact exchange contributions:
alpha = 5.0
exx = HybridXC('EXX', alpha=alpha, ecut=ecut, bands=gwbands_n)
calc.get_xc_difference(exx)

# now extract the desired matrix elements:
gwnband = len(gwbands_n)
gwnkpt = len(gwkpt_k)
nspins = calc.wfs.nspins

e_skn = np.zeros((nspins, gwnkpt, gwnband), dtype=float)
vxc_skn = np.zeros((nspins, gwnkpt, gwnband), dtype=float)
exx_skn = np.zeros((nspins, gwnkpt, gwnband), dtype=float)

for s in range(nspins):
    for i, k in enumerate(gwkpt_k):
        ik = calc.wfs.kd.bz2ibz_k[k]
        for j, n in enumerate(gwbands_n):
            e_skn[s][i][j] = calc.get_eigenvalues(kpt=ik, spin=s)[n] / Hartree
            vxc_skn[s][i][j] = v_xc[s][ik][n] / Hartree
            exx_skn[s][i][j] = exx.exx_skn[s][ik][n]

# and dump them in a pickle file:
data = {
        'e_skn': e_skn,        # in Hartree
        'vxc_skn': vxc_skn,    # in Hartree
        'exx_skn': exx_skn,    # in Hartree
        'gwkpt_k': gwkpt_k,
        'gwbands_n': gwbands_n
       }
if rank == 0:
    pickle.dump(data, open('EXX_ecut025.pckl', 'w'), -1)

# we can also determine the (non-selfconsistent) Hartree-Fock bandstructure with these results:
    print "--------------------------------------"
    print "ecut = ", ecut*Hartree, " eV"
    print "non-selfconsistent HF eigenvalues are:"
    print (e_skn - vxc_skn + exx_skn)*Hartree

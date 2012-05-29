import numpy as np
import pickle
from time import time, ctime
from datetime import timedelta
from ase.structure import bulk
from ase.units import Hartree
from gpaw import GPAW, FermiDirac
from gpaw.response.gw import GW
from gpaw.xc.hybridk import HybridXC
from gpaw.xc.tools import vxc
from gpaw.mpi import serial_comm, world, rank

starttime = time()

a = 5.431
atoms = bulk('Si', 'diamond', a=a)

kpts = (2,2,2)

calc = GPAW(
            h=0.24,
            kpts=kpts,
            xc='LDA',
            txt='Si_gs.txt',
            nbands=10,
            convergence={'bands':8},
            occupations=FermiDirac(0.001)
           )

atoms.set_calculator(calc)
atoms.get_potential_energy()
calc.write('Si_gs.gpw','all')

nbands=8
bands=np.array([3,4])
ecut=25./Hartree

gwkpt_k = calc.wfs.kd.ibz2bz_k
gwnkpt = calc.wfs.kd.nibzkpts
gwnband = len(bands)
nspins = calc.wfs.nspins

file='Si_gs.gpw'
calc = GPAW(
            file,
            communicator=serial_comm,
            parallel={'domain':1},
            txt=None
           )

v_xc = vxc(calc)

alpha = 5.0
exx = HybridXC('EXX', alpha=alpha, ecut=ecut, bands=bands)
calc.get_xc_difference(exx)

e_skn = np.zeros((nspins, gwnkpt, gwnband), dtype=float)
vxc_skn = np.zeros((nspins, gwnkpt, gwnband), dtype=float)
exx_skn = np.zeros((nspins, gwnkpt, gwnband), dtype=float)

for s in range(nspins):
    for i, k in enumerate(range(gwnkpt)):
        for j, n in enumerate(bands):
            e_skn[s][i][j] = calc.get_eigenvalues(kpt=k, spin=s)[n] / Hartree
            vxc_skn[s][i][j] = v_xc[s][k][n] / Hartree
            exx_skn[s][i][j] = exx.exx_skn[s][k][n]

data = {
        'e_skn': e_skn,        # in Hartree
        'vxc_skn': vxc_skn,    # in Hartree
        'exx_skn': exx_skn,    # in Hartree
        'gwkpt_k': gwkpt_k,
        'gwbands_n': bands
       }
if rank == 0:
    pickle.dump(data, open('EXX.pckl', 'w'), -1)

exxfile='EXX.pckl'

gw = GW(
        file=file,
        nbands=8,
        bands=np.array([3,4]),
        w=np.array([10., 30., 0.05]),
        ecut=25.,
        eta=0.1,
        hilbert_trans=False,
        exxfile=exxfile
       )

gw.get_QP_spectrum()

QP_False = gw.QP_skn * Hartree

gw = GW(
        file=file,
        nbands=8,
        bands=np.array([3,4]),
        w=np.array([10., 30., 0.05]),
        ecut=25.,
        eta=0.1,
        hilbert_trans=True,
        exxfile=exxfile
       )

gw.get_QP_spectrum()

QP_True = gw.QP_skn * Hartree

if not (np.abs(QP_False - QP_True) < 0.01).all():
    raise AssertionError("method 1 not equal to method 2")

totaltime = round(time() - starttime)
print "GW test finished in %s " %(timedelta(seconds=totaltime))

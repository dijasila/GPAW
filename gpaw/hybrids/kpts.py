from collections import namedtuple

import numpy as np

from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.projections import Projections
from gpaw.utilities.partition import AtomPartition
from gpaw.wavefunctions.arrays import PlaneWaveExpansionWaveFunctions
from gpaw.wavefunctions.pw import PWDescriptor


KPoint = namedtuple(
    'KPoint',
    ['psit',   # plane-wave expansion of wfs
     'proj',   # projections
     'f_n',    # occupations numbers between 0 and 1
     'k_c',    # k-vector in units of reciprocal cell
     'weight'  # weight of k-point
     ])

RSKPoint = namedtuple(
    'RealSpaceKPoint',
    ['u_nR',  # wfs on a real-space grid
     'proj',  # same as above
     'f_n',   # ...
     'k_c',
     'weight',
     # 'index'  # IBZ k-point index
     ])


def to_real_space(psit, na=0, nb=None):
    pd = psit.pd
    comm = pd.comm
    S = comm.size
    q = psit.kpt
    nbands = len(psit.array)
    nb = nb or nbands
    u_nR = pd.gd.empty(nbands, pd.dtype, global_array=True)
    for n1 in range(0, nbands, S):
        n2 = min(n1 + S, nbands)
        u_G = pd.alltoall1(psit.array[n1:n2], q)
        if u_G is not None:
            n = n1 + comm.rank
            u_nR[n] = pd.ifft(u_G, local=True, safe=False, q=q)
        for n in range(n1, n2):
            comm.broadcast(u_nR[n], n - n1)

    return u_nR[na:nb]


def get_kpt(wfs, k, spin, nocc=-1):
    k_c = wfs.kd.ibzk_kc[k]
    weight = wfs.kd.weight_k[k]

    if wfs.world.size == wfs.gd.comm.size:
        # Easy:
        kpt = wfs.mykpts[wfs.kd.nibzkpts * spin + k]
        psit = kpt.psit
        proj = kpt.projections
        f_n = kpt.f_n
    else:
        # Need to redistribute things:
        gd = wfs.gd.new_descriptor(comm=wfs.world)
        kd = KPointDescriptor([k_c])
        pd = PWDescriptor(wfs.ecut, gd, wfs.dtype, kd, wfs.fftwflags)
        psit = PlaneWaveExpansionWaveFunctions(wfs.bd.nbands,
                                               pd,
                                               dtype=wfs.dtype,
                                               spin=spin)
        for n in range(wfs.bd.nbands):
            psit_G = wfs.get_wave_function_array(n, k, spin, realspace=False)
            psit._distribute(psit_G, psit.array[n])

        P_nI = wfs.collect_projections(k, spin)
        natoms = len(wfs.setups)
        rank_a = np.zeros(natoms, int)
        atom_partition = AtomPartition(wfs.world, rank_a)
        nproj_a = [setup.ni for setup in wfs.setups]
        proj = Projections(wfs.bd.nbands,
                           nproj_a,
                           atom_partition,
                           spin=spin,
                           dtype=wfs.dtype,
                           data=P_nI)

        rank_a = np.linspace(0, wfs.world.size, len(wfs.spos_ac),
                             endpoint=False).astype(int)
        atom_partition = AtomPartition(wfs.world, rank_a)
        proj = proj.redist(atom_partition)

        f_n = wfs.get_occupations(k, spin)

    if nocc != -1:
        psit = psit.view(0, nocc)
        proj = proj.view(0, nocc)

    f_n = f_n[:nocc] / (weight * (2 // kd.nspins))  # scale to [0, 1]

    kpt = KPoint(psit, proj, f_n, k_c, weight)
    return kpt

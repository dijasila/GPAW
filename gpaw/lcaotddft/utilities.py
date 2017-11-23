import numpy as np

from gpaw.blacs import BlacsGrid
from gpaw.blacs import Redistributor

def ranks(wfs):
    import time
    time.sleep(wfs.world.rank * 0.1)
    txt = ''
    comm_i = [wfs.world, wfs.gd.comm, wfs.kd.comm,
              wfs.bd.comm, wfs.ksl.block_comm]
    for comm in comm_i:
        txt += '%2d/%2d ' % (comm.rank, comm.size)
    return txt

def collect_uMM(wfs, a_uMM, s, k):
    return collect_uwMM(wfs, a_uMM, s, k, w=None)

def collect_uwMM(wfs, a_uwMM, s, k, w):
    # This function is based on
    # gpaw/wavefunctions/base.py: WaveFunctions.collect_auxiliary()

    dtype = a_uwMM[0][0].dtype

    ksl = wfs.ksl
    NM = ksl.nao
    kpt_rank, u = wfs.kd.get_rank_and_index(s, k)

    ksl_comm = ksl.block_comm

    if wfs.kd.comm.rank == kpt_rank:
        if w is None:
            a_MM = a_uwMM[u]
        else:
            a_MM = a_uwMM[u][w]

        # Collect within blacs grid
        if ksl.using_blacs:
            a_mm = a_MM
            grid = BlacsGrid(ksl_comm, 1, 1)
            MM_descriptor = grid.new_descriptor(NM, NM, NM, NM)
            mm2MM = Redistributor(ksl_comm,
                                  ksl.mmdescriptor,
                                  MM_descriptor)

            a_MM = MM_descriptor.empty(dtype=dtype)
            mm2MM.redistribute(a_mm, a_MM)

        # Domain master send a_MM to the global master
        if ksl_comm.rank == 0:
            if kpt_rank == 0:
                assert wfs.world.rank == 0
                return a_MM
            else:
                wfs.kd.comm.send(a_MM, 0, 2017)
                return None
    elif ksl_comm.rank == 0 and kpt_rank != 0:
        assert wfs.world.rank == 0
        a_MM = np.empty((NM, NM), dtype=dtype)
        wfs.kd.comm.receive(a_MM, kpt_rank, 2017)
        return a_MM

def distribute_MM(wfs, a_MM):
    ksl = wfs.ksl
    if not ksl.using_blacs:
        return a_MM

    dtype = a_MM.dtype
    ksl_comm = ksl.block_comm
    NM = ksl.nao
    grid = BlacsGrid(ksl_comm, 1, 1)
    MM_descriptor = grid.new_descriptor(NM, NM, NM, NM)
    MM2mm = Redistributor(ksl_comm,
                          MM_descriptor,
                          ksl.mmdescriptor)
    if ksl_comm.rank != 0:
        a_MM = MM_descriptor.empty(dtype=dtype)

    a_mm = ksl.mmdescriptor.empty(dtype=dtype)
    MM2mm.redistribute(a_MM, a_mm)
    return a_mm

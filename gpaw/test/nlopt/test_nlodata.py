import pytest
from gpaw import GPAW
from gpaw.mpi import broadcast, world
from gpaw.nlopt.basic import NLOData
from gpaw.nlopt.matrixel import make_nlodata
import numpy as np


@pytest.mark.skipif(world.size > 1, reason='Serial only')
def test_write_load_serial(in_tmp_dir):
    w_sk = np.random.rand(1, 5)
    f_skn = np.random.rand(1, 5, 30)
    E_skn = np.random.rand(1, 5, 30)
    p_skvnn = np.random.rand(1, 5, 3, 30, 30)

    nlo = NLOData(w_sk, f_skn, E_skn, p_skvnn)
    nlo.write('nlodata.npz')
    del nlo

    newdata = NLOData.load('nlodata.npz')
    assert np.all(newdata.w_sk == w_sk)
    assert np.all(newdata.f_skn == f_skn)
    assert np.all(newdata.E_skn == E_skn)
    assert np.all(newdata.p_skvnn == p_skvnn)


def test_write_load_parallel(in_tmp_dir):
    if world.rank == 0:   
        w_sk = np.random.rand(1, 5)
        f_skn = np.random.rand(1, 5, 30)
        E_skn = np.random.rand(1, 5, 30)
        p_skvnn = np.random.rand(1, 5, 3, 30, 30)
    else:
        w_sk = None
        f_skn = None
        E_skn = None
        p_skvnn = None
    w_sk = broadcast(w_sk, root=0)
    f_skn = broadcast(f_skn, root=0)
    E_skn = broadcast(E_skn, root=0)
    p_skvnn = broadcast(p_skvnn, root=0)

    nlo = NLOData(w_sk, f_skn, E_skn, p_skvnn)
    nlo.write('nlodata.npz')
    del nlo

    newdata = NLOData.load('nlodata.npz')
    k_info = newdata.distribute()

    for k, data in k_info.items():
        assert w_sk[:, k] == data[0]
        assert np.all(f_skn[:, k] == data[1])
        assert np.all(E_skn[:, k] == data[2])
        assert np.all(p_skvnn[:, k] == data[3])

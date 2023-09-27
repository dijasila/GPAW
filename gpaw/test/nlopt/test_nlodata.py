import pytest
import numpy as np

from gpaw.mpi import world
from gpaw.nlopt.basic import NLOData


def generate_testing_data(ns, nk, nb, seed=42):
    rng = np.random.default_rng(seed=42)

    w_sk = rng.random((ns, nk))
    f_skn = rng.random((ns, nk, nb))
    E_skn = rng.random((ns, nk, nb))
    p_skvnn = rng.random((ns, nk, 3, nb, nb)) \
        + 1j * rng.random((ns, nk, 3, nb, nb))

    return w_sk, f_skn, E_skn, p_skvnn


@pytest.mark.skipif(world.size > 1, reason='Serial only')
def test_write_load_serial(in_tmp_dir):
    w_sk, f_skn, E_skn, p_skvnn = generate_testing_data(1, 8, 20)

    nlo = NLOData(w_sk, f_skn, E_skn, p_skvnn, world)
    nlo.write('nlodata.npz')

    newdata = NLOData.load('nlodata.npz', world)
    assert np.all(newdata.w_sk == w_sk)
    assert np.all(newdata.f_skn == f_skn)
    assert np.all(newdata.E_skn == E_skn)
    assert np.all(newdata.p_skvnn == p_skvnn)


def test_serial_file_parallel_data(in_tmp_dir):
    # Random data only on rank = 0
    if world.rank == 0:
        w_sk, f_skn, E_skn, p_skvnn = generate_testing_data(1, 8, 20)
    else:
        w_sk = None
        f_skn = None
        E_skn = None
        p_skvnn = None

    nlo = NLOData(w_sk, f_skn, E_skn, p_skvnn, world)
    nlo.write('nlodata.npz')
    k_info_original = nlo.distribute()

    newdata = NLOData.load('nlodata.npz', world)
    k_info = newdata.distribute()
    for (k, data), (_, original_data) in zip(k_info.items(),
                                             k_info_original.items()):
        assert original_data[0] == data[0]
        assert np.all(original_data[1] == data[1])
        assert np.all(original_data[2] == data[2])
        assert np.all(original_data[3] == data[3])


def test_write_load_parallel(in_tmp_dir):
    # Same random data array on each core
    w_sk, f_skn, E_skn, p_skvnn = generate_testing_data(1, 8, 20)

    nlo = NLOData(w_sk, f_skn, E_skn, p_skvnn, world)
    nlo.write('nlodata.npz')

    newdata = NLOData.load('nlodata.npz', world)
    k_info = newdata.distribute()

    # Compare the distributed data with original data
    for k, data in k_info.items():
        assert w_sk[:, k] == data[0]
        assert np.all(f_skn[:, k] == data[1])
        assert np.all(E_skn[:, k] == data[2])
        assert np.all(p_skvnn[:, k] == data[3])

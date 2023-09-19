import numpy as np
import pytest
from gpaw.core.matrix import Matrix
from gpaw.gpu import cupy as cp, as_np, as_xp
from gpaw.mpi import world
from gpaw.new.c import GPU_AWARE_MPI
from gpaw.gpu.mpi import CuPyMPI


@pytest.mark.gpu
@pytest.mark.serial
def test_zyrk():
    a = np.array([[1, 1 + 2j, 2], [1, 0.5j, -1 - 0.5j]])
    m = Matrix(2, 3, data=a)
    b = m.multiply(m, opb='C', beta=0.0, symmetric=True)
    b.tril2full()
    a = cp.asarray(a)
    m = Matrix(2, 3, data=a)
    b2 = m.multiply(m, opb='C', beta=0.0, symmetric=True)
    b2.tril2full()
    c = b2.to_cpu()
    assert (c.data == b.data).all()


@pytest.mark.gpu
@pytest.mark.serial
def test_eigh():
    H1 = Matrix(2, 2, data=np.array([[2, 42.1 + 42.1j], [0.1 - 0.1j, 3]]))
    S1 = Matrix(2, 2, data=np.array([[1, 42.1 + 42.2j], [0.1 - 0.2j, 0.9]]))
    H2 = Matrix(2, 2, data=cp.asarray(H1.data))
    S2 = Matrix(2, 2, data=cp.asarray(S1.data))

    E1 = H1.eigh(S1)

    S0 = S1.copy()
    S0.tril2full()

    E2 = H2.eigh(S2)
    assert as_np(E2) == pytest.approx(E1)

    C1 = H1.data
    C2 = H2.to_cpu().data

    # Check that eigenvectors are parallel:
    X = C1.conj() @ S0.data @ C2.T
    assert abs(X) == pytest.approx(np.eye(2))


def op(a, o):
    return a


@pytest.mark.gpu
@pytest.mark.parametrize('shape1, shape2, op1, op2, beta, sym',
                         [((5, 19), (5, 19), 'N', 'C', 0.0, True),
                          ((5, 19), (5, 19), 'N', 'C', 0.0, False),
                          ((5, 5), (5, 5), 'N', 'C', 0.0, False),
                          ((5, 5), (5, 5), 'C', 'N', 0.0, False),
                          ((5, 5), (5, 5), 'N', 'C', 1.0, False),
                          ((5, 5), (5, 5), 'C', 'N', 1.0, False),
                          ((5, 5), (5, 5), 'N', 'N', 0.0, False),
                          ])
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('xp', [np, cp])
def test_mul(shape1, shape2, op1, op2, alpha, beta, dtype, xp, sym, rng):
    alpha = 1.234
    comm = world if GPU_AWARE_MPI else CuPyMPI(world)
    shape3 = (shape1[0] if op1 == 'N' else shape1[1],
              shape2[1] if op2 == 'N' else shape1[0])
    m1 = Matrix(*shape1, dtype=dtype, dist=(comm, -1, 1), xp=xp)
    m2 = Matrix(*shape2, dtype=dtype, dist=(comm, -1, 1), xp=xp)
    m3 = Matrix(*shape3, dtype=dtype, dist=(comm, -1, 1), xp=xp)
    for m in [m1, m2, m3]:
        data = m.data.view(float)
        data[:] = as_xp(rng.random(data.shape), xp)
    a1, a2, a3 = (as_np(m.gather().data) for m in [m1, m2, m3])
    m1.multiply(m2, alpha=alpha, opa=op1, opb=op2, beta=beta,
                out=m3, symmetric=sym)
    m4 = m3.gather()
    if m4 is not None:
        if sym:
            a3 = beta * a3 + 0.5 * alpha * (op(a1, op1) @ op(a2, op2) +
                                            op(a2, op1) @ op(a1, op2))
        else:
            a3 = beta * a3 + alpha * op(a1, op1) @ op(a2, op2)
        error = abs(a3 - as_np(m4.data)).max()
        print(error)

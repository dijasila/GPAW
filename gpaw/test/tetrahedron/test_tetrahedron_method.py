import numpy as np
import pytest

import gpaw.tetrahedron as T


@pytest.mark.serial
def test_bja():
    """Test formulas from Blöchl et al. paper."""
    eigs = np.linspace(1.0, 4.0, 4).reshape((4, 1))
    for n in range(1, 4):
        bjan = getattr(T, f'bja{n}')
        bjanb = getattr(T, f'bja{n}b')
        ef = n + 0.5
        E = eigs - ef
        f1, dfde1 = bjan(*E)
        x = 0.0001
        dfde2 = (bjan(*(E - x))[0] - bjan(*(E + x))[0]) / (2 * x)
        f2 = bjanb(*E).sum()
        print(n, f1, f2, dfde1, dfde2)
        assert f1 == pytest.approx(f2)
        assert dfde1 == pytest.approx(dfde2)


@pytest.mark.serial
def test_tetra():
    """Test 2-d BZ exapmle."""
    t = T.TetrahedronMethod(np.diag([1.0, 1.0, 0.1]),
                            [2, 2, 1],
                            False,
                            [0, 1, 2, 1])

    eig_in = np.array([[0.0, 2.0], [0.0, 1.0], [0.0, 1.0]])
    weight_i = [0.25, 0.5, 0.25]
    f_in, (ef,), _ = t.calculate(1.5, eig_in, weight_i, [1.5])
    assert ef == pytest.approx(2 - (2 / 3)**0.5)
    assert f_in.sum(1).dot(weight_i) == pytest.approx(1.5)

import pytest
import numpy as np
import numpy.random as ra
from gpaw.setup import create_setup
from gpaw.xc import XC
from gpaw.test import equal
import cupy as cp

@pytest.mark.parametrize('xp', [np, cp])
@pytest.mark.parametrize('xc', ['LDA', 'PBE'])
def test_xc_xcatom(xp, xc):
    x = 0.000001
    ra.seed(8)
    print(xc)
    xc = XC(xc, xp=xp)
    s = create_setup('N', xc, xp=xp)
    ni = s.ni
    nii = ni * (ni + 1) // 2
    D_p = xp.asarray(0.1 * ra.random(nii) + 0.2)
    H_p = xp.zeros(nii)
    print(D_p)
    D_ii = unpack(D_p)
    H_ii = unpack(H_ii)
    xc.calculate_paw_corrections(
        [s], [unpack(D_p.reshape(1, -1))], [unpack(H_p.reshape(1, -1)])
    
    #xc.calculate_paw_correction(
    #    s, D_p.reshape(1, -1), H_p.reshape(1, -1))
    dD_p = xp.asarray(x * ra.random(nii))
    dE = xp.dot(H_p, dD_p) / x
    D_p += dD_p
    Ep = xc.calculate_paw_correction(s, D_p.reshape(1, -1))
    D_p -= 2 * dD_p
    Em = xc.calculate_paw_correction(s, D_p.reshape(1, -1))
    print(dE, dE - 0.5 * (Ep - Em) / x)
    assert xp.abs(dE - 0.5 * (Ep - Em) / x) < 1e-6

    Ems = xc.calculate_paw_correction(s, xp.array([0.5 * D_p, 0.5 * D_p]))
    print(Em - Ems)
    assert xp.abs(Em - Ems) < 1.0e-12

    D_sp = xp.asarray(0.1 * ra.random((2, nii)) + 0.2)
    H_sp = xp.zeros((2, nii))

    xc.calculate_paw_correction(s, D_sp, H_sp)
    dD_sp = xp.asarray(x * ra.random((2, nii)))
    dE = xp.dot(H_sp.ravel(), dD_sp.ravel()) / x
    D_sp += dD_sp
    Ep = xc.calculate_paw_correction(s, D_sp)
    D_sp -= 2 * dD_sp
    Em = xc.calculate_paw_correction(s, D_sp)
    print(dE, dE - 0.5 * (Ep - Em) / x)
    assert xp.abs(dE - 0.5 * (Ep - Em) / x) < 1e-6

from functools import partial

import pytest
from ase.units import Ha

from gpaw.occupations import (fermi_dirac, marzari_vanderbilt,
                              methfessel_paxton, FermiDirac, ZeroWidth)


funcs = []
for w in [0.1, 0.5]:
    funcs.append(partial(fermi_dirac, fermi_level=0.0, width=w))
    funcs.append(partial(marzari_vanderbilt, fermi_level=0.0, width=w))
    for n in range(4):
        funcs.append(partial(methfessel_paxton,
                             fermi_level=0.0, order=n, width=w))


@pytest.mark.parametrize('func', funcs)
@pytest.mark.ci
def test_occupations(func):
    for e in [-0.3 / Ha, 0, 0.1 / Ha, 1.2 / Ha]:
        n0, d0, S0 = func(e)
        x = 0.000001
        fp, dp, Sp = func(e + x)
        fm, dm, Sm = func(e - x)
        d = -(fp - fm) / (2 * x)
        dS = Sm - Sp
        dn = fp - fm
        print(d - d0, dS - e * dn)
        assert abs(d - d0) < 3e-5
        assert abs(dS - e * dn) < 1e-13


@pytest.mark.parametrize(
    'e_kn,w_k,ne',
    [([[0.0, 1.0]], [1.0], 2),
     ([[0.0, 1.0], [0.0, 2.0]], [0.5, 0.5], 1.5),
     ([[0.0, 1.0, 2.0], [0.0, 2.0, 2.0]], [0.5, 0.5], 2)])
@pytest.mark.ci
def test_occupation_obj(e_kn, w_k, ne):
    for occ in [FermiDirac(0.1), ZeroWidth()]:
        f_kn, fl, s = occ.calculate(ne, e_kn, w_k)
        print(f_kn, fl, s)
        assert f_kn.sum(1).dot(w_k) == pytest.approx(ne, abs=1e-14)

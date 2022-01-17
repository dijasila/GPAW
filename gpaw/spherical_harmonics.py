r"""
Real-valued spherical harmonics


=== === === =======
 L   l   m
=== === === =======
 0   0   0   1
 1   1  -1   y
 2   1   0   z
 3   1   1   x
 4   2  -2   xy
 5   2  -1   yz
 6   2   0   3z2-r2
 7   2   1   zx
 8   2   2   x2-y2
=== === === =======

For a more complete list, see c/bmgs/sharmonic.py


Gaunt coefficients::

                  __
     ^      ^    \   L      ^
  Y (r)  Y (r) =  ) G    Y (r)
   L      L      /__ L L  L
    1      2      L   1 2

"""

import numpy as np

from math import pi
from collections import defaultdict
from _gpaw import spherical_harmonics as Yl

__all__ = ['Y', 'YL', 'nablarlYL', 'Yl']

names = [['1'],
         ['y', 'z', 'x'],
         ['xy', 'yz', '3z2-r2', 'zx', 'x2-y2'],
         ['3x2y-y3', 'xyz', '4yz2-y3-x2y', '2z3-3x2z-3y2z', '4xz2-x3-xy2',
          'x2z-y2z', 'x3-3xy2']]


def Y(L, x, y, z):
    result = 0.0
    for c, n in YL[L]:
        result += c * x**n[0] * y**n[1] * z**n[2]
    return result


def Yarr(L_M, R_Av):
    """
    Calculate spherical harmonics L_M at positions R_Av, where
    A is some array like index.
    """
    Y_MA = np.zeros((len(L_M), *R_Av.shape[:-1]))
    for M, L in enumerate(L_M):
        for c, n in YL[L]:  # could be vectorized further
            Y_MA[M] += c * np.prod(np.power(R_Av, n), axis=-1)
    return Y_MA


def nablarlYL(L, R):
    """Calculate the gradient of a real solid spherical harmonic."""
    x, y, z = R
    dYdx = dYdy = dYdz = 0.0
    terms = YL[L]
    # The 'abs' avoids error in case powx == 0
    for N, (powx, powy, powz) in terms:
        dYdx += N * powx * x**abs(powx - 1) * y**powy * z**powz
        dYdy += N * powy * x**powx * y**abs(powy - 1) * z**powz
        dYdz += N * powz * x**powx * y**powy * z**abs(powz - 1)
    return dYdx, dYdy, dYdz


g = [1.0]
for l in range(16):
    g.append(g[-1] * (l + 0.5))


def gam(n0, n1, n2):
    h0 = n0 // 2
    h1 = n1 // 2
    h2 = n2 // 2
    if 2 * h0 != n0 or 2 * h1 != n1 or 2 * h2 != n2:
        return 0.0
    return 2.0 * pi * g[h0] * g[h1] * g[h2] / g[1 + h0 + h1 + h2]


def Y0(l, m):
    """Sympy version of spherical harmonics."""
    from fractions import Fraction
    from sympy import assoc_legendre, sqrt, simplify, factorial as fac, I, pi
    from sympy.abc import x, y, z
    c = sqrt((2 * l + 1) * fac(l - m) / fac(l + m) / 4 / pi)
    if m > 0:
        return simplify(c * (x + I * y)**m / (1 - z**2)**Fraction(m, 2) *
                        assoc_legendre(l, m, z))
    return simplify(c * (x - I * y)**(-m) / (1 - z**2)**Fraction(-m, 2) *
                    assoc_legendre(l, m, z))


def S(l, m):
    """Sympy version of real valued spherical harmonics."""
    from sympy import I, Number, sqrt
    if m > 0:
        return (Y0(l, m) + (-1)**m * Y0(l, -m)) / sqrt(2) * (-1)**m
    if m < 0:
        return -(Y0(l, m) - Number(-1)**m * Y0(l, -m)) / (sqrt(2) * I)
    return Y0(l, m)


def poly_coeffs(l, m):
    """Sympy coefficients for polynomiunm in x, y and z."""
    from sympy import Poly
    from sympy.abc import x, y, z
    Y = S(l, m)
    coeffs = {}
    for nx, coef in enumerate(reversed(Poly(Y, x).all_coeffs())):
        for ny, coef in enumerate(reversed(Poly(coef, y).all_coeffs())):
            for nz, coef in enumerate(reversed(Poly(coef, z).all_coeffs())):
                if coef:
                    coeffs[(nx, ny, nz)] = coef
    return coeffs


def fix_exponents(coeffs, l):
    """Make sure exponents add up to l."""
    from sympy import Number
    new = defaultdict(lambda: Number(0))
    for (nx, ny, nz), coef in coeffs.items():
        if nx + ny + nz == l:
            new[(nx, ny, nz)] += coef
        else:
            new[(nx + 2, ny, nz)] += coef
            new[(nx, ny + 2, nz)] += coef
            new[(nx, ny, nz + 2)] += coef

    new = {nxyz: coef for nxyz, coef in new.items() if coef}

    if not all(sum(nxyz) == l for nxyz in new):
        new = fix_exponents(new, l)

    return new


def print_YL_table_code():
    """Generate YL table using sympy.

    This will generate slightly more accurate numbers, but we will not update
    right now because then we would also have to update
    c/bmgs/spherical_harminics.c.
    """
    print('# Computer generated table - do not touch!')
    print('YL = [')
    print('    # s, l=0:')
    print(f'    [({(4 * pi)**-0.5}, (0, 0, 0))],')
    for l in range(1, 2):
        s = 'spdfghijklmnopq'[l]
        print(f'    # {s}, l={l}:')
        for m in range(-l, l + 1):
            e = poly_coeffs(l, m)
            e = fix_exponents(e, l)
            if l**2 + m + l < len(YL):
                assert len(e) == len(YL[l**2 + m + l])
                for c0, n in YL[l**2 + m + l]:
                    c = e[n].evalf()
                    assert abs(c0 - c) < 1e-10
            terms = []
            for n, en in e.items():
                c = float(en)
                terms.append(f'({c!r}, {n})')
            print('    [' + ',\n     '.join(terms) + '],')
    print(']')


def write_c_code(l: int) -> None:
    print(f'          else if (l == {l})')
    print('            {')
    for m in range(2 * l + 1):
        terms = []
        for c, n in YL[l**2 + m]:
            terms.append(f'{c!r} * ' + '*'.join('x' * n[0] +
                                                'y' * n[1] +
                                                'z' * n[2]))
        print(f'              Y_m[{m}] = ' + ' + '.join(terms) + ';')
    print('            }')



from gpaw.more_harm import YL as _YL
YL = _YL
 

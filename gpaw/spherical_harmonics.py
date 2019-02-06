# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

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


from math import pi, sqrt  # noqa

from gpaw import debug
from _gpaw import spherical_harmonics as Yl

__all__ = ['Y', 'YL', 'nablarlYL', 'Yl', 'gam']

# Computer generated tables - do not touch!
YL = [# s:
      [(1, (0, 0, 0))],
      # p:
      [(1, (0, 1, 0))],
      [(1, (0, 0, 1))],
      [(1, (1, 0, 0))],
      # d:
      [(1, (1, 1, 0))],
      [(1, (0, 1, 1))],
      [(2, (0, 0, 2)), (-1, (0, 2, 0)), (-1, (2, 0, 0))],
      [(1, (1, 0, 1))],
      [(1, (2, 0, 0)), (-1, (0, 2, 0))],
      # f:
      [(-1, (0, 3, 0)), (3, (2, 1, 0))],
      [(1, (1, 1, 1))],
      [(-1, (0, 3, 0)), (4, (0, 1, 2)), (-1, (2, 1, 0))],
      [(2, (0, 0, 3)), (-3, (2, 0, 1)), (-3, (0, 2, 1))],
      [(4, (1, 0, 2)), (-1, (3, 0, 0)), (-1, (1, 2, 0))],
      [(1, (2, 0, 1)), (-1, (0, 2, 1))],
      [(1, (3, 0, 0)), (-3, (1, 2, 0))],
      # g:
      [(1, (3, 1, 0)), (-1, (1, 3, 0))],
      [(-1, (0, 3, 1)), (3, (2, 1, 1))],
      [(-1, (3, 1, 0)), (-1, (1, 3, 0)), (6, (1, 1, 2))],
      [(-3, (2, 1, 1)), (4, (0, 1, 3)), (-3, (0, 3, 1))],
      [(6, (2, 2, 0)), (-24, (2, 0, 2)), (3, (0, 4, 0)),
       (-24, (0, 2, 2)), (3, (4, 0, 0)), (8, (0, 0, 4))],
      [(4, (1, 0, 3)), (-3, (3, 0, 1)), (-3, (1, 2, 1))],
      [(6, (2, 0, 2)), (1, (0, 4, 0)), (-1, (4, 0, 0)), (-6, (0, 2, 2))],
      [(1, (3, 0, 1)), (-3, (1, 2, 1))],
      [(-6, (2, 2, 0)), (1, (0, 4, 0)), (1, (4, 0, 0))],
      # h:
      [(-10, (2, 3, 0)), (5, (4, 1, 0)), (1, (0, 5, 0))],
      [(1, (3, 1, 1)), (-1, (1, 3, 1))],
      [(-8, (0, 3, 2)), (1, (0, 5, 0)), (-3, (4, 1, 0)), (-2, (2, 3, 0)),
       (24, (2, 1, 2))],
      [(-1, (3, 1, 1)), (-1, (1, 3, 1)), (2, (1, 1, 3))],
      [(-12, (0, 3, 2)), (2, (2, 3, 0)), (-12, (2, 1, 2)), (8, (0, 1, 4)),
       (1, (4, 1, 0)), (1, (0, 5, 0))],
      [(30, (2, 2, 1)), (-40, (0, 2, 3)), (15, (0, 4, 1)), (-40, (2, 0, 3)),
       (15, (4, 0, 1)), (8, (0, 0, 5))],
      [(-12, (3, 0, 2)), (8, (1, 0, 4)), (1, (5, 0, 0)), (2, (3, 2, 0)),
       (-12, (1, 2, 2)), (1, (1, 4, 0))],
      [(-1, (4, 0, 1)), (1, (0, 4, 1)), (2, (2, 0, 3)), (-2, (0, 2, 3))],
      [(8, (3, 0, 2)), (-1, (5, 0, 0)), (2, (3, 2, 0)), (-24, (1, 2, 2)),
       (3, (1, 4, 0))],
      [(1, (4, 0, 1)), (-6, (2, 2, 1)), (1, (0, 4, 1))],
      [(-10, (3, 2, 0)), (1, (5, 0, 0)), (5, (1, 4, 0))],
      # i:
      [(3, (5, 1, 0)), (-10, (3, 3, 0)), (3, (1, 5, 0))],
      [(5, (4, 1, 1)), (-10, (2, 3, 1)), (1, (0, 5, 1))],
      [(10, (3, 1, 2)), (-1, (5, 1, 0)), (1, (1, 5, 0)), (-10, (1, 3, 2))],
      [(-8, (0, 3, 3)), (-6, (2, 3, 1)), (3, (0, 5, 1)), (24, (2, 1, 3)),
       (-9, (4, 1, 1))],
      [(-16, (3, 1, 2)), (16, (1, 1, 4)), (2, (3, 3, 0)), (1, (5, 1, 0)),
       (-16, (1, 3, 2)), (1, (1, 5, 0))],
      [(5, (0, 5, 1)), (-20, (0, 3, 3)), (10, (2, 3, 1)), (-20, (2, 1, 3)),
       (5, (4, 1, 1)), (8, (0, 1, 5))],
      [(90, (4, 0, 2)), (-120, (0, 2, 4)), (-15, (2, 4, 0)), (16, (0, 0, 6)),
       (-15, (4, 2, 0)), (90, (0, 4, 2)), (-5, (0, 6, 0)), (-120, (2, 0, 4)),
       (-5, (6, 0, 0)), (180, (2, 2, 2))],
      [(-20, (3, 0, 3)), (8, (1, 0, 5)), (5, (5, 0, 1)), (10, (3, 2, 1)),
       (-20, (1, 2, 3)), (5, (1, 4, 1))],
      [(16, (2, 0, 4)), (-16, (0, 2, 4)), (-1, (2, 4, 0)), (-16, (4, 0, 2)),
       (1, (4, 2, 0)), (-1, (0, 6, 0)), (1, (6, 0, 0)), (16, (0, 4, 2))],
      [(8, (3, 0, 3)), (-3, (5, 0, 1)), (6, (3, 2, 1)), (-24, (1, 2, 3)),
       (9, (1, 4, 1))],
      [(5, (4, 2, 0)), (10, (0, 4, 2)), (-60, (2, 2, 2)), (-1, (0, 6, 0)),
       (-1, (6, 0, 0)), (10, (4, 0, 2)), (5, (2, 4, 0))],
      [(1, (5, 0, 1)), (-10, (3, 2, 1)), (5, (1, 4, 1))],
      [(-15, (4, 2, 0)), (-1, (0, 6, 0)), (1, (6, 0, 0)), (15, (2, 4, 0))],
      # j:
      [(21, (2, 5, 0)), (7, (6, 1, 0)), (-35, (4, 3, 0)), (-1, (0, 7, 0))],
      [(-10, (3, 3, 1)), (3, (1, 5, 1)), (3, (5, 1, 1))],
      [(-5, (6, 1, 0)), (5, (4, 3, 0)), (-120, (2, 3, 2)), (9, (2, 5, 0)),
       (12, (0, 5, 2)), (60, (4, 1, 2)), (-1, (0, 7, 0))],
      [(10, (3, 1, 3)), (3, (1, 5, 1)), (-10, (1, 3, 3)), (-3, (5, 1, 1))],
      [(-120, (2, 3, 2)), (-3, (0, 7, 0)), (3, (2, 5, 0)), (15, (4, 3, 0)),
       (-80, (0, 3, 4)), (240, (2, 1, 4)), (9, (6, 1, 0)), (60, (0, 5, 2)),
       (-180, (4, 1, 2))],
      [(30, (3, 3, 1)), (15, (1, 5, 1)), (-80, (3, 1, 3)), (48, (1, 1, 5)),
       (-80, (1, 3, 3)), (15, (5, 1, 1))],
      [(-5, (0, 7, 0)), (120, (0, 5, 2)), (-15, (2, 5, 0)), (-15, (4, 3, 0)),
       (-240, (0, 3, 4)), (-240, (2, 1, 4)), (64, (0, 1, 6)), (-5, (6, 1, 0)),
       (240, (2, 3, 2)), (120, (4, 1, 2))],
      [(-168, (2, 0, 5)), (16, (0, 0, 7)), (-168, (0, 2, 5)), (420, (2, 2, 3)),
       (-35, (6, 0, 1)), (-105, (4, 2, 1)), (210, (4, 0, 3)), (-35, (0, 6, 1)),
       (-105, (2, 4, 1)), (210, (0, 4, 3))],
      [(120, (1, 4, 2)), (-5, (7, 0, 0)), (240, (3, 2, 2)), (-15, (5, 2, 0)),
       (-5, (1, 6, 0)), (-240, (3, 0, 4)), (-15, (3, 4, 0)), (64, (1, 0, 6)),
       (-240, (1, 2, 4)), (120, (5, 0, 2))],
      [(48, (2, 0, 5)), (-48, (0, 2, 5)), (15, (6, 0, 1)), (15, (4, 2, 1)),
       (-80, (4, 0, 3)), (-15, (0, 6, 1)), (-15, (2, 4, 1)), (80, (0, 4, 3))],
      [(-9, (1, 6, 0)), (3, (7, 0, 0)), (120, (3, 2, 2)), (-3, (5, 2, 0)),
       (80, (3, 0, 4)), (-15, (3, 4, 0)), (180, (1, 4, 2)), (-240, (1, 2, 4)),
       (-60, (5, 0, 2))],
      [(10, (4, 0, 3)), (-3, (0, 6, 1)), (15, (2, 4, 1)), (-60, (2, 2, 3)),
       (10, (0, 4, 3)), (-3, (6, 0, 1)), (15, (4, 2, 1))],
      [(-5, (1, 6, 0)), (-120, (3, 2, 2)), (-1, (7, 0, 0)), (60, (1, 4, 2)),
       (5, (3, 4, 0)), (12, (5, 0, 2)), (9, (5, 2, 0))],
      [(1, (6, 0, 1)), (-1, (0, 6, 1)), (-15, (4, 2, 1)), (15, (2, 4, 1))],
      [(1, (7, 0, 0)), (-21, (5, 2, 0)), (35, (3, 4, 0)), (-7, (1, 6, 0))]]

norms = ['sqrt(1./4/pi)', 'sqrt(3./4/pi)', 'sqrt(3./4/pi)', 'sqrt(3./4/pi)', 'sqrt(15./4/pi)', 'sqrt(15./4/pi)', 'sqrt(5./16/pi)', 'sqrt(15./4/pi)', 'sqrt(15./16/pi)', 'sqrt(35./32/pi)', 'sqrt(105./4/pi)', 'sqrt(21./32/pi)', 'sqrt(7./16/pi)', 'sqrt(21./32/pi)', 'sqrt(105./16/pi)', 'sqrt(35./32/pi)', 'sqrt(315./16/pi)', 'sqrt(315./32/pi)', 'sqrt(45./16/pi)', 'sqrt(45./32/pi)', 'sqrt(9./256/pi)', 'sqrt(45./32/pi)', 'sqrt(45./64/pi)', 'sqrt(315./32/pi)', 'sqrt(315./256/pi)', 'sqrt(693./512/pi)', 'sqrt(3465./16/pi)', 'sqrt(385./512/pi)', 'sqrt(1155./16/pi)', 'sqrt(165./256/pi)', 'sqrt(11./256/pi)', 'sqrt(165./256/pi)', 'sqrt(1155./64/pi)', 'sqrt(385./512/pi)', 'sqrt(3465./256/pi)', 'sqrt(693./512/pi)', 'sqrt(3003./512/pi)', 'sqrt(9009./512/pi)', 'sqrt(819./64/pi)', 'sqrt(1365./512/pi)', 'sqrt(1365./512/pi)', 'sqrt(273./256/pi)', 'sqrt(13./1024/pi)', 'sqrt(273./256/pi)', 'sqrt(1365./2048/pi)', 'sqrt(1365./512/pi)', 'sqrt(819./1024/pi)', 'sqrt(9009./512/pi)', 'sqrt(3003./2048/pi)', 'sqrt(6435./4096/pi)', 'sqrt(45045./512/pi)', 'sqrt(3465./4096/pi)', 'sqrt(3465./64/pi)', 'sqrt(315./4096/pi)', 'sqrt(315./512/pi)', 'sqrt(105./4096/pi)', 'sqrt(15./1024/pi)', 'sqrt(105./4096/pi)', 'sqrt(315./2048/pi)', 'sqrt(315./4096/pi)', 'sqrt(3465./1024/pi)', 'sqrt(3465./4096/pi)', 'sqrt(45045./2048/pi)', 'sqrt(6435./4096/pi)']
# End of computer generated code

Lmax = len(norms)

# Normalize
for L in range(Lmax):
    YL[L] = [(eval(norms[L]) * c, n) for c, n in YL[L]]

# Only used for debug, and Gaunt coeff. generation
g = [1.0]
for l in range(9):
    g.append(g[-1] * (l + 0.5))


def gam(n0, n1, n2):
    h0 = n0 // 2
    h1 = n1 // 2
    h2 = n2 // 2
    if 2 * h0 != n0 or 2 * h1 != n1 or 2 * h2 != n2:
        return 0.0
    return 2.0 * pi * g[h0] * g[h1] * g[h2] / g[1 + h0 + h1 + h2]


def yLL(L1, L2):
    s = 0.0
    for c1, n1 in YL[L1]:
        for c2, n2 in YL[L2]:
            s += c1 * c2 * gam(n1[0] + n2[0], n1[1] + n2[1], n1[2] + n2[2])
    return s


if debug:
    for L1 in range(Lmax):
        for L2 in range(Lmax):
            r = 0.0
            if L1 == L2:
                r = 1.0
            assert abs(yLL(L1, L2) - r) < 1e-14


def Y(L, x, y, z):
    result = 0.0
    for c, n in YL[L]:
        result += c * x**n[0] * y**n[1] * z**n[2]
    return result


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

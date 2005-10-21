# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""
Y_L:

+----+-----------------+
|  L | l | m |         | 
+----+-----------------+ 
|  0 | 0 | 0 | 1       |  
+----+-----------------+
|  1 | 1 | 0 | x       |
|  2 | 1 | 1 | y       |
|  3 | 1 | 2 | z       |
+----+-----------------+
|  4 | 2 | 0 | xy      |
|  5 | 2 | 1 | yz      |
|  6 | 2 | 2 | zx      |
|  7 | 2 | 3 | x2-y2   |
|  8 | 2 | 4 | 3z2-r2  |
+----+-----------------+

Y_L1 * Y_L2 = sum_L G[L1, L2, L] * Y_L

"""

from math import pi

from gridpaw import debug


YL = [# s:
      [(1, (0, 0, 0))],
      # p:
      [(1, (1, 0, 0))],
      [(1, (0, 1, 0))],
      [(1, (0, 0, 1))],
      # d:
      [(1, (1, 1, 0))],
      [(1, (0, 1, 1))],
      [(1, (1, 0, 1))],
      [(1, (2, 0, 0)), (-1, (0, 2, 0))],
##      [(-1, (0, 0, 0)), (3, (0, 0, 2))],
      [(-1, (2, 0, 0)), (-1, (0, 2, 0)), (2, (0, 0, 2))],
      # f:
##       [(5, (0, 0, 3)), (-3, (0, 0, 1))],
##       [(5, (1, 0, 2)), (-1, (1, 0, 0))],
##       [(5, (0, 1, 2)), (-1, (0, 1, 0))],
##       [(1, (2, 0, 1)), (-1, (0, 2, 1))],
##       [(1, (1, 1, 1))],
##       [(1, (3, 0, 0)), (-3, (1, 2, 0))],
##       [(3, (2, 1, 0)), (-1, (0, 3, 0))],
##       # g:
##       [(3, (0, 0, 0)), (-30, (0, 0, 2)), (35, (0, 0, 4))],
##       [(-3, (1, 0, 1)), (7, (1, 0, 3))],
##       [(-3, (0, 1, 1)), (7, (0, 1, 3))],
##       [(-1, (2, 0, 0)), (1, (0, 2, 0)), (7, (2, 0, 2)), (-7, (0, 2, 2))],
##       [(-1, (1, 1, 0)), (7, (1, 1, 2))],
##       [(1, (3, 0, 1)), (-3, (1, 2, 1))],
##       [(-1, (0, 3, 1)), (3, (2, 1, 1))],
##       [(1, (4, 0, 0)), (-6, (2, 2, 0)), (1, (0, 4, 0))],
##       [(1, (3, 1, 0)), (-1, (1, 3, 0))]]
##       [(5, (0, 0, 3)), (-3, (0, 0, 1))],
      [(2, (0, 0, 3)), (-3, (2, 0, 1)), (-3, (0, 2, 1))],
#      [(5, (1, 0, 2)), (-1, (1, 0, 0))],
      [(4, (1, 0, 2)), (-1, (3, 0, 0)), (-1, (1, 2, 0))],
#      [(5, (0, 1, 2)), (-1, (0, 1, 0))],
      [(4, (0, 1, 2)), (-1, (2, 1, 0)), (-1, (0, 3, 0))],
      [(1, (2, 0, 1)), (-1, (0, 2, 1))],
      [(1, (1, 1, 1))],
      [(1, (3, 0, 0)), (-3, (1, 2, 0))],
      [(3, (2, 1, 0)), (-1, (0, 3, 0))],
      # g:
#      [(3, (0, 0, 0)), (-30, (0, 0, 2)), (35, (0, 0, 4))],
      [(3, (0, 0, 0)), (-30, (0, 0, 2)), (35, (0, 0, 4))],
#      [(-3, (1, 0, 1)), (7, (1, 0, 3))],
      [(-3, (3, 0, 1)), (-3, (1, 2, 1)), (4, (1, 0, 3))],
#      [(-3, (0, 1, 1)), (7, (0, 1, 3))],
      [(-3, (2, 1, 1)), (-3, (0, 3, 1)), (4, (0, 1, 3))],
      [(-1, (2, 0, 0)), (1, (0, 2, 0)), (7, (2, 0, 2)), (-7, (0, 2, 2))],
#      [(-1, (1, 1, 0)), (7, (1, 1, 2))],
      [(-1, (3, 1, 0)), (-1, (1, 3, 0)), (6, (1, 1, 2))],
      [(1, (3, 0, 1)), (-3, (1, 2, 1))],
      [(-1, (0, 3, 1)), (3, (2, 1, 1))],
      [(1, (4, 0, 0)), (-6, (2, 2, 0)), (1, (0, 4, 0))],
      [(1, (3, 1, 0)), (-1, (1, 3, 0))]]

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

for L in range(25):
    s = 1.0 / yLL(L, L)**0.5
    YL[L] = [(s * c, n) for c, n in YL[L]]

if debug:
    for L1 in range(25):
        for L2 in range(25):
            r = 0.0
            if L1 == L2:
                r = 1.0
            assert abs(yLL(L1, L2) - r) < 1e-14

def Y(L, x, y, z):
    result = 0.0
    for c, n in YL[L]:
        result += c * x**n[0] * y**n[1] * z**n[2]
    return result

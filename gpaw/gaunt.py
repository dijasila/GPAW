from functools import lru_cache

import numpy as np

from gpaw.typing import Array3D


@lru_cache()
def gaunt(lmax: int = 2) -> Array3D:
    Lmax = (lmax + 1)**2
    L2max = (2 * lmax + 1)**2

    from gpaw.spherical_harmonics import YL, gam
    G_LLL = np.zeros((Lmax, L2max, L2max))
    for L1 in range(Lmax):
        for L2 in range(L2max):
            for L in range(L2max):
                r = 0.0
                for c1, n1 in YL[L1]:
                    for c2, n2 in YL[L2]:
                        for c, n in YL[L]:
                            nx = n1[0] + n2[0] + n[0]
                            ny = n1[1] + n2[1] + n[1]
                            nz = n1[2] + n2[2] + n[2]
                            r += c * c1 * c2 * gam(nx, ny, nz)
                G_LLL[L1, L2, L] = r
    return G_LLL


@lru_cache()
def nabla(lmax: int = 2) -> Array3D:
    """Create the array of derivative intergrals.

    ::

      /  ^     1-l' d   l'
      | dr Y  r    ---(r Y  )
      /     L      dr     L'
                     v
    """
    Lmax = (lmax + 1)**2
    from gpaw.spherical_harmonics import YL, gam
    Y_LLv = np.zeros((Lmax, Lmax, 3))
    # Insert new values
    for L1 in range(Lmax):
        for L2 in range(Lmax):
            for v in range(3):
                r = 0.0
                for c1, n1 in YL[L1]:
                    for c2, n2 in YL[L2]:
                        n = [0, 0, 0]
                        n[0] = n1[0] + n2[0]
                        n[1] = n1[1] + n2[1]
                        n[2] = n1[2] + n2[2]
                        if n2[v] > 0:
                            # apply derivative
                            n[v] -= 1
                            # add integral
                            r += n2[v] * c1 * c2 * gam(n[0], n[1], n[2])
                Y_LLv[L1, L2, v] = r
    return Y_LLv

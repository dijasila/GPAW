from typing import Tuple

import numpy as np
import ase.units as units

from gpaw.typing import Array2D, ArrayLike2D


def parse_hubbard_string(type: str) -> Tuple[str, 'HubbardU']:

    # Parse DFT+U parameters from type-string:
    # Examples: "type:l,U" or "type:l,U,scale"
    type, lus = type.split(':')
    if type == '':
        type = 'paw'

    l = []
    U = []
    scale = []

    for lu in lus.split(';'):  # Multiple U corrections
        l_, u_, scale_ = (lu + ',,').split(',')[:3]
        l.append('spdf'.find(l_))
        U.append(float(u_) / units.Hartree)
        if scale_:
            scale.append(bool(int(scale_)))
        else:
            scale.append(True)
    return type, HubbardU(U, l, scale)


class HubbardU:
    def __init__(self, U, l, scale=1):
        self.scale = scale
        self.U = U
        self.l = l

    def _tuple(self):
        # Tests use this method to compare to expected values
        return (self.l, self.U, self.scale)

    def calculate(self, setup, D_sii):
        eU = 0.
        dHU_sii = np.zeros_like(D_sii)
        for l, U, scale in zip(self.l, self.U, self.scale):
            eU1, dHU1_sii = hubbard(
                setup.l_j, setup.lq, D_sii,
                l=l, U=U, scale=scale)

            eU += eU1
            dHU_sii += dHU1_sii
        return eU, dHU_sii

    def descriptions(self):
        for U, l, scale in zip(self.U, self.l, self.scale):
            yield f'Hubbard: {{U: {U * units.Ha},  # eV\n'
            yield f'          l: {l},\n'
            yield f'          scale: {bool(scale)}}}'


def hubbard(l_j, lq,
            D_sii,
            l: int,
            U: float,
            scale: bool) -> Tuple[float, ArrayLike2D]:
    nspins = len(D_sii)

    nl = np.where(np.equal(l_j, l))[0]
    nn = (2 * np.array(l_j) + 1)[0:nl[0]].sum()

    eU = 0.
    dHU_sii = []

    s = 0
    for D_ii in D_sii:
        N_mm, dHU_ii = aoom(l_j, lq, D_ii, l, scale)
        N_mm = N_mm / 2 * nspins

        if nspins == 4:
            N_mm = N_mm / 2.0
            if s == 0:
                Eorb = U / 2. * (N_mm -
                                 0.5 * np.dot(N_mm, N_mm)).trace()

                Vorb = U / 2. * (np.eye(2 * l + 1) - N_mm)

            else:
                Eorb = U / 2. * (-0.5 * np.dot(N_mm, N_mm)).trace()

                Vorb = -U / 2. * N_mm
        else:
            Eorb = U / 2. * (N_mm -
                             np.dot(N_mm, N_mm)).trace()

            Vorb = U * (0.5 * np.eye(2 * l + 1) - N_mm)

        eU += Eorb
        if nspins == 1:
            # Add contribution of other spin manifold
            eU += Eorb

        if len(nl) == 2:
            mm = (2 * np.array(l_j) + 1)[0:nl[1]].sum()

            dHU_ii[nn:nn + 2 * l + 1, nn:nn + 2 * l + 1] *= Vorb
            dHU_ii[mm:mm + 2 * l + 1, nn:nn + 2 * l + 1] *= Vorb
            dHU_ii[nn:nn + 2 * l + 1, mm:mm + 2 * l + 1] *= Vorb
            dHU_ii[mm:mm + 2 * l + 1, mm:mm + 2 * l + 1] *= Vorb
        else:
            dHU_ii[nn:nn + 2 * l + 1, nn:nn + 2 * l + 1] *= Vorb

        dHU_sii.append(dHU_ii)
        s += 1

    return eU, dHU_sii


def aoom(l_j, lq,
         D_ii: Array2D,
         l: int,
         scale: bool = True) -> Tuple[Array2D, Array2D]:
    """Atomic orbital occupation matrix.

    Determine the atomic orbital occupation matrix (aoom) for a
    given l-quantum number.

    This function finds the submatrix / submatrices of the density matrix
    (D_sii) which
    represent the overlap of orbitals the selected orbitals (l) upon which the
    the density is expanded (ex <p_x|p*>,<p|p>,<p*|p*> ).

    Returned is only the part of the density matrix which represents the
    orbital occupation matrix. For l=2 this is a 5x5 matrix.
    """

    # j-indices which have the correct angular momentum quantum number
    nl = np.where(np.equal(l_j, l))[0]

    nm_j = 2 * np.array(l_j) + 1
    nm = nm_j[nl[0]]

    # Get relevant entries of the density matrix
    i1 = slice(nm_j[:nl[0]].sum(), nm_j[:nl[0]].sum() + nm)  # Bounded

    dHU_ii = np.zeros_like(D_ii)
    if len(nl) == 2:
        # First get q-indices for the inner products
        q1 = nl[0] * len(l_j) - (nl[0] - 1) * nl[0] // 2  # Bounded-bounded
        q2 = nl[1] * len(l_j) - (nl[1] - 1) * nl[1] // 2  # Unbounded-unbounded
        q12 = q1 + nl[1] - nl[0]  # Bounded-unbounded

        # If the Hubbard correction should be scaled, the three inner products
        # will be divided by the inner product of the bounded partial wave,
        # increasing these inner products since 0 < lq[q1] < 1.
        if scale:
            lq_1 = 1
            lq_12 = lq[q12] / lq[q1]
            lq_2 = lq[q2] / lq[q1]
        else:
            lq_1 = lq[q1]
            lq_12 = lq[q12]
            lq_2 = lq[q2]

        # Get relevant entries of the density matrix (unbounded partial waves)
        i2 = slice(nm_j[:nl[1]].sum(), nm_j[:nl[1]].sum() + nm)

        # Finally, scale and add the four submatrices of the occupation matrix
        N_mm = (D_ii[i1, i1] * lq_1 + D_ii[i1, i2] * lq_12
                + D_ii[i2, i1] * lq_12 + D_ii[i2, i2] * lq_2)

        dHU_ii[i1, i1] = lq_1
        dHU_ii[i1, i2] = lq_12
        dHU_ii[i2, i1] = lq_12
        dHU_ii[i2, i2] = lq_2

        return N_mm, dHU_ii
    elif len(nl) == 1:
        # If there is only 1 partial wave with matching l, we assert that the
        # partial wave is unbounded.
        assert l_j[-1] == l

        assert scale == 0, \
            'DFT+U correction cannot be scaled if there is no bounded state.'

        N_mm = D_ii[i1, i1] * lq[-1]
        dHU_ii[i1, i1] = lq[-1]
        return N_mm, dHU_ii
    else:
        raise NotImplementedError(f'Setup has {len(nl)} partial waves with '
                                  f'angular momentum quantum number {l}. '
                                  'Must be 1 or 2 for DFT+U correction.')

"""Improved tetrahedron method for Brillouin-zone integrations
Peter E. Blöchl, O. Jepsen, and O. K. Andersen
Phys. Rev. B 49, 16223 – Published 15 June 1994

DOI:https://doi.org/10.1103/PhysRevB.49.16223
"""

from math import nan
from typing import List, Tuple

import numpy as np
from scipy.spatial import Delaunay

from gpaw.occupations import (ZeroWidth, findroot, collect_eigelvalues,
                              distribute_occupation_numbers,
                              OccupationNumbers)
from gpaw.mpi import broadcast_float


def bja1(e1, e2, e3, e4):
    """Eq. (A2) from Blöchl, Jepsen and Andersen."""
    x = 1.0 / ((e2 - e1) * (e3 - e1) * (e4 - e1))
    return (-e1**3 * x,
            3 * e1**2 * x)


def bja2(e1, e2, e3, e4):
    x = 1.0 / ((e3 - e1) * (e4 - e1))
    y = (e3 - e1 + e4 - e2) / ((e3 - e2) * (e4 - e2))
    return (x * ((e2 - e1)**2
                 - 3 * (e2 - e1) * e2
                 + 3 * e2**2
                 + y * e2**3),
            x * (3 * (e2 - e1)
                 - 6 * e2
                 - 3 * y * e2**2))


def bja3(e1, e2, e3, e4):
    x = 1.0 / ((e4 - e1) * (e4 - e2) * (e4 - e3))
    return (1 - x * e4**3,
            3 * x * e4**2)


def triangulate_submesh(rcell_cv):
    """Find the 6 tetrahedra."""
    ABC_sc = np.array([[A, B, C]
                       for A in [0, 1] for B in [0, 1] for C in [0, 1]])
    dt = Delaunay(ABC_sc.dot(rcell_cv))
    s_tq = dt.simplices
    assert s_tq.shape == (6, 4)
    ABC_tqc = ABC_sc[s_tq]
    return ABC_tqc


def triangulate_everything(size_c, ABC_tqc, i_k):
    nbzk = size_c.prod()
    ABC_ck = np.unravel_index(np.arange(nbzk), size_c)
    ABC_tqck = ABC_tqc[..., np.newaxis] + ABC_ck
    ABC_cktq = np.transpose(ABC_tqck, (2, 3, 0, 1))
    k_ktq = np.ravel_multi_index(ABC_cktq.reshape((3, nbzk * 6 * 4)),
                                 size_c,
                                 mode='wrap').reshape((nbzk, 6, 4))
    i_ktq = i_k[k_ktq]
    return i_ktq


class TetrahedronMethod(OccupationNumbers):
    def __init__(self,
                 size: Tuple[int, int, int],
                 bz2ibzmap: List[int],
                 rcell: List[List[float]]):
        """ . """

        OccupationNumbers.__init__(self)

        self.size_c = np.asarray(size)
        self.i_k = np.asarray(bz2ibzmap)
        self.rcell_cv = np.asarray(rcell)

        assert self.size_c.shape == (3,)
        assert self.rcell_cv.shape == (3, 3)
        assert self.i_k.shape == (self.size_c.prod(),)

        ABC_tqc = triangulate_submesh(
            self.rcell_cv / self.size_c[:, np.newaxis])

        self.i_ktq = triangulate_everything(self.size_c, ABC_tqc, self.i_k)

    def _calculate(self,
                   nelectrons,
                   eig_qn,
                   weight_q,
                   f_qn,
                   parallel,
                   fermi_level_guess=nan):

        if np.isnan(fermi_level_guess):
            zero = ZeroWidth()
            fermi_level_guess, _ = zero._calculate(
                nelectrons, eig_qn, weight_q, f_qn, parallel)
            print(fermi_level_guess)
            if np.isinf(fermi_level_guess):
                return fermi_level_guess, 0.0

        x = fermi_level_guess

        eig_in, weight_k, nkpts_r = collect_eigelvalues(eig_qn, weight_q,
                                                        parallel)

        assert self.i_k.max() == len(eig_in) - 1

        if eig_in is not None:
            def func(x, eig_in=eig_in):
                n, dn = count(x, eig_in, self.i_ktq)
                return n - nelectrons, dn

            fermi_level, niter = findroot(func, x)
            f_kn = np.zeros_like(eig_in)
        else:
            f_kn = None
            fermi_level = nan

        distribute_occupation_numbers(f_kn, f_qn, nkpts_r, parallel)

        if parallel.kpt_comm.rank == 0 and parallel.bd is not None:
            fermi_level = broadcast_float(fermi_level, parallel.bd.comm)
        fermi_level = broadcast_float(fermi_level, parallel.kpt_comm)

        return fermi_level, 0.0


def count(fermi_level, eig_in, i_ktq):
    eig_in = eig_in - fermi_level
    nocc_i = (eig_in < 0.0).sum(axis=1)
    n1 = nocc_i.min()
    n2 = nocc_i.max()

    ne = n1
    dnedef = 0.0

    if n1 == n2:
        return ne, dnedef

    ntetra = 6 * i_ktq.shape[0]
    eig_Tq = eig_in[i_ktq, n1:n2].transpose((0, 1, 3, 2)).reshape(
        (ntetra * (n2 - n1), 4))
    eig_Tq.sort(axis=1)

    eig_Tq = eig_Tq[eig_Tq[:, 0] < 0.0]

    mask1_T = eig_Tq[:, 1] > 0.0
    mask2_T = ~mask1_T & (eig_Tq[:, 2] > 0.0)
    mask3_T = ~mask1_T & ~mask2_T & (eig_Tq[:, 3] > 0.0)

    for mask_T, bja in [(mask1_T, bja1), (mask2_T, bja2), (mask3_T, bja3)]:
        n_T, dn_T = bja(*eig_Tq[mask_T].T)
        ne += n_T.sum() / ntetra
        dnedef += dn_T.sum() / ntetra

    mask4_T = ~mask1_T & ~mask2_T & ~mask3_T
    ne += mask4_T.sum() / ntetra

    return ne, dnedef


def test():
    t = TetrahedronMethod([2, 2, 1],
                          [0, 1, 2, 1],
                          np.diag([1.0, 1.0, 0.1]))

    eig_in = np.array([[1.0], [0.0], [0.0]])
    """
    x = np.linspace(-0.5, 1.5, 50)
    import matplotlib.pyplot as plt
    plt.plot(x, [count(f, eig_in, t.i_ktq)[0] for f in x])
    plt.show()
    """
    r = t.calculate(0.5, eig_in, [0.25, 0.5, 0.25], None, [0.5])
    print(r)


if __name__ == '__main__':
    test()

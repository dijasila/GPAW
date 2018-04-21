# encoding: utf-8
import numpy as np
import scipy.sparse as sparse
from ase.neighborlist import PrimitiveNeighborList

from gpaw import debug
from gpaw.lcao.overlap import (FourierTransformer, TwoSiteOverlapCalculator,
                               ManySiteOverlapCalculator,
                               AtomicDisplacement, NullPhases, BlochPhases,
                               DerivativeAtomicDisplacement)

def get_cutoffs(f_Ij):
    rcutmax_I = []
    for f_j in f_Ij:
        rcutmax = 0.001  # 'paranoid zero'
        for f in f_j:
            rcutmax = max(rcutmax, f.get_cutoff())
        rcutmax_I.append(rcutmax)
    return rcutmax_I


def get_lvalues(f_Ij):
    return [[f.get_angular_momentum_number() for f in f_j] for f_j in f_Ij]


class AtomPairRegistry:
    def __init__(self, cutoff_a, pbc_c, cell_cv, spos_ac):
        nl = PrimitiveNeighborList(cutoff_a, skin=0, sorted=True,
                                   self_interaction=True,
                                   use_scaled_positions=True)

        nl.update(pbc=pbc_c, cell=cell_cv, coordinates=spos_ac)
        r_and_offset_aao = {}

        def add(a1, a2, R_c, offset):
            r_and_offset_aao.setdefault((a1, a2), []).append((R_c, offset))

        for a1, spos1_c in enumerate(spos_ac):
            a2_a, offsets = nl.get_neighbors(a1)
            for a2, offset in zip(a2_a, offsets):
                spos2_c = spos_ac[a2] + offset

                R_c = np.dot(spos2_c - spos1_c, cell_cv)
                add(a1, a2, R_c, offset)
                if a1 != a2 or offset.any():
                    add(a2, a1, -R_c, -offset)
        self.r_and_offset_aao = r_and_offset_aao

    def get(self, a1, a2):
        R_c_and_offset_a = self.r_and_offset_aao.get((a1, a2))
        return R_c_and_offset_a


class TCI:
    """High-level two-center integral calculator.

    This object is not aware of parallelization.  It works with any
    pair of atoms a1, a2.

    Create the object and calculate any interatomic overlap matrix as below.

      tci = TCI(...)

    Projector/basis overlap <pt_i^a1|phi_mu> between atoms a1, a2:

      P_qim = tci.P(a1, a2)

    Derivatives of the above with respect to movement of a2:

      dPdR_qvim = tci.dPdR(a1, a2)

    Basis/basis overlap and kinetic matrix elements between atoms a1, a2:

      O_qmm, T_qmm = tci.O_T(a1, a2)

    Derivative of the above wrt. position of a2:

      dOdR_qvmm, dTdR_qvmm = tci.dOdR_dTdR(a1, a2)

    """
    def __init__(self, phit_Ij, pt_Ij, I_a, cell_cv, spos_ac, pbc_c, ibzk_qc,
                 dtype):
        assert len(pt_Ij) == len(phit_Ij)
        self.dtype = dtype

        # Cutoffs by species:
        pt_rcmax_I = get_cutoffs(pt_Ij)
        phit_rcmax_I = get_cutoffs(phit_Ij)
        rcmax_I = [max(rc1, rc2) for rc1, rc2
                   in zip(pt_rcmax_I, phit_rcmax_I)]
        # XXX It is somewhat nasty that rcmax depends on how long our
        # longest orbital happens to be
        transformer = FourierTransformer(rcmax=max(rcmax_I + [1e-3]), ng=2**10)
        tsoc = TwoSiteOverlapCalculator(transformer)
        msoc = ManySiteOverlapCalculator(tsoc, I_a, I_a)

        # Cutoffs by atom:
        cutoff_a = [rcmax_I[I] for I in I_a]
        self.pt_rcmax_a = np.array([pt_rcmax_I[I] for I in I_a])
        self.phit_rcmax_a = np.array([phit_rcmax_I[I] for I in I_a])

        self.a1a2 = AtomPairRegistry(cutoff_a, pbc_c, cell_cv, spos_ac)

        self.ibzk_qc = ibzk_qc
        if ibzk_qc.any():
            self.get_phases = BlochPhases
        else:
            self.get_phases = NullPhases

        phit_Ijq = msoc.transform(phit_Ij)
        pt_Ijq = msoc.transform(pt_Ij)

        pt_l_Ij = get_lvalues(pt_Ij)
        phit_l_Ij = get_lvalues(phit_Ij)

        # Avoid two-way for O and T?
        # TODO: lazy msoc
        self.O_expansions = msoc.calculate_expansions(phit_l_Ij, phit_Ijq,
                                                      phit_l_Ij, phit_Ijq)
        self.T_expansions = msoc.calculate_kinetic_expansions(phit_l_Ij,
                                                              phit_Ijq)
        self.P_expansions = msoc.calculate_expansions(pt_l_Ij, pt_Ijq,
                                                      phit_l_Ij, phit_Ijq)

        self.O_T = self._tci_shortcut(True, False, False)
        self.P = self._tci_shortcut(False, True, False)
        self.dOdR_dTdR = self._tci_shortcut(True, False, True)
        self.dPdR = self._tci_shortcut(False, True, True)

    def _tci_shortcut(self, OT, P, derivative):
        def calculate(a1, a2):
            return self._calculate(a1, a2, OT, P, derivative)
        return calculate

    def _calculate(self, a1, a2, OT=False, P=False, derivative=False):
        """Calculate overlap of functions between atoms a1 and a2."""

        R_c_and_offset_a = self.a1a2.get(a1, a2)

        if R_c_and_offset_a is None:
            return None if P else (None, None)

        dtype = self.dtype
        #get_phases = self.overlapcalc.phaseclass
        get_phases = self.get_phases

        displacement = DerivativeAtomicDisplacement if derivative else AtomicDisplacement
        ibzk_qc = self.ibzk_qc
        nq = len(ibzk_qc)
        phit_rcmax_a = self.phit_rcmax_a
        pt_rcmax_a = self.pt_rcmax_a

        shape = (nq, 3) if derivative else (nq,)

        # The caller should give either P or OT.
        if P:
            P_expansion = self.P_expansions.get(a1, a2)
            obj = P_qim = P_expansion.zeros(shape, dtype=dtype)

        if OT:
            O_expansion = self.O_expansions.get(a1, a2)
            T_expansion = self.T_expansions.get(a1, a2)
            O_qmm = O_expansion.zeros(shape, dtype=dtype)
            T_qmm = T_expansion.zeros(shape, dtype=dtype)
            obj = O_qmm, T_qmm

        for R_c, offset in R_c_and_offset_a:
            norm = np.linalg.norm(R_c)
            phases = get_phases(ibzk_qc, offset)

            disp = displacement(None, a1, a2, R_c, offset, phases)

            if P and norm < pt_rcmax_a[a1] + phit_rcmax_a[a2]:
                disp.evaluate_overlap(P_expansion, P_qim)

            if OT and norm < phit_rcmax_a[a1] + phit_rcmax_a[a2]:
                disp.evaluate_overlap(O_expansion, O_qmm)
                disp.evaluate_overlap(T_expansion, T_qmm)

        return obj


class ManyTCI:
    def __init__(self, setups, gd, spos_ac, ibzk_qc, dtype):
        I_setup = {}
        setups_I = list(setups.setups.values())
        for I, setup in enumerate(setups_I):
            I_setup[setup] = I
        I_a = [I_setup[setup] for setup in setups]

        tci = TCI([s.phit_j for s in setups_I], [s.pt_j for s in setups_I],
                  I_a, gd.cell_cv, spos_ac=spos_ac, pbc_c=gd.pbc_c,
                  ibzk_qc=ibzk_qc, dtype=dtype)
        self.tci = tci

        self.setups = setups
        self.dtype = dtype
        self.Pindices = setups.projector_indices()
        self.Mindices = setups.basis_indices()
        self.natoms = len(setups)
        self.nq = len(ibzk_qc)
        self.nao = self.Mindices.max

    def P_aqMi(self, my_atom_indices, derivative=False):
        P_axMi = {}
        if derivative:
            P = self.tci.dPdR
            empty = lambda nI: np.empty((self.nq, 3, self.nao, nI), self.dtype)
        else:
            P = self.tci.P
            empty = lambda nI: np.empty((self.nq, self.nao, nI), self.dtype)

        Mindices = self.Mindices

        for a1 in my_atom_indices:
            P_xMi = empty(self.setups[a1].ni)

            for a2 in range(self.natoms):
                N1, N2 = Mindices[a2]
                P_xmi = P_xMi[..., N1:N2, :]
                P_xim = P(a1, a2)
                if P_xim is None:
                    P_xmi[:] = 0.0
                else:
                    P_xmi[:] = P_xim.swapaxes(-2, -1).conj()
            P_axMi[a1] = P_xMi

        if derivative:
            for a in P_axMi:
                P_axMi[a] *= -1.0
        return P_axMi

    def P_qIM(self, my_atom_indices):
        nq = self.nq
        P = self.tci.P
        P_qIM = [sparse.lil_matrix((self.Pindices.max, self.Mindices.max),
                                   dtype=self.dtype)
                 for _ in range(nq)]

        for a1 in my_atom_indices:
            I1, I2 = self.Pindices[a1]

            # We can stride a2 over e.g. bd.comm and then do bd.comm.sum().
            # How should we do comm.sum() on a sparse matrix though?
            for a2 in range(self.natoms):
                M1, M2 = self.Mindices[a2]
                P_qim = P(a1, a2)
                if P_qim is not None:
                    for q in range(nq):
                        P_qIM[q][I1:I2, M1:M2] = P_qim[q]
        P_qIM = [P_IM.tocsr() for P_IM in P_qIM]
        return P_qIM

    def O_qMM_T_qMM(self, gdcomm, Mstart, Mstop, ignore_upper,
                    derivative=False):
        mynao = Mstop - Mstart
        Mindices = self.Mindices

        if derivative:
            O_T = self.tci.dOdR_dTdR
            shape = (self.nq, 3, mynao, self.nao)
        else:
            O_T = self.tci.O_T
            shape = (self.nq, mynao, self.nao)

        O_xMM = np.zeros(shape, self.dtype)
        T_xMM = np.zeros(shape, self.dtype)

        # XXX the a1/a2 loops are not yet well load balanced.
        for a1 in range(self.natoms):
            M1, M2 = Mindices[a1]
            if M2 <= Mstart or M1 >= Mstop:
                continue

            myM1 = max(M1 - Mstart, 0)
            myM2 = min(M2 - Mstart, mynao)
            nM = myM2 - myM1

            assert nM > 0, nM

            a2max = a1 + 1 if ignore_upper else self.natoms

            for a2 in range(gdcomm.rank, a2max, gdcomm.size):
                O_xmm, T_xmm = O_T(a1, a2)
                if O_xmm is None:
                    continue

                N1, N2 = Mindices[a2]
                m1 = max(Mstart - M1, 0)
                m2 = m1 + nM  # (Slice may go beyond end of matrix but OK)
                O_xmm = O_xmm[..., m1:m2, :]
                T_xmm = T_xmm[..., m1:m2, :]
                O_xMM[..., myM1:myM2, N1:N2] = O_xmm
                T_xMM[..., myM1:myM2, N1:N2] = T_xmm

        if debug:
            assert not np.isnan(O_qMM).any()
            assert not np.isnan(T_qMM).any()
        return O_xMM, T_xMM

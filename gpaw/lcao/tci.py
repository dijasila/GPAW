# encoding: utf-8
from ase.neighborlist import PrimitiveNeighborList

from gpaw.lcao.overlap import (FourierTransformer, TwoSiteOverlapCalculator,
                               TwoCenterIntegralCalculator)

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


class Overlap:
    def __init__(self):
        self.O_ii = None
        self.T_ii = None
        self.P_ii = None

class TCI:
    def __init__(self, pt_Ij, phit_Ij, I_a, cell_cv, spos_ac, pbc_c, ibzk_qc,
                 gamma):
        assert len(pt_Ij) == len(phit_Ij)

        # Cutoffs by species:
        pt_rcmax_I = get_cutoffs(pt_Ij)
        phit_rcmax_I = get_cutoffs(phit_Ij)
        rcmax_I = [max(rc1, rc2) for rc1, rc2
                        in zip(pt_rcmax_I, phit_rcmax_I)]
        transformer = FourierTransformer(rcut=max(0.001, *rcmax_I), ng=2**10)
        tsoc = TwoSiteOverlapCalculator(transformer)

        # Cutoffs by atom:
        cutoff_a = [rcmax_I[I] for I in I_a]
        pt_rcmax_a = np.array([self.pt_rcmax_I[I] for I in I_a])
        phit_rcmax_a = np.array([self.phit_rcmax_I[I] for I in I_a])

        self.a1a2 = AtomPairRegistry(cutoff_a, pbc_c, cell_cv, spos_ac)

        self.overlapcalc = TwoCenterIntegralCalculator(wfs.kd.ibzk_qc,
                                                       derivative=False)

        pt_l_Ij = get_lvalues(pt_Ij)
        phit_l_Ij = get_lvalues(phit_Ij)
        self.P_expansions = msoc.calculate_expansions(pt_l_Ij, pt_Ijq,
                                                      phit_l_Ij, phit_Ijq)

    def calculate(a1, a2, OT=False, P=False):
        """Calculate overlap of functions between atoms a1 and a2."""

        o = Overlap()
        R_c_and_offset_a = self.a1a2.get(a1, a2)

        if R_c_and_offset_a is None:
            return o

        dtype = self.dtype
        get_phases = self.overlapcalc.phaseclass

        if P:
            P_expansion = self.P_expansions.get(a1, a2)
            o.P_qim = P_qim = expansion.zeros((nq,), dtype=dtype)

        if OT:
            O_expansion = self.O_expansions.get(a1, a2)
            T_expansion = self.T_expansions.get(a1, a2)
            o.O_qmm = O_qmm = O_expansion.zeros((nq,), dtype=dtype)
            o.T_qmm = T_qmm = T_expansion.zeros((nq,), dtype=dtype)

        for R_c, offset in R_c_and_offset_a:
            norm = np.linalg.norm(R_c, R_c)
            phases = get_phases(overlapcalc.ibzk_qc, offset)
            disp = AtomicDisplacement(None, a1, a2, R_c, offset, phases)

            if P and pt_rcmax_a[a1] + phit_rcmax_a[a2] < norm:
                disp.evaluate_overlap(P_expansion, P_qim)

            if OT and phit_rcmax_a[a1] + phit_rcmax_a[a2] < norm:
                disp.evaluate_overlap(O_expansion, O_qim)
                disp.evaluate_overlap(T_expansion, T_qim)

        return o

import numpy as np
from gpaw.lcao.tci import get_cutoffs, split_setups_to_types, AtomPairRegistry,\
                          get_lvalues

from gpaw.auxlcao.utilities import get_compensation_charge_splines,\
                                   get_wgauxphit_product_splines,\
                                   get_auxiliary_splines

from gpaw.lcao.overlap import (FourierTransformer, TwoSiteOverlapCalculator,
                               ManySiteOverlapCalculator,
                               AtomicDisplacement, NullPhases)
from gpaw.gaunt import gaunt

G_LLL = gaunt(3) # XXX

class MatrixElements:
    def __init__(self, lmax=2):
        self.lmax = lmax

    def initialize(self, density, ham, wfs):
        self.setups = setups = wfs.setups

        self.M_a = setups.M_a.copy()
        self.M_a.append(setups.nao)

        # I_a is an index for each atom identifying which setup type it has.
        # setup_for_atom_a = setups_I[I_a[a]]
        I_a, setups_I = split_setups_to_types(setups)

        # Build a list of basis function splines for each setup, and obtain
        # the maximum cut off for each setup type, phit_rcmax_I.
        phit_rcmax_I = get_cutoffs([s.phit_j for s in setups_I])

        # Obtain the maximum cutoff on per atom basis
        self.rcmax_a = [phit_rcmax_I[I] for I in I_a]

        for I, (setup, rcmax) in enumerate(zip(setups_I, phit_rcmax_I)):
            # Compensation charges
            setup.gaux_l, setup.wgaux_l = get_compensation_charge_splines(setup, self.lmax, rcmax)

            # Single center Hartree of compensation charges * one phit_j
            setup.wgauxphit_x = get_wgauxphit_product_splines(setup, setup.wgaux_l, setup.phit_j, rcmax)

            # Auxiliary basis functions
            setup.auxt_j, setup.wauxt_j = get_auxiliary_splines(setup, self.lmax, rcmax)

            setup.Naux = sum([ 2*spline.l+1 for spline in setup.auxt_j])

            # Single center Hartree of auxiliary basis function * one phit_j
            setup.wauxtphit_x = get_wgauxphit_product_splines(setup, setup.wauxt_j, setup.phit_j, rcmax)


        transformer = FourierTransformer(rcmax=max(phit_rcmax_I)+1e-3, ng=2**14) # XXX Used to be 2**10, add accuracy for debug purposes
        tsoc = TwoSiteOverlapCalculator(transformer)
        msoc = ManySiteOverlapCalculator(tsoc, I_a, I_a)

        wauxtphit_Ix = [ setup.wauxtphit_x for setup in setups_I]
        wauxtphit_Ixq = msoc.transform(wauxtphit_Ix)
        wauxtphit_l_Ix = get_lvalues(wauxtphit_Ix)

        wgauxphit_Ix = [ setup.wgauxphit_x for setup in setups_I]
        wgauxphit_Ixq = msoc.transform(wgauxphit_Ix)
        wgauxphit_l_Ix = get_lvalues(wgauxphit_Ix)

        phit_Ij = [ setup.phit_j for setup in setups_I] 
        phit_Ijq = msoc.transform(phit_Ij)
        phit_l_Ij = get_lvalues(phit_Ij)

        auxt_Ij = [ setup.auxt_j for setup in setups_I] 
        auxt_Ijq = msoc.transform(auxt_Ij)
        auxt_l_Ij = get_lvalues(phit_Ij)

        # X = Combined index of compensation l, phit j, selected gaunt expansion l channels, gaunt expansion m of those chanels
        self.W_XM_expansions = msoc.calculate_expansions(wgauxphit_l_Ix, wgauxphit_Ixq,
                                                         phit_l_Ij, phit_Ijq)

        # Y = Combined index of auxiliary j, phit j, selected gaunt expansion l channels, gaunt expansion m of those channels
        self.W_YM_expansions = msoc.calculate_expansions(wauxtphit_l_Ix, wauxtphit_Ixq,
                                                         phit_l_Ij, phit_Ijq)


    def set_positions_and_cell(self, spos_ac, cell_cv, pbc_c, ibzq_qc, dtype):
        self.spos_ac = spos_ac
        self.cell_cv = cell_cv
        self.pbc_c = pbc_c
        self.ibzq_qc = ibzq_qc
        self.dtype = dtype
      
        # Build an atom pair registry for basis function overlap
        self.apr = AtomPairRegistry(self.rcmax_a, pbc_c, cell_cv, spos_ac)

        self.a1a2_p = [ (a1,a2) for a1,a2 in self.apr.get_atompairs() ]

        print(self.a1a2_p)
        print(self.rcmax_a)
        def log(name, quantity):
            print('%-70s %-5d' % (name, quantity))

        log('Number of atoms', len(self.rcmax_a))
        log('Atom pairs with overlapping basis functions', len(self.a1a2_p))
        self.a1a2_p = [ (a1,a2) for a1,a2 in self.apr.get_atompairs() if a1<=a2 ]

        #for a1, a2 in self.a1a2_p:
        #    for R_c in self.apr.get(a1,a2):
        #        print(a1,a2,np.linalg.norm(R_c), R_c)

        def a1a2a3a4_q():
            for p1, (a1,a2) in enumerate(self.a1a2_p):
                for a3, a4 in self.a1a2_p[p1:]:
                    yield a1,a2,a3,a4
 
        self.a1a2a3a4_q = a1a2a3a4_q

        # Even this takes minutes on large systems
        # log('Atom quadtuples with exchange of pairs', len([ aaaa for aaaa in self.a1a2a3a4_q() ]))

        log('Atom quadtuples with exchange of pairs', len(self.a1a2_p)*(len(self.a1a2_p)+1)/2)

    def set_parameters(self, parameters):
        self.parameters = parameters


    """

             /       ||               \
     I    =  | g (r) || φ (r') φ (r') |
      LMM    \  L    ||  M     M'     /,

      where double bar stands for Coulomb integral.

      a1 is always either a2 or a3.
      a2 and a3 are overlapping centers.

    """

    def evaluate_3ci_AMM(self, a1, a2, a3):
        print('Calling 3ci_AMM', a1,a2,a3)
        if a1 != a2:
            if a1 != a3:
                raise NotImplementedError('Only 3-center integrals spanning 2 centers are supported. Got: (%d, %d, %d).' % (a1,a2,a3))
            return self.evaluate_3ci_AMM(a1, a3, a2)

        R_c_and_offset_a = self.apr.get(a2, a3)
        if R_c_and_offset_a is None:
            return None

        # We do not support q-points yet
        nq = len(self.ibzq_qc)

        W_YM_expansion = self.W_YM_expansions.get(a1, a2)
        obj = W_qYM = W_YM_expansion.zeros((nq,), dtype=self.dtype)

        for R_c, offset in R_c_and_offset_a:
            norm = np.linalg.norm(R_c)
            phases = NullPhases(self.ibzq_qc, offset)
            disp = AtomicDisplacement(None, a2, a3, R_c, offset, phases)
            disp.evaluate_overlap(W_YM_expansion, W_qYM)

        setup1 = self.setups[a1]

        M_a = self.M_a

        W_YM = W_qYM[0]
        print("W_XM", a1,a2,a3,W_YM)


        local_I_AMM = np.zeros( (setup1.Na, M_a[a2+1]-M_a[a2], M_a[a3+1]-M_a[a3]) ) 
        # 1) Loop over L
        A = 0
        X = 0
        Astart = 0
        M2start = 0 
        for ja, auxt in enumerate(setup1.auxt_j):
            la = auxt.l
            M1start = 0
            for j1, phit1 in enumerate(setup1.phit_j):
                for l in range((la + phit1.l) % 2, la + phit1.l + 1, 2):
                    for m in range(2*l+1):
                        LX = l**2 + m
                        for m1 in range(2*phit1.l+1):
                            M1 = M1start + m1
                            L1 = phit1.l**2 + m1
                            for mA in range(2*lg+1):
                                LA = lg**2 + mA
                                A = Astart + mA
                                local_I_AMM[A, M1, :] += G_LLL[LA,L1,LX] * W_YM[X, :]
                        X += 1
                M1start += 2*phit1.l+1
            Astart += 2*lg+1

        return local_I_AMM

    def evaluate_3ci_LMM(self, a1, a2, a3):
        print('Calling', a1,a2,a3)
        if a1 != a2:
            return self.evaluate_3ci_LMM(a1, a3, a2)

        R_c_and_offset_a = self.apr.get(a2, a3)
        if R_c_and_offset_a is None:
            return None

        # We do not support q-points yet
        nq = len(self.ibzq_qc)

        W_XM_expansion = self.W_XM_expansions.get(a1, a2)
        obj = W_qXM = W_XM_expansion.zeros((nq,), dtype=self.dtype)

        for R_c, offset in R_c_and_offset_a:
            norm = np.linalg.norm(R_c)
            phases = NullPhases(self.ibzq_qc, offset)
            disp = AtomicDisplacement(None, a1, a3, R_c, offset, phases)
            disp.evaluate_overlap(W_XM_expansion, W_qXM)

        setup1 = self.setups[a1]
        M_a = self.M_a

        W_XM = W_qXM[0]
        print("W_XM", a1,a2,a3,W_XM)
        local_I_AMM = np.zeros( ( (self.lmax+1)**2, M_a[a2+1]-M_a[a2], M_a[a3+1]-M_a[a3]) ) 
        # 1) Loop over L
        A = 0
        X = 0
        Astart = 0
        M2start = 0 
        for lg, gaux in enumerate(setup1.gaux_l):
            M1start = 0
            for j1, phit1 in enumerate(setup1.phit_j):
                for l in range((lg + phit1.l) % 2, lg + phit1.l + 1, 2):
                    for m in range(2*l+1):
                        LX = l**2 + m
                        for m1 in range(2*phit1.l+1):
                            M1 = M1start + m1
                            L1 = phit1.l**2 + m1
                            for mA in range(2*lg+1):
                                LA = lg**2 + mA
                                A = Astart + mA
                                local_I_AMM[A, M1, :] += G_LLL[LA,L1,LX] * W_XM[X, :]
                        X += 1

                M1start += 2*phit1.l+1
            Astart += 2*lg+1

        return local_I_AMM

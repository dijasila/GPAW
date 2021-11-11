import numpy as np
from gpaw.lcao.tci import get_cutoffs, split_setups_to_types, AtomPairRegistry,\
                          get_lvalues

from gpaw.auxlcao.utilities import get_compensation_charge_splines,\
                                   get_wgauxphit_product_splines
from gpaw.lcao.overlap import (FourierTransformer, TwoSiteOverlapCalculator,
                               ManySiteOverlapCalculator,
                               AtomicDisplacement, NullPhases)

class MatrixElements:
    def __init__(self, lmax=2):
        self.lmax = lmax

    def initialize(self, density, ham, wfs):
        setups = wfs.setups

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
            setup.gaux_l, setup.wgaux_j = get_compensation_charge_splines(setup, self.lmax, rcmax)
            setup.wauxtphit_x = get_wgauxphit_product_splines(setup, setup.wgaux_j, setup.phit_j, rcmax)

        transformer = FourierTransformer(rcmax=max(phit_rcmax_I)+1e-3, ng=2**14) # XXX Used to be 2**10, add accuracy for debug purposes
        tsoc = TwoSiteOverlapCalculator(transformer)
        msoc = ManySiteOverlapCalculator(tsoc, I_a, I_a)

        wauxtphit_Ix = [ setup.wauxtphit_x for setup in setups_I]
        phit_Ij = [ setup.phit_j for setup in setups_I] 

        phit_Ijq = msoc.transform(phit_Ij)
        wauxtphit_Ixq = msoc.transform(wauxtphit_Ix)
	
        wauxtphit_l_Ix = get_lvalues(wauxtphit_Ix)
        phit_l_Ij = get_lvalues(phit_Ij)

        self.W_XM_expansions = msoc.calculate_expansions(wauxtphit_l_Ix, wauxtphit_Ixq,
                                                         phit_l_Ij, phit_Ijq)

    def set_positions_and_cell(self, spos_ac, cell_cv, pbc_c, ibzq_qc, dtype):
        self.spos_ac = spos_ac
        self.cell_cv = cell_cv
        self.pbc_c = pbc_c
        self.ibzq_qc = ibzq_qc
        self.dtype = dtype
      
        # Build an atom pair registry for basis function overlap
        self.apr = AtomPairRegistry(self.rcmax_a, pbc_c, cell_cv, spos_ac)

        self.a1a2_p = [ (a1,a2) for a1,a2 in self.apr.get_atompairs() if a1<=a2 ]

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

    def evaluate_3ci_LMM(self, a1, a2, a3):
        R_c_and_offset_a = self.apr.get(a2, a3)
        if R_c_and_offset_a is None:
            return None

        # We do not support q-points yet
        nq = len(self.ibzq_qc)

        W_XM_expansion = self.W_XM_expansions.get(a1, a2)
        obj = W_qXM = W_XM_expansion.zeros((nq,), dtype=self.dtype)
        print(obj)
        print(a1,a2,a3)
        print(R_c_and_offset_a)
        raise NotImplementedError
        return 0.0

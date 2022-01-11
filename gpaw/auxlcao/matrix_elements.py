import numpy as np
from gpaw.lcao.tci import get_cutoffs, split_setups_to_types, AtomPairRegistry,\
                          get_lvalues

from gpaw.auxlcao.utilities import get_compensation_charge_splines,\
                                   get_compensation_charge_splines_screened,\
                                   get_wgauxphit_product_splines,\
                                   get_auxiliary_splines,\
                                   get_auxiliary_splines_screened,\
                                   safe_inv
from gpaw.auxlcao.procedures import get_A_a
from gpaw.lcao.overlap import (FourierTransformer, TwoSiteOverlapCalculator,
                               ManySiteOverlapCalculator,
                               AtomicDisplacement, NullPhases, BlochPhases)
from gpaw.auxlcao.screenedcoulombkernel import ScreenedCoulombKernel

from gpaw.gaunt import gaunt

G_LLL = gaunt(3) # XXX


r"""

                 sr.       lr.    
     φ (r)    = φ (r)  +  φ (r)  
       A         A         A      


      sr.                 
     φ (r) =  φ (r) - M  g (r)
      A        A       A  L_A    


      lr.         
     φ (r) =  M  g (r)
      A        A  L_A    


            /  lr   ||  lr    \
     W   =  | φ (r) || φ (r') |
      LL'   \  L    ||  L'    /

             /   sr.   ||  lr.    \
     M    =  |  φ (r)  || g (r')  |
      AL     \   A     ||  L      /


             /   sr.   ||  sr.    \
     S    =  |  φ (r)  || φ (r')  |
      AA'    \   A     ||  A'     /


       S    P_AMM
        AA
------------------------------------------------

             SS    SL    LS    LL
 P_AMM   (  W   + W   + W   + W   )  P_AMM 
             AA    AA    AA    AA

 W_AL

             SL    LL
 P_AMM   (  W   + W   )  P_LMM 
             AL    AL     

             LS    LL
 P_LMM   (  W   + W   )  P_AMM 
             LA    LL     

                               LL      
 P_LMM   (                    W   )  P_LMM
                               LL


           '
  What if P     = P    + M   P
           LMM     LMM    LA  AMM


and
          SS
 P_AMM ( W   ) P_AMM
          AA

          SL
 P_AMM ( W   ) P'_LMM
          AL

  '       LS
 P_LMM ( W   ) P_AMM
          LA


           LL                   LL                     LL                        LL               LL
 P'_LMM ( W   ) P'_LMM =  P    W    P      + P    M   W    M    P    + P    M   W   P     + P    W   M   P
           LL              LMM  LL   LMM      AMM  AL  LL   LA   AMM    AMM  AL  LL  LMM     LMM  LL  LA  AMM


--------------------------------------
             LL        LL
            W   = M   W   M_LA
             AA    AL  LL

             LL
            W   = M   W
             AL    AL  LL

             LS        LS
            W   = M   W
             AA    AL  LA

             SL     SL
            W   =  W   M
             AA     LA  LA
-----------------------------------------------

             SS       LS    SL            LL
 P_AMM   (  W   + M  W   + W   M   + M   W   M   )  P_AMM 
             AA    AL LA    LA  LA    AL  LL  LA

             SL       LL
 P_AMM   (  W   + M  W    )  P_LMM 
             AL    AL LL     

             LS      LL
 P_LMM   (  W    +  W   M   )  P_AMM 
             LA      LL  AL   

                               LL      
 P_LMM   (                    W   )  P_LMM 
                               LL
----------------------------------

 
NO!


Only viable option is
-----------------------------------------------
      P_LMM 
     PS_LMM = M_LA P_AMM
      P_AMM


-------------------------
-------------------------
          SS    SL    LS
 P_AMM ( W   + W   + W    ) P_AMM
          AA    AA    AA

             SL  
 P_AMM   (  W    )  P_LMM 
             AL       

             LS  
 P_LMM   (  W    )  P_AMM 
             LA

                     LL
 ( PS_LMM + P_LMM ) W   ( PS_LMM + P_LMM)
                     LL


-------------------------
-------------------------
          SS  
 P_AMM ( W    ) P_AMM
          AA   

           SL    
 P_AMM (  W   PS_LMM ) 
           AL    

            LS    
 PS_LMM (  W    P_AMM ) 
            LA    
 
        SL
 P_AMM W   P_LMM
        AL


        LS
 P_LMM W   P_AMM
        LA


----------------------
         LL
 PS_LMM W    PS_LMM
         LL

         LL
 PS_LMM W    P_LMM
         LL

         LL
 PC_LMM W   PC_LMM
         LL

         LL
 PS_LMM W   PS_LMM 
         LL
-------------------
------------------



             SS    SL    LS    LL
 P_AMM   (  W   + W   + W   + W   )  P_AMM 
             AA    AA    AA    AA

             SL    LL
 P_AMM   (  W   + W   )  P_LMM 
             AL    AL     


             LS    LL
 P_LMM   (  W   + W   )  P_AMM 
             LA    LA     

                               LL      
 P_LMM   (                    W   )  P_LMM 
                               LL


---------------------------------------------
      P_LMM


      W_AA P_AMM

    """




class MatrixElements:
    def __init__(self, lmax=2, lcomp=2, screening_omega=0.0, threshold=None):
        assert threshold is not None
        self.lmax = lmax
        self.lcomp = lcomp
        self.screening_omega = screening_omega
        self.threshold = threshold

    def initialize(self, density, ham, wfs):
        self.setups = setups = wfs.setups
        self.wfs = wfs

        self.my_a = range(len(setups.M_a))

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

            if self.screening_omega != 0.0:
                setup.screened_coulomb = ScreenedCoulombKernel(setup.rgd, self.screening_omega)

                # Compensation charges
                setup.gaux_l, setup.wgaux_l, setup.W_LL = get_compensation_charge_splines_screened(setup, self.lmax, rcmax)

                # Auxiliary basis functions
                setup.auxt_j, setup.wauxt_j, setup.sauxt_j, setup.wsauxt_j, setup.M_j, setup.W_AA = \
                   get_auxiliary_splines_screened(setup, self.lmax, rcmax, threshold=self.threshold)

            else:
                # Compensation charges
                setup.gaux_l, setup.wgaux_l, setup.W_LL = get_compensation_charge_splines(setup, self.lmax, rcmax)

                # Auxiliary basis functions
                setup.auxt_j, setup.wauxt_j, setup.sauxt_j, setup.wsauxt_j, setup.M_j, setup.W_AA = \
                     get_auxiliary_splines(setup, self.lmax, rcmax, threshold=self.threshold)

            # Single center Hartree of compensation charges * one phit_j
            setup.wgauxphit_x = get_wgauxphit_product_splines(setup, setup.wgaux_l, setup.phit_j, rcmax)

            setup.Naux = sum([ 2*spline.l+1 for spline in setup.auxt_j])

            # Single center Hartree of auxiliary basis function * one phit_j
            setup.wauxtphit_x = get_wgauxphit_product_splines(setup, setup.wauxt_j, setup.phit_j, rcmax)

        transformer = FourierTransformer(rcmax=max(phit_rcmax_I)+1e-3, ng=2**10)
        tsoc = TwoSiteOverlapCalculator(transformer)
        msoc = ManySiteOverlapCalculator(tsoc, I_a, I_a)

        # Poisson of auxiliary function times a basis function expanded on product angular momentum channels
        wauxtphit_Ix = [ setup.wauxtphit_x for setup in setups_I]
        wauxtphit_Ixq = msoc.transform(wauxtphit_Ix)
        wauxtphit_l_Ix = get_lvalues(wauxtphit_Ix)

        # Poisson of gaussian function times a basis function expanded on product angular momentum channels
        wgauxphit_Ix = [ setup.wgauxphit_x for setup in setups_I]
        wgauxphit_Ixq = msoc.transform(wgauxphit_Ix)
        wgauxphit_l_Ix = get_lvalues(wgauxphit_Ix)

        # Basis functions
        phit_Ij = [ setup.phit_j for setup in setups_I] 
        phit_Ijq = msoc.transform(phit_Ij)
        phit_l_Ij = get_lvalues(phit_Ij)

        # Generalied gaussian functions
        ghat_Il = [ setup.ghat_l for setup in setups_I] 
        ghat_Ilq = msoc.transform(ghat_Il)
        ghat_l_Il = get_lvalues(ghat_Il)

        # Auxiliary functions
        auxt_Ij = [ setup.auxt_j for setup in setups_I] 
        auxt_Ijq = msoc.transform(auxt_Ij)
        auxt_l_Ij = get_lvalues(auxt_Ij)

        # Generalized gaussian screened auxiliary functions
        sauxt_Ij = [ setup.sauxt_j for setup in setups_I]
        sauxt_Ijq = msoc.transform(sauxt_Ij)
        sauxt_l_Ij = get_lvalues(sauxt_Ij)

        # Potential from generalized gaussian screened auxiliary functions
        wsauxt_Ij = [ setup.wsauxt_j for setup in setups_I] 
        wsauxt_Ijq = msoc.transform(wsauxt_Ij)
        wsauxt_l_Ij = get_lvalues(wsauxt_Ij)

        # Screened Coulomb integrals between auxiliary functions
        self.S_AA_expansions = msoc.calculate_expansions(wsauxt_l_Ij, wsauxt_Ijq,
                                                         sauxt_l_Ij, sauxt_Ijq)

        # Overlap between Poisson solution of screened auxiliary functions and gaussians
        self.M_AL_expansions = msoc.calculate_expansions(wsauxt_l_Ij, wsauxt_Ijq,
                                                         ghat_l_Il, ghat_Ilq)

        # X = Combined index of compensation l, phit j, selected gaunt expansion l channels, gaunt expansion m of those chanels
        self.W_XM_expansions = msoc.calculate_expansions(wgauxphit_l_Ix, wgauxphit_Ixq,
                                                         phit_l_Ij, phit_Ijq)

        # Y = Combined index of auxiliary j, phit j, selected gaunt expansion l channels, gaunt expansion m of those channels
        self.W_YM_expansions = msoc.calculate_expansions(wauxtphit_l_Ix, wauxtphit_Ixq,
                                                         phit_l_Ij, phit_Ijq)

        self.A_a = get_A_a( [ setup.auxt_j for setup in setups ] )
        Lsize = (self.lmax+1)**2
        self.L_a = np.arange(0, len(self.A_a)*Lsize, Lsize)

    def calculate(self, W_LL=None, W_AA=None, P_AMM=None, P_LMM=None, W_AL=None):
        self.calculate_sparse_W_LL(W_LL) 
        self.calculate_sparse_P_AMM(P_AMM)
        self.calculate_sparse_P_LMM(P_LMM, P_AMM)
        self.calculate_sparse_W_AL(W_AL)
        self.calculate_sparse_W_AA(W_AA, W_AL, W_LL)


    def calculate_sparse_W_LL(self, W_LL):
        for a, setup in enumerate(self.setups):
            W_LL += (a,a), setup.W_LL
    

    r"""
     Production implementation of two center auxiliary RI-V projection.

                                -1
               /       ||       \  /       ||             \
      P	     = | φ (r) || φ (r) |  | φ (r) || φ (r) φ (r) |
       AM1M2   \  A    ||  A'   /  \  A'   ||  M1    M2   /

       Where A and A' ∈  a(M1) ∪ a(M2).
    """
    def calculate_sparse_P_AMM(self, P_AMM):

        # Single center projections
        for a in self.my_a:
            iW_AA = safe_inv(self.setups[a].W_AA)
            Iloc_AMM = self.evaluate_3ci_AMM(a, a, a)
            print(iW_AA.shape, Iloc_AMM.shape, 'shapesxxx')
            P_AMM += (a,a,a), np.einsum('AB,Bij->Aij', iW_AA, Iloc_AMM)

        print('TODO: Two center projections')

        return

        # Two center projections
        for a1, (A1start, A1end, M1start, M1end) in enumerate(zip(A_a[:-1], A_a[1:], M_a[:-1], M_a[1:])):
            for a2, (A2start, A2end, M2start, M2end) in enumerate(zip(A_a[:-1], A_a[1:], M_a[:-1], M_a[1:])):
                if a1 == a2:
                    continue
                locW_AA = np.block( [ [ W_AA[A1start:A1end, A1start:A1end], W_AA[A1start:A1end, A2start:A2end] ],
                                      [ W_AA[A2start:A2end, A1start:A1end], W_AA[A2start:A2end, A2start:A2end] ] ] )
                iW_AA = safe_inv(locW_AA)
                I_AMM = [matrix_elements.evaluate_3ci_AMM(a1, a1, a2),
                         matrix_elements.evaluate_3ci_AMM(a2, a1, a2) ]
                if I_AMM[0] is None or I_AMM[1] is None:
                    continue
                I_AMM = np.vstack(I_AMM)
                Ploc_AMM = np.einsum('AB,Bij', iW_AA, I_AMM, optimize=True)
                P_AMM[A1start:A1end, M1start:M1end, M2start:M2end] += Ploc_AMM[:A1end-A1start]
                P_AMM[A2start:A2end, M1start:M1end, M2start:M2end] += Ploc_AMM[A1end-A1start:]

        P_kkAMM[(0,0)] = P_AMM
        return P_kkAMM

    r"""
     Production implementation of compensation charge projection
    
    
                   __     a      /  a    |       \  /  a  |       \
     P           = \     D      | p (r) | φ (r) |  | p   | φ (r) |
      (L,a)M1M2    /_     Li1i2  \  i1   |  M1   /  \  i2 |  M2   /
                   ai1i2


    """
    def calculate_sparse_P_LMM(self, P_LMM, P_AMM):
        for a, setup in enumerate(self.setups):
            P_Mi = self.wfs.atomic_correction.P_aqMi[a][0]
            Ploc_LMM = np.einsum('ijL,Mi,Nj->LMN', setup.Delta_iiL, P_Mi, P_Mi, optimize=True)

            for a1, (M1start, M1end) in enumerate(zip(self.M_a[:-1], self.M_a[1:])):
                for a2, (M2start, M2end) in enumerate(zip(self.M_a[:-1], self.M_a[1:])): 
                    P_LMM += (a, a1, a2), Ploc_LMM[:, M1start:M1end, M2start:M2end]
            #print('Modified compensation charges')

        for i, block_xx in P_AMM.block_i.items():
            a1, a2, a3 = i
            Ploc_LMM = np.zeros( (9,) + block_xx.shape[1:] )
            Aloc = 0
            setup = self.setups[a1]
            for auxt, M in zip(self.setups[a1].auxt_j, setup.M_j):
                S = 2*auxt.l+1
                L = (auxt.l)**2
                Ploc_LMM[L:L+S, :, :] = M * block_xx[Aloc:Aloc+S, :, :]
                Aloc += S
            P_LMM += (a1, a2, a3), Ploc_LMM

    def calculate_sparse_W_AL(self, W_AL):
        for a1 in self.my_a:
            for a2 in self.my_a:
                Mloc_qAL = self.evaluate_2ci_M_qAL(a1, a2)
                if Mloc_qAL is None:
                    continue
                W_AL += (a1, a2), Mloc_qAL[0]

    def calculate_sparse_W_AA(self, W_AA, W_AL, W_LL):
        A_a = self.A_a
        for a1 in self.my_a:
            NA1 = A_a[a1+1] - A_a[a1]
            for a2 in self.my_a:
                NA2 = A_a[a2+1] - A_a[a2]
                Mloc_qAA = self.evaluate_2ci_S_qAA(a1, a2)
                if Mloc_qAA is not None:
                    W_AA += (a1, a2), Mloc_qAA[0]

                """                
                Mloc_AL = self.evaluate_2ci_M_qAL(a1, a2)[0]
                if Mloc_AL is not None:
                    Mloc_AA = np.zeros( (NA1, NA2) )
                    setup = self.setups[a2]
                    A2 = 0
                    for M, auxt in zip(setup.M_j, setup.auxt_j):
                        for m2 in range(2*auxt.l+1):
                            L = auxt.l**2 + m2
                            if L < (self.lcomp+1)**2:
                                Mloc_AA[:, A2] = M*Mloc_AL[:, L]
                            A2 += 1
                    W_AA += (a1, a2), Mloc_AA[0]

                Mloc_LA = self.evaluate_2ci_M_qAL(a2, a1)[0].T
                if Mloc_LA is not None:
                    Mloc_AA = np.zeros( (NA1, NA2) )
                    setup = self.setups[a1]
                    A2 = 0
                    for M, auxt in zip(setup.M_j, setup.auxt_j):
                        for m2 in range(2*auxt.l+1):
                            L = auxt.l**2 + m2
                            if L < (self.lcomp+1)**2:
                                Mloc_AA[A2, :] = M*Mloc_LA[L, :]
                            A2 += 1
                """

    def set_positions_and_cell(self, spos_ac, cell_cv, pbc_c, ibz_qc, bzq_qc, dtype):
        self.spos_ac = spos_ac
        self.cell_cv = cell_cv
        self.pbc_c = pbc_c
        self.ibz_qc = ibz_qc
        self.dtype = dtype

        self.bzq_qc = bzq_qc # q-points
      
        # Build an atom pair registry for basis function overlap
        self.apr = AtomPairRegistry(self.rcmax_a, pbc_c, cell_cv, spos_ac)

        self.a1a2_p = [ (a1,a2) for a1,a2 in self.apr.get_atompairs() ]

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


    r"""

             /   sr.   ||  sr.    \
     S    =  |  φ (r)  || φ (r')  |
      AA'    \   A     ||  A'     /

    """

    def evaluate_2ci_S_qAA(self, a1, a2):
        R_c_and_offset_a = self.apr.get(a1, a2)
        if R_c_and_offset_a is None:
            return None

        # We do not support q-points yet
        nq = len(self.bzq_qc)

        S_AA_expansion = self.S_AA_expansions.get(a1, a2)
        obj = S_qAA = S_AA_expansion.zeros((nq,), dtype=self.dtype)

        if self.bzq_qc.any():
            get_phases = BlochPhases
        else:
            get_phases = NullPhases

        for R_c, offset in R_c_and_offset_a:
            norm = np.linalg.norm(R_c)
            phases = get_phases(self.bzq_qc, offset)
            disp = AtomicDisplacement(None, a1, a2, R_c, offset, phases)
            disp.evaluate_overlap(S_AA_expansion, S_qAA)

        return S_qAA


    r"""

             /   sr.   ||  lr.    \
     M    =  |  φ (r)  || g (r')  |
      AL     \   A     ||  L      /

    """

    def evaluate_2ci_M_qAL(self, a1, a2):
        R_c_and_offset_a = self.apr.get(a1, a2)
        if R_c_and_offset_a is None:
            return None

        nq = len(self.bzq_qc)

        M_AL_expansion = self.M_AL_expansions.get(a1, a2)
        obj = M_qAL = M_AL_expansion.zeros((nq,), dtype=self.dtype)

        if self.bzq_qc.any():
            get_phases = BlochPhases
        else:
            get_phases = NullPhases


        for R_c, offset in R_c_and_offset_a:
            norm = np.linalg.norm(R_c)
            phases = get_phases(self.bzq_qc, offset)
            disp = AtomicDisplacement(None, a1, a2, R_c, offset, phases)
            disp.evaluate_overlap(M_AL_expansion, M_qAL)

        return M_qAL


    r"""

             /       ||               \
     I    =  | φ (r) || φ (r') φ (r') |
      AMM    \  A    ||  M     M'     /,

      where double bar stands for Coulomb integral.

      a1 is always either a2 or a3.

      a2 and a3 are overlapping centers.

    """

    def evaluate_3ci_AMM(self, a1, a2, a3):
        if a1 != a2:
            if a1 != a3:
                raise NotImplementedError('Only 3-center integrals spanning up to 2 centers are supported. Got atom indices: (%d, %d, %d).' % (a1,a2,a3))
            # The first and third atom are the same. Swap second and third atom, so that first and second are same.
            locI_AMM = self.evaluate_3ci_AMM(a1, a3, a2)
            if locI_AMM is None:
                return None
            return np.transpose(locI_AMM, axes=(0,2,1))

        # From here on, it is quaranteed, that a1 and a2 are the same

        R_c_and_offset_a = self.apr.get(a2, a3)
        if R_c_and_offset_a is None:
            return None

        # We do not support q-points yet
        nq = len(self.ibz_qc)

        W_YM_expansion = self.W_YM_expansions.get(a2, a3)
        obj = W_qYM = W_YM_expansion.zeros((nq,), dtype=self.dtype)

        for R_c, offset in R_c_and_offset_a:
            norm = np.linalg.norm(R_c)
            phases = NullPhases(self.bzq_qc, offset)
            disp = AtomicDisplacement(None, a2, a3, R_c, offset, phases)
            #print(disp.overlap_without_phases(W_YM_expansion))
            disp.evaluate_overlap(W_YM_expansion, W_qYM)


        setup1 = self.setups[a1]

        M_a = self.M_a
        A_a = self.A_a

        W_YM = W_qYM[0]
        #print("W_XM", a1,a2,a3,W_YM)


        local_I_AMM = np.zeros( (A_a[a1+1]-A_a[a1], M_a[a2+1]-M_a[a2], M_a[a3+1]-M_a[a3]) )
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
                            for mA in range(2*auxt.l+1):
                                LA = auxt.l**2 + mA
                                A = Astart + mA
                                #print(G_LLL.shape, LA,L1,LX, W_YM.shape, X, A, M1, local_I_AMM.shape)
                                local_I_AMM[A, M1, :] += G_LLL[LA,L1,LX] * W_YM[X, :]
                        X += 1
                M1start += 2*phit1.l+1
            Astart += 2*auxt.l+1

        return local_I_AMM



    def evaluate_3ci_LMM(self, a1, a2, a3):
        #print('Calling', a1,a2,a3)
        if a1 != a2:
            return self.evaluate_3ci_LMM(a1, a3, a2)

        R_c_and_offset_a = self.apr.get(a2, a3)
        if R_c_and_offset_a is None:
            return None

        # We do not support q-points yet
        nq = len(self.bzq_qc)

        W_XM_expansion = self.W_XM_expansions.get(a1, a2)
        obj = W_qXM = W_XM_expansion.zeros((nq,), dtype=self.dtype)

        for R_c, offset in R_c_and_offset_a:
            norm = np.linalg.norm(R_c)
            phases = NullPhases(self.bzq_qc, offset)
            disp = AtomicDisplacement(None, a1, a3, R_c, offset, phases)
            disp.evaluate_overlap(W_XM_expansion, W_qXM)

        setup1 = self.setups[a1]
        M_a = self.M_a

        W_XM = W_qXM[0]
        #print("W_XM", a1,a2,a3,W_XM)
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

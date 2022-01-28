import numpy as np
from gpaw.lcao.tci import get_cutoffs, split_setups_to_types, AtomPairRegistry,\
                          get_lvalues
from gpaw.lfc import LFC
from ase.neighborlist import PrimitiveNeighborList
from gpaw.auxlcao.generatedcode import generated_W_LL_screening
from gpaw.auxlcao.generatedcode2 import generated_W_LL

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

G_LLL = gaunt(4) # XXX

class MatrixElements:
    def __init__(self, laux=2, lcomp=2, screening_omega=0.0, threshold=None):
        assert threshold is not None
        self.laux = laux
        self.lcomp = lcomp
        self.screening_omega = screening_omega
        self.threshold = threshold
    
    def direct_V_LL(self, a1, a2, disp_c):
        R_v = np.dot(self.spos_ac[a1, :] - self.spos_ac[a2, :] + disp_c, self.cell_cv)
        d = (R_v[0]**2 + R_v[1]**2 + R_v[2]**2)**0.5
        Lmax = (self.lcomp+1)**2
        locV_LL = np.zeros((Lmax,Lmax))
        if self.screening_omega != 0.0:
            generated_W_LL_screening(locV_LL, d, R_v[0], R_v[1], R_v[2], self.screening_omega)
        else:
            generated_W_LL(self.lcomp, locV_LL, d, R_v[0], R_v[1], R_v[2])
        if np.any(np.isnan(locV_LL)):
            print(a1,a2,locV_LL)
            xxx
        #print(a1, a2, locV_LL)
        return locV_LL

    def direct_W_AA(self, a1, a2, disp_c, return_parts=False):
        # Multipole reduced auxiliary   x    multipole reduced auxiliary
        disp_c = np.array(disp_c)
        S_AA = self.direct_S_AA(a1, a2, disp_c)
        V_LL = self.direct_V_LL(a1, a2, disp_c)
        M_AL = self.direct_M_AL(a1, a2, disp_c)
        M_LA = self.direct_M_AL(a2, a1, disp_c).T    
        m_AL = self.multipole_AL(a1)
        m_LA = self.multipole_AL(a2).T
        #print(m_AL,'m_AL')
        #print(m_LA,'m_LA')
        #print(M_AL,'M_AL')
        #print(M_LA,'M_LA')
        P1 = S_AA
        P2 = m_AL @ V_LL @ m_LA
        P3 = m_AL @ M_LA
        P4 = M_AL @ m_LA
        #print(a1,a2)
        #print(P1,'S_AA')
        #print(P2,'m_AL V_LL m_LA')
        #print(P3,'m_AL @ M_LA')
        #print(P4,'M_AL @ m_LA')
        #print('V_LL',V_LL)
        #print('tot', P1+P2+P3+P4)
        #print(a1,a2, P1.shape)
        if 0: #not return_parts:
            X1,X2,X3,X4 = self.direct_W_AA(a2, a1, -disp_c, return_parts=True)
            #print('P1',np.linalg.norm(X1-P1.T))
            #print(np.linalg.norm(X2-P2.T))
            #print(np.linalg.norm(X3-P3.T))
            #print(np.linalg.norm(X2-P3.T),'ow')
            #print(np.linalg.norm(X3-P2.T),'ow')
            #print('P4',np.linalg.norm(X4-P4.T))
            tot = P1+P2+P3+P4
            tot2 = X1+X2+X3+X4
            #print(np.linalg.norm(tot-tot2.T),'tot norm')
            with open('status.txt','a') as f:
                print('P1', file=f)
                np.savetxt(f, P1)
                print('X1', file=f)
                np.savetxt(f, X1)
                print('P2', file=f)
                np.savetxt(f, P2)
                print('X2', file=f)
                np.savetxt(f, X2)
                print('P3', file=f)
                np.savetxt(f, P1)
                print('X3', file=f)
                np.savetxt(f, X3)
                print('P4', file=f)
                np.savetxt(f, P4)
                print('X4', file=f)
                np.savetxt(f, X4)
            #input('asd')
            
        return P1+P2+P3+P4

    def direct_W_AL(self, a1, a2, disp_c):
        V_LL = self.direct_V_LL(a1, a2, disp_c)
        m_AL = self.multipole_AL(a1)
        return m_AL @ V_LL

    def direct_P_iM(self, a1, a2, disp_c):
        return self.direct_matrix_element(self.P_iM_expansions, a1, a2, disp_c)

    def direct_S_AA(self, a1, a2, disp_c):
        return self.direct_matrix_element(self.S_AA_expansions, a1, a2, disp_c)

    def direct_M_AL(self, a1, a2, disp_c):
        return self.direct_matrix_element(self.M_AL_expansions, a1, a2, disp_c)

    def multipole_AL(self, a):
        setup = self.setups[a]
        Lmax = (self.lcomp+1)**2
        #print(Lmax,'Lmax')
        m_AL = np.zeros( (setup.Naux, Lmax) )
        Aloc = 0
        for auxt, M in zip(setup.auxt_j, setup.M_j):
            if auxt.l > self.lcomp:
                continue
            S = 2*auxt.l+1
            L = (auxt.l)**2
            #print(S, M,'S M')
            m_AL[Aloc:Aloc+S, L:L+S] = np.eye(S) * M
            Aloc += S
        return m_AL


    def initialize(self, density, ham, wfs):
        self.timer = ham.timer
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

        with self.timer('Construct RI-bases'):
            for I, (setup, rcmax) in enumerate(zip(setups_I, phit_rcmax_I)):
                if self.screening_omega != 0.0:
                    setup.screened_coulomb = ScreenedCoulombKernel(setup.rgd, self.screening_omega)

                    # Compensation charges
                    setup.gaux_l, setup.wgaux_l, setup.W_LL = get_compensation_charge_splines_screened(setup, self.lcomp, rcmax)

                    # Auxiliary basis functions
                    setup.auxt_j, setup.wauxt_j, setup.sauxt_j, setup.wsauxt_j, setup.M_j, setup.W_AA = \
                        get_auxiliary_splines_screened(setup, self.lcomp, self.laux, rcmax, threshold=self.threshold)

                else:
                    # Compensation charges
                    setup.gaux_l, setup.wgaux_l, setup.W_LL = get_compensation_charge_splines(setup, self.lcomp, rcmax)

                    # Auxiliary basis functions
                    setup.auxt_j, setup.wauxt_j, setup.sauxt_j, setup.wsauxt_j, setup.M_j, setup.W_AA = \
                        get_auxiliary_splines(setup, self.lcomp, self.laux, rcmax, threshold=self.threshold)

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
        ghat_Il = [ setup.gaux_l for setup in setups_I] 
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

        # Projectors
        pt_Ij = [s.pt_j for s in setups_I]
        pt_Ijq = msoc.transform(pt_Ij)
        pt_l_Ij = get_lvalues(pt_Ij)

        with self.timer('Calcualte expansions'):
            # Projectors to pseudo basis functions
            self.P_iM_expansions = msoc.calculate_expansions(pt_l_Ij, pt_Ijq,
                                                             phit_l_Ij, phit_Ijq)

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
        Lsize = (self.lcomp+1)**2
        self.L_a = np.arange(0, len(self.A_a)*Lsize, Lsize)


    def calculate(self, W_LL=None, W_AA=None, P_AMM=None, P_LMM=None, W_AL=None, only_ghat=False, no_ghat=False,
                        only_ghat_aux_interaction=False):
        if only_ghat_aux_interaction:
            self.calculate_sparse_W_LL(W_LL) 
            self.calculate_sparse_W_AL(W_AL, W_LL)
            self.calculate_sparse_W_AA(W_AA)
            self.calculate_sparse_P_AMM(P_AMM, W_AA)
            self.calculate_sparse_P_LMM(P_LMM, P_AMM)
            W_LL.zero()
            W_AA.zero()
            return
        if no_ghat:
            #self.calculate_sparse_W_LL(W_LL) 
            self.calculate_sparse_P_AMM(P_AMM, W_AA)
            #self.calculate_sparse_P_LMM(P_LMM, P_AMM)
            #self.calculate_sparse_W_AL(W_AL)
            self.calculate_sparse_W_AA(W_AA)
            return
        if only_ghat:
            self.calculate_sparse_W_LL(W_LL)
            self.calculate_sparse_P_LMM(P_LMM, P_AMM)
            return

        with self.timer('W_LL'):
            self.calculate_sparse_W_LL(W_LL) 
        with self.timer('W_AL'):
            self.calculate_sparse_W_AL(W_AL, W_LL)
        with self.timer('W_AA'):
            self.calculate_sparse_W_AA(W_AA)
        with self.timer('P_AMM'):
            self.calculate_sparse_P_AMM(P_AMM, W_AA)
        with self.timer('P_LMM'):
            self.calculate_sparse_P_LMM(P_LMM, P_AMM)
        #print('disabled P_LMM?')
        return

    def calculate_sparse_W_LL(self, W_LL):
        if self.sparse_periodic:
            for a1, setup in enumerate(self.setups):
                a1R = (a1, (0, 0, 0))
                W_LL += (a1R, a1R), setup.W_LL
                a2_a, disp_xc = self.nl.get_neighbors(a1)
                for a2, disp_c in zip(a2_a, disp_xc):
                    if a2 == a1 and not np.any(disp_c):
                        continue
                    W_LL += ( (a1, (0,0,0) ), (a2, tuple(disp_c)) ), self.direct_V_LL(a1, a2, disp_c)
                    W_LL += ( (a2, (0,0,0) ), (a1, tuple(disp_c)) ), self.direct_V_LL(a2, a1, disp_c)
        else:
            for a1, setup in enumerate(self.setups):
                W_LL += (a1,a1), setup.W_LL
                a2_a, disp_xc = self.nl.get_neighbors(a1)
                for a2, disp_c in zip(a2_a, disp_xc):
                    if a2 == a1 and not np.any(disp_c):
                        continue
                    assert not np.any(disp_c) # This is a non periodic system
                    W_LL += (a1, a2), self.direct_V_LL(a1, a2, disp_c)
                    W_LL += (a2, a1), self.direct_V_LL(a2, a1, disp_c)


    r"""
     Production implementation of two center auxiliary RI-V projection.

                                -1
               /       ||       \  /       ||             \
      P	     = | φ (r) || φ (r) |  | φ (r) || φ (r) φ (r) |
       AM1M2   \  A    ||  A'   /  \  A'   ||  M1    M2   /

       Where A and A' ∈  a(M1) ∪ a(M2).
    """
    def calculate_sparse_P_AMM(self, P_AMM, Wfull_AA):
        # Single center projections
        for a in self.my_a:
            iW_AA = safe_inv(self.setups[a].W_AA)
            Iloc_AMM = self.evaluate_3ci_AMM(a, a, a)
            locP_AMM = np.einsum('AB,Bij->Aij', iW_AA, Iloc_AMM)
            #print(np.max(locP_AMM.ravel()),'loc P_AMM')
            if self.sparse_periodic:
                aR = (a, (0, 0, 0))
                P_AMM += (aR, aR, aR), locP_AMM
            else:
                P_AMM += (a,a,a), locP_AMM

            for a2 in self.my_a:
                if a == a2:
                    continue
                #f = open('asd.txt','a')
                #print('Not doing two center projections')
                             
                locW_AA = np.block( [ [ self.setups[a].W_AA,      self.direct_W_AA(a, a2, [0,0,0]) ],
                                      [ self.direct_W_AA(a2, a, [0,0,0]), self.setups[a2].W_AA     ] ] )
                #np.savetxt(f, locW_AA)
                #f.write('^locW_AA\n')
                #print(locW_AA, locW_AA.shape)
                
                iW_AA = safe_inv(locW_AA)
                #np.savetxt(f, iW_AA)
                #f.write('^iW_AA\n')

                I_AMM = [self.evaluate_3ci_AMM(a, a, a2),
                         self.evaluate_3ci_AMM(a2, a, a2) ]
                #for A in range(I_AMM[0].shape[0]):
                #    f.write('%d' % A)
                #    np.savetxt(f, I_AMM[0][A])
                #f.write('^I_AMM[0]\n')
                #for A in range(I_AMM[1].shape[0]):
                #    f.write('%d' % A)
                #    np.savetxt(f, I_AMM[1][A])
                #f.write('^I_AMM[1]\n')

                #print(I_AMM, 'I_AMM')
                if I_AMM[0] is None or I_AMM[1] is None:
                    continue
                I_AMM = np.vstack(I_AMM)
                #print(I_AMM, 'I_AMM stacked')
                locP_AMM = np.einsum('AB,Bij', iW_AA, I_AMM, optimize=True)
                #print(iW_AA,'iW_AA')
                #print(locP_AMM,'locP_AMM')
                NA1 = len(self.setups[a].W_AA)
                NA2 = len(self.setups[a2].W_AA)
                M1 = self.M_a[a+1]-self.M_a[a]
                #print(np.max(locP_AMM.ravel()),'2sit P_AMM')
                #f.close()
                if not self.sparse_periodic:
                    P_AMM += (a, a, a2),   locP_AMM[:NA1, :, :]
                    P_AMM += (a2, a, a2),  locP_AMM[NA1:, :, :]
                else:
                    R0 = (0,0,0)
                    P_AMM += ((a,R0), (a,R0), (a2,R0)),   locP_AMM[:NA1, :, :]
                    P_AMM += ((a2,R0), (a,R0), (a2,R0)),  locP_AMM[NA1:, :, :]
                #input("Press Enter to continue...")

        """
        for a1, (A1start, A1end, M1start, M1end) in enumerate(zip(A_a[:-1], A_a[1:], M_a[:-1], M_a[1:])):
            for a2, (A2start, A2end, M2start, M2end) in enumerate(zip(A_a[:-1], A_a[1:], M_a[:-1], M_a[1:])):


                R_c_and_offset_a = self.apr.get(a1, a2)
                for R_c, offset in R_c_and_offset_a:
                    locW_AA = np.block( [ [ self.setups[a].W_AA, W_AA[A1start:A1end, A2start:A2end] ],
                                          [ W_AA[A2start:A2end, A1start:A1end], W_AA[A2start:A2end, A2start:A2end] ] ] )
                    

                raise NotImplementedError('TODO: Two center projections')

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
        """

    r"""
     Production implementation of compensation charge projection
    
    
                   __     a      /  a    |       \  /  a  |       \
     P           = \     D      | p (r) | φ (r) |  | p   | φ (r) |
      (L,a)M1M2    /_     Li1i2  \  i1   |  M1   /  \  i2 |  M2   /
                   ai1i2


    """


    r"""

        direct_W_AA is the full coulomb matrix element between two auxiliary basis functions

        In practice, the basis function is split into multipole screened short range part,
        and a generalized gaussian.

                      sr.       lr.    
          φ (r)    = φ (r)  +  φ (r)  
            A         A         A      


           sr.                 
          φ (r) =  φ (r) - M   g (r)
           A        A       A   L_A    

        
        We have following matrix elements available

                  /   sr.   ||  sr.    \
          S    =  |  φ (r)  || φ (r')  |
           AA'    \   A     ||  A'     /,
          

                  /   sr.   ||  lr.    \
          M    =  |  φ (r)  || g (r')  |
           AL     \   A     ||  L      /,

   
        and

                  /   lr.   ||  lr.    \
          V    =  |  g (r)  || g (r')  |
           LL     \   L     ||  L      /.
 
        Thus, we find that

         W   = S   + M   m    + m   M   + V  .
          AA    SS    AL  LA     AL  LA    LL

        where m   maps the multipole moments.
               LA

    """


    def direct_matrix_element(self, expansions, a1, a2, disp_c):
        expansion = expansions.get(a1, a2)
        R_c = np.dot(self.spos_ac[a2, :] - self.spos_ac[a1, :] + disp_c, self.cell_cv)
        disp = AtomicDisplacement(None, a1, a2, R_c, None, None)
        return disp.evaluate_direct_without_phases(expansion)

    def fix_broken_ase_neighborlist(self, a):
        a_a, disp_xc = self.nl.get_neighbors(a)
        yield (a, [0,0,0])
        for a2, disp_c in zip(a_a, disp_xc):
            yield a2, disp_c
            yield a2, -disp_c

    def calculate_sparse_P_LMM(self, P_LMM, P_AMM):
        Lmax = (self.lcomp+1)**2
        if self.sparse_periodic:
            #P_LMM.show()
            for a1, setup in enumerate(self.setups):
                #print('corrections for atom', a1,'Note: Using too large neighbourlist')
                a2_a, disp_xc = self.nl.get_neighbors(a1)
                #print(a2_a, disp_xc,' Neighbourlist of', a1)

                P_aiM = {}
                for a2, disp2_c in self.fix_broken_ase_neighborlist(a1):
                    P2_iM = self.direct_P_iM(a1, a2, disp2_c)
                    for a3, disp3_c in self.fix_broken_ase_neighborlist(a1):
                        P3_iM = self.direct_P_iM(a1, a3, disp3_c) # XXX Too much calculation
                        Ploc_LMM = np.einsum('ijL,Mi,Nj->LMN', setup.Delta_iiL[:,:,:Lmax], P2_iM.T, P3_iM.T, optimize=True)
                        P_LMM += ( (a1, (0, 0, 0)), (a2, tuple(disp2_c)), (a3, tuple(disp3_c)) ), Ploc_LMM * (4*np.pi)**0.5
        else:
            for a, setup in enumerate(self.setups):
                P_Mi = self.wfs.atomic_correction.P_aqMi[a][0]
                Ploc_LMM = np.einsum('ijL,Mi,Nj->LMN', setup.Delta_iiL[:,:,:Lmax], P_Mi, P_Mi, optimize=True)
                for a1, (M1start, M1end) in enumerate(zip(self.M_a[:-1], self.M_a[1:])):
                    for a2, (M2start, M2end) in enumerate(zip(self.M_a[:-1], self.M_a[1:])):
                        P_LMM += (a, a1, a2), Ploc_LMM[:, M1start:M1end, M2start:M2end] * (4*np.pi)**0.5

        """
        return
        for i, block_xx in P_AMM.block_i.items():
            a1, a2, a3 = i
            Lmax = (self.lcomp+1)**2
            Ploc_LMM = np.zeros( (Lmax,) + block_xx.shape[1:] )
            Aloc = 0
            setup = self.setups[a1]
            for auxt, M in zip(self.setups[a1].auxt_j, setup.M_j):
                S = 2*auxt.l+1
                L = (auxt.l)**2
                Ploc_LMM[L:L+S, :, :] = M * block_xx[Aloc:Aloc+S, :, :]
                Aloc += S
            P_LMM += (a1, a2, a3), Ploc_LMM

        if no_ghat:
            return
    """

    def calculate_sparse_W_AL(self, W_AL, W_LL, no_ghat = False):
        if self.sparse_periodic:
            if no_ghat:
                return
            raise NotImplementedError
        
        for a1 in self.my_a:
            for a2 in self.my_a:
                Mloc_qAL = self.evaluate_2ci_M_qAL(a1, a2)
                if Mloc_qAL is not None:
                    W_AL += (a1, a2), Mloc_qAL[0]
    
                locW_LL = W_LL.get( (a1, a2) )
                #print(a1, locW_LL)
                #print(self.multipole_AL(a1).shape)
                #print(locW_LL.shape,'s')

                W_AL += (a1, a2), ( self.multipole_AL(a1) @ locW_LL )

        """      
        mul_AL = W_AL.__class__('mul','AL')
        for a2 in self.my_a:
            setup = self.setups[a2]
            Lmax = (self.lcomp+1)**2
            loc_AL = np.zeros( (setup.Naux, Lmax) )
            Aloc = 0
            for auxt, M in zip(setup.auxt_j, setup.M_j):
                S = 2*auxt.l+1
                L = (auxt.l)**2
                loc_AL[Aloc:Aloc+S, L:L+S] = np.eye(S) * M #/ (4*np.pi)**0.5
                Aloc += S
            mul_AL += (a2,a2), loc_AL 

        #mul_AL.show()
        #W_LL.show()
        # Long range part of W_AL
        tmp= mul_AL.meinsum('mul_AL*W_LL', 'AL,LK->AK', mul_AL, W_LL)
        #tmp.show()
        W_AL += tmp
        """
    def calculate_sparse_W_AA(self, W_AA):
        if self.sparse_periodic:
            print('Fake non periodic W_AA data')
            for a1 in self.my_a:
                for a2 in self.my_a:
                    if a1 == a2:
                        W_AA += ( (a1,(0,0,0)), (a2,(0,0,0))), self.setups[a1].W_AA
                    else:
                        W_AA += ( (a1,(0,0,0)), (a2,(0,0,0))), self.direct_W_AA(a1, a2, [0,0,0])
        else:
            for a1 in self.my_a:
                for a2 in self.my_a:
                    if a1 == a2:
                        W_AA += (a1, a2), self.setups[a1].W_AA
                    else:
                        W_AA += (a1, a2), self.direct_W_AA(a1, a2, [0,0,0])
                        #temp = self.direct_W_AA(a1, a2, [0,0,0])
                        #temp2 = self.direct_W_AA(a2, a1, [0,0,0]).T
                        #print(temp, temp2, temp-temp2)
                        #print(a1,a2)
                        #input('wait')
        """
        return S_qAA

                    print('Skipping two atom W_AA')
                    #raise ValueError('Two atoms not yet supported')
        return
        A_a = self.A_a
        for a1 in self.my_a:
            NA1 = A_a[a1+1] - A_a[a1]
            for a2 in self.my_a:
                NA2 = A_a[a2+1] - A_a[a2]
                Mloc_qAA = self.evaluate_2ci_S_qAA(a1, a2)
                if Mloc_qAA is not None:
                    W_AA += (a1, a2), Mloc_qAA[0]

                                
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

        # Use ASE neighbour list to enumerate the supercells which need to be enumerated
        if self.screening_omega != 0.0:
            cutoff = 2.5 / self.screening_omega
        else:
            cutoff = np.max(np.sum(cell_cv**2, axis=1)**0.5)

        self.nl = PrimitiveNeighborList([ cutoff ] * len(spos_ac), skin=0,
                                        self_interaction=False, bothways=False,
                                        use_scaled_positions=True)

        #print('Updating neighbourlist with', spos_ac)
        self.nl.update(pbc=pbc_c, cell=cell_cv, coordinates=spos_ac)

        self.sparse_periodic = np.any(pbc_c)
        if self.sparse_periodic:
            print('Initializing periodic implementation of RI')

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

d      where double bar stands for Coulomb integral.

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


        local_I_AMM = np.zeros( (A_a[a1+1]-A_a[a1], M_a[a2+1]-M_a[a2], M_a[a3+1]-M_a[a3]), dtype=W_YM.dtype )
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



from typing import Tuple, Dict
import numpy as np
from gpaw.auxlcao.algorithm import RIAlgorithm
from gpaw.hybrids.coulomb import ShortRangeCoulomb
from gpaw.grid_descriptor import GridDescriptor
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.pw.descriptor import PWDescriptor
from gpaw.lfc import LFC
from gpaw.utilities import pack, unpack2
from gpaw.pw.lfc import PWLFC
from gpaw.transformers import Transformer
from gpaw.response.wstc import WignerSeitzTruncatedCoulomb as WSTC
import scipy
from gpaw.auxlcao.multipole import calculate_W_qLL
from gpaw.auxlcao.procedures import calculate_V_qAA,\
                                    calculate_S_qAA,\
                                    calculate_M_qAA,\
                                    calculate_P_kkAMM,\
                                    calculate_P_kkLMM,\
                                    calculate_W_qAL

from gpaw.auxlcao.matrix_elements import MatrixElements

from collections import defaultdict

import matplotlib.pyplot as plt
from numpy.matlib import repmat


"""

      W_LL W_LA     [ 
      _AL W_AA

"""


flops = 0

def meinsum(output_name, index_str, T1, T2, timer=None):
    input_index_str, output_index_str = index_str.split('->')
    index1, index2 = input_index_str.split(',') 
    contraction_indices = list(set(index1)&set(index2))

    outref = []
    output_index_types = ''
    for idx in output_index_str:
        t1_idx = index1.find(idx)
        if t1_idx>=0:
            outref.append( (1, t1_idx) )
            output_index_types += T1.indextypes[t1_idx]
        else:
            t2_idx = index2.find(idx)
            outref.append( (2, t2_idx) )
            output_index_types += T2.indextypes[t2_idx]

    T3 = SparseTensor(output_name, output_index_types)

    #print('%s[%s] = %s[%s] * %s[%s] ' % (T3.get_name(), output_index_str, T1.get_name(), index1, T2.get_name(), index2))

    T1_i = tuple([ index1.find(idx) for idx in contraction_indices ])
    T2_i = tuple([ index2.find(idx) for idx in contraction_indices ])

    def get_out(indices1, indices2):
        s = []
        for id, index in outref:
            if id == 1:
                s.append(indices1[index])
            else:
                s.append(indices2[index])
        return tuple(s)

    for i1, block1 in T1.block_i.items():
        for i2, block2 in T2.block_i.items():
            fail = False
            for ii1, ii2 in zip(T1_i, T2_i):
                if i1[ii1] != i2[ii2]:
                    fail = True
            if fail:
                continue
            out_indices = get_out(i1, i2)
            #print('Einsum', i1, i2, index_str, block1.shape, block2.shape)
            #print('in',block1, 'in2', block2)
            if timer:
                with timer('einsum'):
                    value = np.einsum(index_str, block1, block2)
            else:
                value = np.einsum(index_str, block1, block2)
            #print('out', np.max(np.abs(value)))
            #input('prod')
            T3 += out_indices, value
    return T3


class SparseTensor:
    def __init__(self, name, indextypes):
        self.name = name
        self.indextypes = indextypes
        self.zero()
        self.meinsum = meinsum
        self.dtype = None

    def zero(self):
        self.block_i = defaultdict(float)        

    def to_full2d(self, M1_a, M2_a):
        T_MM = np.zeros( (M1_a[-1], M2_a[-1]), dtype=self.dtype )
        for index, block_xx in self.block_i.items():
            a1, a2 = index
            T_MM[ M1_a[a1]:M1_a[a1+1], M2_a[a2]:M2_a[a2+1] ] += block_xx
        return T_MM

    def to_full3d(self, M1_a, M2_a, M3_a):
        T_MMM = np.zeros( (M1_a[-1], M2_a[-1], M3_a[-1]), dtype=self.dtype )
        for index, block_xx in self.block_i.items():
            a1, a2,a3 = index
            T_MMM[ M1_a[a1]:M1_a[a1+1], M2_a[a2]:M2_a[a2+1], M3_a[a3]:M3_a[a3+1] ] += block_xx
        return T_MMM

    def to_full3d_R(self, M1_a, M2_a, M3_a):
        T_MMM = np.zeros( (M1_a[-1], M2_a[-1], M3_a[-1]), dtype=self.dtype )
        for index, block_xx in self.block_i.items():
            print(index)
            (a1, R1), (a2, R2),(a3, R3) = index
            T_MMM[ M1_a[a1]:M1_a[a1+1], M2_a[a2]:M2_a[a2+1], M3_a[a3]:M3_a[a3+1] ] += block_xx
        return T_MMM

    def __iadd__(self, index_and_block):
        if isinstance(index_and_block, SparseTensor):
            for index, block_xx in index_and_block.block_i.items():
                if self.dtype is None:
                    self.dtype = block_xx.dtype
                self.block_i[index] += block_xx.copy()
        else:
            index, block_xx = index_and_block
            if self.dtype is None:
                self.dtype = block_xx.dtype
            #print('added', index, block_xx, self.name)
            self.block_i[index] += block_xx.copy() #xxx

        return self

    def get(self, index):
        if index in self.block_i:
            return self.block_i[index]

    def get_name(self):
        return '%s_%s' % (self.name, self.indextypes)

    def show(self):
        s = self.get_name() + ' {\n'
        for i, block in self.block_i.items():
            s += '   ' + repr(i) + ' : ' + repr(block.shape) + ' ' + repr(block.dtype) + ' ' + repr(block) +'\n'
        s += '}\n'
        print(s)
       

    def __repr__(self):
        s = self.get_name() + ' {\n'
        for i, block in self.block_i.items():
            s += '   ' + repr(i) + ' : ' + repr(block.shape) + ' ' + repr(block.dtype) + '\n'
        s += '}\n'
        return s        

"""
Phase 1: Generate all displacements
"""

class P_AMM_generator:
    def __init__(self, P_AMM):
        self.matrix_elements = matrix_elements

    #def select_by_M2(self, M2set):
        
    

"""
    def get(self, index):
        if self.cache_i.haskey(index):
            return self.cache_i[index]
        else:
            value = self.evaluate(index)
            self.cache[index] = value
            return value

    def lazy_evaluate(self, index):
        raise NotImplementedError

    def get_index_list(self):
        raise NotImplementedError
"""

class RIBasisMaker:
    def __init__(self, setup):
        self.setup = setup

class Struct:
    pass

class RIBase(RIAlgorithm):
    def __init__(self, name, exx_fraction=None, screening_omega=None, lcomp=2, laux=2, threshold=1e-5):
        RIAlgorithm.__init__(self, name, exx_fraction, screening_omega)

        self.lcomp = lcomp
        self.laux = laux 

        self.threshold = threshold
        #print('ribase', self.threshold)
        #print('RI base screening omega', screening_omega)
        self.matrix_elements = MatrixElements(laux = self.laux, lcomp=self.lcomp, screening_omega = screening_omega, threshold=threshold)

class RIR(RIBase):
    def __init__(self, exx_fraction=None, screening_omega=None, lcomp=2, laux=2, threshold=1e-5):
        RIBase.__init__(self, 'RI-R', exx_fraction, screening_omega, lcomp, laux, threshold)
        self.only_ghat = False
        self.no_ghat = False
        self.only_ghat_aux_interaction = False

    def cube_debug(self, atoms, a1, M1, a2, M2):
        from ase.io.cube import write_cube
        from gpaw.lfc import LFC
        auxt_lfc_coarse = LFC(self.density.gd, [setup.auxt_j for setup in self.wfs.setups])
        auxt_lfc_coarse.set_positions(self.spos_ac)
        Q_aA = auxt_lfc_coarse.dict(zero=True)
        for a, setup in enumerate(self.wfs.setups):
            locP_AMM = self.P_AMM.get( (a, a1, a2) )
            if locP_AMM is None:
                print(a,a1,a2,'none')
                continue
            Aloc = 0
            for j, aux in enumerate(setup.auxt_j):
                for m in range(2*aux.l+1):
                    print(locP_AMM.shape,'pamm shape')
                    print(Q_aA,'Q_aA')
                    Q_aA[a][Aloc] = locP_AMM[Aloc, M1, M2]
                    Aloc += 1
        fit_G = self.density.gd.zeros()
        auxt_lfc_coarse.add(fit_G, Q_aA)
        write_cube(open('fit_a%03d_%03d_a%03d_%03d.cube' % (a1,M1, a2,M2),'w'), atoms, data=fit_G, comment='')

        wfs = self.wfs
        density = self.density
        nao = self.wfs.setups.nao
        # Put wave functions to grid
        phit1_MG = density.gd.zeros(nao)
        self.wfs.basis_functions.lcao_to_grid(np.eye(nao), phit1_MG, 0)
       
        write_cube(open('ref_a%03d_%03d_a%03d_%03d.cube' % (a1, M1, a2, M2),'w'), atoms, data=phit1_MG[self.M_a[a1]+M1] * phit1_MG[self.M_a[a2]+M2], comment='')

    def local_K_MMMM(self, a1, i1, a2, i2, a3, i3, a4, i4):
        M2start, M2end = self.M_a[a2], self.M_a[a2+1]
        M4start, M4end = self.M_a[a4], self.M_a[a4+1]
        rho_MM = SparseTensor('rho', 'MM')
        locrho_MM = np.zeros( (M2end-M2start, M4end-M4start) )
        locrho_MM[i2, i4] = 1.0
        rho_MM += (a2, a4), locrho_MM
        F_MM = self.contractions(rho_MM)
        return F_MM.get( (a1, a3) )[i1, i3]

    def ref_V_LL(self, a1, L1, a2, L2):
        finegd = self.density.finegd
        ghat = self.density.ghat
        Q1_aM = ghat.dict(zero=True)
        Q1_aM[a1][L1] = 1.0 / (4*np.pi)**0.5
        Q2_aM = ghat.dict(zero=True)
        Q2_aM[a2][L2] = 1.0 / (4*np.pi)**0.5

        g1_g = finegd.zeros()
        g2_g = finegd.zeros()
        ghat.add(g1_g, Q1_aM)
        ghat.add(g2_g, Q2_aM)
        v_g = finegd.zeros()
        self.hamiltonian.poisson.solve(v_g, g1_g, charge=None)
        return finegd.integrate(g2_g*v_g)

    def ref_W_AA(self, a1, L1, a2, L2):
        finegd = self.density.finegd

        Alfc = LFC(finegd, [setup.auxt_j for setup in self.density.setups])
        Alfc.set_positions(self.spos_ac)
        A1_aM = Alfc.dict(zero=True)
        A1_aM[a1][L1] = 1.0
        A2_aM = Alfc.dict(zero=True)
        A2_aM[a2][L2] = 1.0

        g1_g = finegd.zeros()
        g2_g = finegd.zeros()
        Alfc.add(g1_g, A1_aM)
        Alfc.add(g2_g, A2_aM)
        v_g = finegd.zeros()
        self.hamiltonian.poisson.solve(v_g, g1_g, charge=None)
        return finegd.integrate(g2_g*v_g)

    def ref_local_K_MMMM(self, a1, i1, a2, i2, a3, i3, a4, i4):
        setups = self.wfs.setups
        M1 = setups.M_a[a1] + i1
        M2 = setups.M_a[a2] + i2
        M3 = setups.M_a[a3] + i3
        M4 = setups.M_a[a4] + i4

        wfs = self.wfs
        density = self.density

        nao = self.wfs.setups.nao
        # Put wave functions to grid
        phit1_MG = density.gd.zeros(nao)
        self.wfs.basis_functions.lcao_to_grid(np.eye(nao), phit1_MG, 0)

        gd = self.density.gd
        finegd = self.density.finegd
        interpolator = self.density.interpolator
        restrictor = self.hamiltonian.restrictor

        if self.screening_omega != 0.0:
            pbcfinegd = GridDescriptor(self.density.finegd.N_c,
                                       self.density.finegd.cell_cv,
                                       pbc_c=True)
            coulomb = ShortRangeCoulomb(self.screening_omega)
            qd = KPointDescriptor([np.array([0.0,0.0,0.0])])
            pd12 = PWDescriptor(None, pbcfinegd, float, kd=qd) #10 Ha =  270 eV cutoff
            coulomb_G = coulomb.get_potential(pd12)

        def pair_density_and_potential(Ma, Mb, include_pseudo=True, include_comp=True):
            # Construct the product of two basis functions
            rhot_G = phit1_MG[Ma] * phit1_MG[Mb]

            rhot_g = finegd.zeros()
            interpolator.apply(rhot_G, rhot_g)

            #print('real space norm', finegd.integrate(rhot_g))

            Q_aL = {}
            D_ap = {}
            for a in self.wfs.P_aqMi:
                P1_i = self.wfs.P_aqMi[a][0][Ma]
                P2_i = self.wfs.P_aqMi[a][0][Mb]
                D_ii = np.outer(P1_i, P2_i.conjugate())
                D_p = pack(D_ii)
                D_ap[a] = D_p
                Q_aL[a] = np.dot(D_p, self.density.setups[a].Delta_pL)


            if not include_pseudo:
                rhot_g[:] = 0.0

            if include_comp:
                self.density.ghat.add(rhot_g, Q_aL)

            V_g = finegd.zeros()
            if self.screening_omega == 0.0:
                self.hamiltonian.poisson.solve(V_g, rhot_g, charge=None)
            else:
                pbcrhot_g = pbcfinegd.zeros()
                pbcrhot_g[:-1,:-1,:-1] = rhot_g
                rhot_G = pd12.fft(pbcrhot_g)
                Vrho_G = rhot_G * coulomb_G
                V_g[:] = pd12.ifft(Vrho_G)[:-1,:-1,:-1]

            return rhot_g, V_g

        if self.only_ghat_aux_interaction:
            rhot12_g, V12_g = pair_density_and_potential(M1, M2, include_pseudo=False)
            rhot34_g, V34_g = pair_density_and_potential(M3, M4, include_comp=False)
            K = finegd.integrate(rhot12_g*V34_g)
            rhot12_g, V12_g = pair_density_and_potential(M1, M2, include_comp=False)
            rhot34_g, V34_g = pair_density_and_potential(M3, M4, include_pseudo=False)
            K += finegd.integrate(rhot12_g*V34_g)
        elif self.only_ghat:
            rhot12_g, V12_g = pair_density_and_potential(M1, M2, include_pseudo=False)
            rhot34_g, V34_g = pair_density_and_potential(M3, M4, include_pseudo=False)
            K = finegd.integrate(rhot12_g*V34_g)
        elif self.no_ghat:
            rhot12_g, V12_g = pair_density_and_potential(M1, M2, include_comp=False)
            rhot34_g, V34_g = pair_density_and_potential(M3, M4, include_comp=False)
            K = finegd.integrate(rhot12_g*V34_g)
        else:
            rhot12_g, V12_g = pair_density_and_potential(M1, M2)
            rhot34_g, V34_g = pair_density_and_potential(M3, M4)
            K = finegd.integrate(rhot12_g*V34_g)

        altK = finegd.integrate(rhot34_g*V12_g)
        if abs(K-altK)>1e-5:
            print('Warning ', K, altK)
        return K

    def prepare_setups(self, setups):
        RIAlgorithm.prepare_setups(self, setups)
        self.M_a = setups.M_a.copy()
        self.M_a.append(setups.nao)

    def set_positions(self, spos_ac):
        RIAlgorithm.set_positions(self, spos_ac)

        Na = len(spos_ac)

        with self.timer('RI-V: Auxiliary Fourier-Bessel initialization'):
            self.matrix_elements.initialize(self.density, self.hamiltonian, self.wfs)

        gd = self.hamiltonian.gd
        ibzq_qc = np.array([[0.0, 0.0, 0.0]])
        bzq_qc = np.array([[0.0, 0.0, 0.0]])
        dtype = self.wfs.dtype
        with self.timer('RI Matrix elements set positions'):
            self.matrix_elements.set_positions_and_cell(spos_ac,
                                                        gd.cell_cv,
                                                        gd.pbc_c,
                                                        ibzq_qc,
                                                        bzq_qc, # q-points
                                                        dtype)

        self.P_AMM = SparseTensor('P', 'AMM')
        self.P_LMM = SparseTensor('P', 'LMM')
        #self.S_LMM = SparseTensor('S', 'LMM')
        self.W_AA = SparseTensor('W', 'AA')
        self.W_AL = SparseTensor('W', 'AL')
        self.W_LL = SparseTensor('W', 'LL')

        with self.timer('Calculate matrix elements'):
            # Order does not matter, the data is deduces from the name
            self.matrix_elements.calculate(W_AA = self.W_AA,
                                           W_AL = self.W_AL, 
                                           W_LL = self.W_LL,
                                           P_AMM = self.P_AMM,
                                           P_LMM = self.P_LMM, only_ghat=self.only_ghat, no_ghat=self.no_ghat,
                                           only_ghat_aux_interaction=self.only_ghat_aux_interaction)

        if self.no_ghat:
            self.P_LMM.zero()

        with self.timer('Pre contractions'):
            self.WP_AMM = meinsum('WP', 'AB,Bij->Aij', self.W_AA, self.P_AMM, self.timer)
            self.WP_AMM += meinsum('WP', 'AB,Bij->Aij', self.W_AL, self.P_LMM, self.timer)

            self.WP_LMM = meinsum('WP', 'BA,Bij->Aij', self.W_AL, self.P_AMM, self.timer)
            self.WP_LMM += meinsum('WP', 'AB,Bij->Aij', self.W_LL, self.P_LMM, self.timer)

            A_a = self.matrix_elements.A_a
            M_a = self.matrix_elements.M_a
            L_a = self.matrix_elements.L_a
            if not self.matrix_elements.sparse_periodic:
                self.WP_AMM = self.WP_AMM.to_full3d(A_a, M_a, M_a)
                self.WP_LMM = self.WP_LMM.to_full3d(L_a, M_a, M_a)
                self.P_AMM = self.P_AMM.to_full3d(A_a, M_a, M_a)
                self.P_LMM = self.P_LMM.to_full3d(L_a, M_a, M_a)
            else:
                self.WP_AMM = self.WP_AMM.to_full3d_R(A_a, M_a, M_a)
                self.WP_LMM = self.WP_LMM.to_full3d_R(L_a, M_a, M_a)
                self.P_AMM = self.P_AMM.to_full3d_R(A_a, M_a, M_a)
                self.P_LMM = self.P_LMM.to_full3d_R(L_a, M_a, M_a)
 

    def calculate_exchange_per_kpt_pair(self, kpt1, k_c, rho1_MM, kpt2, krho_c, rho2_MM):
        if 0:
            rho_MM = SparseTensor('rho', 'MM')
            for a1, (M1start, M1end) in enumerate(zip(self.M_a[:-1], self.M_a[1:])):
                for a2, (M2start, M2end) in enumerate(zip(self.M_a[:-1], self.M_a[1:])):
                    block = rho2_MM[M1start:M1end, M2start:M2end]
                    if self.matrix_elements.sparse_periodic:
                        rho_MM += ( ((a1,(0,0,0)), (a2,(0,0,0))), block.copy() )
                    else:
                        rho_MM += ( (a1, a2), block )

        with self.timer('Contractions'):
            #F_MM = self.contractions(rho_MM)
            fock_MM = -(self.nspins/2.)*self.exx_fraction*self.fullcontractions(rho2_MM)

        if 0:
            fock_MM = np.zeros_like(rho2_MM)
            if self.matrix_elements.sparse_periodic:
                for index, block_xx in F_MM.block_i.items():
                    (a1,disp1_c), (a2,disp2_c) = index
                    if np.any(disp1_c):
                        continue
                    if np.any(disp1_c):
                        continue # Emulate non-periodic system
                    #a1, a2 = index
                    M1start, M1end = self.M_a[a1], self.M_a[a1+1]
                    M2start, M2end = self.M_a[a2], self.M_a[a2+1]
                    fock_MM[M1start:M1end, M2start:M2end] += -0.5*self.exx_fraction*block_xx
                    print('Implicit gamma', a1, disp1_c, a2, disp2_c)
            else:
                for index, block_xx in F_MM.block_i.items():
                    a1, a2 = index
                    M1start, M1end = self.M_a[a1], self.M_a[a1+1]
                    M2start, M2end = self.M_a[a2], self.M_a[a2+1]
                    fock_MM[M1start:M1end, M2start:M2end] += -0.5*self.exx_fraction*block_xx
   
        evv = 0.5 * np.einsum('ij,ij', fock_MM, rho1_MM, optimize=True)
        #print(fock_MM)
        #print('Overlap eig', scipy.linalg.eigh(kpt2.S_MM))
        #print('diagonal density matrix', kpt2.C_nM @ kpt2.S_MM @ rho2_MM @ kpt2.S_MM @ kpt2.C_nM.T )
        #print(np.linalg.norm(fock_MM-fock_MM.T),'Fock non-symmetric')
        #print(np.linalg.norm(rho2_MM-rho2_MM.T),'rho non-symmetric')
        #with open('fock.txt','a') as f:
        #    np.savetxt(f,fock_MM)
        #fock_MM = (fock_MM + fock_MM.T) /2

        return evv, fock_MM

    def fullcontractions(self, rho_MM):
        with self.timer('RI-V: 1st contraction AMM MM'):
            WP_AMM_RHO_MM = np.einsum('Ajl,kl->Ajk',
                                       self.WP_AMM,
                                       rho_MM)
        with self.timer('RI-V: 2nd contraction AMM AMM'):
            F_MM = np.einsum('Aik,Ajk->ij',
                              self.P_AMM,
                              WP_AMM_RHO_MM)
            WP_AMM_RHO_MM = None

        with self.timer('RI-V: 1st contraction LMM MM'):
            WP_LMM_RHO_MM = np.einsum('Ajl,kl->Ajk',
                                    self.WP_LMM,
                                    rho_MM)

        with self.timer('RI-V: 2nd contraction LMM LMM'):
            F_MM += np.einsum('Aik,Ajk->ij',
                             self.P_LMM,
                             WP_LMM_RHO_MM)
            WP_LMM_RHO_MM = None

        return F_MM

    def contractions(self, rho_MM):
        with self.timer('RI-V: 1st contraction AMM MM'):
            WP_AMM_RHO_MM = meinsum('WP_RHO', 'Ajl,kl->Ajk',
                                       self.WP_AMM,
                                       rho_MM, self.timer)
        with self.timer('RI-V: 2nd contraction AMM AMM'):
            F_MM = meinsum('initialF_MM', 'Aik,Ajk->ij',
                              self.P_AMM,
                              WP_AMM_RHO_MM, self.timer)
            WP_AMM_RHO_MM = None

        with self.timer('RI-V: 1st contraction LMM MM'):
            WP_LMM_RHO_MM = meinsum('WP_RHO', 'Ajl,kl->Ajk',
                                    self.WP_LMM,
                                    rho_MM, self.timer)

        with self.timer('RI-V: 2nd contraction LMM LMM'):
            F_MM += meinsum('F_MM', 'Aik,Ajk->ij',
                             self.P_LMM,
                             WP_LMM_RHO_MM, self.timer)
            WP_LMM_RHO_MM = None

        return F_MM

        #with self.timer('RI-V: 1st contraction (lr-lr) LMM MM'):
        #    WL_LMM_RHO_MM = meinsum('WLRHO', 'Ajl,kl->Ajk',
        #                             self.WL_LMM,
        #                             rho_MM)
        #with self.timer('RI-V: 2nd contraction (lr-lr) LMM LMM'):
        #    F_MM += meinsum('F_MM', 'Aik,Ajk->ij',
        #                      self.S_LMM,
        #                      WL_LMM_RHO_MM)
        #    WL_LMM_RHO_MM = None


class RILVL(RIAlgorithm):
    def __init__(self, exx_fraction=None, screening_omega=None, lcomp=2, laux=2, threshold=1e-2):
        RIAlgorithm.__init__(self, 'RI-LVL', exx_fraction, screening_omega)

        self.lcomp = lcomp
        assert self.lcomp == 2

        self.laux = laux 
        assert self.laux == 2

        self.threshold = threshold

        self.matrix_elements = MatrixElements(self.laux, screening_omega, threshold=threshold)

        self.K_kkMMMM={}


    def prepare_setups(self, setups):
        RIAlgorithm.prepare_setups(self, setups)

    def set_positions(self, spos_ac):
        RIAlgorithm.set_positions(self, spos_ac)

        with self.timer('RI-V: Auxiliary Fourier-Bessel initialization'):
            self.matrix_elements.initialize(self.density, self.hamiltonian, self.wfs)

        auxt_aj = [ setup.auxt_j for setup in self.wfs.setups ]
        M_aj = [ setup.M_j for setup in self.wfs.setups ]

        kd = self.wfs.kd
        self.bzq_qc = bzq_qc = kd.get_bz_q_points()
        #print('Number of q-points: ', len(bzq_qc))

        with self.timer('RI-V: calculate W_qLL'):
             self.W_qLL = calculate_W_qLL(self.density.setups,\
                                          self.hamiltonian.gd.cell_cv,
                                          spos_ac,
                                          self.hamiltonian.gd.pbc_c,
                                          bzq_qc,
                                          self.wfs.dtype,
                                          self.lcomp, omega=self.screening_omega)
             #print(self.W_qLL,'W_qLL')
        gd = self.hamiltonian.gd
        ibzq_qc = np.array([[0.0, 0.0, 0.0]])
        dtype = self.wfs.dtype
        self.matrix_elements.set_positions_and_cell(spos_ac,
                                                    gd.cell_cv,
                                                    gd.pbc_c,
                                                    ibzq_qc,
                                                    self.bzq_qc, # q-points
                                                    dtype)


       
        for q, bzq_c in enumerate(bzq_qc):
            with self.timer('RI-V: calculate V_qAA'):
                self.V_qAA = calculate_V_qAA(auxt_aj, M_aj, self.W_qLL, self.lcomp)
                assert not np.isnan(self.V_qAA).any()
                #print(self.V_qAA,'V_AA')

            with self.timer('RI-V: calculate S_AA'):
                self.S_qAA = calculate_S_qAA(self.matrix_elements)
                #print(self.S_qAA,'S_AA')
                assert not np.isnan(self.S_qAA).any()

            with self.timer('RI-V: calculate M_AA'):
                self.M_qAA = calculate_M_qAA(self.matrix_elements, auxt_aj, M_aj, self.lcomp)
                #print(self.M_qAA,'M_qAA')
                self.W_qAA = self.V_qAA + self.S_qAA + self.M_qAA + self.M_qAA.T
                #print(self.W_qAA,'W_qAA')
                assert not np.isnan(self.M_qAA).any()

        self.kpt_pairs = []
        self.q_p = []
        for kpt1 in self.wfs.kpt_u:
            for kpt2 in self.wfs.kpt_u:
                self.kpt_pairs.append((kpt1.q, kpt2.q))
                self.q_p.append(0) # XXX

        assert(len(self.q_p)==1)

        with self.timer('RI-V: Calculate P_kkAMM'):
            self.P_kkAMM = calculate_P_kkAMM(self.matrix_elements, self.W_qAA)
            #print('P_kkAMM', self.P_kkAMM)

        with self.timer('RI-V: Calculate P_kkLMM'):
            self.P_kkLMM = calculate_P_kkLMM(self.matrix_elements, self.wfs.setups, self.wfs.atomic_correction)

        with self.timer('RI-V: Calculate W_qAL'):
            self.W_qAL = calculate_W_qAL(self.matrix_elements, auxt_aj, M_aj, self.W_qLL)


        with self.timer('RI-V: Calculate WP_AMM'):
            self.WP_kkAMM = {}
            for pair, q in zip(self.kpt_pairs, self.q_p):
                self.WP_kkAMM[pair] = np.einsum('AB,Bij',self.W_qAA[q], self.P_kkAMM[pair], optimize=True)
                self.WP_kkAMM[pair] += np.einsum('AB,Bij',self.W_qAL[q], self.P_kkLMM[pair], optimize=True)

        with self.timer('RI-V: Calculate WP_LMM'):
            self.WP_kkLMM = {}
            for pair, q in zip(self.kpt_pairs, self.q_p):
                self.WP_kkLMM[pair] = np.einsum('BA,Bij',self.W_qAL[q], self.P_kkAMM[pair], optimize=True)
                self.WP_kkLMM[pair] += np.einsum('AB,Bij',self.W_qLL[q], self.P_kkLMM[pair], optimize=True)


    def calculate_exchange_per_kpt_pair(self, kpt1, k_c, rho1_MM, kpt2, krho_c, rho2_MM):
        kpt_pair = (kpt1.q, kpt2.q)

        K_MMMM = np.einsum('Aij,AB,Bkl', self.P_kkAMM[kpt_pair], self.W_qAA[0], self.P_kkAMM[kpt_pair])
        #print('My K_MMMM wo comp', K_MMMM)
        with self.timer('RI-V: 1st contraction AMM MM'):
            WP_AMM_RHO_MM = np.einsum('Ajl,kl',
                                        self.WP_kkAMM[kpt_pair],
                                        rho2_MM, optimize=True)

        with self.timer('RI-V: 2nd contraction AMM AMM'):
            F_MM = np.einsum('Aik,Ajk',
                              self.P_kkAMM[kpt_pair],
                              WP_AMM_RHO_MM,
                              optimize=True)
            WP_AMM_RHO_MM = None

        with self.timer('RI-V: 1st contraction LMM MM'):
            WP_LMM_RHO_MM = np.einsum('Ajl,kl',
                                       self.WP_kkLMM[kpt_pair],
                                       rho2_MM, optimize=True)

        with self.timer('RI-V: 2nd contraction LMM LMM'):
            F_MM += np.einsum('Aik,Ajk',
                              self.P_kkLMM[kpt_pair],
                              WP_LMM_RHO_MM,
                              optimize=True)
            WP_LMM_RHO_MM = None

        #print(self.P_kkAMM[kpt_pair],'P_AMM')

        F_MM *= -0.5*self.exx_fraction
        #print('Not adding F_MM')
        evv = 0.5 * np.einsum('ij,ij', F_MM, rho1_MM, optimize=True)
        #print('F_MM', F_MM)
        #print('rho_MM', rho2_MM)
        return evv.real, F_MM

    def get_K_MMMM(self, kpt1, k1_c, kpt2, k2_c):
        k12_c = str((k1_c, k2_c))
        if k12_c in self.K_kkMMMM:
            return self.K_kkMMMM[k12_c]

        if self.screening_omega != 0.0 or np.all(self.density.gd.pbc_c):
            K_MMMM = self.get_K_MMMM_pbc(kpt1, k1_c, kpt2, k2_c)
        elif self.screening_omega == 0.0 and not np.any(self.density.gd.pbc_c):
            K_MMMM = self.get_K_MMMM_finite(kpt1, k1_c, kpt2, k2_c)
        else:
            raise NotImplementedError    

        self.K_kkMMMM[k12_c] = K_MMMM
        return K_MMMM

    def get_K_MMMM_pbc(self, kpt1, k1_c, kpt2, k2_c):
        wfs = self.wfs
        density = self.density

        q_c = (k1_c - k2_c)
        #print(q_c,'q_c',k1_c,k2_c)

        finegd = GridDescriptor(self.density.finegd.N_c, 
                                self.density.finegd.cell_cv,
                                pbc_c=True)

        # Create descriptors for the charge density (with symmetry q_c)
        if self.screening_omega != 0.0:
            coulomb = ShortRangeCoulomb(self.screening_omega)
        else:
            coulomb = WSTC(finegd.cell_cv, np.array([3,3,3]))


        gd = self.density.gd

        interpolator = Transformer(gd, finegd, 1, complex)
        
        #qd = KPointDescriptor([0*q_c])
        qd = KPointDescriptor([-q_c])
        pd12 = PWDescriptor(10.0, finegd, complex, kd=qd) #10 Ha =  270 eV cutoff
        v_G = coulomb.get_potential(pd12)

        # Exchange compensation charges
        ghat = PWLFC([data.ghat_l for data in wfs.setups], pd12)
        ghat.set_positions(wfs.spos_ac)

        nao = self.wfs.setups.nao
        K_MMMM = np.zeros( (nao, nao, nao, nao ), dtype=complex )
        #print('Allocating ', K_MMMM.itemsize / 1024.0**2,' MB')

        #gd = self.density.gd

        pairs_p = []
        for M1 in range(nao):
            for M2 in range(nao):
                pairs_p.append((M1,M2))

        # Since this is debug, do not care about memory use
        rho_pG = pd12.zeros( len(pairs_p), dtype=complex, q=0)
        V_pG = pd12.zeros( len(pairs_p), dtype=complex, q=0 )
        #print('Allocating ', 2*rho_pG.itemsize / 1024.0**2,' MB')

        # Put wave functions of the two basis functions to the grid
        phit1_MG = gd.zeros(nao, dtype=complex)
        self.wfs.basis_functions.lcao_to_grid(np.eye(nao, dtype=complex), phit1_MG, kpt1.q)

        phit2_MG = gd.zeros(nao, dtype=complex)
        self.wfs.basis_functions.lcao_to_grid(np.eye(nao, dtype=complex), phit2_MG, kpt2.q)

        for p1, (M1, M2) in enumerate(pairs_p):
            if p1 % 100 == 0:
                print('Potentials for pair %d/%d' % (p1, len(pairs_p)))
            rhot_xG = self.density.gd.zeros((1,), dtype=complex) 
            # Construct the product of two basis functions
            rhot_xG[0][:] = phit1_MG[M1].conjugate() * phit2_MG[M2] * density.gd.plane_wave(q_c)

            slice_G = rhot_xG[0,0,0,:]
            slice_G = np.concatenate((slice_G, slice_G))

            rhot_xg = finegd.zeros((1,), dtype=complex)
            interpolator.apply(rhot_xG, rhot_xg, np.ones((3, 2), complex))

            #if 1:
            #    N = np.linalg.norm(rhot_xg - ghat.pd.ifft(ghat.pd.fft(rhot_xg)))
            #    print('Norms', N)
            #    assert N<1e-7

            #print(self.density.gd.integrate(rhot_xG),'Real space norm')

            rho_xG = pd12.fft(rhot_xg) # Fourier transform the Bloch function product
            rho_xG = rho_xG.reshape( (1,) + rho_xG.shape)
            # Add compensation charges in reciprocal space
            Q_aL = {}
            D_ap = {}
            for a in self.wfs.P_aqMi:
                P1_i = self.wfs.P_aqMi[a][kpt1.q][M1]
                P2_i = self.wfs.P_aqMi[a][kpt2.q][M2]
                D_ii = np.outer(P1_i, P2_i.conjugate())
                D_p = pack(D_ii)
                D_ap[a] = D_p
                tmp = np.dot(D_p, self.density.setups[a].Delta_pL)
                Q_aL[a] = tmp.reshape((1,) + tmp.shape)
            ghat.add(rho_xG, Q_aL)

            Vrho_G = rho_xG[0] * v_G

            pot = pd12.ifft(Vrho_G)

            rho_pG[p1][:] = rho_xG[0]
            V_pG[p1][:] = Vrho_G

        K_pp = pd12.integrate(rho_pG, V_pG)
        for p1, (M1, M2) in enumerate(pairs_p):
            for p2, (M3, M4) in enumerate(pairs_p):
                K = K_pp[p1, p2]
                K_MMMM[M1,M2,M3,M4] = K

        return K_MMMM

    def get_K_MMMM_finite(self, kpt1, k1_c, kpt2, k2_c):
        assert kpt1.q == 0
        assert kpt2.q == 0
        wfs = self.wfs
        density = self.density

        q_c = (k1_c - k2_c)
        assert np.all(q_c == 0.0)

        finegd = self.density.finegd 

        # Create descriptors for the charge density (with symmetry q_c)
        if self.screening_omega != 0.0:
            raise NotImplementedError

        gd = self.density.gd
        interpolator = self.density.interpolator
        restrictor = self.hamiltonian.restrictor
        
        nao = self.wfs.setups.nao
        K_MMMM = np.zeros( (nao, nao, nao, nao )  )

        pairs_p = []
        for M1 in range(nao):
            for M2 in range(nao):
                pairs_p.append((M1,M2))

        # Since this is debug, do not care about memory use
        rho_pg = finegd.zeros( len(pairs_p) )
        V_pg = finegd.zeros( len(pairs_p) )
        #print('Allocating ', 2*rho_pg.itemsize / 1024.0**2,' MB')

        # Put wave functions of the two basis functions to the grid
        phit1_MG = gd.zeros(nao)
        self.wfs.basis_functions.lcao_to_grid(np.eye(nao), phit1_MG, kpt1.q)

        for p1, (M1, M2) in enumerate(pairs_p):
            if p1 % 100 == 0:
                print('Potentials for pair %d/%d' % (p1, len(pairs_p)))
            # Construct the product of two basis functions
            rhot_G = phit1_MG[M1] * phit1_MG[M2]

            rhot_g = finegd.zeros()
            interpolator.apply(rhot_G, rhot_g)

            # Add compensation charges in reciprocal space
            Q_aL = {}
            D_ap = {}
            for a in self.wfs.P_aqMi:
                P1_i = self.wfs.P_aqMi[a][kpt1.q][M1]
                P2_i = self.wfs.P_aqMi[a][kpt2.q][M2]
                D_ii = np.outer(P1_i, P2_i.conjugate())
                D_p = pack(D_ii)
                D_ap[a] = D_p
                Q_aL[a] = np.dot(D_p, self.density.setups[a].Delta_pL)

            self.density.ghat.add(rhot_g, Q_aL)

            V_g = finegd.zeros()
            self.hamiltonian.poisson.solve(V_g, rhot_g, charge=None)

            rho_pg[p1][:] = rhot_g
            V_pg[p1][:] = V_g

        K_pp = finegd.integrate(rho_pg, V_pg)
        for p1, (M1, M2) in enumerate(pairs_p):
            for p2, (M3, M4) in enumerate(pairs_p):
                K = K_pp[p1, p2]
                K_MMMM[M1,M2,M3,M4] = K

        return K_MMMM

    def get_description(self):
        return 'RI-LVL'

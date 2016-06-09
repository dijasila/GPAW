'''A class for computing the 
parameters for Marcus theory 
from to constrained DFT
wave functions

Computes: 
-the coupling matrix Hab
 Hab = <Psi_a|H|Psi_b>
-reorganization energy lambda
lambda = E_a(Rb)-E_a(Ra)
'''

import numpy as np
from gpaw.cdft import *
from ase.units import kB as kb
from gpaw.utilities import pack
from gpaw import GPAW
from gpaw.utilities.ps2ae import PS2AE
from ase.data import atomic_numbers
from ase.units import Hartree

class Marcus_parameters:

    def __init__(self, cdft_a = None, cdft_b = None,
               calc_a = None, calc_b = None, wfs_a = 'initial.gpw',
               wfs_b = 'final.gpw', gd = None,
               FA = None, FB = None, Va = None, Vb = None,
               NA = None, NB = None, weightA = None, weightB = None,
               h = 0.05,
               E_KS_A = None, E_KS_B = None,
               freq = None, temp = None, E_tst = None):
        
        '''cdft_a cdft_b: cdft calculators
        freq = effective frequency
        temp = temperature
        E_tst = adiabatic reaction barrier
        '''
        if cdft_a is not None and cdft_b is not None:
            self.cdft_A = cdft_a
            self.cdft_B = cdft_b
        
            self.calc_A = self.cdft_A.calc
            self.calc_B = self.cdft_B.calc
            self.gd = self.cdft_A.get_grid()
            
            # cDFT free energies
            self.FA = self.cdft_A.cdft_energy()
            self.FB = self.cdft_B.cdft_energy()
        
            # cDFT energies
            self.EA = self.cdft_A.dft_energy()
            self.EB = self.cdft_B.dft_energy()

            # lagrange multipliers
            self.Va = self.cdft_A.get_lagrangians() * Hartree
            self.Vb = self.cdft_B.get_lagrangians() * Hartree
            # Weight functions
            self.weightA = self.cdft_A.get_weight()
            self.weightB = self.cdft_B.get_weight()
            
            # constraint values
            self.NA = self.cdft_A.get_constraints()
            self.NB = self.cdft_B.get_constraints()
            
            # number of charge regions
            self.n_charge_regionsA = self.cdft_A.n_charge_regions
            self.n_charge_regionsB = self.cdft_B.n_charge_regions
        
        else:
            self.calc_A = calc_a
            self.calc_B = calc_b
            self.gd = self.calc_A.density.finegd
            # cDFT energies
            self.FA = FA
            self.FB = FB
        
            # Weights, i.e., lagrange multipliers
            self.Va = Va * Hartree
            self.Vb = Va * Hartree
        
            # Weight functions
            self.weightA = weightA
            self.weightB = weightB
            self.NA = NA
            self.NB = NB            

        # KS energies with cDFT densities
        self.E_KS_A = E_KS_A
        self.E_KS_B = E_KS_B

        # wave functions
        if wfs_a is not None and wfs_b is not None:
            self.wfs_A = wfs_a
            self.wfs_B = wfs_b
        else:
            self.wfs_A = self.calc_A.wfs
            self.wfs_B = self.calc_B.wfs
        
        ham = self.calc_A.hamiltonian
        self.H_AA = (self.calc_A.hamiltonian.Ekin + self.calc_A.hamiltonian.Epot + self.calc_A.hamiltonian.Ebar + 
                        self.calc_A.hamiltonian.Exc - self.calc_A.hamiltonian.S)*Hartree
            
        ham = self.calc_B.hamiltonian
        self.H_BB = (ham.Ekin + ham.Epot + ham.Ebar + 
                    ham.Exc - ham.S)*Hartree
        
        self.density = self.calc_A.density.finegd.zeros()
        self.atoms = self.calc_A.atoms
        self.natoms = len(self.atoms)
        self.h = h # all-electron interpolation grid spacing
        
        # controls for rate calculation
        
        self.E_tst = E_tst # adiabatic barrier
        
        # effective frequency
        if freq:
            self.freq =freq
        else:
            self.freq = 1.e12
        # temperature
        if temp:
            self.temp = temp
        else:
            self.temp = 298
    
    def get_coupling_term(self):
        
        '''solves the generalized eigen-equation
         W*S = S*V*L
         
         Then the non-orthogonal Hamiltonian
         is made orthogonal using similarity transform
         --> from this orthogonal Hamiltonian
         the coupling is extracted'''
         
        from scipy.linalg import eigvals
        
        # check that needed terms exist
        if (hasattr(self,'H') and
            hasattr(self,'W') and
            hasattr(self,'S')):
            H = self.H
            S = self.S
            W = self.W
        
        else:
            self.make_hamiltonian_matrix()
            H = self.H
            S = self.S
            W = self.W
             
        V, L = eigvals(W,S)
        # similarity transform
        H = np.dot((V.conj()).T, np.dot(H,V))
        # use the average
        self.ct =np.real(H[0][1])

        return self.ct, H

    def get_coupling_term_from_lowdin(self):
        self.get_ae_overlap_matrix()
        self.get_ae_weight_matrix2()
        self.get_diagonal_H_elements()

        H_AA = self.H_AA
        H_BB = self.H_BB

        H_AB = self.FB * self.S[0][1] - self.W[0][1]
        H_BA = self.FA * self.S[1][0] - self.W[1][0]

        self.H = np.array([[H_AA,H_AB],
                          [H_BA,H_BB]])

        H_orth = self.lowdin_orthogonalize_cdft_hamiltonian()

        return 1./2.*(H_orth [0][1] + H_orth [1][0]), H_orth 

    def lowdin_orthogonalize_cdft_hamiltonian(self):
        # H_orth = S^(-1/2)*H_cdft*S^(-1/2)
        
        U,D,V =np.linalg.svd(self.S, full_matrices=True) #V=U(+)
        s = np.diag(D**(-0.5))
        S_square = np.dot(U,np.dot(s,V)) # S^(-1/2)
        
        H_orth = np.dot(S_square,np.dot(self.H,S_square))

        return H_orth

    def make_hamiltonian_matrix(self):
        # this returns a 2x2 cDFT Hamiltonian
        # in a non-orthogonal basis --> 
        # made orthogonal in get_coupling_term 
        #self.get_diagonal_H_elements()
        H_AA = self.H_AA # diabat A with H^KS_A
        H_BB = self.H_BB # diabat B with H^KS_B
        
        if hasattr(self,'W'):
            VW = self.VW
            W = self.W
        else:
            W,VW = self.get_ae_weight_matrix()
        VbW_AB = VW[0][1]
        VaW_BA = VW[1][0]
        
        if hasattr(self, 'S'):
            S = self.S
        else:
            S = self.get_ae_overlap_matrix()
        
        S_AB = S[0][1]
        S_BA = S[1][0]
        
        h_AB = self.FB*S_AB - VbW_AB
        h_BA = self.FA*S_BA - VaW_BA

        # Ensure that H is hermitian
        H_AB = 1./2. * (h_AB + h_BA)
        H_BA = 1./2. * (h_AB + h_BA).conj()

        self.H = np.array([[H_AA, H_BA],[H_AB,H_BB]])
        print 'Hamiltonian', self.H
        return self.H
    

    def get_ae_weight_matrix(self):
    	''' Compute W_AB =  <Psi_A|sum_i w_i^B(r)| Psi_B>
    	with all-electron pair density
        '''
        
        ns = self.calc_A.wfs.nspins
        nk = len(self.calc_A.wfs.kd.weight_k)
        
        psi_A = PS2AE(self.calc_A, h = self.h, n = 1, grid = self.weightA[0])
        psi_B = PS2AE(self.calc_B, h = self.h, n = 1, grid = self.weightB[0])
        
        self.W = np.zeros((2,2))
        # V weighted weight matrix
        self.VW = np.zeros((2,2))
        
        # fill diagonal
        for a,b in map(None, range(len(self.Va)),range(len(self.Vb))):
            # W_AA = Va * Na
            if a is not None:
                self.W[0][0] += self.NA[a]
            if b is not None:    
                self.W[1][1] += self.NB[b]        
        
        n_occup_A = self.get_n_occupied_bands(self.calc_A) # total of filled a and b bands
        n_occup_B = self.get_n_occupied_bands(self.calc_B)
    	
        n_bands_A = self.calc_A.get_number_of_bands() # total a or b bands
        n_bands_B = self.calc_B.get_number_of_bands()
    	if n_bands_A == n_bands_B:
            n_bands = n_bands_A
        else:
            n_bands = np.max((n_bands_A,n_bands_B))
    	
    	# list to store k-dependent weight matrices
    	w_ij_AB =[]
    	w_ij_BA = []
    	Vbw_ij_AB = []
    	Vaw_ij_BA = []
    	
    	# form weight matrices of correct size for each kpt	
    	for k in range(nk):
    	    nA = n_occup_A[k].sum() # sum spins
    	    nB = n_occup_B[k].sum()
    	    if nA == nB:
                n_occup = nA
            else:
                n_occup = np.max((nA,nB))
    	    
    	    occup_k = n_occup
    	    w_ij_AB.append(np.zeros((occup_k,occup_k)))
    	    w_ij_BA.append(np.zeros((occup_k,occup_k)))
            Vbw_ij_AB.append(np.zeros((occup_k,occup_k)))
            Vaw_ij_BA.append(np.zeros((occup_k,occup_k)))
        
        w_k = []
        
        # get weight matrix at for each ij band at kpt and spin
        # the resulting matrix is organized in alpha and beta blocks
        #    |       |    |
        #    |  a    | 0  |    a:<psi_a|w|psi_a> != 0 
        # W =|_______|____|  , <psi_a|w|psi_b> = 0
        #    |   0   |  b |    b:<psi_b|w|psi_b> != 0
        #    |       |    |
        #
        # a = nAa x nAa, b = nAb x nAb        
        
        for spin in range(ns):      
            for kpt_A, kpt_B in zip(self.calc_A.wfs.kpt_u[spin*nk:(spin+1)*nk], 
                 self.calc_B.wfs.kpt_u[spin*nk:(spin+1)*nk]):
                               
                k = kpt_A.k
                nAa = n_occup_A[k][0]
                nAb =  n_occup_A[k][1]
                nBa, nBb = n_occup_B[k][0], n_occup_B[k][1]

                nA = nAa + nAb
                
                nB = nBa + nBb

                # check that a and b cDFT states have similar spin state
                if np.sign(nAa-nAb) != np.sign(nBa-nBb):
                    raise ValueError('The cDFT wave functions have'
                        'different spin states! Similar'
                        'spin states are required for coupling constant'
                        'calculation!')

                if nA==nB:
                    n_occup=nA
                else:
                    n_occup = np.max((nA,nB))
                
                # size of alpha block
                # ideally A and B contain equivalent amount of alphas...
                if nAa >= nAb: # more alphas
                    nas = n_occup - np.min((nAb,nBb))
                else: # more betas
                    nas = n_occup - np.min((nAa,nBa))
                
                for i in range(n_occup): 
                    for j in range(n_occup):
                        # take only the bands which contain electrons
                        if i < n_occup_A[k][spin] and j < n_occup_B[k][spin]:
                            psi_kA = psi_A.get_wave_function(n = i, k = k, s = spin,ae = True)
                            f_nA = kpt_A.f_n[i]
                            # ae orbital b 
                            psi_kB = psi_B.get_wave_function(n = j, k = k, s = spin,ae = True)
                            f_nB = kpt_B.f_n[j]

                        
                            # ae pair density
                            n_ij_ae = np.multiply(psi_kA.conj(), psi_kB)
                      
                            w_AB = np.zeros(len(self.Va)) #<psi_a|W_b|psi_b>
                            w_BA = np.zeros(len(self.Vb))
                            VbwAB = np.zeros(len(self.Vb)) #V_b*W_ab = V_b <psi_a|W_b|psi_b>
                            VawBA = np.zeros(len(self.Va))
                        
                            for a,b in map(None, range(len(self.Va)),range(len(self.Vb))):

                                if b is not None:
                                    w_AB[b] = (psi_A.gd.integrate(
                                        self.weightB[b]*n_ij_ae, 
                                        global_integral=True))    
                                if a is not None:
                                    w_BA[a] = (psi_A.gd.integrate(
                                        self.weightA[a]*n_ij_ae.conj(), 
                                        global_integral=True))
                                                       
                            # correct with kpoint weight and band occupation
                            w_AB *=  (f_nB/kpt_B.weight * f_nA/kpt_A.weight)
                            w_BA *= (f_nB/kpt_B.weight * f_nA/kpt_A.weight)
                       
                            # store k-point weights
                            w_k.append(kpt_A.weight * kpt_B.weight)
                            print (i,j,spin,np.dot(self.Vb , w_AB))
                            # fill the pair density-weight matrix
                            # with spin blocks
                            if spin == 0:
                                w_ij_AB[k][i][j] = w_AB.sum()
                                w_ij_BA[k][j][i] = w_BA.sum()
                                Vbw_ij_AB[k][i][j] = np.dot(self.Vb , w_AB)
                                Vaw_ij_BA[k][j][i] = np.dot(self.Va , w_BA)
                            
                            else:
                                #charge constraints
                                w_ij_AB[k][nas+i][nas+j] = w_AB[0:self.n_charge_regionsB].sum()
                                w_ij_BA[k][nas+j][nas+i] = w_BA[0:self.n_charge_regionsA].sum()
                                
                                Vbw_ij_AB[k][nas+i][nas+j] = (self.Vb[0:self.n_charge_regionsB] *\
                                                      w_AB[0:self.n_charge_regionsB]).sum()
                                Vaw_ij_BA[k][nas+j][nas+i] = (self.Va[0:self.n_charge_regionsA] *\
                                                      w_BA[0:self.n_charge_regionsA]).sum()
                                
                                # spin constraints, beta spins switch sign in spin constraints
                                if len(w_AB != self.n_charge_regionsB):
                                    w_ij_AB[k][nas+i][nas+j] += -w_AB[self.n_charge_regionsB:].sum()
                                    Vbw_ij_AB[k][nas+i][nas+j] += -(self.Vb[self.n_charge_regionsB:] *\
                                                      w_AB[self.n_charge_regionsB:]).sum()
                                
                                if len(w_BA != self.n_charge_regionsA):
                                    w_ij_BA[k][nas+j][nas+i] += -w_BA[self.n_charge_regionsA:].sum()
                                    Vaw_ij_BA[k][nas+j][nas+i] += -(self.Va[self.n_charge_regionsA:] *\
                                                      w_AB[self.n_charge_regionsA:]).sum()
                            
        #get determinants for each kpt
        W_k_AB = np.zeros(nk)
        W_k_BA = np.zeros(nk)
        VbW_k_AB = np.zeros(nk)
        VaW_k_BA = np.zeros(nk)
        
        for k in range(nk):
            W_k_AB[k] = w_k[k] * np.linalg.det(w_ij_AB[k])
            W_k_BA[k] = w_k[k] * np.linalg.det(w_ij_BA[k])
            VbW_k_AB[k] = w_k[k] * np.linalg.det(Vbw_ij_AB[k])
            VaW_k_BA[k] = w_k[k] * np.linalg.det(Vaw_ij_BA[k])
        
        # sum kpts
        W_AB = W_k_AB.sum()
        W_BA = W_k_BA.sum()
        VbW_AB = VbW_k_AB.sum()
        VaW_BA = VaW_k_BA.sum()
        
        # fill 2x2 hermitian weight matrix
        self.W[0][1] += 1./2. * (W_AB+W_BA)
        self.W[1][0] += 1./2. * (W_AB+W_BA).conj()
        print ('weight matrix',self.W)
        # ... and V multiplied weights
        self.VW[0][1] += 1./2. *(VbW_AB+VaW_BA)
        self.VW[1][0] += 1./2. *(VbW_AB+VaW_BA).conj()
        print('Vw matrix'), self.VW
        return self.W, self.VW
        

    def get_ae_weight_matrix2(self):
        # the weight matrix
        self.W = np.zeros((2,2))
        # fill diagonal
        
        for a,b in map(None, range(len(self.Va)),range(len(self.Vb))):
            # W_AA = Va * Na
            if a is not None:
                self.W[0][0] += self.NA[a]
            if b is not None:    
                self.W[1][1] += self.NB[b]  

        # pseudo wfs to all-electron wfs
        psi_A = PS2AE(self.calc_A, h = self.h, n = 1, grid = self.weightA[0])
        psi_B = PS2AE(self.calc_B, h = self.h, n = 1, grid = self.weightB[0]) 

        n_spin_regionsA = len(self.Va) - self.n_charge_regionsA
        n_spin_regionsB = len(self.Vb) - self.n_charge_regionsB
        
        ns = self.calc_A.wfs.nspins
        nk = len(self.calc_A.wfs.kd.weight_k)
        
        # spin dependent weight functions
        wa = [[],[]]
        wb = [[],[]]
        # with more than one constraint this seems to be the only way...
        for a in range(self.n_charge_regionsA):
            wa[0].append(self.weightA[a])
            wa[1].append(self.weightA[a])
        
        for a in range(n_spin_regionsA):
            wa[0].append(self.weightA[a+self.n_charge_regionsA])
            wa[1].append(-self.weightA[a+self.n_charge_regionsA])
        
        for b in range(self.n_charge_regionsB):
            wb[0].append(self.weightB[b])
            wb[1].append(self.weightB[b])
        
        for b in range(n_spin_regionsB):
            wb[0].append(self.weightB[b+self.n_charge_regionsB])
            wb[1].append(-self.weightB[b+self.n_charge_regionsB])
        
        wa = np.asarray(wa)
        wb = np.asarray(wb)

        # check number of occupied and total number of bands
        n_occup_A = self.get_n_occupied_bands(self.calc_A) # total of filled a and b bands
        n_occup_B = self.get_n_occupied_bands(self.calc_B)
        
        n_bands_A = self.calc_A.get_number_of_bands() # total a or b bands
        n_bands_B = self.calc_B.get_number_of_bands()
        
        if n_bands_A == n_bands_B:
            n_bands = n_bands_A
        else:
            n_bands = np.max((n_bands_A,n_bands_B))
        
        # place to store <i_A(k)|w|j_B(k)>
        w_kij_AB =[]
        w_kij_BA =[]
        
        for k in range(nk):
            na = n_occup_A[k].sum() # sum spins
            nb = n_occup_B[k].sum()
            if na == nb:
                n_occup = na
            else:
                n_occup = np.max((na,nb))       

            w_kij_AB.append(np.zeros((n_occup,n_occup)))
            w_kij_BA.append(np.zeros((n_occup,n_occup)))

        # k-point weights
        w_k = []
        
        w_kij_BA = np.asarray(w_kij_BA)
        w_kij_AB = np.asarray(w_kij_AB)
        # get weight matrix at for each ij band at kpt and spin
        # the resulting matrix is organized in alpha and beta blocks
        #    |       |    |
        #    |  a    | 0  |    a:<psi_a|w|psi_a> != 0 
        # W =|_______|____|  , <psi_a|w|psi_b> = 0
        #    |   0   |  b |    b:<psi_b|w|psi_b> != 0
        #    |       |    |
        #
        # a = nAa x nAa, b = nAb x nAb        
        
        for spin in range(ns):      
            for kpt_A, kpt_B in zip(self.calc_A.wfs.kpt_u[spin*nk:(spin+1)*nk], 
                 self.calc_B.wfs.kpt_u[spin*nk:(spin+1)*nk]):
                               
                k = kpt_A.k
                # number of occupied alpha and beta bands 
                nAa,nAb  = n_occup_A[k][0],n_occup_A[k][1]
                nBa, nBb = n_occup_B[k][0], n_occup_B[k][1]

                nA = nAa + nAb  
                nB = nBa + nBb

                # check that a and b cDFT states have similar spin state
                if np.sign(nAa-nAb) != np.sign(nBa-nBb):
                    raise ValueError('The cDFT wave functions have'
                        'different spin states! Similar'
                        'spin states are required for coupling constant'
                        'calculation!')

                if nA==nB:
                    n_occup=nA
                else:
                    n_occup = np.max((nA,nB))
                
                # size of alpha block
                # ideally A and B contain equivalent amount of alphas...
                if nAa >= nAb: # more alphas
                    nas = n_occup - np.min((nAb,nBb))
                else: # more betas
                    nas = n_occup - np.min((nAa,nBa))
                # collect kpt weight, only once per kpt
                if spin == 0:
                    w_k.append(kpt_A.weight * kpt_B.weight)
                
                for i in range(n_occup): 
                    for j in range(n_occup):
                        # take only the bands which contain electrons
                        if i < n_occup_A[k][spin] and j < n_occup_B[k][spin]:
                            psi_kA = psi_A.get_wave_function(n = i, k = k, s = spin,ae = True)
                            f_nA = kpt_A.f_n[i]
                            # ae orbital b 
                            psi_kB = psi_B.get_wave_function(n = j, k = k, s = spin,ae = True)
                            f_nB = kpt_B.f_n[j]
                            
                            # with more than one constraint this seems to be the only way...
                            w_ij_AB = []
                            w_ji_BA = []
                            
                            for b in range(len(self.Vb)):
                                w_ij_AB.append(self.gd.integrate( psi_kA.conj() * wb[spin][b] * psi_kB, 
                                        global_integral = True))
                            
                            for a in range(len(self.Va)):
                                w_ji_BA.append(self.gd.integrate( psi_kB.conj() * wa[spin][a] * psi_kA, 
                                        global_integral = True))                          
                            
                            w_ij_AB = np.asarray(w_ij_AB)
                            w_ji_BA = np.asarray(w_ji_BA)

                            # correct with occupation
                            w_ij_AB *= (f_nB/kpt_B.weight * f_nA/kpt_A.weight)
                            w_ji_BA *= (f_nB/kpt_B.weight * f_nA/kpt_A.weight)
                            
                            if spin == 0: # alpha block
                                
                                # multiply first to row by Vb
                                #       | a b c|      |Vb*a Vb*b Vb*c |
                                #Vb*det | d e f| = det|  d   e    f   |
                                #       | g h i|      |  g   h    i   |
                                # makes the determinant calculation more simple
                                # the first row is nonzero only for alpha
                                # then sum each weight region
                                
                                if i == 0:
                                    w_kij_AB[k][i][j] = np.dot(w_ij_AB,self.Vb)
                                else: 
                                    w_kij_AB[k][i][j] = w_ij_AB.sum()
                                
                                if j == 0:
                                    w_kij_BA[k][j][i] = np.dot(w_ji_BA,self.Va)
                                else:
                                    w_kij_BA[k][j][i] = w_ji_BA.sum()

                            else: # beta block
                                w_kij_AB[k][nas+i][nas+j] = w_ij_AB.sum()
                                w_kij_BA[k][nas+j][nas+i] = w_ji_BA.sum()

        W_k_AB = np.zeros(nk)
        W_k_BA = np.zeros(nk)
        
        for k in range(nk):
            W_k_AB[k] = np.linalg.det(w_kij_AB[k])
            W_k_AB[k] *= w_k[k]
            W_k_BA[k] = np.linalg.det(w_kij_BA[k].conj())
            W_k_BA[k] *= w_k[k]
        
        #make sure W is hermitian
        
        self.W[0][1] = 1./2.*(W_k_AB.sum() + W_k_BA.sum())
        self.W[1][0] = self.W[0][1]
        print ('weight matrix',self.W)
        return self.W

    def get_ae_overlap_matrix(self):                       
        ''' <Psi_A|Psi_B> using the all-electron pair density'''
        
        psi_A = PS2AE(self.calc_A, h = self.h, n = 2)
        psi_B = PS2AE(self.calc_B, h = self.h, n = 2)
    	
    	ns = self.calc_A.wfs.nspins
    	nk = len(self.calc_A.wfs.kd.weight_k)        
        self.S = np.identity(2)
        
        # total of filled a and b bands for each spin and kpt 
        n_occup_A = self.get_n_occupied_bands(self.calc_A) 
        n_occup_B = self.get_n_occupied_bands(self.calc_B)
    	# total a or b bands
        n_bands_A = self.calc_A.get_number_of_bands() 
        n_bands_B = self.calc_B.get_number_of_bands()
    	
    	if n_bands_A == n_bands_B:
            n_bands = n_bands_A
        else:
            n_bands = np.max((n_bands_A,n_bands_B))
    	
    	# list to store k-dependent overlap matrices
    	S_kn = []
    	
    	# form overlap matrices of correct size for each kpt
    	for k in range(nk):
    	    na = n_occup_A[k].sum() # sum spins
    	    nb = n_occup_B[k].sum()
    	    if na == nb:
                n_occup = na
            else:
                n_occup = np.max((na,nb))	    

    	    S_kn.append(np.zeros((n_occup,n_occup)))

        w_k = [] #store kpt weights
        
        # get overlap at for each ij band at kpt and spin
        # the resulting matrix is organized in alpha and beta blocks
        #    |       |    |
        #    |  a    | 0  |    a:<psi_a|psi_a> != 0 
        # S =|_______|____|  , <psi_a|psi_b> = 0
        #    |   0   |  b |    b:<psi_b|psi_b> != 0
        #    |       |    |
        #
        # a = naa x naa, b = nab x nab

        for spin in range(ns): 
            for kpt_A, kpt_B in zip(self.calc_A.wfs.kpt_u[spin*nk:(spin+1)*nk], 
                self.calc_B.wfs.kpt_u[spin*nk:(spin+1)*nk]):
                
                k = kpt_A.k
                nAa,nAb = n_occup_A[k][0], n_occup_A[k][1]
                nBa, nBb = n_occup_B[k][0], n_occup_B[k][1]

                nA = nAa + nAb
                nB = nBa + nBb

                # check that a and b cDFT states have similar spin state
                if np.sign(nAa-nAb) != np.sign(nBa-nBb):
                    raise ValueError('The cDFT wave functions have'
                        'different spin states! Similar'
                        'spin states are required for coupling constant'
                        'calculation!')

                if nA==nB:
                    n_occup=nA
                else:
                    n_occup = np.max((nA,nB))
                
                # size of alpha block
                # ideally A and B contain equivalent amount of alphas...
                if nAa >= nAb: # more alphas
                    nas = n_occup - np.min((nAb,nBb))
                else: # more betas
                    nas = n_occup - np.min((nAa,nBa))
                
                # loop over all occupied spin orbitals                
                for i in range(n_occup): 
                    for j in range(n_occup):   
                        # take only the bands which contain electrons in spin-orbital
                        if i < n_occup_A[k][spin] and j < n_occup_B[k][spin]:
                            psi_kA = psi_A.get_wave_function(n = i, k = k, s = spin,ae = True)
                            f_nA = kpt_A.f_n[i]
                            # ae orbital b 
                            psi_kB = psi_B.get_wave_function(n = j, k = k, s = spin,ae = True)
                            f_nB = kpt_B.f_n[j]

                            n_ij = psi_A.gd.integrate(psi_kA.conj(), psi_kB, global_integral= True)

                            # correct with band occupation
                            n_ij = n_ij * (f_nB/kpt_B.weight * f_nA/kpt_A.weight)
                            # store k-point weights
                            w_k.append(kpt_A.weight * kpt_B.weight)
                               
                            # fill the pair density matrix with spin blocks
                            # contains both a and b spin indices
                            print 'overlap, i,j, spin',n_ij, i, j,spin
                            # spin blocks
                            if spin == 0:
                                S_kn[k][i][j] = n_ij
                            elif spin==1:
                                S_kn[k][nas + i][nas + j] = n_ij
                        
        print ('pair density matrix',np.round(S_kn,3))
        S_k_AB = np.zeros(nk)
        S_k_BA = np.zeros(nk)
        
        for k in range(nk):
            S_k_AB[k] = w_k[k] * np.linalg.det(S_kn[k])
            # determinant of the complex conjugate
            S_k_BA[k] = w_k[k] * np.linalg.det(np.transpose(S_kn[k]).conj())
        
        S_AB = S_k_AB.sum()
        S_BA = S_k_BA.sum()        
        
        # fill 2x2 overlap matrix
        self.S[0][1] = 1./2.*(S_AB+S_BA)
        self.S[1][0] = 1./2.*(S_AB+S_BA)
        print 'overlap matrix',self.S
        return self.S   
    
    def get_diagonal_H_elements(self):
        # do a non-self consistent calculation
        # without external potential using 
        # density/wave function from cDFT calculation
        # KS calculators (not cDFT) for non-self consistent
        # calculation using cDFT density
        
        if self.E_KS_A is not None and self.E_KS_B is not None:
            self.H_AA = self.E_KS_A
            self.H_BB = self.E_KS_B
        else:

            ham = self.calc_A.hamiltonian
            self.H_AA = (ham.Ekin + ham.Epot + ham.Ebar + 
                        ham.Exc - ham.S)
            
            ham = self.calc_B.hamiltonian
            self.H_BB = (ham.Ekin + ham.Epot + ham.Ebar + 
                        ham.Exc - ham.S)
    
    def get_reorganization_energy(self):
        # get Ea (Rb) - Ea(Ra) -->
        # cdft energy at geometry Rb
        # with charge constraint A
        geometry = self.calc_B.get_atoms()
        
        cdft = self.cdft_A
        # set cdft_a on geometry of B
        geometry.set_calculator(cdft)
        self.reorg_energy = geometry.get_potential_energy()
        self.reorg_energy -= self.EA
        
        return self.reorg_energy
             
    def get_landau_zener(self):
        # computes the Landau-Zener factor
        
        planck = 4.135667e-15 #eV s
        
        if hasattr(self, 'reorg_energy'):
            Lambda = self.reorg_energy
        else:
            Lambda = self.get_reorganization_energy()
            
        if hasattr(self,'ct'):
            Hab = self.ct
        else:
            Hab = self.get_coupling_term()
                    
        self.P_lz = 1 - np.exp(-np.pi**(3./2.) * (np.abs(Hab))**2 / \
                   (planck * v_eff * np.sqrt(temp*Lambda*kb)))
        
        return self.P_lz
        
    def get_marcus_rate(self):
        # adiabatic transition state energy
        if self.E_tst:
            E_tst = self.E_tst
        else:
            # compute the activation energy
            # from the classical marcus
            # parabolas
            E_tst = self.get_marcus_barrier()
            
        # electron transmission coeffiecient
        # is the reaction diabatic or adiabatic?
        P_lz = self.get_landau_zener()
        dE = self.EA - self.EB
        if  dE>= -self.reorg_energy:
            # normal
            kappa = 2. * P_lz / (1 + P_lz)
        else:
            # inverted
            kappa = 2 * P_lz(1 - P_lz)
            
        rate = kappa * self.freq * np.exp(-E_tst/(kb * self.temp))
        
        return rate
            
    def get_marcus_barrier(self):
        # approximate barrier from
        # two marcus parabolas 
        # and an adiabatic correction
        
        # reaction energy
        dE = self.EA - self.EB
        
        # crossing of the parabolas
        barrier = 1. / (4. * self.reorg_energy) * \
                   np.exp(self.reorg_energy + dE)**2
        
        # adiabatic correction
        correction = np.abs(self.ct) + \
                    (self.reorg_energy + dE) / 2. -\
                    np.sqrt((1. / (4. * self.reorg_energy))**2 + \
                    (np.abs(self.ct))**2)
        
        return barrier - correction
    
    def get_n_occupied_bands(self, calc, partial = 0.01):
        ''' how many occupied bands?'''
        ns = calc.wfs.nspins
        occup_ks = np.zeros((len(calc.wfs.kd.weight_k),ns),dtype = np.int)
        for kpt in calc.wfs.kpt_u:
            s = kpt.s
            k = kpt.k
            f_n = kpt.f_n / kpt.weight
            full_and_partial = f_n > partial
            # get number of full and partial
            occup_ks[k][s] += full_and_partial.sum()
        return occup_ks
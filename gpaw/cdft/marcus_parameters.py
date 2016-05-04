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
from ConstrainedDft import CDFT
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
            self.cdft_a = cdft_a
            self.cdft_b = cdft_b
        
            self.calc_a = self.cdft_a.calc
            self.calc_b = self.cdft_b.calc
            self.gd = self.cdft_a.get_grid()
            # cDFT free energies
            self.FA = self.cdft_a.cdft_energy()
            self.FB = self.cdft_b.cdft_energy()
            
            # cDFT energies
            self.EA = self.cdft_A.dft_energy()
            self.EB = self.cdft_B.dft_energy()

            # lagrange multipliers
            self.Va = self.cdft_a.get_lagrangians() * Hartree
            self.Vb = self.cdft_b.get_lagrangians() * Hartree
            # Weight functions
            self.weightA = self.cdft_a.get_weight()
            self.weightB = self.cdft_b.get_weight()
            
            #constraint values
            self.NA = self.cdft_a.get_constraints()
            self.NB = self.cdft_b.get_constraints()

            # KS calculators (not cDFT) for non-self consistent
            # calculation using cDFT density
        
            self.KS_calc_a = cdft_a.calc.set(external = None, fixdensity = True)
            self.KS_calc_b = cdft_b.calc.set(external = None, fixdensity = True)
        
        else:
            self.calc_a = calc_a
            self.calc_b = calc_b
            self.gd = self.calc_a.density.finegd
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

        # initialize
        if wfs_a and wfs_b:
            self.wfs_a = wfs_a
            self.wfs_b = wfs_b
        else:
            self.wfs_a = self.calc_a.wfs
            self.wfs_b = self.calc_b.wfs
        
        self.density = self.calc_a.density.finegd.zeros()
        self.atoms = self.calc_a.atoms
        self.natoms = len(self.atoms)
        self.h = h # ae interpolation grid spacing
        
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

    def make_hamiltonian_matrix(self):
        # this returns a 2x2 cDFT Hamiltonian
        # in a non-orthogonal basis --> 
        # made orthogonal in get_coupling_term 
        self.get_diagonal_H_elements()
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
        
        ns = self.calc_a.wfs.nspins
        nk = len(self.calc_a.wfs.kd.weight_k)
        
        psi_a = PS2AE(self.calc_a, h = self.h, n = 1, grid = self.weightA[0])
        psi_b = PS2AE(self.calc_b, h = self.h, n = 1, grid = self.weightB[0])
        
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
        
        n_occup_a = self.get_n_occupied_bands(self.calc_a) # total of filled a and b bands
        n_occup_b = self.get_n_occupied_bands(self.calc_b)
        
        n_bands_a = self.calc_a.get_number_of_bands() # total a or b bands
        n_bands_b = self.calc_b.get_number_of_bands()
        if n_bands_a == n_bands_b:
            n_bands = n_bands_a
        else:
            n_bands = np.max((n_bands_a,n_bands_b))
        
        # list to store k-dependent weight matrices
        w_ij_AB =[]
        w_ij_BA = []
        Vbw_ij_AB = []
        Vaw_ij_BA = []
        
        # form weight matrices of correct size for each kpt 
        for k in range(nk):
            na = n_occup_a[k].sum() # sum spins
            nb = n_occup_b[k].sum()
            if na == nb:
                n_occup = na
            else:
                n_occup = np.max((na,nb))
            
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
            for kpt_a, kpt_b in zip(self.calc_a.wfs.kpt_u[spin*nk:(spin+1)*nk], 
                 self.calc_b.wfs.kpt_u[spin*nk:(spin+1)*nk]):
                               
                k = kpt_a.k
                nAa = n_occup_a[k][0]
                nAb =  n_occup_a[k][1]
                nBa, nBb = n_occup_b[k][0], n_occup_b[k][1]

                nA = nAa + nAb
                
                nB = nBa + nBb

                # check that a and b cDFT states have similar spin state
                if np.sign(nAa-nAb) != np.sign(nBa-nBb):
                    raise ValueError('The cDFT wave functions have'
                        'different spin states! Similar'
                        'spin states are required for coupling constant'
                        'calculation!')

                if na==nb:
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
                        if i < n_occup_a[k][spin] and j < n_occup_b[k][spin]:
                            psi_ka = psi_a.get_wave_function(n = i, k = k, s = spin,ae = True)
                            f_na = kpt_a.f_n[i]
                            # ae orbital b 
                            psi_kb = psi_b.get_wave_function(n = j, k = k, s = spin,ae = True)
                            f_nb = kpt_b.f_n[j]

                        
                            # ae pair density
                            n_ij_ae = np.multiply(psi_ka.conj(), psi_kb)
                      
                            w_ab = np.zeros(len(self.Va))
                            w_ba = np.zeros(len(self.Vb))
                            Vbwab = np.zeros(len(self.Vb)) #V_b*W_ab
                            Vawba = np.zeros(len(self.Va))
                        
                            for a,b in map(None, range(len(self.Va)),range(len(self.Vb))):

                                if a is not None:
                                    w_ab[a] = (psi_a.gd.integrate(
                                        self.weightA[a]*n_ij_ae, 
                                        global_integral=True))    
                                if b is not None:
                                    w_ba[b] = (psi_a.gd.integrate(
                                        self.weightB[b]*n_ij_ae.conj(), 
                                        global_integral=True))
                                                       
                            # correct with kpoint weight and band occupation
                            w_ab *=  (f_nb/kpt_b.weight * f_na/kpt_a.weight)
                            w_ba *= (f_nb/kpt_b.weight * f_na/kpt_a.weight)
                       
                            # store k-point weights
                            w_k.append(kpt_a.weight * kpt_b.weight)
                        
                            print 'w_ab, w_ba', w_ab, w_ba
                            # fill the pair density-weight matrix
                            if spin == 0:
                                w_ij_AB[k][i][j] = w_ab.sum()
                                w_ij_BA[k][j][i] = w_ba.sum()
                                Vbw_ij_AB[k][i][j] = (self.Vb * w_ab).sum()
                                Vaw_ij_BA[k][j][i] = (self.Va * w_ba).sum()
                            else:
                                w_ij_AB[k][nas+i][nas+j] = w_ab.sum()
                                w_ij_BA[k][nas+j][nas+i] = w_ba.sum()
                                Vbw_ij_AB[k][nas+i][nas+j] = (self.Vb * w_ab).sum()
                                Vaw_ij_BA[k][nas+j][nas+i] = (self.Va * w_ba).sum()
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
        print 'weight matrix',self.W
        # ... and V multiplied weights
        self.VW[0][1] += 1./2. *(VbW_AB+VaW_BA)
        self.VW[1][0] += 1./2. *(VbW_AB+VaW_BA).conj()
        return self.W, self.VW
    
    def get_ae_overlap_matrix(self):                       
        ''' <Psi_A|Psi_B> using the all-electron pair density'''
        
        psi_a = PS2AE(self.calc_a, h = self.h, n = 2)
        psi_b = PS2AE(self.calc_b, h = self.h, n = 2)
        
        ns = self.calc_a.wfs.nspins
        nk = len(self.calc_a.wfs.kd.weight_k)        
        self.S = np.identity(2)
        
        # total of filled a and b bands for each spin and kpt 
        n_occup_a = self.get_n_occupied_bands(self.calc_a) 
        n_occup_b = self.get_n_occupied_bands(self.calc_b)
        # total a or b bands
        n_bands_a = self.calc_a.get_number_of_bands() 
        n_bands_b = self.calc_b.get_number_of_bands()
        
        if n_bands_a == n_bands_b:
            n_bands = n_bands_a
        else:
            n_bands = np.max((n_bands_a,n_bands_b))
        
        # list to store k-dependent overlap matrices
        S_kn = []
        
        # form overlap matrices of correct size for each kpt
        for k in range(nk):
            na = n_occup_a[k].sum() # sum spins
            nb = n_occup_b[k].sum()
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
            for kpt_a, kpt_b in zip(self.calc_a.wfs.kpt_u[spin*nk:(spin+1)*nk], 
                self.calc_b.wfs.kpt_u[spin*nk:(spin+1)*nk]):
                
                k = kpt_a.k
                nAa = n_occup_a[k][0]
                nAb =  n_occup_a[k][1]
                nBa, nBb = n_occup_b[k][0], n_occup_b[k][1]

                nA = nAa + nAb
                
                nB = nBa + nBb

                # check that a and b cDFT states have similar spin state
                if np.sign(nAa-nAb) != np.sign(nBa-nBb):
                    raise ValueError('The cDFT wave functions have'
                        'different spin states! Similar'
                        'spin states are required for coupling constant'
                        'calculation!')

                if na==nb:
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
                        if i < n_occup_a[k][spin] and j < n_occup_b[k][spin]:
                            psi_ka = psi_a.get_wave_function(n = i, k = k, s = spin,ae = True)
                            f_na = kpt_a.f_n[i]
                            # ae orbital b 
                            psi_kb = psi_b.get_wave_function(n = j, k = k, s = spin,ae = True)
                            f_nb = kpt_b.f_n[j]

                            n_ij = psi_a.gd.integrate(psi_ka.conj(), psi_kb, global_integral= True)

                            # correct with band occupation
                            n_ij = n_ij * (f_nb/kpt_b.weight * f_na/kpt_a.weight)
                            # store k-point weights
                            w_k.append(kpt_a.weight * kpt_b.weight)
                               
                            # fill the pair density matrix
                            # contains both a and b spin indices
                            print 'overlap, i,j, spin',n_ij, i, j,spin
                            print S_kn
                            # spin blocks
                            if spin == 0:
                                S_kn[k][i][j] += n_ij
                            elif spin==1:
                                S_kn[k][nas + i][nas + j] += n_ij
                                
            # sum over kpts
        print S_kn
        S_k_AB = np.zeros(nk)
        S_k_BA = np.zeros(nk)
        
        for k in range(nk):
            S_k_AB[k] = w_k[k] * np.linalg.det(S_kn[k])
            # determinant of the complex conjugate
            S_k_BA[k] = w_k[k] * np.linalg.det(np.transpose(S_kn[k]).conj())
        
        S_AB = S_k_AB.sum()
        S_BA = S_k_BA.sum()        
        
        # fill 2x2 overlap matrix
        self.S[0][1] = S_AB
        self.S[1][0] = S_BA
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
            
            self.KS_calc_a = GPAW(self.wfs_a,external = None, 
                fixdensity = True, txt = 'Initial_KS.txt')
            
            self.KS_calc_b = GPAW(self.wfs_b, external = None,
                fixdensity = True, txt = 'Final_KS.txt')
            
            self.KS_calc_a.scf.converged = False  
            self.KS_calc_b.scf.converged = False

            atomsA = self.calc_a.atoms.copy()
            atomsB = self.calc_b.atoms.copy()
        
            atomsA.set_calculator(self.KS_calc_a)
            self.H_AA = atomsA.get_potential_energy()
        
            atomsB.set_calculator(self.KS_calc_b)
            self.H_BB = atomsB.get_potential_energy()
        
    def get_reorganization_energy(self):
        # get Ea (Rb) - Ea(Ra) -->
        # cdft energy at geometry Rb
        # with charge constraint A
        geometry = self.calc_b.get_atoms()
        
        cdft = self.cdft_a
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
            print full_and_partial
            # get number of full and partial
            occup_ks[k][s] += full_and_partial.sum()
        return occup_ks
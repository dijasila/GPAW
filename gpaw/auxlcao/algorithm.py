import numpy as np

from gpaw.auxlcao.procedures import calculate_W_LL_offdiagonals_multipole,\
                                    get_W_LL_diagonals_from_setups,\
                                    calculate_local_I_LMM,\
                                    grab_local_W_LL,\
                                    add_to_global_P_LMM,\
                                    calculate_V_AA,\
                                    calculate_S_AA,\
                                    calculate_M_AA,\
                                    calculate_I_AMM,\
                                    calculate_P_AMM,\
                                    reference_I_AMM,\
                                    reference_W_AA
   


from gpaw.utilities import (pack_atomic_matrices, unpack_atomic_matrices,
                            unpack2, unpack, packed_index, pack, pack2)


from gpaw.auxlcao.matrix_elements import MatrixElements

class RIAlgorithm:
    def __init__(self, exx_fraction):
        self.exx_fraction = exx_fraction

class RIMPV(RIAlgorithm):
    def __init__(self, exx_fraction):
        RIAlgorithm.__init__(self, exx_fraction)
        self.lmax = 2
        self.matrix_elements = MatrixElements()

    def initialize(self, density, hamiltonian, wfs):
        self.density = density
        self.hamiltonian = hamiltonian
        self.wfs = wfs
        self.matrix_elements.initialize(density, hamiltonian, wfs)
        self.timer = hamiltonian.timer

    def set_positions(self, spos_ac):
        self.spos_ac = spos_ac
        with self.timer('calculate W_LL'):
            self.W_LL = calculate_W_LL_offdiagonals_multipole(\
                           self.hamiltonian.gd.cell_cv, 
                           spos_ac,
                           self.hamiltonian.gd.pbc_c,
                           self.wfs.kd.ibzk_qc,
                           self.wfs.dtype,
                           self.lmax)
            get_W_LL_diagonals_from_setups(self.W_LL, self.lmax, self.density.setups)

        auxt_aj = [ setup.auxt_j for setup in self.wfs.setups ]
        M_aj = [ setup.M_j for setup in self.wfs.setups ]


        gd = self.hamiltonian.gd
        ibzq_qc = np.array([[0.0, 0.0, 0.0]])
        dtype = self.wfs.dtype
        self.matrix_elements.set_positions_and_cell(spos_ac, 
                                                    gd.cell_cv,
                                                    gd.pbc_c,
                                                    ibzq_qc,
                                                    dtype)

        """
            Screened Coulomb integrals

        """
        with self.timer('calculate W_AA'):
            self.V_AA = calculate_V_AA(auxt_aj, M_aj, self.W_LL, self.lmax)

        with self.timer('calculate S_AA'):
            self.S_AA = calculate_S_AA(self.matrix_elements)

        with self.timer('calculate M_AA'):
            self.M_AA = calculate_M_AA(self.matrix_elements, auxt_aj, M_aj, self.lmax)

        self.W_AA = self.V_AA + self.S_AA + self.M_AA + self.M_AA.T


        with self.timer('Calculate I_AMM'):
            self.I_AMM = calculate_I_AMM(self.matrix_elements)

        with self.timer('Calculate P_AMM'):
            self.P_AMM = calculate_P_AMM(self.matrix_elements, self.W_AA)

        with self.timer('Calculate reference I_AMM'):
            self.Iref_AMM = reference_I_AMM(self.wfs, self.density, self.hamiltonian, self.hamiltonian.poisson, auxt_aj, spos_ac)

        self.iW_AA = np.linalg.inv(self.W_AA)
        self.Pref_AMM = np.einsum('AB,Bij->Aij', self.iW_AA, self.Iref_AMM)
      
        with open('RIMPV-Pref_AMM.npy', 'wb') as f:
            np.save(f, self.Pref_AMM)
        with open('RIMPV-P_AMM.npy', 'wb') as f:
            np.save(f, self.P_AMM)

        with open('RIMPV-Iref_AMM.npy', 'wb') as f:
            np.save(f, self.Iref_AMM)
        with open('RIMPV-I_AMM.npy', 'wb') as f:
            np.save(f, self.I_AMM)


        with self.timer('calculate reference W_AA'):
            self.Wref_AA = reference_W_AA(self.density, self.hamiltonian.poisson, auxt_aj, spos_ac)
            print(self.V_AA,'V_AA')
            print(self.W_AA,'V_AA+S_AA+M_AA+M_AA.T')
            print(self.Wref_AA,'W_REF')
            print(np.linalg.norm(self.W_AA-self.Wref_AA),'norm error')

        with open('RIMPV-Wref_AA.npy', 'wb') as f:
            np.save(f, self.Wref_AA)
        with open('RIMPV-W_AA.npy', 'wb') as f:
            np.save(f, self.W_AA)


        xxx
        self.Wh_LL = np.linalg.cholesky(self.W_LL)

        # Debugging inverse
        self.iW_LL = np.linalg.inv(self.W_LL)
        assert np.linalg.norm(self.iW_LL-self.iW_LL.T)<1e-10

        # Debugging: Reconstruct W_LL from Wh_LL
        print(self.Wh_LL)
        assert np.linalg.norm(self.Wh_LL @ self.Wh_LL.T - self.W_LL)<1e-10
        print(self.Wh_LL @ self.Wh_LL.T - self.W_LL)

        # Number of atoms
        Na = len(spos_ac)

        # Number of compensation charges per atom
        locLmax = (self.lmax+1)**2

        # Number of total compensation charges
        Ltot = Na * locLmax

        # Number of atomic orbitals
        nao = self.wfs.setups.nao

        """

            P_LMM is a projection operator to the auxiliary compensation charge subspace

        """

        self.P_LMM = np.zeros( (Ltot, nao, nao) )

        with open('RIMPV-I_LMM.npy', 'wb') as f:
            np.save(f, self.P_LMM)
            xxx

        
        Msize = 5
        for a1 in range(2):
            locP_LMM = self.P_LMM[ a1*9:(a1+1)*9 ]
            for a2 in range(2):
                for a3 in range(2):
                    if a2 == a1 or a3 == a1:
                        locP_LMM[:,  a2*Msize: (a2+1)*Msize, a3*Msize:(a3+1)*Msize ] = self.matrix_elements.evaluate_3ci_LMM(a1,a2,a3)
                    else:
                        locP_LMM[:, a2*Msize: (a2+1)*Msize, a3*Msize:(a3+1)*Msize ] = np.nan


        self.I_LMM = self.P_LMM.copy()


        #with self.timer('3ci: build I_LMM'):
        if 0:
            a1a2_p = self.matrix_elements.a1a2_p
            for apair in a1a2_p:
                """                 
                     loc (a1,a2)    [  W[a1,a1]_LL   W[a1, a2]_LL ]
                    W    =          [                             ]
                     LL             [  W[a2,a1]_LL   W[a2, a2]_LL ]
                """
                Wloc_LL, Lslices = grab_local_W_LL(self.W_LL, apair, self.lmax)

                Iloc_LMM, slicing_internals = calculate_local_I_LMM(self.matrix_elements, apair, self.lmax)
                iWloc_LL = np.linalg.inv(Wloc_LL)
                result = np.einsum('LO,OMN->LMN', iWloc_LL, Iloc_LMM)
                add_to_global_P_LMM(self.P_LMM, result, slicing_internals)


        """

            Compensation charge contributions to P_LMM

        """
        #with self.timer('3ci: Compensation charge corrections'):
        if 0:
            for a, setup in enumerate(self.wfs.setups):
                Delta_iiL = setup.Delta_iiL
                P_Mi = self.wfs.atomic_correction.P_aqMi[a][0]
                self.P_LMM[a*locLmax:(a+1)*locLmax, :, :] += \
                          np.einsum('Mi,ijL,Nj->LMN', 
                          P_Mi, Delta_iiL, P_Mi, optimize=True)


        """
            
            Premultiply with cholesky
        """
        print('xxx not premultiplying with cholesky')
        #self.P_LMM = np.einsum('LO,OMN->LMN', self.Wh_LL, self.P_LMM)

    def cube_debug(self, atoms):
        from ase.io.cube import write_cube

        from gpaw.lfc import LFC
        C_AMM = np.einsum('AB,Bkl',
                           self.iW_LL,
                           self.I_LMM, optimize=True)


        gaux_lfc_coarse = LFC(self.density.gd, [setup.gaux_l for setup in self.wfs.setups])
        gaux_lfc_coarse.set_positions(self.spos_ac)
        nao = self.wfs.setups.nao

        for M1 in range(nao):
            for M2 in range(nao):
                A = 0
                Q_aM = gaux_lfc_coarse.dict(zero=True)
                for a, setup in enumerate(self.wfs.setups):
                    Aloc = 0
                    for j, aux in enumerate(setup.gaux_l):
                        for m in range(2*aux.l+1):
                            Q_aM[a][Aloc] = C_AMM[A, M1, M2]
                            A += 1
                            Aloc += 1
                fit_G = self.density.gd.zeros()
                gaux_lfc_coarse.add(fit_G, Q_aM)
                write_cube(open('new_%03d_%03d.cube' % (M1, M2),'w'), atoms, data=fit_G, comment='')



        xxx

    def nlxc(self, H_MM, dH_asp, wfs, kpt):
        evc = 0.0
        evv = 0.0
        ekin = 0.0

        rho_MM = wfs.ksl.calculate_density_matrix(kpt.f_n, kpt.C_nM)
        rho_MM[:] = 0.0
        rho_MM[0,0] = 1.0

        C_AMM = np.einsum('Bkl,jl',
                          self.P_LMM,
                          rho_MM, optimize=True)

        with open('RMPV-C_AMM.npy', 'wb') as f:
            np.save(f, C_AMM)


        F_MM = -0.5 * np.einsum('Aij,Akl,jl',
                                self.P_LMM,
                                self.P_LMM,
                                rho_MM, optimize=True)


        with open('RIMPV-F_MM.npy', 'wb') as f:
            np.save(f, F_MM)
            xxx
        H_MM += self.exx_fraction * F_MM
        evv = 0.5 * self.exx_fraction * np.einsum('ij,ij', F_MM, rho_MM)

        for a in dH_asp.keys():
            #print(a)
            D_ii = unpack2(self.density.D_asp[a][0]) / 2 # Check 1 or 2
            # Copy-pasted from hybrids/pw.py
            ni = len(D_ii)
            V_ii = np.empty((ni, ni))
            for i1 in range(ni):
                for i2 in range(ni):
                    V = 0.0
                    for i3 in range(ni):
                        p13 = packed_index(i1, i3, ni)
                        for i4 in range(ni):
                            p24 = packed_index(i2, i4, ni)
                            V += self.density.setups[a].M_pp[p13, p24] * D_ii[i3, i4]
                    V_ii[i1, i2] = +V
            V_p = pack2(V_ii)
            dH_asp[a][0] += (-V_p - self.density.setups[a].X_p) * self.exx_fraction

            evv -= self.exx_fraction * np.dot(V_p, self.density.D_asp[a][0]) / 2
            evc -= self.exx_fraction * np.dot(self.density.D_asp[a][0], self.density.setups[a].X_p)

        ekin = -2*evv -evc

        return evv, evc, ekin 

    def get_description(self):
        return 'RI-V FullMetric: Resolution of identity Coulomb-metric fit to full auxiliary space RI-V'

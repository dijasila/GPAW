import numpy as np

from gpaw.auxlcao.procedures import calculate_W_LL_offdiagonals_multipole,\
                                    get_W_LL_diagonals_from_setups,\
                                    calculate_local_I_LMM,\
                                    grab_local_W_LL,\
                                    add_to_global_P_LMM,\
                                    calculate_V_AA,\
                                    calculate_S_AA,\
                                    calculate_M_AA,\
                                    calculate_W_AL,\
                                    calculate_I_AMM,\
                                    calculate_P_AMM,\
                                    calculate_P_LMM,\
                                    reference_I_AMM,\
                                    reference_W_AA
   
from gpaw.auxlcao.utilities import safe_inv

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
        self.timer = hamiltonian.timer
        with self.timer('Auxiliary Fourier-Bessel initialization'):
            self.matrix_elements.initialize(density, hamiltonian, wfs)

    def set_positions(self, spos_ac, debug):
        self.spos_ac = spos_ac
        print('W_LL')
        with self.timer('calculate W_LL'):
            self.W_LL = calculate_W_LL_offdiagonals_multipole(\
                           self.hamiltonian.gd.cell_cv, 
                           spos_ac,
                           self.hamiltonian.gd.pbc_c,
                           self.wfs.kd.ibzk_qc,
                           self.wfs.dtype,
                           self.lmax)
            get_W_LL_diagonals_from_setups(self.W_LL, self.lmax, self.density.setups)
            assert not np.isnan(self.W_LL).any()

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
        print('V_AA')
        with self.timer('calculate V_AA'):
            self.V_AA = calculate_V_AA(auxt_aj, M_aj, self.W_LL, self.lmax)
            assert not np.isnan(self.V_AA).any()

        print('S_AA')
        with self.timer('calculate S_AA'):
            self.S_AA = calculate_S_AA(self.matrix_elements)
            assert not np.isnan(self.S_AA).any()

        print('M_AA')
        with self.timer('calculate M_AA'):
            self.M_AA = calculate_M_AA(self.matrix_elements, auxt_aj, M_aj, self.lmax)
            self.W_AA = self.V_AA + self.S_AA + self.M_AA + self.M_AA.T
            assert not np.isnan(self.M_AA).any()

        print('P_AMM')
        with self.timer('Calculate P_AMM'):
            self.P_AMM = calculate_P_AMM(self.matrix_elements, self.W_AA)
            assert not np.isnan(self.P_AMM).any()

        print('P_LMM')
        with self.timer('Calculate P_LMM'):
            self.P_LMM = calculate_P_LMM(self.matrix_elements, self.wfs.setups, self.wfs.atomic_correction)
            assert not np.isnan(self.P_LMM).any()

        print('W_AL')
        with self.timer('Calculate W_AL'):
            self.W_AL = calculate_W_AL(self.matrix_elements, auxt_aj, M_aj, self.W_LL)
            assert not np.isnan(self.W_AL).any()

        with self.timer('Calculate WP_AMM'):
            print('W_AA @ P_AMM')
            self.WP1_AMM = np.einsum('AB,Bij',self.W_AA, self.P_AMM, optimize=True)
            print('W_AL @ P_LMM')
            self.WP2_AMM = np.einsum('AB,Bij',self.W_AL, self.P_LMM, optimize=True)

        with self.timer('Calculate WP_LMM'):
            print('W_LA @ P_AMM')
            self.WP3_LMM = np.einsum('BA,Bij',self.W_AL, self.P_AMM, optimize=True)
            print('W_LL @ P_LMM')
            self.WP4_LMM = np.einsum('AB,Bij',self.W_LL, self.P_LMM, optimize=True)

        """

              P     W    P     rho
               AMM   AA   AMM     MM

               
              P      W   P     rho
               LMM    LA  AMM     MM

                [   W       W    ]
                [    AA      AL  ]
                [                ]
                [   W       W    ]
                [    LA      LL  ]


            P     =
             LMM

                      a        a     a
           P     =   Δ        P     P      
            LMM       Li1i2    i1μ   i2μ'

                    a       a    a       a'       a'    a'
           W       Δ       P    P       Δ        P     P      ρ
            LL'     Li1i2   i1μ  i2μ'    L'i3i4   i3ν   i4ν'    μ'ν'




                    a       a    a       a'       a'    a'
           W       Δ       P    P       Δ        P     P      ρ
            LL'     Li1i2   i1μ  i2μ'    L'i3i4   i3ν   i4ν'    μ'ν'




        """

        if debug['ref']:
            with self.timer('Calculate reference I_AMM'):
                self.Iref_AMM = reference_I_AMM(self.wfs, self.density, self.hamiltonian, self.hamiltonian.poisson, auxt_aj, spos_ac)

            with self.timer('Calculate I_AMM'):
                self.I_AMM = calculate_I_AMM(self.matrix_elements)

            self.iW_AA = safe_inv(self.W_AA)
            self.Pref_AMM = np.einsum('AB,Bij->Aij', self.iW_AA, self.Iref_AMM)
            self.K_MMMM = np.einsum('Aij,AB,Bkl', self.P_AMM, self.W_AA, self.P_AMM, optimize=True)
            self.Kref_MMMM = np.einsum('Aij,AB,Bkl', self.Iref_AMM, self.iW_AA, self.Iref_AMM, optimize=True)
      
            with open('RIMPV-K_MMMM.npy', 'wb') as f:
                np.save(f, self.K_MMMM)
            with open('RIMPV-Kref_MMMM.npy', 'wb') as f:
                np.save(f, self.Kref_MMMM)

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

            with open('RIMPV-Wref_AA.npy', 'wb') as f:
                np.save(f, self.Wref_AA)
            with open('RIMPV-W_AA.npy', 'wb') as f:
                np.save(f, self.W_AA)


        """
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

        with self.timer('Calculate rho'):
            rho_MM = wfs.ksl.calculate_density_matrix(kpt.f_n, kpt.C_nM)

        with self.timer('1st contractions'):
            WP1_AMM_RHO_MM = np.einsum('Ajl,kl',
                                        self.WP1_AMM,
                                        rho_MM, optimize=True)
            WP2_AMM_RHO_MM = np.einsum('Ajl,kl',
                                        self.WP2_AMM,
                                        rho_MM, optimize=True)
            WP3_LMM_RHO_MM = np.einsum('Ajl,kl',
                                       self.WP3_LMM,
                                       rho_MM, optimize=True)
            WP4_LMM_RHO_MM = np.einsum('Ajl,kl',
                                       self.WP4_LMM,
                                       rho_MM, optimize=True)

        with self.timer('2nd contractions'):
            F1_MM = np.einsum('Aij,Ajl',
                               self.P_LMM,
                               WP4_LMM_RHO_MM,
                               optimize=True)
            F2_MM = np.einsum('Aij,Ajl',
                              self.P_AMM,
                              WP2_AMM_RHO_MM,
                               optimize=True) 
            F3_MM = np.einsum('Aij,Ajl',
                               self.P_LMM,
                               WP3_LMM_RHO_MM,
                               optimize=True)
            F4_MM = np.einsum('Aij,Ajl',
                              self.P_AMM,
                              WP1_AMM_RHO_MM,
                               optimize=True) 
        print('F1',F1_MM, '\nF2',F2_MM, '\nF3',F3_MM, '\nF4',F4_MM,'Fs')
        F_MM = -0.5*(F1_MM+F2_MM+F3_MM+F4_MM)
        H_MM += (self.exx_fraction) * F_MM 

        evv = 0.5 * self.exx_fraction * np.einsum('ij,ij', F_MM, rho_MM, optimize=True)
        print('evv',evv)
        with self.timer('RI Local atomic corrections'):
            for a in dH_asp.keys():
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

import numpy as np
from typing import Tuple, Dict
from gpaw.auxlcao.procedures import calculate_W_LL_multipole,\
                                    calculate_W_LL_multipole_screened,\
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
                                    reference_W_AA,\
                                    reference_W_AA_screened

from gpaw.auxlcao.utilities import safe_inv
from gpaw.utilities import (pack_atomic_matrices, unpack_atomic_matrices,
                            unpack2, unpack, packed_index, pack, pack2)

from gpaw.auxlcao.matrix_elements import MatrixElements

class RIAlgorithm:
    def __init__(self, exx_fraction, screening_omega):
        self.exx_fraction = exx_fraction
        self.screening_omega = screening_omega


class RILVL(RIAlgorithm):
    def __init__(self, exx_fraction=None, screening_omega=None, threshold=None):
        RIAlgorithm.__init__(self, exx_fraction, screening_omega)
        self.lmax = 2
        assert exx_fraction is not None
        assert screening_omega is not None
        assert threshold is not None

        self.matrix_elements = MatrixElements(self.lmax, screening_omega, threshold=threshold)

        if self.screening_omega == 0.0:
            self.calculate_W_LL = calculate_W_LL_multipole
        else:
            self.calculate_W_LL = calculate_W_LL_multipole_screened

    def initialize(self, density, hamiltonian, wfs):
        self.density = density
        self.hamiltonian = hamiltonian
        self.wfs = wfs
        self.timer = hamiltonian.timer

    def set_positions(self, spos_ac, debug):
        with self.timer('RI-V: Auxiliary Fourier-Bessel initialization'):
            self.matrix_elements.initialize(self.density, self.hamiltonian, self.wfs)
        self.spos_ac = spos_ac
        print('W_LL')
        with self.timer('RI-V: calculate W_LL'):
            self.W_LL = self.calculate_W_LL(self.density.setups,\
                               self.hamiltonian.gd.cell_cv, 
                               spos_ac,
                               self.hamiltonian.gd.pbc_c,
                               self.wfs.kd.ibzk_qc,
                               self.wfs.dtype,
                               self.lmax, omega=self.screening_omega)
                
            #get_W_LL_diagonals_from_setups(self.W_LL, self.lmax, self.density.setups)

            with open('RIMPV-W_LL_%.5f.npy' % self.screening_omega, 'wb') as f:
                np.save(f, self.W_LL)

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
        with self.timer('RI-V: calculate V_AA'):
            self.V_AA = calculate_V_AA(auxt_aj, M_aj, self.W_LL, self.lmax)
            assert not np.isnan(self.V_AA).any()

        with open('RIMPV-V_AA_%.5f.npy' % self.screening_omega, 'wb') as f:
            np.save(f, self.V_AA)


        print('S_AA')
        with self.timer('RI-V: calculate S_AA'):
            self.S_AA = calculate_S_AA(self.matrix_elements)
            assert not np.isnan(self.S_AA).any()

        with open('RIMPV-S_AA_%.5f.npy' % self.screening_omega, 'wb') as f:
            np.save(f, self.S_AA)

        print('M_AA')
        with self.timer('RI-V: calculate M_AA'):
            self.M_AA = calculate_M_AA(self.matrix_elements, auxt_aj, M_aj, self.lmax)
            self.W_AA = self.V_AA + self.S_AA + self.M_AA + self.M_AA.T
            print(self.W_AA)
            assert not np.isnan(self.M_AA).any()

        with open('RIMPV-M_AA_%.5f.npy' % self.screening_omega, 'wb') as f:
            np.save(f, self.M_AA)


        print('P_AMM')
        with self.timer('RI-V: Calculate P_AMM'):
            self.P_AMM = calculate_P_AMM(self.matrix_elements, self.W_AA)
            assert not np.isnan(self.P_AMM).any()

        print('P_LMM')
        with self.timer('RI-V: Calculate P_LMM'):
            self.P_LMM = calculate_P_LMM(self.matrix_elements, self.wfs.setups, self.wfs.atomic_correction)
            assert not np.isnan(self.P_LMM).any()

        print('W_AL')
        with self.timer('RI-V: Calculate W_AL'):
            self.W_AL = calculate_W_AL(self.matrix_elements, auxt_aj, M_aj, self.W_LL)
            assert not np.isnan(self.W_AL).any()

        with self.timer('RI-V: Calculate WP_AMM'):
            print('W_AA @ P_AMM')
            self.WP_AMM = np.einsum('AB,Bij',self.W_AA, self.P_AMM, optimize=True)
            print('W_AL @ P_LMM')
            self.WP_AMM += np.einsum('AB,Bij',self.W_AL, self.P_LMM, optimize=True)

        with self.timer('RI-V: Calculate WP_LMM'):
            print('W_LA @ P_AMM')
            self.WP_LMM = np.einsum('BA,Bij',self.W_AL, self.P_AMM, optimize=True)
            print('W_LL @ P_LMM')
            self.WP_LMM += np.einsum('AB,Bij',self.W_LL, self.P_LMM, optimize=True)

        r"""

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

            with self.timer('calculate reference W_AA'):
                self.Wref_AA = reference_W_AA_screened(self.density, self.hamiltonian.poisson, auxt_aj, spos_ac, self.screening_omega)

            with open('RIMPV-Wref_AA.npy', 'wb') as f:
                np.save(f, self.Wref_AA)
            with open('RIMPV-W_AA.npy', 'wb') as f:
                np.save(f, self.W_AA)

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

            xxx

            with self.timer('calculate reference W_AA'):
                self.Wref_AA = reference_W_AA(self.density, self.hamiltonian.poisson, auxt_aj, spos_ac)

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

    def nlxc(self, 
             H_MM:np.ndarray,
             dH_asp:Dict[int,np.ndarray],
             wfs,
             kpt) -> Tuple[float, float, float]:
        evc = 0.0
        evv = 0.0
        ekin = 0.0

        with self.timer('Calculate rho'):
            rho_MM = wfs.ksl.calculate_density_matrix(kpt.f_n, kpt.C_nM)

        with self.timer('RI-V: 1st contraction AMM MM'):
            WP_AMM_RHO_MM = np.einsum('Ajl,kl',
                                        self.WP_AMM,
                                        rho_MM, optimize=True)

        """ Just testing wh
            sA, sM1, sM2 = self.WP_AMM.shape
        with self.timer('RI-V: 1st contraction AMM MM with np.dot'):
            TEMPWP_AMM_RHO_MM = np.reshape(self.WP_AMM, (sA*sM1, sM2)) @ rho_MM
        print(np.max(np.abs(WP_AMM_RHO_MM.ravel()-TEMPWP_AMM_RHO_MM.ravel())),'ERROR dot')

        
        from gpaw.utilities.blas import mmm        
        with self.timer('RI-V: 1st contraction AMM MM width mmm'):
            WP_ZM = np.reshape(self.WP_AMM, (sA*sM1, sM2))
            print(WP_ZM.strides)
            print(rho_MM.strides)
            WR_ZM = np.zeros_like(WP_ZM)
            mmm(1.0, WP_ZM, 'N',
                     rho_MM, 'N',
                0.0, WR_ZM)
        print(np.max(np.abs(WR_ZM.ravel()-TEMPWP_AMM_RHO_MM.ravel())),'ERROR mmm')
        """

        with self.timer('RI-V: 2nd contraction AMM AMM'):
            F_MM = np.einsum('Aik,Ajk',
                              self.P_AMM,
                              WP_AMM_RHO_MM,
                              optimize=True) 
            WP_AMM_RHO_MM = None

        with self.timer('RI-V: 1st contraction LMM MM'):
            WP_LMM_RHO_MM = np.einsum('Ajl,kl',
                                       self.WP_LMM,
                                       rho_MM, optimize=True)

        with self.timer('RI-V: 2nd contraction LMM LMM'):
            F_MM += np.einsum('Aik,Ajk',
                              self.P_LMM,
                              WP_LMM_RHO_MM,
                              optimize=True)
            WP_LMM_RHO_MM = None

        H_MM += -0.5*(self.exx_fraction) * F_MM
        evv = -0.5 * 0.5 * self.exx_fraction * np.einsum('ij,ij', F_MM, rho_MM, optimize=True)

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

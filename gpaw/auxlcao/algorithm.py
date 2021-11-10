import numpy as np

from gpaw.auxlcao.procedures import calculate_W_LL_offdiagonals_multipole,\
                                    get_W_LL_diagonals_from_setups

class RIAlgorithm:
    def __init__(self, exx_fraction):
        self.exx_fraction = exx_fraction

class RIMPV(RIAlgorithm):
    def __init__(self, exx_fraction):
        RIAlgorithm.__init__(self, exx_fraction)
        self.lmax = 2

    def initialize(self, density, hamiltonian, wfs):
        self.density = density
        self.hamiltonian = hamiltonian
        self.wfs = wfs

    def set_positions(self, spos_ac):
        self.W_LL = calculate_W_LL_offdiagonals_multipole(\
                        self.hamiltonian.gd.cell_cv, 
                        spos_ac,
                        self.hamiltonian.gd.pbc_c,
                        self.wfs.kd.ibzk_qc,
                        self.wfs.dtype,
                        self.lmax)
        get_W_LL_diagonals_from_setups(self.W_LL, self.lmax, self.density.setups)
        with open('RIMPV-W_LL.npy', 'wb') as f:
            np.save(f, self.W_LL)
        
    def nlxc(self, H_MM, dH_asp, wfs, kpt):
        rho_MM = wfs.ksl.calculate_density_matrix(kpt.f_n, kpt.C_nM)
        F_MM = -0.5 * np.einsum('Aij,AB,Bkl,jl',
                                self.I_AMM,
                                self.iW_AA,
                                self.I_AMM,
                                rho_MM, optimize=True)
        H_MM += self.exx_fraction * F_MM
        self.evv = 0.5 * self.exx_fraction * np.einsum('ij,ij', F_MM, rho_MM)

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

            #print("Atomic Ex correction", np.dot(V_p, self.density.D_asp[a][0]) / 2)
            #print("Atomic Ex correction", np.trace(V_ii @ D_ii))
            self.evv -= self.exx_fraction * np.dot(V_p, self.density.D_asp[a][0]) / 2
            self.evc -= self.exx_fraction * np.dot(self.density.D_asp[a][0], self.density.setups[a].X_p)

        self.ekin = -2*self.evv -self.evc


    def get_description(self):
        return 'RI-V FullMetric: Resolution of identity Coulomb-metric fit to full auxiliary space RI-V'

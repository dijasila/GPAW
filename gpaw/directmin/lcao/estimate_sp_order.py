from ase.units import Hartree
import numpy as np
from gpaw.utilities import pack, unpack
from gpaw.lfc import LFC
from gpaw.transformers import Transformer
from gpaw.poisson import PoissonSolver


class EstimateSPOrder(object):
    def __init__(self, wfs, dens, ham, poisson_solver='FPS'):

        self.name = 'Estimator'
        self.setups = wfs.setups
        self.bfs = wfs.basis_functions
        self.cgd = wfs.gd
        self.finegd = dens.finegd
        self.ghat = dens.ghat
        self.ghat_cg = None
        self.xc = ham.xc

        if poisson_solver == 'FPS':
            self.poiss = PoissonSolver(use_charge_center=True,
                                       use_charged_periodic_corrections=True)
        elif poisson_solver == 'GS':
            self.poiss = PoissonSolver(name='fd',
                                       relax=poisson_solver,
                                       eps=1.0e-16,
                                       use_charge_center=True,
                                       use_charged_periodic_corrections=True)

        self.poiss.set_grid_descriptor(self.finegd)

        self.interpolator = Transformer(self.cgd, self.finegd, 3)
        self.restrictor = Transformer(self.finegd, self.cgd, 3)
        self.dtype = wfs.dtype
        self.eigv_s = {}
        self.lagr_diag_s = {}
        self.e_sic_by_orbitals = {}
        self.n_kps = wfs.kd.nibzkpts

def get_orbital_potential_matrix(self, f_n, C_nM, kpt,
                                 wfs, setup, m, timer):
    """
    :param f_n:
    :param C_nM:
    :param kpt:
    :param wfs:
    :param setup:
    :return: F_i = <m|v_i + PAW_i|n > for occupied
             F_i = 0 for unoccupied,
             SI energies

    To calculate this, we need to calculate
    orbital-depended potential and PAW corrections to it.

    Fot this, we need firstly to get orbitals densities.

    """
    kpoint = self.n_kps * kpt.s + kpt.q
    n_set = C_nM.shape[1]
    F_MM = np.zeros(shape=(n_set, n_set),
                    dtype=self.dtype)
    # get orbital-density
    timer.start('Construct Density, Charge, adn DM')
    nt_G, Q_aL, D_ap = \
        self.get_density(f_n,
                         C_nM, kpt,
                         wfs, setup, m)
    timer.stop('Construct Density, Charge, adn DM')

    # calculate sic energy,
    # sic pseudo-potential and Hartree
    timer.start('Get Pseudo Potential')
    e_sic_m, vt_mG, vHt_g = \
        self.get_pseudo_pot(nt_G, Q_aL, m, kpoint, timer)
    timer.stop('Get Pseudo Potential')

    # calculate PAW corrections
    timer.start('PAW')
    e_sic_paw_m, dH_ap = \
        self.get_paw_corrections(D_ap, vHt_g, timer)
    timer.stop('PAW')

    # total sic:
    e_sic_m += e_sic_paw_m

    # now calculate potential matrix F_i
    # calculate pseudo-part
    # Vt_MM = \
    #     self.bfs.calculate_potential_matrices(vt_mG)[0]

    # TODO: sum over cell? see calculate_hamiltonian_matrix in
    # eigensolver.py
    timer.start('ODD Potential Matrices')
    Vt_MM = np.zeros_like(F_MM)
    self.bfs.calculate_potential_matrix(vt_mG, Vt_MM, kpt.q)
    # make matrix hermitian
    ind_l = np.tril_indices(Vt_MM.shape[0], -1)
    Vt_MM[(ind_l[1], ind_l[0])] = Vt_MM[ind_l].conj()
    timer.stop('ODD Potential Matrices')

    # np.save('Vt_MM1', Vt_MM)
    #
    # wfs.timer.start('Potential matrix')
    # Vt_xMM = self.bfs.calculate_potential_matrices(vt_mG)
    # wfs.timer.stop('Potential matrix')
    #
    # if self.bfs.gamma and self.dtype is float:
    #     yy = 1.0
    #     Vt_MM = Vt_xMM[0]
    # else:
    #     wfs.timer.start('Sum over cells')
    #     yy = 0.5
    #     k_c = wfs.kd.ibzk_qc[kpt.q]
    #     Vt_MM = (0.5 + 0.0j) * Vt_xMM[0]
    #     for sdisp_c, Vt1_MM in zip(self.bfs.sdisp_xc[1:], Vt_xMM[1:]):
    #         Vt_MM += np.exp(2j * np.pi * np.dot(sdisp_c, k_c)) * Vt1_MM
    #     wfs.timer.stop('Sum over cells')
    #
    # # make matrix hermitian
    # ind_l = np.tril_indices(Vt_MM.shape[0], -1)
    # Vt_MM[(ind_l[1], ind_l[0])] = Vt_MM[ind_l].conj()
    #
    # np.save('Vt_MM2', Vt_MM)

    # PAW:
    timer.start('Potential matrix - PAW')
    for a, dH_p in dH_ap.items():
        P_Mj = wfs.P_aqMi[a][kpt.q]
        dH_ij = unpack(dH_p)
        # dH_ij = yy * unpack(dH_p)

        # K_iM = np.zeros((dH_ij.shape[0], n_set),
        #                 dtype=self.dtype)

        if self.dtype is complex:
            # gemm(1.0, P_Mj,
            #      dH_ij.astype(complex),
            #      0.0, K_iM, 'c')
            # gemm(1.0, K_iM,
            #      P_Mj,
            #      1.0, F_MM)
            F_MM += P_Mj @ dH_ij @ P_Mj.T.conj()
            # K_iM = np.dot(dH_ij, P_Mj.T.conj())
            # F_MM += np.dot(P_Mj, K_iM)

        else:
            # gemm(1.0, P_Mj, dH_ij, 0.0, K_iM, 't')
            # gemm(1.0, K_iM, P_Mj, 1.0, F_MM)

            # K_iM = np.dot(dH_ij, P_Mj.T)
            # F_MM += np.dot(P_Mj, K_iM)
            F_MM += P_Mj @ dH_ij @ P_Mj.T

    if self.dtype is complex:
        F_MM += Vt_MM.astype(complex)
    else:
        F_MM += Vt_MM
    #
    # wfs.timer.start('Distribute overlap matrix')
    # F_MM = wfs.ksl.distribute_overlap_matrix(
    #     F_MM, root=-1, add_hermitian_conjugate=(yy == 0.5))
    # wfs.timer.stop('Distribute overlap matrix')
    #
    if self.sic_coarse_grid:
        self.cgd.comm.sum(F_MM)
    else:
        self.finegd.comm.sum(F_MM)
    timer.stop('Potential matrix - PAW')

    return F_MM, e_sic_m * f_n[m]


def get_density(self, f_n, C_nM, kpt,
                wfs, setup, m):
    # construct orbital density matrix
    if f_n[m] > 1.0 + 1.0e-4:
        occup_factor = f_n[m] / (3.0 - wfs.nspins)
    else:
        occup_factor = f_n[m]
    rho_MM = occup_factor * np.outer(C_nM[m].conj(), C_nM[m])

    nt_G = self.cgd.zeros()
    self.bfs.construct_density(rho_MM, nt_G, kpt.q)

    # calculate  atomic density matrix and
    # compensation charges
    D_ap = {}
    Q_aL = {}

    for a in wfs.P_aqMi.keys():
        P_Mi = wfs.P_aqMi[a][kpt.q]
        # rhoP_Mi = np.zeros_like(P_Mi)
        # gemm(1.0, P_Mi, rho_MM, 0.0, rhoP_Mi)
        D_ii = np.zeros((wfs.P_aqMi[a].shape[2],
                         wfs.P_aqMi[a].shape[2]),
                        dtype=self.dtype)
        rhoP_Mi = rho_MM @ P_Mi
        # if self.dtype is complex:
        #     gemm(1.0, rhoP_Mi, P_Mi.T.conj().copy(), 0.0, D_ii)
        # else:
        #     gemm(1.0, rhoP_Mi, P_Mi.T.copy(), 0.0, D_ii)
        D_ii = P_Mi.T.conj() @ rhoP_Mi
        # FIXME: What to do with complex part, which are not zero
        if self.dtype is complex:
            D_ap[a] = D_p = pack(D_ii.real)
        else:
            D_ap[a] = D_p = pack(D_ii)

        Q_aL[a] = np.dot(D_p, setup[a].Delta_pL)

    return nt_G, Q_aL, D_ap


def get_pseudo_pot(self, nt, Q_aL, i, kpoint, timer):
    if self.sic_coarse_grid is False:
        # change to fine grid
        vt_sg = self.finegd.zeros(2)
        vHt_g = self.finegd.zeros()
        nt_sg = self.finegd.zeros(2)
    else:
        vt_sg = self.cgd.zeros(2)
        vHt_g = self.cgd.zeros()
        nt_sg = self.cgd.zeros(2)

    if self.sic_coarse_grid is False:
        self.interpolator.apply(nt, nt_sg[0])
        nt_sg[0] *= self.cgd.integrate(nt) / \
                    self.finegd.integrate(nt_sg[0])
    else:
        nt_sg[0] = nt

    timer.start('ODD XC 3D grid')
    if self.sic_coarse_grid is False:
        e_xc = self.xc.calculate(self.finegd, nt_sg, vt_sg)
    else:
        e_xc = self.xc.calculate(self.cgd, nt_sg, vt_sg)
    timer.stop('ODD XC 3D grid')
    vt_sg[0] *= -self.beta_x

    # Hartree
    if self.sic_coarse_grid is False:
        self.ghat.add(nt_sg[0], Q_aL)
    else:
        self.ghat_cg.add(nt_sg[0], Q_aL)

    timer.start('ODD Poisson')
    if self.store_potentials:
        if self.sic_coarse_grid:
            vHt_g = self.old_pot[kpoint][i]
        else:
            self.interpolator.apply(self.old_pot[kpoint][i],
                                    vHt_g)
    self.poiss.solve(vHt_g, nt_sg[0],
                     zero_initial_phi=self.store_potentials,
                     timer=timer)
    if self.store_potentials:
        if self.sic_coarse_grid:
            self.old_pot[kpoint][i] = vHt_g.copy()
        else:
            self.restrictor.apply(vHt_g, self.old_pot[kpoint][i])

    timer.stop('ODD Poisson')

    timer.start('ODD Hartree integrate')
    if self.sic_coarse_grid is False:
        ec = 0.5 * self.finegd.integrate(nt_sg[0] * vHt_g)
    else:
        ec = 0.5 * self.cgd.integrate(nt_sg[0] * vHt_g)

    timer.stop('ODD Hartree integrate')
    vt_sg[0] -= vHt_g * self.beta_c
    if self.sic_coarse_grid is False:
        vt_G = self.cgd.zeros()
        self.restrictor.apply(vt_sg[0], vt_G)
    else:
        vt_G = vt_sg[0]

    return np.array([-ec * self.beta_c,
                     -e_xc * self.beta_x]), vt_G, vHt_g


def get_paw_corrections(self, D_ap, vHt_g, timer):
    # XC-PAW
    timer.start('xc-PAW')
    dH_ap = {}
    exc = 0.0
    for a, D_p in D_ap.items():
        setup = self.setups[a]
        dH_sp = np.zeros((2, len(D_p)))
        D_sp = np.array([D_p, np.zeros_like(D_p)])
        exc += self.xc.calculate_paw_correction(setup, D_sp,
                                                dH_sp,
                                                addcoredensity=False,
                                                a=a)
        dH_ap[a] = -dH_sp[0] * self.beta_x
    timer.stop('xc-PAW')

    # Hartree-PAW
    timer.start('Hartree-PAW')
    ec = 0.0
    timer.start('ghat-PAW')
    if self.sic_coarse_grid is False:
        W_aL = self.ghat.dict()
        self.ghat.integrate(vHt_g, W_aL)
    else:
        W_aL = self.ghat_cg.dict()
        self.ghat_cg.integrate(vHt_g, W_aL)
    timer.stop('ghat-PAW')

    for a, D_p in D_ap.items():
        setup = self.setups[a]
        M_p = np.dot(setup.M_pp, D_p)
        ec += np.dot(D_p, M_p)
        dH_ap[a] += -(2.0 * M_p + np.dot(setup.Delta_pL,
                                         W_aL[a])) * self.beta_c
    timer.stop('Hartree-PAW')

    timer.start('Wait for sum')
    if self.sic_coarse_grid is False:
        ec = self.finegd.comm.sum(ec)
        exc = self.finegd.comm.sum(exc)
    else:
        ec = self.cgd.comm.sum(ec)
        exc = self.cgd.comm.sum(exc)
    timer.stop('Wait for sum')

    return np.array([-ec * self.beta_c, -exc * self.beta_x]), dH_ap

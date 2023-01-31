from ase.units import Hartree
import numpy as np
from gpaw.utilities import pack, unpack
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

    def get_orbital_potential_matrix(self, f_n, C_nM, kpt,
                                 wfs, setup, m, timer):

        n_set = C_nM.shape[1]
        F_MM = np.zeros(shape=(n_set, n_set), dtype=self.dtype)

        timer.start('Construct Density, Charge, adn DM')
        nt_G, Q_aL, D_ap = self.get_density(f_n, C_nM, kpt, wfs, setup, m)
        timer.stop('Construct Density, Charge, adn DM')

        timer.start('Get Pseudo Potential')
        e_sic_m, vt_mG, vHt_g = self.get_pseudo_pot(nt_G, Q_aL, timer)
        timer.stop('Get Pseudo Potential')

        timer.start('PAW')
        e_sic_paw_m, dH_ap = self.get_paw_corrections(D_ap, vHt_g, timer)
        timer.stop('PAW')

        e_sic_m += e_sic_paw_m

        timer.start('ODD Potential Matrices')
        Vt_MM = np.zeros_like(F_MM)
        self.bfs.calculate_potential_matrix(vt_mG, Vt_MM, kpt.q)
        ind_l = np.tril_indices(Vt_MM.shape[0], -1)
        Vt_MM[(ind_l[1], ind_l[0])] = Vt_MM[ind_l].conj()
        timer.stop('ODD Potential Matrices')

        timer.start('Potential matrix - PAW')
        for a, dH_p in dH_ap.items():
            P_Mj = wfs.P_aqMi[a][kpt.q]
            dH_ij = unpack(dH_p)
            F_MM += P_Mj @ dH_ij @ np.conj(P_Mj.T)

        if self.dtype is complex:
            F_MM += Vt_MM.astype(complex)
        else:
            F_MM += Vt_MM
        self.finegd.comm.sum(F_MM)
        timer.stop('Potential matrix - PAW')

        return F_MM, e_sic_m * f_n[m]

    def get_density(self, f_n, C_nM, kpt, wfs, setup, m):

        if f_n[m] > 1.0 + 1.0e-4:
            occup_factor = f_n[m] / (3.0 - wfs.nspins)
        else:
            occup_factor = f_n[m]
        rho_MM = occup_factor * np.outer(C_nM[m].conj(), C_nM[m])

        nt_G = self.cgd.zeros()
        self.bfs.construct_density(rho_MM, nt_G, kpt.q)

        D_ap = {}
        Q_aL = {}
        for a in wfs.P_aqMi.keys():
            P_Mi = wfs.P_aqMi[a][kpt.q]
            rhoP_Mi = rho_MM @ P_Mi
            D_ii = P_Mi.T.conj() @ rhoP_Mi
            D_ap[a] = D_p = pack(np.real(D_ii))
            Q_aL[a] = np.dot(D_p, setup[a].Delta_pL)

        return nt_G, Q_aL, D_ap

    def get_pseudo_pot(self, nt, Q_aL, timer):
        vt_sg = self.finegd.zeros(2)
        vHt_g = self.finegd.zeros()
        nt_sg = self.finegd.zeros(2)

        self.interpolator.apply(nt, nt_sg[0])
        nt_sg[0] *= self.cgd.integrate(nt) / self.finegd.integrate(nt_sg[0])

        timer.start('ODD XC 3D grid')
        e_xc = self.xc.calculate(self.finegd, nt_sg, vt_sg)
        timer.stop('ODD XC 3D grid')

        # Hartree
        self.ghat.add(nt_sg[0], Q_aL)

        timer.start('ODD Poisson')
        self.poiss.solve(vHt_g, nt_sg[0],
                         zero_initial_phi=False,
                         timer=timer)
        timer.stop('ODD Poisson')

        timer.start('ODD Hartree integrate')
        ec = 0.5 * self.finegd.integrate(nt_sg[0] * vHt_g)
        timer.stop('ODD Hartree integrate')

        vt_sg[0] -= vHt_g
        vt_G = self.cgd.zeros()
        self.restrictor.apply(vt_sg[0], vt_G)

        return np.array([-ec, -e_xc]), vt_G, vHt_g

    def get_paw_corrections(self, D_ap, vHt_g, timer):
        timer.start('xc-PAW')
        dH_ap = {}
        exc = 0.0
        for a, D_p in D_ap.items():
            setup = self.setups[a]
            dH_sp = np.zeros((2, len(D_p)))
            D_sp = np.array([D_p, np.zeros_like(D_p)])
            exc += self.xc.calculate_paw_correction(
                setup, D_sp, dH_sp, addcoredensity=False, a=a)
            dH_ap[a] = -dH_sp[0]
        timer.stop('xc-PAW')

        timer.start('Hartree-PAW')
        ec = 0.0
        timer.start('ghat-PAW')
        W_aL = self.ghat.dict()
        self.ghat.integrate(vHt_g, W_aL)
        timer.stop('ghat-PAW')

        for a, D_p in D_ap.items():
            setup = self.setups[a]
            M_p = np.dot(setup.M_pp, D_p)
            ec += np.dot(D_p, M_p)
            dH_ap[a] += -(2.0 * M_p + np.dot(setup.Delta_pL, W_aL[a]))
        timer.stop('Hartree-PAW')

        ec = self.finegd.comm.sum(ec)
        exc = self.finegd.comm.sum(exc)

        return np.array([-ec, -exc]), dH_ap

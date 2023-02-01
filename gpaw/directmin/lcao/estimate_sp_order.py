from ase.units import Hartree
import numpy as np
from gpaw.utilities import pack
from gpaw.transformers import Transformer
from gpaw.poisson import PoissonSolver
from copy import deepcopy


class EstimateSPOrder(object):
    def __init__(self, wfs, dens, ham, poisson_solver='FPS'):

        self.name = 'Estimator'
        self.setups = wfs.setups
        self.bfs = wfs.basis_functions
        self.cgd = wfs.gd
        self.finegd = dens.finegd
        self.ghat = dens.ghat
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
        self.dtype = wfs.dtype

    def run(self, calc, occ_ex):
        nkpt = len(calc.wfs.kpt_u)
        n_bands = calc.wfs.kpt_u[0].C_nM.shape[1]
        timer = calc.wfs.timer
        assert len(occ_ex) == nkpt, 'Occupation numbers do not match number ' \
                                    'of K-points'
        occ_gs = [deepcopy(calc.wfs.kpt_u[x].f_n) for x in range(nkpt)]
        dens = calc.density
        vHt_g, vt_sg = self.get_coulomb_and_exchange_pseudo_pot(
            dens.rho_tg, timer)
        for k, kpt in enumerate(calc.wfs.kpt_u):
            for n in range(n_bands):
                nt_n, Q_aLn, D_apn = self.get_orbital_density(
                    occ_gs[k][n], kpt.C_nM[n], kpt, calc.wfs, calc.wfs.setups)
                ec_gs = self.integrate_coulomb_and_exchange_per_orbital(
                    vHt_g, vt_sg, nt_n, Q_aLn)

    def coulomb_and_exchange_paw_per_orbital(self, D_apn, timer):

        timer.start('xc-PAW')
        exc = 0.0
        for a, D_pn in D_apn.items():
            setup = self.setups[a]
            dH_spn = np.zeros((2, len(D_pn)))
            D_spn = np.array([D_pn, np.zeros_like(D_pn)])
            exc += self.xc.calculate_paw_correction(
                setup, D_spn, dH_spn, addcoredensity=False, a=a)
        timer.stop('xc-PAW')

        timer.start('Hartree-PAW')
        ec = 0.0
        # timer.start('ghat-PAW')
        # W_aL = self.ghat.dict()
        # self.ghat.integrate(vHt_g, W_aL)
        # timer.stop('ghat-PAW')

        for a, D_p in D_apn.items():
            setup = self.setups[a]
            M_p = np.dot(setup.M_pp, D_p)
            ec += np.dot(D_p, M_p)
        timer.stop('Hartree-PAW')

        ec = self.finegd.comm.sum(ec)
        exc = self.finegd.comm.sum(exc)

        return np.array([-ec, -exc])

    def integrate_coulomb_and_exchange_per_orbital(
        self, vHt_g, vt_sg, nt_n, Q_aLn):

        nt_sg = self.finegd.zeros(2)
        self.interpolator.apply(nt_n, nt_sg[0])
        nt_sg[0] *= self.cgd.integrate(nt_n) / self.finegd.integrate(nt_sg[0])
        self.ghat.add(nt_sg[0], Q_aLn)
        ec = 0.5 * self.finegd.integrate(nt_sg[0] * vHt_g)
        #exc = 0.5 * self.finegd.integrate(nt_sg[0] * vt_sg) how?
        return ec

    def get_coulomb_and_exchange_pseudo_pot(self, nt_sg, rhot_g, timer):
        vt_sg = self.finegd.zeros(2)
        vHt_g = self.finegd.zeros(2)

        timer.start('ODD XC 3D grid')
        e_xc_tot = self.xc.calculate(self.finegd, nt_sg, vt_sg)
        timer.stop('ODD XC 3D grid')

        timer.start('ODD Poisson')
        self.poiss.solve(vHt_g, rhot_g,
                         zero_initial_phi=False,
                         timer=timer)
        timer.stop('ODD Poisson')

        return vHt_g, vt_sg

    def get_sic(self, nt_n, Q_aLn, D_apn, timer):

        timer.start('Get Pseudo Potential')
        e_sic_m, vHt_g = self.get_pseudo_pot(nt_n, Q_aLn, timer)
        timer.stop('Get Pseudo Potential')

        timer.start('PAW')
        e_sic_paw_m = self.get_electron_hole_sic_paw(D_apn, vHt_g, timer)
        timer.stop('PAW')

        e_sic_m += e_sic_paw_m

        return e_sic_m

    def get_orbital_density(self, f, C, kpt, wfs, setup):

        occup_factor = f / (3.0 - wfs.nspins)
        rho_MMn = occup_factor * np.outer(C.conj(), C)

        nt_n = self.cgd.zeros()
        self.bfs.construct_density(rho_MMn, nt_n, kpt.q)

        D_apn = {}
        Q_aLn = {}
        for a in wfs.P_aqMi.keys():
            P_Mi = wfs.P_aqMi[a][kpt.q]
            rhoP_Mi = rho_MMn @ P_Mi
            D_iin = P_Mi.T.conj() @ rhoP_Mi
            D_apn[a] = D_pn = pack(np.real(D_iin))
            Q_aLn[a] = np.dot(D_pn, setup[a].Delta_pL)

        return nt_n, Q_aLn, D_apn

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

        return np.array([-ec, -e_xc]), vHt_g

    def get_electron_hole_sic_paw(self, D_ap, vHt_g, timer):
        timer.start('xc-PAW')
        exc = 0.0
        for a, D_p in D_ap.items():
            setup = self.setups[a]
            dH_sp = np.zeros((2, len(D_p)))
            D_sp = np.array([D_p, np.zeros_like(D_p)])
            exc += self.xc.calculate_paw_correction(
                setup, D_sp, dH_sp, addcoredensity=False, a=a)
        timer.stop('xc-PAW')

        timer.start('Hartree-PAW')
        ec = 0.0
        #timer.start('ghat-PAW')
        #W_aL = self.ghat.dict()
        #self.ghat.integrate(vHt_g, W_aL)
        #timer.stop('ghat-PAW')

        for a, D_p in D_ap.items():
            setup = self.setups[a]
            M_p = np.dot(setup.M_pp, D_p)
            ec += np.dot(D_p, M_p)
        timer.stop('Hartree-PAW')

        ec = self.finegd.comm.sum(ec)
        exc = self.finegd.comm.sum(exc)

        return np.array([-ec, -exc])

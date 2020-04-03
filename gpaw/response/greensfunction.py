import numpy as np
from ase.parallel import parprint
from time import time
from math import pi

class GreensFunction:
    def __init__(self, gw_object):
        self.gw_object = gw_object
        self.gw_object.ite = 0

    def _report(self, msg):
        parprint(f"Greensfunction: {msg}", flush=True)


    def calculate(self, domegas, k_index=None):
        domegas = np.array(domegas)
        gw = self.gw_object

        kd = gw.calc.wfs.kd

        gw.calculate_ks_xc_contribution()
        gw.calculate_exact_exchange()

        if gw.restartfile is not None:
            assert False
            loaded = gw.load_restart_file()
            if not loaded:
                gw.last_q = -1
                gw.previous_sigma = 0.0
                gw.previous_dsigma = 0.0

            else:
                print('Reading ' + str(gw.last_q + 1) +
                      ' q-point(s) from the previous calculation: ' +
                      gw.restartfile + '.sigma.pckl', file=gw.fd)
        else:
            gw.last_q = -1
            gw.previous_sigma = 0.0
            gw.previous_dsigma = 0.0
        gw.fd.flush()


        self.sigma_wskn = np.zeros((len(domegas),) + gw.shape, dtype=np.complex128)
        self.dsigma_wskn = np.zeros((len(domegas),) + gw.shape, dtype=np.complex128)
        
        

        b1, b2 = gw.bands
        nibzk = gw.calc.wfs.kd.nibzkpts
        for i, k in enumerate(gw.kpts):
            for s in range(gw.nspins):
                u = s*nibzk + k
                kpt = gw.calc.wfs.kpt_u[u]
                gw.eps_skn[s, i] = kpt.eps_n[b1:b2]
                gw.f_skn[s, i] = kpt.f_n[b1:b2]/kpt.weight

        gw.qp_skn = gw.eps_skn.copy()
        gw.qp_iskn = np.array([gw.qp_skn])

                
        mykpts = [gw.get_k_point(s, K, n1, n2)
                  for s, K, n1, n2 in gw.mysKn1n2]
        if k_index is not None:
            mykpts = [kpt for kpt in mykpts if gw.kpts.index(kd.bz2ibz_k[kpt.K]) == k_index]
            print("Number of kpts: ", len(mykpts))

        self._report("Finished setup")
        for iterc, (ie, pd0, W0, q_c, m2, W0_GW) in enumerate(gw.calculate_screened_potential()):
            t1 = time()
            self._report(f"At ie = {ie} in calc W")
            for u, kpt1 in enumerate(mykpts):
                K2 = kd.find_k_plus_q(q_c, [kpt1.K])[0]
                kpt2 = gw.get_k_point(kpt1.s, K2, 0, m2, block=True)
                
                k1 = kd.bz2ibz_k[kpt1.K]
                i = gw.kpts.index(k1)
                self.calculate_q(domegas, ie, i, kpt1, kpt2, pd0, W0, W0_GW)

            t2 = time()
            self._report(f"Calculation of iterc = {iterc} took {t2 - t1} seconds.")
            
        self._report("Finished calculation")
        gw.world.sum(self.sigma_wskn)
        gw.world.sum(self.dsigma_wskn)

        na = np.newaxis
        G_wskn = 1/(domegas[na, na, na].T - self.sigma_wskn)

        return G_wskn, self.sigma_wskn, self.dsigma_wskn

    def calculate_q(self, domegas, ie, k, kpt1, kpt2, pd0, W0, W0_GW=None):
        gw = self.gw_object

        if W0_GW is None:
            Ws = [W0]
        else:
            Ws = [W0, W0_GW]

        wfs = gw.calc.wfs

        N_c = pd0.gd.N_c
        i_cG = gw.sign * np.dot(gw.U_cc, np.unravel_index(pd0.Q_qG[0], N_c))

        q_c = wfs.kd.bzk_kc[kpt2.K] - wfs.kd.bzk_kc[kpt1.K]

        shift0_c = q_c - gw.sign * np.dot(gw.U_cc, pd0.kd.bzk_kc[0])
        assert np.allclose(shift0_c.round(), shift0_c)
        shift0_c = shift0_c.round().astype(int)
        
        shift_c = kpt1.shift_c - kpt2.shift_c - shift0_c
        I_G = np.ravel_multi_index(i_cG + shift_c[:, None], N_c, "wrap")
        
        G_Gv = pd0.get_reciprocal_vectors()
        pos_av = np.dot(gw.spos_ac, pd0.gd.cell_cv)
        M_vv = np.dot(pd0.gd.cell_cv.T,
                      np.dot(gw.U_cc.T, np.linalg.inv(pd0.gd.cell_cv).T))

        Q_aGii = []

        for a, Q_Gii in enumerate(gw.Q_aGii):
            x_G = np.exp(1j * np.dot(G_Gv, (pos_av[a] - np.dot(M_vv, pos_av[a]))))
            U_ii = gw.calc.wfs.setups[a].R_sii[gw.s]
            Q_Gii = np.dot(np.dot(U_ii, Q_Gii*x_G[:, None, None]), U_ii.T).transpose(1, 0, 2)
            
            if gw.sign == -1:
                Q_Gii = Q_Gii.conj()
            Q_aGii.append(Q_Gii)

        for n in range(kpt1.n2 - kpt1.n1):
            ut1cc_R = kpt1.ut_nR[n].conj()
            eps1 = kpt1.eps_n[n]
            C1_aGi = [np.dot(Qa_Gii, P1_ni[n].conj())
                      for Qa_Gii, P1_ni in zip(Q_aGii, kpt1.P_ani)]
            n_mG = gw.calculate_pair_densities(ut1cc_R, C1_aGi, kpt2, pd0, I_G)

            if gw.sign == 1:
                n_mG = n_mG.conj()
        
            f_m = kpt2.f_n                
                               
            deps_wm = np.array([eps1 + dw - kpt2.eps_n for dw in domegas])
            
            nn = kpt1.n1 + n - gw.bands[0]

            W = Ws[0]
            sigma_w, dsigma_w = self._calculate_sigma(n_mG, deps_wm, f_m, W, full_sigma=True)
            self.sigma_wskn[:, kpt1.s, k, nn] += sigma_w
            self.dsigma_wskn[:, kpt1.s, k, nn] += dsigma_w



    def calculate_sigma(self, n_mG, deps_wm, f_m, C_swGG):
        gw = self.gw_object
        sigma_w = np.zeros(deps_wm.shape[0], dtype=np.complex128)
        dsigma_w = np.zeros(deps_wm.shape[0], dtype=np.complex128)
        for w_index, deps_m in enumerate(deps_wm):
            sigma, dsigma = gw.calculate_sigma(n_mG, deps_m, f_m, C_swGG, full_sigma=True)
            sigma_w[w_index] += -np.real(sigma)*1j + np.imag(sigma)
            dsigma_w[w_index] += dsigma
        return sigma_w, dsigma_w

    def _calculate_sigma(self, n_mG, deps_wm, f_m, C_swGG, full_sigma=False):
        # parprint("Using optimized calculate sigma", flush=True)
        """Calculates a contribution to the self-energy and its derivative for
        a given (k, k-q)-pair from its corresponding pair-density and
        energy."""
        gw = self.gw_object
        C_swGG = np.array(C_swGG)
        o_wm = abs(deps_wm)
        # Add small number to avoid zeros for degenerate states:
        sgn_wm = np.sign(deps_wm + 1e-15)

        # Pick +i*eta or -i*eta:
        s_wm = (1 + sgn_wm * np.sign(0.5 - f_m[np.newaxis, :])).astype(int) // 2

        beta = (2**0.5 - 1) * gw.domega0 / gw.omega2
        w_wm = (o_wm / (gw.domega0 + beta * o_wm)).astype(int)
        index_inb = np.where(w_wm < len(gw.omega_w) - 1)
        o1_wm = np.empty(o_wm.shape)
        o2_wm = np.empty(o_wm.shape)
        o1_wm[index_inb] = gw.omega_w[w_wm[index_inb]]
        o2_wm[index_inb] = gw.omega_w[w_wm[index_inb] + 1]

        x = 1.0 / (gw.qd.nbzkpts * 2 * pi * gw.vol)
        sigma_w = np.zeros(deps_wm.shape[0], dtype=np.complex128)
        dsigma_w = np.zeros(deps_wm.shape[0], dtype=np.complex128)
        # Performing frequency integration
        for o_w, o1_w, o2_w, sgn_w, s_w, w_w, n_G in zip(o_wm.T, o1_wm.T, o2_wm.T,
                                             sgn_wm.T, s_wm.T, w_wm.T, n_mG):

            # if w >= len(self.omega_w) - 1:
            #     continue

            prefac_w = np.ones(len(w_w))
            prefac_w[w_w >= len(gw.omega_w) - 1] = 0

            C1_wGG = C_swGG[s_w, w_w]
            C2_wGG = C_swGG[s_w, w_w + 1]
            p_w = x * sgn_w
            myn_G = n_G[gw.Ga:gw.Gb]
            
            if full_sigma:
                sigma1_w = p_w * np.dot(myn_G,
                                        (np.dot(C1_wGG, n_G.conj()).T))
                sigma2_w = p_w * np.dot(myn_G,
                                        (np.dot(C2_wGG, n_G.conj()).T))
            else:
                raise NotImplementedError
            sigma_w += ((o_w - o1_w) * sigma2_w + (o2_w - o_w) * sigma1_w) / (o2_w - o1_w) * prefac_w
            dsigma_w += sgn_w * (sigma2_w - sigma1_w) / (o2_w - o1_w) * prefac_w

        return sigma_w, dsigma_w

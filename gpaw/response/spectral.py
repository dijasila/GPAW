#from g0w0 import G0W0
import numpy as np
from time import time



class SpectralFunction:
    def __init__(self, gw_object):
        self.gw_object = gw_object
        self.gw_object.ite = 0


    def calculate(self, domegas, k_index=None):
        gw = self.gw_object

        kd = gw.calc.wfs.kd

        gw.calculate_ks_xc_contribution()
        gw.calculate_exact_exchange()

        if gw.restartfile is not None:
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
        
        
        

        b1, b2 = gw.bands
        nibzk = gw.calc.wfs.kd.nibzkpts
        for i, k in enumerate(gw.kpts):
            for s in range(gw.nspins):
                u = s*nibzk + k
                kpt = gw.calc.wfs.kpt_u[u]
                gw.eps_skn[s, i] = kpt.eps_n[b1:b2]
                gw.f_skn[s, i] = kpt.f_n[b1:b2]/kpt.weight



                
        mykpts = [gw.get_k_point(s, K, n1, n2)
                  for s, K, n1, n2 in gw.mysKn1n2]
        if k_index is not None:
            mykpts = [kpt for kpt in mykpts if gw.kpts.index(kd.bz2ibz_k[kpt.K]) == k_index]
            print("Number of kpts: ", len(mykpts))

        for ie, pd0, W0, q_c, m2, W0_GW in gw.calculate_screened_potential():
            for u, kpt1 in enumerate(mykpts):
                K2 = kd.find_k_plus_q(q_c, [kpt1.K])[0]
                kpt2 = gw.get_k_point(kpt1.s, K2, 0, m2, block=True)
                
                k1 = kd.bz2ibz_k[kpt1.K]
                i = gw.kpts.index(k1)
                self.calculate_q(domegas, ie, i, kpt1, kpt2, pd0, W0, W0_GW)

        gw.world.sum(self.sigma_wskn)

        return self.sigma_wskn

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
            assert np.allclose(deps_wm[0], eps1 - kpt2.eps_n)
            
            nn = kpt1.n1 + n - gw.bands[0]

            #for jj, W in enumerate(Ws):
            W = Ws[0]
            sigma_w = self.calculate_sigma(n_mG, deps_wm, f_m, W)
            self.sigma_wskn[:, kpt1.s, k, nn] += sigma_w



    def calculate_sigma(self, n_mG, deps_wm, f_m, C_swGG):
        gw = self.gw_object
        sigma_w = np.zeros(deps_wm.shape[0], dtype=np.complex128)
        for w_index, deps_m in enumerate(deps_wm):
            sigma, _ = gw.calculate_sigma(n_mG, deps_m, f_m, C_swGG, full_sigma=True)
            sigma_w[w_index] += -np.real(sigma)*1j + np.imag(sigma)
        return sigma_w

    

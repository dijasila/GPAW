import numpy as np

from gpaw.response.pw_parallelization import Blocks1D
from gpaw.response.gamma_int import GammaIntegrator

class EpsilonInverse:
    def __init__(self, chi0, qd, coulomb, kd, volume, fxc_mode, *, integrate_gamma):
        # Unpack data
        pd = chi0.pd
        chi0_wGG = chi0.copy_array_with_distribution('wGG')
        chi0_Wvv = chi0.chi0_Wvv
        chi0_WxvG = chi0.chi0_WxvG
        V0 = None
        sqrtV0 = None
        q_c = pd.q_c
        wblocks1d = Blocks1D(chi0.blockdist.blockcomm, len(chi0.wd))
        delta_GG, fv = self.basic_dyson_arrays(pd, fxc_mode)
        if integrate_gamma != 0:
            reduced = (integrate_gamma == 2)
            V0, sqrtV0 = coulomb.integrated_kernel(pd=pd, reduced=reduced)
        elif integrate_gamma == 0 and np.allclose(q_c, 0):
            bzvol = (2 * np.pi)**3 / volume / qd.nbzkpts
            Rq0 = (3 * bzvol / (4 * np.pi))**(1. / 3.)
            V0 = 16 * np.pi**2 * Rq0 / bzvol
            sqrtV0 = (4 * np.pi)**(1.5) * Rq0**2 / bzvol / 2

        self.V0 = V0
        self.sqrtV0 = sqrtV0

        # Generate fine grid in vicinity of gamma
        # Use optical_limit check on chi0_data in the future XXX
        if np.allclose(q_c, 0) and len(chi0_wGG) > 0:
            gamma_int = GammaIntegrator(
                truncation=coulomb.truncation,
                kd=kd, pd=pd,
                chi0_wvv=chi0_Wvv[wblocks1d.myslice],
                chi0_wxvG=chi0_WxvG[wblocks1d.myslice])

        self.einv_wGG = chi0_wGG
        for iw, chi0_GG in enumerate(chi0_wGG):
            if np.allclose(q_c, 0):
                einv_GG = np.zeros(delta_GG.shape, complex)
                for iqf in range(len(gamma_int.qf_qv)):
                    gamma_int.set_appendages(chi0_GG, iw, iqf)

                    sqrtV_G = coulomb.sqrtV(
                        pd=pd, q_v=gamma_int.qf_qv[iqf])

                    dfc = DielectricFunctionCalculator(
                        sqrtV_G, chi0_GG, mode=fxc_mode, fv_GG=fv)
                    einv_GG += dfc.get_einv_GG() * gamma_int.weight_q
            else:
                sqrtV_G = coulomb.sqrtV(pd=pd, q_v=None)
                dfc = DielectricFunctionCalculator(
                    sqrtV_G, chi0_GG, mode=fxc_mode, fv_GG=fv)
                einv_GG = dfc.get_einv_GG()
        self.sqrtV_G = sqrtV_G  # sqrtV_G escapes the iqf loop, thus sqrtV_G[0] is corrupted
        if sqrtV_G is not None:
            sqrtV_G[0] = 0 # JUST IN CASE. See line above! ???
        self.q_c = q_c
        self.wblocks1d = wblocks1d
        self.pd = pd
        
    def basic_dyson_arrays(self, pd, fxc_mode):
        delta_GG = np.eye(pd.ngmax)

        if fxc_mode == 'GW':
            fv = delta_GG
        else:
            fv = self.xckernel.calculate(pd)

        return delta_GG, fv

class DielectricFunctionCalculator:
    def __init__(self, sqrtV_G, chi0_GG, mode, fv_GG=None):
        self.sqrtV_G = sqrtV_G
        self.chiVV_GG = chi0_GG * sqrtV_G * sqrtV_G[:, np.newaxis]

        self.I_GG = np.eye(len(sqrtV_G))

        self.fv_GG = fv_GG
        self.chi0_GG = chi0_GG
        self.mode = mode

    def _chiVVfv_GG(self):
        assert self.mode != 'GW'
        assert self.fv_GG is not None
        return self.chiVV_GG @ self.fv_GG

    def e_GG_gwp(self):
        gwp_inv_GG = np.linalg.inv(self.I_GG - self._chiVVfv_GG() +
                                   self.chiVV_GG)
        return self.I_GG - gwp_inv_GG @ self.chiVV_GG

    def e_GG_gws(self):
        # Note how the signs are different wrt. gwp.
        # Nobody knows why.
        gws_inv_GG = np.linalg.inv(self.I_GG + self._chiVVfv_GG() -
                                   self.chiVV_GG)
        return gws_inv_GG @ (self.I_GG - self.chiVV_GG)

    def e_GG_plain(self):
        return self.I_GG - self.chiVV_GG

    def e_GG_w_fxc(self):
        return self.I_GG - self._chiVVfv_GG()

    def get_e_GG(self):
        mode = self.mode
        if mode == 'GWP':
            return self.e_GG_gwp()
        elif mode == 'GWS':
            return self.e_GG_gws()
        elif mode == 'GW':
            return self.e_GG_plain()
        elif mode == 'GWG':
            return self.e_GG_w_fxc()
        raise ValueError(f'Unknown mode: {mode}')

    def get_einv_GG(self):
        e_GG = self.get_e_GG()
        return np.linalg.inv(e_GG)

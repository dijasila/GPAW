import numpy as np


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

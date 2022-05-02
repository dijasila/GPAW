import numpy as np


class DielectricFunctionCalculator:
    def __init__(self, sqrV_G, chi0_GG, mode, fv_GG=None):
        self.sqrV_G = sqrV_G
        self.chiVV_GG = chi0_GG * sqrV_G * sqrV_G[:, np.newaxis]

        self.I_GG = np.eye(len(sqrV_G))

        self.chi0_GG = chi0_GG
        self.mode = mode

    def _chiVVfv(self):
        assert self.mode != 'GW'
        assert self.fv_GG is not None
        return self.chiVV_GG @ self.fv_GG

    def e_GG_gwp(self):
        gwp_inv_GG = np.linalg.inv(self.I_GG - self._chiVVfv_GG() + self.chiVV_GG)
        return self.I_GG - gwp_inv_GG @ self.chiVV_GG

    def e_GG_gws(self):
        # Note how the signs are different wrt. gwp.
        # Nobody knows why.
        gws_inv_GG = np.linalg.inv(self.I_GG + self._chiVVfv_GG() - self.chiVV_GG)
        return gws_inv_GG @ (self.I_GG - self.chiVV_GG)

    def e_GG_plain(self):
        return self.I_GG - self.chiVV_GG

    def get_e_GG(self):
        mode = self.mode
        if mode == 'GWP':
            return self.e_GG_gwp()
        elif mode == 'GWS':
            return self.e_GG_gws()
        elif mode == 'GW':
            return self.e_GG_plain()
        raise ValueError(f'Unknown mode: {mode}')

    def get_einv_GG(self):
        e_GG = self.get_e_GG()
        return np.linalg.inv(e_GG)

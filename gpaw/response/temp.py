import numpy as np


class DielectricFunctionCalculator:
    def __init__(self, sqrV_G, chi0_GG, fv):
        self.sqrV_G = sqrV_G
        self.chiVV_GG = chi0_GG * sqrV_G * sqrV_G[:, np.newaxis]
        self.chiVVfv_GG = self.chiVV_GG @ fv
        self.I_GG = np.eye(len(sqrV_G))

        self.chi0_GG = chi0_GG

    def e_GG_gwp(self):
        gwp_inverse_GG = np.linalg.inv(self.I_GG - self.chiVVfv_GG + self.chiVV_GG)
        return self.I_GG - gwp_inverse_GG @ self.chiVV_GG

    def e_GG_gws(self):
        gws_inverse_GG = np.linalg.inv(self.I_GG + self.chiVVfv_GG - self.chiVV_GG)
        return gws_inverse_GG @ (self.I_GG - self.chiVV_GG)

    def e_GG_plain(self):
        return self.I_GG - self.chiVV_GG

    def get_e_GG(self, mode):
        if mode == 'GWP':
            return self.e_GG_gwp()
        elif mode == 'GWS':
            return self.e_GG_gws()
        elif mode == 'GW':
            return self.e_GG_plain()
        raise ValueError(f'Unknown mode: {mode}')

import numpy as np

def calculate_inveps(sqrV_G, chi0_GG, mode, fv_GG=None, weight=1.0, out_GG=None, lib=np):
    """

        Calculates the symmetric inverse dielectric function, and adds it to out-array
        with given weight.

        out_GG[:] += weight * inveps_GG

    """
    N = len(sqrV_G)

    if mode == 'GW':
        """
            For GW method, the symmetric dielectric matrix is given as
                  /        ½          ½  \-1
            ε   = | I   - V   chi0   V   |
             GG   \  GG    GG     GG  GG /

        """
        e_GG = -chi0_GG * sqrV_G * sqrV_G[:, lib.newaxis]
        e_GG.flat[::N+1] += 1.0  # Fast way to add an identity matrix
    elif mode == 'GWP':
        raise NotImplementedError
    elif mode == 'GWS':
        raise NotImplementedError
    elif mode == 'GWG':
         raise NotImplementedError
    else:
        raise ValueError(f'Unknown mode: {mode}')

    out_GG += weight * lib.linalg.inv(e_GG)

"""
    def inv(self, a):
        return self.lib.linalg.inv(a)



    def e_GG_gws(self):
        # Note how the signs are different wrt. gwp.
        # Nobody knows why.
        gws_inv_GG = self.inv(self.I_GG + self._chiVVfv_GG() - self.chiVV_GG)
        return gws_inv_GG @ (self.I_GG - self.chiVV_GG)


    def e_GG_w_fxc(self):
        return self.I_GG - self._chiVVfv_GG()
"""

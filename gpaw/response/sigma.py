import numpy as np


class SigmaCalculator:
    def __init__(self, wd, factor):
        self.wd = wd
        self.factor = factor

    def calculate_sigma(self, n_mG, deps_m, f_m, C_swGG, blocks1d):
        wd = self.wd
        o_m = abs(deps_m)
        # Add small number to avoid zeros for degenerate states:
        sgn_m = np.sign(deps_m + 1e-15)

        # Pick +i*eta or -i*eta:
        s_m = (1 + sgn_m * np.sign(0.5 - f_m)).astype(int) // 2

        w_m = wd.get_floor_index(o_m, safe=False)
        m_inb = np.where(w_m < len(wd) - 1)[0]
        o1_m = np.empty(len(o_m))
        o2_m = np.empty(len(o_m))
        o1_m[m_inb] = wd.omega_w[w_m[m_inb]]
        o2_m[m_inb] = wd.omega_w[w_m[m_inb] + 1]

        sigma = 0.0
        dsigma = 0.0
        # Performing frequency integration
        for o, o1, o2, sgn, s, w, n_G in zip(o_m, o1_m, o2_m,
                                             sgn_m, s_m, w_m, n_mG):

            if w >= len(wd.omega_w) - 1:
                continue

            C1_GG = C_swGG[s][w]
            C2_GG = C_swGG[s][w + 1]
            p = self.factor * sgn
            myn_G = n_G[blocks1d.myslice]

            sigma1 = p * np.dot(np.dot(myn_G, C1_GG), n_G.conj()).imag
            sigma2 = p * np.dot(np.dot(myn_G, C2_GG), n_G.conj()).imag
            sigma += ((o - o1) * sigma2 + (o2 - o) * sigma1) / (o2 - o1)
            dsigma += sgn * (sigma2 - sigma1) / (o2 - o1)

        return sigma, dsigma


class PPASigmaCalculator:
    def __init__(self, eta, factor):
        self.eta = eta
        self.factor = factor

    def calculate_sigma(self, n_mG, deps_m, f_m, W):
        # XXX It is completely impossible to infer the meaning of these
        # arrays since they're often named "_m" but then later
        # multiplied with "_GG" arrays.
        W_GG, omegat_GG = W

        sigma = 0.0
        dsigma = 0.0

        for m in range(len(n_mG)):
            deps_GG = deps_m[m]
            sign_GG = 2 * f_m[m] - 1
            x1_GG = 1 / (deps_GG + omegat_GG - 1j * self.eta)
            x2_GG = 1 / (deps_GG - omegat_GG + 1j * self.eta)
            x3_GG = 1 / (deps_GG + omegat_GG - 1j * self.eta * sign_GG)
            x4_GG = 1 / (deps_GG - omegat_GG - 1j * self.eta * sign_GG)
            x_GG = W_GG * (sign_GG * (x1_GG - x2_GG) + x3_GG + x4_GG)
            dx_GG = W_GG * (sign_GG * (x1_GG**2 - x2_GG**2) +
                            x3_GG**2 + x4_GG**2)
            nW_G = np.dot(n_mG[m], x_GG)
            sigma += np.vdot(n_mG[m], nW_G).real
            nW_G = np.dot(n_mG[m], dx_GG)
            dsigma -= np.vdot(n_mG[m], nW_G).real

        return self.factor * sigma, self.factor * dsigma

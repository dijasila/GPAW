import numpy as np
from math import sqrt, pi, exp
from scipy.special import erfcx

from ase.units import Ha


K_G = 8 * sqrt(2) / (3 * pi**2)  # 0.382106112167171


class Coefficients:
    def __init__(self, eps=0.05, width=None, metallic=False):
        self.eps = eps / Ha
        self.metallic = metallic
        if width is not None:
            width = width / Ha
        self.width = width

    def initialize(self, wfs):
        self.wfs = wfs

    def initialize_1d(self, ae):
        self.ae = ae

    def get_description(self):
        desc = 'eps={:.4f} eV'.format(self.eps * Ha)
        if self.metallic:
            desc += 'metallic'
        if self.width is not None:
            desc += 'width={:.4f} eV'.format(self.width * Ha)
        return desc

    def f(self, f):
        if self.width is None:
            if f > self.eps:
                return sqrt(f)
            else:
                return 0.0
        else:
            width = self.width
            if f > 0:
                K = sqrt(f) + 0.5 * sqrt(pi * width) * erfcx(sqrt(f / width))
            else:
                K = 0.5 * sqrt(pi * width) * exp(f / width)
            return K

    def get_coefficients(self, e_j, f_j):
        homo_e = max([np.where(f > 1e-3, e, -1000) for f, e in zip(f_j, e_j)])
        return [f * K_G * self.f(homo_e - e) for e, f in zip(e_j, f_j)]

    def get_coefficients_1d(self, smooth=False, lumo_perturbation=False):
        homo_e = max([np.where(f > 1e-3, e, -1000)
                      for f, e in zip(self.ae.f_j, self.ae.e_j)])
        if not smooth:
            if lumo_perturbation:
                lumo_e = min([np.where(f < 1e-3, e, 1000)
                              for f, e in zip(self.ae.f_j, self.ae.e_j)])
                return np.array([f * K_G * (self.f(lumo_e - e)
                                            - self.f(homo_e - e))
                                 for e, f in zip(self.ae.e_j, self.ae.f_j)])
            else:
                return np.array([f * K_G * (self.f(homo_e - e))
                                 for e, f in zip(self.ae.e_j, self.ae.f_j)])
        else:
            return [[f * K_G * self.f(homo_e - e) for e, f in zip(e_n, f_n)]
                    for e_n, f_n in zip(self.ae.e_ln, self.ae.f_ln)]

    def get_coefficients_by_kpt(self, kpt_u, lumo_perturbation=False,
                                homolumo=None, nspins=1):
        eref_s = []
        eref_lumo_s = []
        if self.metallic:
            # Use Fermi level as reference levels
            assert homolumo is None
            fermilevel = self.wfs.fermi_level
            assert isinstance(fermilevel, float), \
                'GLLBSCM supports only a single Fermi level'
            for s in range(nspins):
                eref_s.append(fermilevel)
                eref_lumo_s.append(fermilevel)
        elif homolumo is None:
            # Find homo and lumo levels for each spin
            for s in range(nspins):
                homo, lumo = self.wfs.get_homo_lumo(s)
                # Check that homo and lumo are reasonable
                if homo > lumo:
                    m = ("GLLBSC error: HOMO is higher than LUMO. "
                         "Are you using `xc='GLLBSC'` for a metallic system? "
                         "If yes, try using `xc='GLLBSCM'` instead.")
                    raise RuntimeError(m)

                eref_s.append(homo)
                eref_lumo_s.append(lumo)
        else:
            eref_s, eref_lumo_s = homolumo
            if not isinstance(eref_s, (list, tuple)):
                eref_s = [eref_s]
                eref_lumo_s = [eref_lumo_s]

        if lumo_perturbation:
            return [np.array([f * K_G * (self.f(eref_lumo_s[kpt.s] - e)
                                         - self.f(eref_s[kpt.s] - e))
                              for e, f in zip(kpt.eps_n, kpt.f_n)])
                    for kpt in kpt_u]
        else:
            coeff = [np.array([f * K_G * self.f(eref_s[kpt.s] - e)
                     for e, f in zip(kpt.eps_n, kpt.f_n)])
                     for kpt in kpt_u]
            return coeff

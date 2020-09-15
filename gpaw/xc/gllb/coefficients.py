import numpy as np
from scipy.special import erfcx

from ase.units import Ha


K_G = 8 * np.sqrt(2) / (3 * np.pi**2)  # 0.382106112167171


class Coefficients:
    r"""Coefficient calculator for GLLB functionals.

    This class implements the calculation of sqrt(E) coefficients as given by
    Eq. (16) of https://doi.org/10.1103/PhysRevB.82.115106.

    Parameters:

    eps (in eV):
        This parameter cuts sqrt(E) to zero for E < eps.
        The cut is supposed to help convergence with degenerate systems.
        This parameter should be small.
    width (in eV):
        If this parameter is set, then a smoothed variant of the sqrt(E)
        expression is used. This parameter sets the energy width of
        the smoothing.
        The eps parameter is ignored when the width parameter is set.
    metallic:
        If this parameter is set, then Fermi level is used as the reference
        energy in the sqrt(E) expression instead of the HOMO energy.
        This is necessary to get sensible results in metallic systems.
    """
    def __init__(self,
                 eps: float = 0.05,
                 width: float = None,
                 metallic: bool = False):
        self.eps = eps / Ha
        self.metallic = metallic
        if width is not None:
            width = width / Ha
            self.eps = None  # Make sure that eps is not used with width
        self.width = width

    def initialize(self, wfs):
        self.wfs = wfs

    def initialize_1d(self, ae):
        self.ae = ae

    def get_description(self):
        desc = []
        if self.eps is not None:
            desc += ['eps={:.4f} eV'.format(self.eps * Ha)]
        if self.metallic:
            desc += ['metallic']
        if self.width is not None:
            desc += ['width={:.4f} eV'.format(self.width * Ha)]
        return ', '.join(desc)

    def f(self, energy_n):
        """Calculate the sqrt(E)-like coefficient.

        See the class description for details.
        """
        w_n = np.zeros_like(energy_n, dtype=float)
        if self.width is None:
            flt_n = energy_n > self.eps
            w_n[flt_n] = np.sqrt(energy_n[flt_n])
        else:
            prefactor = 0.5 * np.sqrt(np.pi * self.width)
            rel_energy_n = energy_n / self.width
            # Evaluate positive energies
            flt_n = energy_n > 0.0
            w_n[flt_n] = (np.sqrt(energy_n[flt_n])
                          + prefactor * erfcx(np.sqrt(rel_energy_n[flt_n])))
            # Evaluate negative energies
            flt_n = np.logical_not(flt_n)
            w_n[flt_n] = prefactor * np.exp(rel_energy_n[flt_n])
        return w_n

    def get_reference_homo_1d(self):
        e_j = np.asarray(self.ae.e_j)
        f_j = np.asarray(self.ae.f_j)
        homo = np.max(e_j[f_j > 1e-3])
        return homo

    def get_reference_lumo_1d(self):
        e_j = np.asarray(self.ae.e_j)
        f_j = np.asarray(self.ae.f_j)
        lumo = np.min(e_j[f_j < 1e-3])
        return lumo

    def get_coefficients_1d(self, smooth=False):
        homo = self.get_reference_homo_1d()
        if smooth:
            w_ln = []
            for e_n, f_n in zip(self.ae.e_ln, self.ae.f_ln):
                e_n = np.asarray(e_n)
                f_n = np.asarray(f_n)
                w_n = f_n * K_G * self.f(homo - e_n)
                w_ln.append(w_n)
            return w_ln
        else:
            e_j = np.asarray(self.ae.e_j)
            f_j = np.asarray(self.ae.f_j)
            return f_j * K_G * self.f(homo - e_j)

    def get_coefficients_1d_for_lumo_perturbation(self, smooth=False):
        homo = self.get_reference_homo_1d()
        lumo = self.get_reference_lumo_1d()
        e_j = np.asarray(self.ae.e_j)
        f_j = np.asarray(self.ae.f_j)
        return f_j * K_G * (self.f(lumo - e_j) - self.f(homo - e_j))

    def get_reference_energies(self, homolumo=None):
        eref_s = []
        eref_lumo_s = []
        if self.metallic:
            # Use Fermi level as reference levels
            assert homolumo is None
            fermilevel = self.wfs.fermi_level
            assert isinstance(fermilevel, float), \
                'GLLBSCM supports only a single Fermi level'
            for s in range(self.wfs.nspins):
                eref_s.append(fermilevel)
                eref_lumo_s.append(fermilevel)
        elif homolumo is None:
            # Find homo and lumo levels for each spin
            for s in range(self.wfs.nspins):
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
            if not isinstance(eref_s, (list, tuple, np.ndarray)):
                eref_s = [eref_s]
                eref_lumo_s = [eref_lumo_s]
        return eref_s, eref_lumo_s

    def get_coefficients(self, kpt_u, homolumo=None):
        eref_s, eref_lumo_s = self.get_reference_energies(homolumo=homolumo)
        w_kn = []
        for kpt in kpt_u:
            w_n = self.f(eref_s[kpt.s] - kpt.eps_n)
            w_n *= kpt.f_n * K_G
            w_kn.append(w_n)
        return w_kn

    def get_coefficients_for_lumo_perturbation(self, kpt_u, homolumo=None):
        eref_s, eref_lumo_s = self.get_reference_energies(homolumo=homolumo)
        w_kn = []
        for kpt in kpt_u:
            w_n = (self.f(eref_lumo_s[kpt.s] - kpt.eps_n)
                   - self.f(eref_s[kpt.s] - kpt.eps_n))
            w_n *= kpt.f_n * K_G
            w_kn.append(w_n)
        return w_kn

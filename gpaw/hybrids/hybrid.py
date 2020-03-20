# type: ignore
from math import nan

import numpy as np
from ase.units import Ha

from gpaw.xc.exx import pawexxvv
from gpaw.xc.tools import _vxc
from gpaw.utilities import unpack2, unpack
from .kpts import KPoint
from .coulomb import ShortRangeCoulomb


class EXX:
    ftol = 1e-9

    def __init__(self, gd, kd, nspins, pt, setups, omega, exx_fraction, xc):
        if -1 in kd.bz2bz_ks:
            raise ValueError(
                'Your k-points are not as symmetric as your crystal!')

        assert kd.comm.size == 1

        self.gd = gd
        self.kd = kd
        self.nspins = nspins
        self.pt = pt
        self.setups = setups
        self.omega = omega
        self.exx_fractions = exx_fraction
        self.xc = xc

        self.ecc = sum(setup.ExxC for setup in setups) * exx_fraction

        # PAW-correction stuff:
        self.Delta_aiiL = []
        self.VC_aii = {}
        for a, data in enumerate(setups):
            self.Delta_aiiL.append(data.Delta_iiL)
            self.VC_aii[a] = unpack(data.X_p)

        self.v_knG = {}

        self.evv = np.nan
        self.evc = np.nan
        self.ekin = np.nan

        self.spos_ac = None

    def calculate_local_potential_and_energy(self, gd, nt_sr, vt_sr):
        energy = self.ecc + self.evv + self.evc
        if self.xc:
            e_r = gd.empty()
            self.xc.calculate(gd, nt_sr, vt_sr, e_r)
            energy += gd.integrate(e_r)
        return energy

    def apply1(self, kpt, psit_xG, Htpsit_xG):
        deg = 2 / self.nspins

        v_nG = self.v_knG.pop(kpt.k)
        if v_nG is not None:
            Htpsit_xG += v_nG * self.exx_fraction
            return

        if kpt.s == 0:
            self.evv = 0.0
            self.evc = 0.0
            self.ekin = 0.0

        kd = self.kd
        VV_aii = self.calculate_valence_valence_paw_corrections(kpt.s)
        K = kd.nibzkpts
        k1 = kpt.s * K
        k2 = k1 + K
        kpts = [KPoint(kpt.psit,
                       kpt.projections,
                       kpt.f_n / kpt.weight,  # scale to [0, 1] range
                       kd.ibzk_kc[kpt.k],
                       kd.weight_k[kpt.k])
                for kpt in self.wfs.mykpts[k1:k2]]
        evv, evc, ekin, self.v_knG = self.xx.calculate(
            kpts, kpts,
            VV_aii,
            derivatives=True)
        self.evv += evv * deg * self.exx_fraction
        self.evc += evc * deg * self.exx_fraction
        self.ekin += ekin * deg * self.exx_fraction

    def apply2(self, kpt, psit_xG, Htpsit_xG):
        VV_aii = self.calculate_valence_valence_paw_corrections(spin)

        K = kd.nibzkpts
        k1 = (spin - kd.comm.rank) * K
        k2 = k1 + K
        kpts2 = [KPoint(kpt.psit,
                        kpt.projections,
                        kpt.f_n / kpt.weight,  # scale to [0, 1] range
                        kd.ibzk_kc[kpt.k],
                        kd.weight_k[kpt.k])
                 for kpt in self.wfs.mykpts[k1:k2]]

        psit = kpt.psit.new(buf=psit_xG)
        P = kpt.projections.new()
        psit.matrix_elements(self.wfs.pt, out=P)

        kpts1 = [KPoint(psit,
                        P,
                        kpt.f_n + nan,
                        kd.ibzk_kc[kpt.k],
                        nan)]
        _, _, _, v_1xG = self.xx.calculate(
            kpts1, kpts2,
            VV_aii,
            derivatives=True)
        Htpsit_xG += self.exx_fraction * v_1xG[0]


    def test(self):
        self._initialize()

        wfs = self.wfs
        kd = wfs.kd

        evv = 0.0
        evc = 0.0

        e_skn = np.zeros((wfs.nspins, kd.nibzkpts, wfs.bd.nbands), complex)

        for spin in range(wfs.nspins):
            VV_aii = self.calculate_valence_valence_paw_corrections(spin)
            K = kd.nibzkpts
            k1 = spin * K
            k2 = k1 + K
            kpts = [KPoint(kpt.psit,
                           kpt.projections,
                           kpt.f_n / kpt.weight,  # scale to [0, 1]
                           kd.ibzk_kc[kpt.k],
                           kd.weight_k[kpt.k])
                    for kpt in wfs.mykpts[k1:k2]]
            e1, e2, _, v_knG = self.xx.calculate(kpts, kpts, VV_aii,
                                                 derivatives=True)
            evv += e1
            evc += e2

            for e_n, kpt, v_nG in zip(e_skn[spin], kpts, v_knG.values()):
                e_n[:] = [kpt.psit.pd.integrate(v_G, psit_G) *
                          self.exx_fraction
                          for v_G, psit_G in zip(v_nG, kpt.psit.array)]

        if self.xc:
            vxc_skn = _vxc(self.xc, self.ham, self.dens, self.wfs) / Ha
            e_skn += vxc_skn

        deg = 2 / wfs.nspins
        evv = kd.comm.sum(evv) * deg
        evc = kd.comm.sum(evc) * deg

        return evv * Ha, evc * Ha, e_skn * Ha

    def calculate_energy(self, spin, nocc):
        wfs = self.wfs
        kd = wfs.kd
        VV_aii = self.calculate_valence_valence_paw_corrections(spin)
        K = kd.nibzkpts
        k1 = spin * K
        k2 = k1 + K
        kpts = [KPoint(kpt.psit.view(0, nocc),
                       kpt.projections.view(0, nocc),
                       kpt.f_n[:nocc] / kpt.weight,  # scale to [0, 1]
                       kd.ibzk_kc[kpt.k],
                       kd.weight_k[kpt.k])
                for kpt in wfs.mykpts[k1:k2]]
        evc, evv = self.xx.calculate_energy(kpts, VV_aii)
        deg = 2 / wfs.nspins
        return evc * deg, evv * deg

    def calculate_eigenvalue_contribution(self,
                                          spin,
                                          k,
                                          n1, n2, nocc,
                                          VV_aii) -> np.ndarray:
        wfs = self.wfs
        kd = wfs.kd
        k1 = spin * kd.nibzkpts
        k2 = (spin + 1) * kd.nibzkpts

        kpt = wfs.mykpts[k1 + k]
        kpt1 = KPoint(kpt.psit.view(n1, n2),
                      kpt.projections.view(n1, n2),
                      kpt.f_n[n1:n2] / kpt.weight,  # scale to [0, 1]
                      kd.ibzk_kc[kpt.k],
                      kd.weight_k[kpt.k])

        kpts2 = [KPoint(kpt.psit.view(0, nocc),
                        kpt.projections.view(0, nocc),
                        kpt.f_n[:nocc] / kpt.weight,  # scale to [0, 1]
                        kd.ibzk_kc[kpt.k],
                        kd.weight_k[kpt.k])
                 for kpt in wfs.mykpts[k1:k2]]

        e_n = self.xx.calculate_eigenvalues(kpt1, kpts2, VV_aii)
        self.xx.comm.sum(e_n)
        e_n *= self.exx_fraction
        return e_n

    def calculate_eigenvalues0(self, n1, n2, kpts):
        self._initialize()
        wfs = self.wfs
        kd = wfs.kd

        nocc = max(((kpt.f_n / kpt.weight) > self.ftol).sum()
                   for kpt in wfs.mykpts)

        if kpts is None:
            kpts = range(len(wfs.mykpts) // wfs.nspins)

        self.e_skn = np.zeros((wfs.nspins, len(kpts), n2 - n1))

        for spin in range(wfs.nspins):
            VV_aii = self.calculate_valence_valence_paw_corrections(spin)
            K = kd.nibzkpts
            k1 = spin * K
            k2 = k1 + K
            kpts1 = [KPoint(kpt.psit.view(n1, n2),
                            kpt.projections.view(n1, n2),
                            kpt.f_n[n1:n2] / kpt.weight,  # scale to [0, 1]
                            kd.ibzk_kc[kpt.k],
                            kd.weight_k[kpt.k])
                     for kpt in (wfs.mykpts[k] for k in kpts)]
            kpts2 = [KPoint(kpt.psit.view(0, nocc),
                            kpt.projections.view(0, nocc),
                            kpt.f_n[:nocc] / kpt.weight,  # scale to [0, 1]
                            kd.ibzk_kc[kpt.k],
                            kd.weight_k[kpt.k])
                     for kpt in wfs.mykpts[k1:k2]]
            self.xx.calculate(
                kpts1, kpts2,
                VV_aii,
                e_kn=self.e_skn[spin])

        self.xx.comm.sum(self.e_skn)
        self.e_skn *= self.exx_fraction

        if self.xc:
            vxc_skn = _vxc(self.xc, self.ham, self.dens, self.wfs) / Ha
            self.e_skn += vxc_skn[:, kpts, n1:n2]

        return self.e_skn * Ha

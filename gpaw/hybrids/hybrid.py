from math import nan
from typing import Tuple
from io import StringIO

import numpy as np
from ase.units import Ha

from gpaw.xc import XC
from gpaw.xc.exx import pawexxvv
from gpaw.xc.tools import _vxc
from gpaw.utilities import unpack2
from gpaw.response.wstc import WignerSeitzTruncatedCoulomb as WSTC
from .exx import EXX, KPoint
from .coulomb import ShortRangeCoulomb


def parse_name(name: str) -> Tuple[str, float, float]:
    if name == 'EXX':
        return 'null', 1.0, 0.0
    if name == 'PBE0':
        return 'HYB_GGA_XC_PBEH', 0.25, 0.0
    if name == 'HSE03':
        return 'HYB_GGA_XC_HSE03', 0.25, 0.106
    if name == 'HSE06':
        return 'HYB_GGA_XC_HSE06', 0.25, 0.11
    if name == 'B3LYP':
        return 'HYB_GGA_XC_B3LYP', 0.2, 0.0


class HybridXC:
    orbital_dependent = True
    type = 'HYB'
    ftol = 1e-9

    def __init__(self,
                 name: str = None,
                 xc=None,
                 exx_fraction: float = None,
                 omega: float = None,
                 mix_all: bool = True):
        if name is not None:
            assert xc is None and exx_fraction is None and omega is None
            xc, exx_fraction, omega = parse_name(name)
            self.name = name
        else:
            assert xc is not None and exx_fraction is not None
            self.name = '???'

        if xc:
            xc = XC(xc)

        self.xc = xc
        self.exx_fraction = exx_fraction
        self.omega = omega
        self.mix_all = mix_all

        self.initialized = False

        self.dens = None
        self.ham = None
        self.wfs = None

        self.xx = None
        self.coulomb = None
        self.v_knG = {}
        self.spin = -1

        self.evv = np.nan
        self.evc = np.nan
        self.ecc = np.nan
        self.ekin = np.nan

        self.vt_sR = None
        self.spos_ac = None

        self.description = ''

    def read(self, reader):
        pass

    def get_setup_name(self):
        return 'PBE'
        return 'LDA'

    def write(self, writer):
        pass

    def initialize(self, dens, ham, wfs, occupations):
        self.dens = dens
        self.ham = ham
        self.wfs = wfs
        assert wfs.world.size == wfs.gd.comm.size

    def get_description(self):
        return self.description

    def set_grid_descriptor(self, gd):
        pass

    def set_positions(self, spos_ac):
        self.spos_ac = spos_ac

    def calculate(self, gd, nt_sr, vt_sr):
        if not self.xc:
            return self.evv + self.evc
        e_r = gd.empty()
        self.xc.calculate(gd, nt_sr, vt_sr, e_r)
        # print(self.ecc, self.evv, self.evc, gd.integrate(e_r))
        return self.ecc + self.evv + self.evc + gd.integrate(e_r)

    def calculate_paw_correction(self, setup, D_sp, dH_sp=None, a=None):
        if not self.xc:
            return 0.0
        return self.xc.calculate_paw_correction(setup, D_sp, dH_sp, a=a)

    def get_kinetic_energy_correction(self):
        # print(self.ekin)
        return self.ekin

    def _initialize(self):
        if self.initialized:
            return

        wfs = self.wfs
        kd = wfs.kd

        if -1 in kd.bz2bz_ks:
            raise ValueError(
                'Your k-points are not as symmetric as your crystal!')

        assert kd.comm.size == 1 or kd.comm.size == 2 and wfs.nspins == 2
        assert wfs.bd.comm.size == 1

        if self.omega:
            coulomb = ShortRangeCoulomb(self.omega)
        else:
            # Wigner-Seitz truncated Coulomb:
            output = StringIO()
            coulomb = WSTC(wfs.gd.cell_cv, wfs.kd.N_c, txt=output)
            self.description += output.getvalue()

        self.xx = EXX(wfs.kd, wfs.setups, wfs.pt, coulomb, self.spos_ac,
                      wfs.timer)

        self.ecc = sum(setup.ExxC for setup in wfs.setups) * self.exx_fraction

        self.initialized = True

    def apply_orbital_dependent_hamiltonian(self, kpt, psit_xG,
                                            Htpsit_xG=None, dH_asp=None):
        self._initialize()

        kd = self.wfs.kd
        spin = kpt.s

        if kpt.f_n is None:
            if self.vt_sR is None:
                from gpaw.xc import XC
                lda = XC('LDA')
                nt_sr = self.dens.nt_sg
                vt_sr = np.zeros_like(nt_sr)
                self.vt_sR = self.dens.gd.zeros(self.wfs.nspins)
                lda.calculate(self.dens.finegd, nt_sr, vt_sr)
                for vt_R, vt_r in zip(self.vt_sR, vt_sr):
                    vt_R[:], _ = self.dens.pd3.restrict(vt_r, self.dens.pd2)

            pd = kpt.psit.pd
            for psit_G, Htpsit_G in zip(psit_xG, Htpsit_xG):
                Htpsit_G += pd.fft(self.vt_sR[kpt.s] *
                                   pd.ifft(psit_G, kpt.k), kpt.q)
            return

        self.vt_sR = None
        deg = 2 / self.wfs.nspins

        if kpt.psit.array.base is psit_xG.base:
            if not self.v_knG:
                self.spin = spin
                if spin == 0:
                    self.evv = 0.0
                    self.evc = 0.0
                    self.ekin = 0.0
                VV_aii = self.calculate_valence_valence_paw_corrections(spin)
                K = kd.nibzkpts
                k1 = (spin - kd.comm.rank) * K
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
            assert self.spin == spin
            Htpsit_xG += self.v_knG.pop(kpt.k) * self.exx_fraction
        else:
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

    def correct_hamiltonian_matrix(self, kpt, H_nn):
        if self.mix_all or kpt.f_n is None:
            return

        n = (kpt.f_n > kpt.weight * self.ftol).sum()
        H_nn[n:, :n] = 0.0
        H_nn[:n, n:] = 0.0

    def rotate(self, kpt, U_nn):
        pass  # 1 / 0

    def add_correction(self, kpt, psit_xG, Htpsit_xG, P_axi, c_axi, n_x,
                       calculate_change=False):
        pass  # 1 / 0

    def calculate_valence_valence_paw_corrections(self, spin):
        VV_aii = {}
        for a, D_sp in self.dens.D_asp.items():
            data = self.wfs.setups[a]
            D_p = D_sp[spin]
            D_ii = unpack2(D_p) * (self.wfs.nspins / 2)
            VV_ii = pawexxvv(data, D_ii)
            VV_aii[a] = VV_ii
        return VV_aii

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

    def calculate_energy(self):
        self._initialize()

        wfs = self.wfs
        kd = wfs.kd
        # rank = kd.comm.rank
        # size = kd.comm.size

        nocc = max(((kpt.f_n / kpt.weight) > self.ftol).sum()
                   for kpt in wfs.mykpts)

        evv = 0.0
        evc = 0.0

        for spin in range(wfs.nspins):
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
            e1, e2, ekin, _ = self.xx.calculate(kpts, kpts, VV_aii)
            evv += e1
            evc += e2

        deg = 2 / wfs.nspins
        evv = kd.comm.sum(evv) * deg
        evc = kd.comm.sum(evc) * deg

        if self.xc:
            pass

        return evv * Ha, evc * Ha

    def calculate_eigenvalues(self, n1, n2, kpts):
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

        kd.comm.sum(self.e_skn)
        self.e_skn *= self.exx_fraction

        if self.xc:
            vxc_skn = _vxc(self.xc, self.ham, self.dens, self.wfs) / Ha
            self.e_skn += vxc_skn[:, kpts, n1:n2]

        return self.e_skn * Ha

    def summary(self, log):
        log(self.description)

    def add_forces(self, F_av):
        pass


if __name__ == '__main__':
    from ase import Atoms
    from gpaw import GPAW, PW
    h = Atoms('H', cell=(3, 3, 3), pbc=(1, 1, 1))
    h = Atoms('H2', cell=(3, 3, 3), pbc=1, positions=[[0, 0, 0], [0, 0, 0.75]])
    h.calc = GPAW(mode=PW(100, force_complex_dtype=True),
                  setups='ae',
                  kpts=(1, 1, 2),
                  # spinpol=True,
                  txt=None
                  )
    h.get_potential_energy()
    x = HybridXC('EXX')

    # h.calc.get_xc_difference(exx)
    # e = exx.evv + exx.evc + exx.exx.ecc
    # print(e * Ha, exx.e_skn * Ha)

    c = h.calc
    x.initialize(c.density, c.hamiltonian, c.wfs, c.occupations)
    x.set_positions(c.spos_ac)
    e = x.calculate_energy()
    print(e)
    x.calculate_eigenvalues(0, 1, [0])
    print(x.e_skn * Ha)

    from gpaw.xc.exx import EXX as EXX0
    xx = EXX0(c, bands=(0, 1))
    xx.calculate()
    e0 = xx.get_exx_energy()
    eps0 = xx.get_eigenvalue_contributions()
    print(e0, eps0)
    print(e0 - e[0] - e[1])
    print(eps0 - x.e_skn * Ha)
    # print(e * Ha - e0, xx.e_skn * Ha - eps0)
    print(x.description)

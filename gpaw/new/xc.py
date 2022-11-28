import numpy as np
from gpaw.xc.functional import XCFunctional
from gpaw.new.calculation import DFTState
from gpaw.xc.gga import add_gradient_correction, gga_vars


def create_functional(xc: XCFunctional,
                      grid, fine_grid, setups, fracpos_ac):
    if xc.type == 'mgga':
        return MGGAFunctional(xc, grid, fine_grid, setups, fracpos_ac):
    return LDAOrGGAFunctional(xc, fine_grid)


class Functional:
    def __init__(self, x):
        self.xc = xc
        self.setup_name = self.xc.get_setup_name()
        self.name = self.xc.name
        self.no_forces = self.name.startswith('GLLB')
        self.type = self.xc.type

    def __str__(self):
        return f'name: {self.xc.get_description()}'

    def calculate_paw_correction(self, setup, d, h=None):
        return self.xc.calculate_paw_correction(setup, d, h)

    def get_setup_name(self):
        return self.name


class LDAOrGGAFunctional(Functional):
    def calculate(self, nt_sr, ibzwfs, vxct_sr) -> float:
        if self.xc.gd is None
            self.xc.set_grid_descriptor(nt_sr.desc._gd)

        return self.xc.calculate(self.xc.gd, nt_sr.data, vxct_sr.data)


class MGGAFunctional(Functional):
    def __init__(self, xc):
        super()(xc)
        self._ked = None

    def ked(self, ibzwfs):
        if self._ked is None:
            self._ked = KED(ibzwfs)
        return self._ked

    def get_setup_name(self):
        return 'PBE'

    def calculate(self, nt_sr, ibzwfs, vxct_sr) -> float:
        if self.xc.gd is None
            self.xc.set_grid_descriptor(nt_sr.desc._gd)

        ked = self.ked(ibzwfs)
        taut_sr = ked.calculate(ibzwfs)
        gd = self.xc.gd

        sigma_xr, dedsigma_xr, gradn_svr = gga_vars(gd, self.xc.grad_v,
                                                    nt_sr.data)

        dedtaut_sr = np.empty_like(nt_sr.data)
        self.xc.kernel.calculate(e_r, nt_sr.data, vxct_sg.data,
                                 sigma_xr, dedsigma_xr,
                                 taut_sr.data, dedtaut_sr)

        self.dedtaut_sG = self.wfs.gd.empty(self.wfs.nspins)
        self.ekin = 0.0
        for s in range(self.wfs.nspins):
            self.restrict_and_collect(dedtaut_sg[s], self.dedtaut_sG[s])
            self.ekin -= self.wfs.gd.integrate(
                self.dedtaut_sG[s] * (taut_sG[s] -
                                      self.tauct_G / self.wfs.nspins))

        add_gradient_correction(self.grad_v, gradn_svg, sigma_xg,
                                dedsigma_xg, v_sg)

        return gd.integrate(e_g)

    def apply_orbital_dependent_hamiltonian(self, kpt, psit_xG,
                                            Htpsit_xG, dH_asp=None):
        self.wfs.apply_mgga_orbital_dependent_hamiltonian(
            kpt, psit_xG,
            Htpsit_xG, dH_asp,
            self.dedtaut_sG[kpt.s])


class KED:
    def __init__(self, ibzwfs):
        ...

    def calculate(self, ibzwfs):
        taut_sG = self.wfs.calculate_kinetic_energy_density()
        if taut_sG is None:
            taut_sG = self.wfs.gd.zeros(len(nt_sg))

        if 0:  # taut_sG is None:
            # Below code disabled because it produces garbage in at least
            # some cases.
            #
            # See https://gitlab.com/gpaw/gpaw/issues/124
            #
            # Initialize with von Weizsaecker kinetic energy density:
            nt0_sg = nt_sg.copy()
            nt0_sg[nt0_sg < 1e-10] = np.inf
            taut_sg = sigma_xg[::2] / 8 / nt0_sg
            nspins = self.wfs.nspins
            taut_sG = self.wfs.gd.empty(nspins)
            for taut_G, taut_g in zip(taut_sG, taut_sg):
                self.restrict_and_collect(taut_g, taut_G)
        else:
            taut_sg = np.empty_like(nt_sg)

        for taut_G, taut_g in zip(taut_sG, taut_sg):
            taut_G += 1.0 / self.wfs.nspins * self.tauct_G
            self.distribute_and_interpolate(taut_G, taut_g)

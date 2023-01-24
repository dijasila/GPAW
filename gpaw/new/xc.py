import numpy as np
from gpaw.xc.functional import XCFunctional as OldXCFunctional
from gpaw.xc.gga import add_gradient_correction, gga_vars


def create_functional(xc: OldXCFunctional,
                      grid, coarse_grid, interpolate_domain,
                      setups, fracpos_ac, atomdist):
    if xc.type == 'MGGA':
        return MGGAFunctional(xc, grid, coarse_grid,
                              interpolate_domain, setups, fracpos_ac, atomdist)
    return LDAOrGGAFunctional(xc, grid)


class Functional:
    def __init__(self, xc, grid):
        self.xc = xc
        self.grid = grid
        self.setup_name = self.xc.get_setup_name()
        self.name = self.xc.name
        self.no_forces = self.name.startswith('GLLB')
        self.type = self.xc.type
        self.xc.set_grid_descriptor(grid._gd)

    def __str__(self):
        return f'name: {self.xc.get_description()}'

    def calculate_paw_correction(self, setup, d, h=None):
        return self.xc.calculate_paw_correction(setup, d, h)

    def get_setup_name(self):
        return self.name


class LDAOrGGAFunctional(Functional):
    def calculate(self,
                  nt_sr,
                  vxct_sr,
                  ibzwfs=None,
                  interpolate=None,
                  restrict=None) -> float:
        return self.xc.calculate(self.xc.gd, nt_sr.data, vxct_sr.data)


class MGGAFunctional(Functional):
    def __init__(self,
                 xc,
                 grid,
                 coarse_grid,
                 interpolation_domain,
                 setups,
                 fracpos_ac,
                 atomdist):
        super().__init__(xc, grid)
        self.coarse_grid = coarse_grid
        self.tauct_aX = setups.create_pseudo_core_kinetic_energy_densities(
            interpolation_domain,
            fracpos_ac,
            atomdist)
        self.tauct_R = None
        self.ekin = np.nan
        self.dedtaut_sR = None

    def get_setup_name(self):
        return 'PBE'

    def calculate(self,
                  nt_sr,
                  vxct_r,
                  ibzwfs,
                  interpolate,
                  restrict) -> float:
        nspins = nt_sr.dims[0]

        if self.tauct_R is None:
            self.tauct_R = self.coarse_grid.empty()
            self.tauct_aX.to_uniform_grid(out=self.tauct_R, scale=1.0 / nspins)

        if ibzwfs is None:
            taut_sR = self.coarse_grid.zeros(nspins)
        else:
            taut_sR = ibzwfs.calculate_kinetic_energy_density()

        taut_sr = self.grid.empty(len(taut_sR))
        for taut_R, taut_r in zip(taut_sR, taut_sr):
            taut_R += self.tauct_R
            interpolate(taut_R, taut_r)

        gd = self.xc.gd

        sigma_xr, dedsigma_xr, gradn_svr = gga_vars(gd, self.xc.grad_v,
                                                    nt_sr.data)

        e_r = self.grid.empty()
        dedtaut_sr = taut_sr.new()
        vxct_sr = np.empty_like(nt_sr.data)
        self.xc.kernel.calculate(e_r.data, nt_sr.data, vxct_sr,
                                 sigma_xr, dedsigma_xr,
                                 taut_sr.data, dedtaut_sr.data)

        self.dedtaut_sR = taut_sR.new()
        self.ekin = 0.0
        for dedtaut_R, dedtaut_r, taut_R in zip(self.dedtaut_sR,
                                                dedtaut_sr,
                                                taut_sR):
            restrict(dedtaut_r, dedtaut_R)
            self.ekin -= dedtaut_R.integrate(taut_R - self.tauct_R)

        add_gradient_correction(self.grad_v, gradn_svr, sigma_xr,
                                dedsigma_xr, vxct_sr)

        return e_r.integrate()

    def apply_orbital_dependent_hamiltonian(self, kpt, psit_xG,
                                            Htpsit_xG, dH_asp=None):
        self.wfs.apply_mgga_orbital_dependent_hamiltonian(
            kpt, psit_xG,
            Htpsit_xG, dH_asp,
            self.dedtaut_sG[kpt.s])




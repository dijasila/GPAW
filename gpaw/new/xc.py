import numpy as np
from gpaw.xc.functional import XCFunctional as OldXCFunctional
from gpaw.xc.gga import add_gradient_correction, gga_vars


def create_functional(xc: OldXCFunctional,
                      grid, coarse_grid, setups, fracpos_ac, atomdist):
    if xc.type == 'MGGA':
        return MGGAFunctional(xc, grid,
                              coarse_grid, setups, fracpos_ac, atomdist)
    return LDAOrGGAFunctional(xc, grid)


class Functional:
    def __init__(self, xc, grid):
        self.xc = xc
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
                 interpolation_domain,
                 setups,
                 fracpos_ac,
                 atomdist):
        super()(xc, grid)

        self.tauct_aX = setups.create_pseudo_core_kinetic_energy_densities(
            interpolation_domain,
            fracpos_ac,
            atomdist)

        self.tauct_aX.to_uniform_grid(out=out,
                                      scale=1.0 / (self.ncomponents % 3))

    def get_setup_name(self):
        return 'PBE'

    def calculate(self,
                  nt_sr,
                  vxct_r,
                  ibzwfs,
                  interpolate,
                  restrict) -> float:
        taut_sR = ibzwfs.calculate_kinetic_energy_density()

        for taut_R, taut_r in zip(taut_sR, taut_sr):
            taut_R += 1.0 / self.wfs.nspins * self.tauct_G
            interpolate(taut_R, taut_r)

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
            restrict(dedtaut_sg[s], self.dedtaut_sG[s])
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




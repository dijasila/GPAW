from gpaw.xc import XC


def RI(name, **args):
    if name == 'HSE06WIP':
        return RIXC(name, exx_fraction=0.25,
                    omega=0.11, localxc='HYB_GGA_XC_HSE06')
    raise NotImplementedError(f'RI backend does not support '
                              f'functional called: {name}.')


EXXCW_NONE_ERROR = 'The setup does not contain erfc-screened exchange '\
                   'core-core energy for given omega.'


class RIXC:
    orbital_dependent = False  # SIC!
    type = 'ri'
    # orbital_dependent_lcao = True
    # type = 'auxhybrid'

    def __init__(self, name, *, exx_fraction=None, omega=None, localxc=None):
        self.name = name
        self.exx_fraction = exx_fraction
        self.omega = omega
        self.localxc = XC(localxc)

    def initialize(self, density, hamiltonian, wfs):
        self.ecc = 0
        for setup in wfs.setups:
            if self.omega is not None:
                if setup.ExxC_w is None:
                    raise RuntimeError(EXXCW_NONE_ERROR)
                self.ecc += setup.ExxC_w[self.omega] * self.exx_fraction
            else:
                self.ecc += setup.ExxC * self.exx_fraction
        # self.ri_algorithm.initialize(density, hamiltonian, wfs)

    def get_setup_name(self):
        return 'PBE'

    def set_grid_descriptor(self, gd):
        pass

    def set_positions(self, spos_ac):
        pass

    def get_description(self):
        return f'Resolution of identity (RI) for {self.name} functional.\n'\
               f'   EXX fraction: {self.exx_fraction}\n'\
               f'   Kernel      : erfc(wr)/r\n'\
               f'   omega (w)   : {self.omega}\n'

    def calculate(self, gd, nt_sr, vt_sr):
        """
             Calculate the semi-local potential of the hybrid.
        """
        # if self.use_lda:
        #     return self.ldaxc.calculate(gd, nt_sr, vt_sr)
        energy = self.ecc  # + self.evv + self.evc
        energy += self.localxc.calculate(gd, nt_sr, vt_sr)
        return energy

    def calculate_paw_correction(self, setup, D_sp, dH_sp=None, a=None):
        return self.localxc.calculate_paw_correction(setup, D_sp, dH_sp, a=a)

    def get_kinetic_energy_correction(self):
        return 0  # self.ekin

    def summary(self, log):
        # Take no changes
        # if self.use_lda:
        #     raise ValueError('Error: Due to an
        # internal error, LDA was used thorough the calculation.')

        log(self.get_description())

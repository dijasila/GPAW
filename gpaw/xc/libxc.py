import os

try:
    import pylibxc
except ImportError:
    pylibxc = None

import _gpaw
from gpaw.xc.kernel import XCKernel
from gpaw import debug

short_names = {
    'LDA': 'LDA_X+LDA_C_PW',
    'PW91': 'GGA_X_PW91+GGA_C_PW91',
    'PBE': 'GGA_X_PBE+GGA_C_PBE',
    'PBEsol': 'GGA_X_PBE_SOL+GGA_C_PBE_SOL',
    'revPBE': 'GGA_X_PBE_R+GGA_C_PBE',
    'RPBE': 'GGA_X_RPBE+GGA_C_PBE',
    'BLYP': 'GGA_X_B88+GGA_C_LYP',
    'HCTH407': 'GGA_XC_HCTH_407',
    'WC': 'GGA_X_WC+GGA_C_PBE',
    'AM05': 'GGA_X_AM05+GGA_C_AM05',
    # 'M06-L': 'MGGA_X_M06_L+MGGA_C_M06_L',
    # 'TPSS': 'MGGA_X_TPSS+MGGA_C_TPSS',
    # 'revTPSS': 'MGGA_X_REVTPSS+MGGA_C_REVTPSS',
    'mBEEF': 'MGGA_X_MBEEF+GGA_C_PBE_SOL',
    'SCAN': 'MGGA_X_SCAN+MGGA_C_SCAN'}


if pylibxc:
    LDA = {pylibxc.flags.XC_FAMILY_LDA,
           pylibxc.flags.XC_FAMILY_HYB_LDA}
    GGA = {pylibxc.flags.XC_FAMILY_GGA,
           pylibxc.flags.XC_FAMILY_HYB_GGA}
    MGGA = {pylibxc.flags.XC_FAMILY_MGGA,
            pylibxc.flags.XC_FAMILY_HYB_MGGA}


def LibXC(name):
    if os.environ.get('GPAW_USE_PYLIBXC'):
        return PyLibXC(name)
    return GPAWLibXC(name)


class PyLibXC(XCKernel):
    """Functionals from libxc."""
    def __init__(self, name):
        self.name = name
        self.omega = None
        self.nspins = 0
        self.xcs = []
        name = short_names.get(name, name)
        number = pylibxc.util.xc_functional_get_number(name)
        if number != -1:
            self.numbers = [number]
        else:
            try:
                x, c = name.split('+')
            except ValueError:
                raise NameError('Unknown functional: "%s".' % name)
            x = pylibxc.util.xc_functional_get_number(x)
            c = pylibxc.util.xc_functional_get_number(c)
            if x == -1 or c == -1:
                raise NameError('Unknown functional: "%s".' % name)
            self.numbers = [x, c]

        self.families = []
        for id in self.numbers:
            family = pylibxc.util.xc_family_from_id(id)[0]
            assert family in (LDA | GGA | MGGA)
            self.families.append(family)

        self.type = 'LDA'
        for family in self.families:
            if family in MGGA:
                self.type = 'MGGA'
                break
            if family in GGA:
                self.type = 'GGA'

    def initialize(self, nspins):
        self.nspins = nspins
        self.xcs = [pylibxc.functional.LibXCFunctional(n, nspins)
                    for n in self.numbers]

    def calculate(self, e_g, n_sg, dedn_sg,
                  sigma_xg=None, dedsigma_xg=None,
                  tau_sg=None, dedtau_sg=None):
        if debug:
            self.check_arguments(e_g, n_sg, dedn_sg, sigma_xg, dedsigma_xg,
                                 tau_sg, dedtau_sg)
        nspins = len(n_sg)
        if self.nspins != nspins:
            self.initialize(nspins)

        if self.type == 'GGA':
            dedsigma_xg[:] = 0.0
        if self.type == 'MGGA':
            dedtau_sg[:] = 0.0

        for xc, family in zip(self.xcs, self.families):
            inp = {}
            if nspins == 1:
                inp['rho'] = n_sg[0]
                if family in GGA or family in MGGA:
                    inp['sigma'] = sigma_xg[0]
                    if family in MGGA:
                        inp['tau'] = tau_sg[0]
            else:
                inp['rho'] = n_sg.T.copy()
                if family in GGA or family in MGGA:
                    inp['sigma'] = sigma_xg.T.copy()
                    if family in MGGA:
                        inp['tau'] = tau_sg.T.copy()
            out = xc.compute(inp)
            if nspins == 1:
                e_g[:] = out['zk'][0].reshape(e_g.shape)
                dedn_sg += out['vrho'].reshape(n_sg.shape)
                if family in GGA or family in MGGA:
                    dedsigma_xg += out['vsigma'].reshape(sigma_xg.shape)
                    if family in MGGA:
                        dedtau_sg += out['vtau'].reshape(tau_sg.shape)

            else:
                e_g[:] = out['zk'][0].reshape(e_g.shape[::-1]).T
                dedn_sg += out['vrho'].reshape(n_sg.shape[::-1]).T
                if family in GGA or family in MGGA:
                    dedsigma_xg += out['vsigma'].reshape(
                        sigma_xg.shape[::-1]).T
                    if family in MGGA:
                        dedtau_sg += out['vtau'].reshape(tau_sg.shape[::-1]).T

        e_g *= n_sg.sum(0)

    def calculate_fxc_spinpaired(self, n_g, fxc_g):
        if self.nspins != 1:
            self.initialize(nspins=1)

        fxc_g[:] = 0.0

        for xc, family in zip(self.xcs, self.families):
            assert family in LDA
            inp = {'rho': n_g}
            out = xc.compute(inp, do_exc=False, do_vxc=False, do_fxc=True)
            fxc_g += out['v2rho2'].reshape(n_g.shape)

    def set_omega(self, omega):
        for xc, family in zip(self.xcs, self.families):
            xc.xc_func.contents.cam_omega = omega


class GPAWLibXC(XCKernel):
    """Functionals from libxc."""
    def __init__(self, name):
        self.name = name
        self.omega = None
        self.initialize(nspins=1)

    def initialize(self, nspins):
        self.nspins = nspins
        name = short_names.get(self.name, self.name)
        number = _gpaw.lxcXCFuncNum(name)
        if number is not None:
            f = number
            xc = -1
            x = -1
            c = -1
            if '_XC_' in name:
                xc = f
            elif '_C_' in name:
                c = f
            else:
                x = f
        else:
            try:
                x, c = name.split('+')
            except ValueError:
                raise NameError('Unknown functional: "%s".' % name)
            xc = -1
            x = _gpaw.lxcXCFuncNum(x)
            c = _gpaw.lxcXCFuncNum(c)
            if x is None or c is None:
                raise NameError('Unknown functional: "%s".' % name)

        self.xc = _gpaw.lxcXCFunctional(xc, x, c, nspins)
        self.set_omega()

        if self.xc.is_mgga():
            self.type = 'MGGA'
        elif self.xc.is_gga():
            self.type = 'GGA'
        else:
            self.type = 'LDA'

    def calculate(self, e_g, n_sg, dedn_sg,
                  sigma_xg=None, dedsigma_xg=None,
                  tau_sg=None, dedtau_sg=None):
        if debug:
            self.check_arguments(e_g, n_sg, dedn_sg, sigma_xg, dedsigma_xg,
                                 tau_sg, dedtau_sg)
        nspins = len(n_sg)
        if self.nspins != nspins:
            self.initialize(nspins)

        self.xc.calculate(e_g.ravel(), n_sg, dedn_sg,
                          sigma_xg, dedsigma_xg,
                          tau_sg, dedtau_sg)

    def calculate_fxc_spinpaired(self, n_g, fxc_g):
        self.xc.calculate_fxc_spinpaired(n_g.ravel(), fxc_g)

    def set_omega(self, omega=None):
        """Set the value of gamma/omega in RSF."""
        if omega is not None:
            self.omega = omega
        if self.omega is not None:
            if not self.xc.set_omega(self.omega):
                raise ValueError('Tried setting omega on non RSF hybrid.')

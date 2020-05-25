"""Python wrapper for GPAW's native XC functionals."""

import _gpaw
from gpaw import debug

codes = {
    'LDA': -1,
    'PBE': 0,
    'revPBE': 1,
    'RPBE': 2,
    'PW91': 14}


class XCNull:
    type = 'LDA'
    name = 'null'

    def calculate(self, e_g, n_sg, dedn_sg):
        e_g[:] = 0.0


class XCKernel:
    def __init__(self, name):
        self.name = name
        if name == 'LDA':
            self.type = 'LDA'
        else:
            self.type = 'GGA'
        self.xc = _gpaw.XCFunctional(codes[name])

    def calculate(self, e_g, n_sg, dedn_sg,
                  sigma_xg=None, dedsigma_xg=None,
                  tau_sg=None, dedtau_sg=None):
        """Calculate energy and derivatives from density and gradients.

         * e_g is the energy density.  Values are overwritten.
         * n_sg is the density and is an input.
         * dedn_sg is the partial derivative of the energy with
           respect to the density (any gradients constant).  Values are
           added to this array.
         * sigma_xg is the squared norm of the gradient and is an input.
         * dedsigma_xg is the partial derivative of the energy with respect
           to the squared gradient norm.  Values are overwritten.
         * tau_sg and dedtau_sg probably behave similarly but correspond to
           2nd-order derivatives for MGGAs.  XXX verify and document this.
        """
        if debug:
            self.check_arguments(e_g, n_sg, dedn_sg, sigma_xg, dedsigma_xg)
        self.xc.calculate(e_g, n_sg, dedn_sg, sigma_xg, dedsigma_xg)

    def check_arguments(self, e_g, n_sg, dedn_sg, sigma_xg, dedsigma_xg):
        S = n_sg.shape[0]
        G = n_sg.shape[1:]
        assert e_g.shape == G
        assert e_g.flags.contiguous and e_g.dtype == float
        assert dedn_sg.shape == (S,) + G
        assert dedn_sg.flags.contiguous
        assert dedn_sg.dtype == float
        if self.type != 'LDA':
            assert sigma_xg.shape == (2 * S - 1,) + G
            assert dedsigma_xg.shape == (2 * S - 1,) + G
            assert sigma_xg.flags.contiguous and sigma_xg.dtype == float
            assert (dedsigma_xg.flags.contiguous and
                    dedsigma_xg.dtype == float)

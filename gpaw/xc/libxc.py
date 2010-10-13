import _gpaw
from gpaw.xc.kernel import XCKernel
from gpaw import debug

short_names = {
    'LDA':     'LDA_X,LDA_C_PW',
    'PW91':    'GGA_X_PW91,GGA_C_PW91',
    'PBE':     'GGA_X_PBE,GGA_C_PBE',
    'revPBE':  'GGA_X_PBE_R,GGA_C_PBE',
    'RPBE':    'GGA_X_RPBE,GGA_C_PBE',
    'HCTH407': 'GGA_XC_HCTH_407',
    'TPSS':    'MGGA_X_TPSS,MGGA_C_TPSS',
    'M06L':    'MGGA_X_M06L,MGGA_C_M06L'}


class LibXC(XCKernel):
    def __init__(self, name):
        self.name = name
        self.initialize(nspins=1)

    def initialize(self, nspins):
        self.nspins = nspins
        name = short_names.get(self.name, self.name)
        if name in libxc_functionals:
            f = libxc_functionals[name]
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
            x, c = name.split(',')
            xc = -1
            x = libxc_functionals[x]
            c = libxc_functionals[c]

        if xc != -1:
            # The C code can't handle this case!
            c = xc
            xc = -1

        self.xc = _gpaw.lxcXCFunctional(xc, x, c, nspins)

        if self.xc.is_mgga():
            self.type = 'MGGA'
        elif self.xc.is_gga() or self.xc.is_hyb_gga():
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

        if nspins == 1:
            self.xc.calculate_spinpaired(e_g.ravel(), n_sg, dedn_sg,
                                         sigma_xg, dedsigma_xg,
                                         tau_sg, dedtau_sg)
        else:
            if self.type == 'LDA':
                self.xc.calculate_spinpolarized(
                    e_g.ravel(),
                    n_sg[0], dedn_sg[0],
                    n_sg[1], dedn_sg[1])
            elif self.type == 'GGA':
                self.xc.calculate_spinpolarized(
                    e_g.ravel(),
                    n_sg[0], dedn_sg[0],
                    n_sg[1], dedn_sg[1],
                    sigma_xg[0], sigma_xg[1], sigma_xg[2],
                    dedsigma_xg[0], dedsigma_xg[1], dedsigma_xg[2])
            else:
                self.xc.calculate_spinpolarized(
                    e_g.ravel(),
                    n_sg[0], dedn_sg[0],
                    n_sg[1], dedn_sg[1],
                    sigma_xg[0], sigma_xg[1], sigma_xg[2],
                    dedsigma_xg[0], dedsigma_xg[1], dedsigma_xg[2],
                    tau_sg[0], tau_sg[1],
                    dedtau_sg[0], dedtau_sg[1])


# libxc: svn version 4179
# http://www.tddft.org/programs/octopus/wiki/index.php/Libxc
libxc_functionals = {
    'LDA_X': 1,
    'LDA_C_WIGNER': 2,
    'LDA_C_RPA': 3,
    'LDA_C_HL': 4,
    'LDA_C_GL': 5,
    'LDA_C_XALPHA': 6,
    'LDA_C_VWN': 7,
    'LDA_C_VWN_RPA': 8,
    'LDA_C_PZ': 9,
    'LDA_C_PZ_MOD': 10,
    'LDA_C_OB_PZ': 11,
    'LDA_C_PW': 12,
    'LDA_C_PW_MOD': 13,
    'LDA_C_OB_PW': 14,
    'LDA_C_AMGB': 15,
    'LDA_XC_TETER93': 20,
    'GGA_X_PBE': 101,
    'GGA_X_PBE_R': 102,
    'GGA_X_B86': 103,
    'GGA_X_B86_R': 104,
    'GGA_X_B86_MGC': 105,
    'GGA_X_B88': 106,
    'GGA_X_G96': 107,
    'GGA_X_PW86': 108,
    'GGA_X_PW91': 109,
    'GGA_X_OPTX': 110,
    'GGA_X_DK87_R1': 111,
    'GGA_X_DK87_R2': 112,
    'GGA_X_LG93': 113,
    'GGA_X_FT97_A': 114,
    'GGA_X_FT97_B': 115,
    'GGA_X_PBE_SOL': 116,
    'GGA_X_RPBE': 117,
    'GGA_X_WC': 118,
    'GGA_X_mPW91': 119,
    'GGA_X_AM05': 120,
    'GGA_X_PBEA': 121,
    'GGA_X_MPBE': 122,
    'GGA_X_XPBE': 123,
    'GGA_C_PBE': 130,
    'GGA_C_LYP': 131,
    'GGA_C_P86': 132,
    'GGA_C_PBE_SOL': 133,
    'GGA_C_PW91': 134,
    'GGA_C_AM05': 135,
    'GGA_C_XPBE': 136,
    'GGA_C_PBE_REVTPSS': 137,
    'GGA_XC_LB': 160,
    'GGA_XC_HCTH_93': 161,
    'GGA_XC_HCTH_120': 162,
    'GGA_XC_HCTH_147': 163,
    'GGA_XC_HCTH_407': 164,
    'GGA_XC_EDF1': 165,
    'GGA_XC_XLYP': 166,
    'HYB_GGA_XC_B3PW91': 401,
    'HYB_GGA_XC_B3LYP': 402,
    'HYB_GGA_XC_B3P86': 403,
    'HYB_GGA_XC_O3LYP': 404,
    'HYB_GGA_XC_PBEH': 406,
    'HYB_GGA_XC_X3LYP': 411,
    'HYB_GGA_XC_B1WC': 412,
    'MGGA_X_TPSS': 201,
    'MGGA_C_TPSS': 202,
    'MGGA_X_M06L': 203,
    'MGGA_C_M06L': 204,
    'MGGA_X_REVTPSS': 205,
    'MGGA_C_REVTPSS': 206
    }

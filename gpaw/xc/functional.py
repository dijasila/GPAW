from gpaw.xc.libxc import LibXC
from gpaw.xc.lda import LDA
from gpaw.xc.gga import GGA
from gpaw.xc.mgga import MGGA


class XCNull:
    type = 'LDA'
    name = 'null'
    def calculate(self, n_sg, e_g, dedn_sg):
        e_g[:] = 0.0


def XC(xckernel, parameters=None):
    if isinstance(xckernel, str):
        name = xckernel
        if name == 'vdW-DF':
            from gpaw.xc.vdw import FFTVDWFunctional
            return FFTVDWFunctional()
        if name == 'BEE1':
            from gpaw.xc.bee import BEE1
            xckernel = BEE1(parameters)
        else:
            xckernel = LibXC(xckernel)
    if xckernel.type == 'LDA':
        return LDA(xckernel)
    elif xckernel.type == 'GGA':
        return GGA(xckernel)
    else:
        return MGGA(xckernel)


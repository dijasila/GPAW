from gpaw.xc.libxc import LibXC
from gpaw.xc.lda import LDA
from gpaw.xc.gga import GGA


class XCNull:
    type = 'LDA'
    name = 'null'
    def calculate(self, n_sg, e_g, dedn_sg):
        e_g[:] = 0.0


def XC(xckernel):
    if isinstance(xckernel, str):
        name = xckernel
        if name == 'vdW-DF':
            from gpaw.xc.vdw import FFTVDWFunctional
            return FFTVDWFunctional()
        xckernel = LibXC(xckernel)
    if xckernel.type == 'LDA':
        return LDA(xckernel)
    elif xckernel.type == 'GGA':
        return GGA(xckernel)
    else:
        sdfgdfg

from gpaw.xc.libxc import LibXC
from gpaw.xc.lda import LDA
from gpaw.xc.gga import GGA
from gpaw.xc.mgga import MGGA


def XC(kernel, parameters=None):
    if isinstance(kernel, str):
        name = kernel
        if name == 'vdW-DF':
            from gpaw.xc.vdw import FFTVDWFunctional
            return FFTVDWFunctional()
        elif name == 'EXX':
            from gpaw.xc.hybrid import HybridXC
            return HybridXC()
        elif name == 'BEE1':
            from gpaw.xc.bee import BEE1
            kernel = BEE1(parameters)
        else:
            kernel = LibXC(kernel)
    if kernel.type == 'LDA':
        return LDA(kernel)
    elif kernel.type == 'GGA':
        return GGA(kernel)
    else:
        return MGGA(kernel)


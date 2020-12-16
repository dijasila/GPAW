from math import pi
from typing import Dict, Tuple, Sequence

import numpy as np

from gpaw.wavefunctions.mode import Mode
from gpaw.tb.wavefunctions import TBWaveFunctions
from gpaw.tb.repulsion import Repulsion
from gpaw.tb.parameters import DefaultParameters
from gpaw.spline import Spline
from gpaw.setup import Setup
from gpaw.xc.functional import XCFunctional


class TB(Mode):
    name = 'tb'
    interpolation = 1
    force_complex_dtype = False

    def __init__(self, parameters: Dict[Tuple[str, str], Repulsion] = None):
        if parameters is None:
            parameters = DefaultParameters()
        self.parameters = parameters

    def __call__(self, ksl, **kwargs) -> TBWaveFunctions:
        return TBWaveFunctions(ksl, **kwargs)

    def fix_old_setups(self,
                       setups: Sequence[Setup],
                       xc: XCFunctional) -> None:
        for setup in setups:
            print('W', setup.data.W)
            vt, W = calculate_potential(setup, xc)
            print('W', W);asdfg


def calculate_potential(setup: Setup,
                        xc: XCFunctional) -> Tuple[Spline, float]:
    phit_jg = np.array(setup.data.phit_jg)
    rgd = setup.rgd

    # Density:
    nt_g = np.einsum('jg, j, jg -> g',
                     phit_jg, setup.f_j, phit_jg) / (4 * pi)
    nt_g += setup.data.nct_g * (1 / (4 * pi)**0.5)

    # XC:
    vt_g = rgd.zeros()
    xc.calculate_spherical(rgd, nt_g[np.newaxis], vt_g[np.newaxis])
    vt_g += setup.vbar.map(rgd.r_g) / (4 * pi)**0.5

    # Coulomb:
    g_g = setup.ghat_l[0].map(rgd.r_g)
    Q = -rgd.integrate(nt_g) / rgd.integrate(g_g)
    rhot_g = nt_g + Q * g_g
    vHtr_g = rgd.poisson(rhot_g)
    vt_g[1:] += vHtr_g[1:] / rgd.r_g[1:]
    vt_g[0] += vHtr_g[1] / rgd.r_g[1]

    W = rgd.integrate(g_g * vHtr_g, n=-1) / (4 * pi)**0.5
    print(rgd.integrate(g_g) / (4 * pi))
    rgd.plot(g_g)
    rgd.plot(vHtr_g, show=1)
    return rgd.spline(vt_g * (4 * pi)**0.5, points=300), W

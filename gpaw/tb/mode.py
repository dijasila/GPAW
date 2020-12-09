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

    def __call__(self, ksl, xc, **kwargs) -> TBWaveFunctions:
        return TBWaveFunctions(xc, ksl, **kwargs)

    def fix_setups(self,
                   setups: Sequence[Setup],
                   xc: XCFunctional) -> None:
        for setup in setups:
            print(setup.data.eps_j)
            print(setup)
            if setup.vt is None:
                setup.vt = calculate_potential(setup, xc)


def calculate_potential(setup: Setup,
                        xc: XCFunctional) -> Spline:
    phit_jg = np.array(setup.data.phit_jg)
    rgd = setup.rgd

    # Densities with frozen core:
    nt_g = np.einsum('jg, j, jg -> g',
                     phit_jg, setup.f_j, phit_jg) / (4 * pi)
    nt_g += setup.data.nct_g * (1 / (4 * pi)**0.5)

    # XC potential:
    nt_sg = nt_g[np.newaxis]
    vt_sg = np.zeros_like(nt_sg)
    xc.calculate_spherical(rgd, nt_sg, vt_sg)
    vt_sg[0] += setup.vbar.map(rgd.r_g) / (4 * pi)**0.5
    vtr_g = vt_sg[0] * rgd.r_g
    g_g = setup.ghat_l[0].map(rgd.r_g)
    Q = -rgd.integrate(nt_g) / rgd.integrate(g_g)
    nt_g += Q * g_g
    vtr_g += rgd.poisson(nt_g)
    vtr_g[1:] /= rgd.r_g[1:]
    vtr_g[0] = vtr_g[1]
    return rgd.spline(vtr_g * (4 * pi)**0.5, points=300)

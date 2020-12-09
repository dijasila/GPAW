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
from gpaw.hints import Array1D
from gpaw.utilities import pack


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

    def manipulate_setups_hook(self,
                               setups: Sequence[Setup],
                               xc: XCFunctional) -> None:
        for setup in setups:
            setup.vt, delta_eig_j = calculate_potential(setup, xc)
            K_i = sum(([e] * (2 * l + 1)
                      for e, l in zip(delta_eig_j, setup.l_j)),
                      [])
            setup.K_p = pack(np.diag(K_i))
            setup.M_p[:] = 0.0
            setup.MB_p[:] = 0.0
            setup.M_pp[:] = 0.0
            setup.M = 0.0
            setup.MB = 0.0
            setup.Kc = 0.0


def calculate_potential(setup: Setup,
                        xc: XCFunctional) -> Tuple[Spline, Array1D]:
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
    vt_g = vtr_g
    vt_g[1:] /= rgd.r_g[1:]
    vt_g[0] = vt_g[1]

    delta_eig_j = (rgd.integrate(phit_jg**2 * vt_g) / (4 * pi) -
                   setup.data.eps_j)

    return rgd.spline(vtr_g * (4 * pi)**0.5, points=300), delta_eig_j

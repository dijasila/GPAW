import pytest

import numpy as np

from gpaw import GPAW
from gpaw.mpi import world
from gpaw.response import ResponseGroundStateAdapter, ResponseContext
from gpaw.response.chiks import ChiKSCalculator, SelfEnhancementCalculator
from gpaw.response.frequencies import ComplexFrequencyDescriptor
from gpaw.response.dyson import DysonEnhancer

from gpaw.test.conftest import response_band_cutoff


@pytest.mark.kspair
@pytest.mark.response
def test_response_iron_sf_pawALDA(in_tmp_dir, gpw_files, scalapack):
    # ---------- Inputs ---------- #

    # Magnetic response calculation
    q_qc = [[0.0, 0.0, 0.0], [0.0, 0.0, 1. / 4.]]  # Two q-points along G-N
    frq_qw = [np.linspace(-0.06, 0.1, 21),
              np.linspace(0.280, 0.440, 21)]
    ecut = 100
    eta = 0.1
    rshewmin = 1e-8

    if world.size > 1:
        nblocks = 2
    else:
        nblocks = 1

    # ---------- Script ---------- #

    context = ResponseContext(txt='iron_susceptibility.txt')
    calc = GPAW(gpw_files['fe_pw'], parallel=dict(domain=1))
    nbands = response_band_cutoff['fe_pw']
    gs = ResponseGroundStateAdapter(calc)

    calc_args = (gs,)
    calc_kwargs = dict(context=context,
                       nbands=nbands,
                       ecut=ecut,
                       gammacentered=True,
                       nblocks=nblocks)
    chiks_calc = ChiKSCalculator(*calc_args, **calc_kwargs)
    xi_calc = SelfEnhancementCalculator(*calc_args,
                                        rshewmin=rshewmin,
                                        **calc_kwargs)
    dyson_enhancer = DysonEnhancer(context)

    for q_c, frq_w in zip(q_qc, frq_qw):
        # Calculate χ_KS^+- and Ξ^++
        zd = ComplexFrequencyDescriptor.from_array(frq_w + 1j * eta)
        chiks = chiks_calc.calculate('+-', q_c, zd)
        xi = xi_calc.calculate(q_c, zd)

        # Distribute frequencies and invert dyson equation
        chiks = chiks.copy_with_global_frequency_distribution()
        xi = xi.copy_with_global_frequency_distribution()
        chi = dyson_enhancer(chiks, xi)

    context.write_timer()

    # XXX To do XXX
    # * Extract and save macroscopic component
    # * Extract and save acoustic mode
    # * Plot component and mode
    # * Extract magnon frequency
    # * Test against reference values

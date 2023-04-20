# General modules
import pytest
import numpy as np

# Script modules
from gpaw import GPAW
from gpaw.response import ResponseGroundStateAdapter, ResponseContext
from gpaw.response.chiks import ChiKSCalculator
from gpaw.response.susceptibility import ChiFactory
from gpaw.response.localft import LocalFTCalculator
from gpaw.response.fxc_kernels import FXCScaling


@pytest.mark.kspair
@pytest.mark.response
def test_response_cobalt_sf_gssALDA(in_tmp_dir, gpw_files):
    # ---------- Inputs ---------- #

    fxc = 'ALDA'
    q_qc = [[0.0, 0.0, 0.0], [0.0, 0.0, 1. / 4.]]  # Two q-points along G-N
    frq_qw = [np.linspace(-0.080, 0.120, 26), np.linspace(0.250, 0.450, 26)]
    eta = 0.1

    rshelmax = 0
    fxc_scaling = FXCScaling('fm')
    ecut = 100
    reduced_ecut = 50
    nblocks = 'max'

    # ---------- Script ---------- #

    # Initialize objects to calculat Chi
    context = ResponseContext()
    calc = GPAW(gpw_files['co_pw_wfs'], parallel=dict(domain=1))
    nbands = calc.parameters.convergence['bands']
    gs = ResponseGroundStateAdapter(calc)
    chiks_calc = ChiKSCalculator(gs,
                                 nbands=nbands,
                                 ecut=ecut,
                                 gammacentered=True,
                                 nblocks=nblocks)
    chi_factory = ChiFactory(chiks_calc)
    localft_calc = LocalFTCalculator.from_rshe_parameters(gs, context,
                                                          rshelmax=rshelmax)

    for q, (q_c, frq_w) in enumerate(zip(q_qc, frq_qw)):
        complex_frequencies = frq_w + 1.j * eta
        if q == 0:
            chi = chi_factory('+-', q_c, complex_frequencies,
                              fxc_scaling=fxc_scaling,
                              fxc=fxc, localft_calc=localft_calc)
            fxc_kernel = chi.fxc_kernel
        else:
            chi = chi_factory('+-', q_c, complex_frequencies,
                              fxc_scaling=fxc_scaling,
                              fxc_kernel=fxc_kernel)
        chi.write_reduced_arrays(f'chiwGG_q{q}.pckl',
                                 reduced_ecut=reduced_ecut)

    context.write_timer()
    context.comm.barrier()

    # XXX First step XXX #
    # - Plot components G=0 and G=1
    # - Extract magnon energies and test values
    # - Save diagonal instead of array
    # XXX Second step XXX #
    # - Reformulate the documentation, such that a pair function can be an
    #   interacting one
    # - Move calculation responsibility to dyson solver
    # - Rename ChiFactory to SusceptibilityFactory and make it return both
    #   ChiKS and Chi.
    # - Rename ChiKS to Susceptibility and make chiks and chi individual
    #   instances of it
    # XXX Third step XXX #
    # - Make it possible to calculate the dissipative part of a Susceptibility
    # - Save the full spectral weight of KS and MB spectra (trace of the
    #   respective dissipative parts)
    # - Compare full spectral weight to the G=0 and G=1 components
    # XXX Fourth step XXX #
    # - Introduce the EigendecomposedSpectrum into GPAW
    # - Compare to full spectral weight and the G=0 and G=1 components

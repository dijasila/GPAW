"""Simple magnons calculation in a 1D hydrogen chain."""

# General modules
import pytest
import numpy as np

# Script modules
from gpaw import GPAW
from gpaw.mpi import world
from gpaw.test import findpeak

from gpaw.response import ResponseGroundStateAdapter
from gpaw.response.frequencies import ComplexFrequencyDescriptor
from gpaw.response.chiks import ChiKSCalculator
from gpaw.response.susceptibility import ChiFactory
from gpaw.response.fxc_kernels import FXCScaling
from gpaw.response.df import read_response_function


@pytest.mark.kspair
@pytest.mark.response
def test_response_afm_hchain_gssALDA(in_tmp_dir, gpw_files):
    # ---------- Inputs ---------- #

    # Magnetic response calculation
    q_qc = [[0., 0., 0.],
            [1. / 6., 0., 0.],
            [1. / 3., 0., 0.]]
    fxc = 'ALDA'
    fxc_scaling = FXCScaling('afm')
    rshelmax = -1
    rshewmin = 1e-8
    ecut = 120
    frq_w = np.linspace(-0.6, 0.6, 41)
    eta = 0.24
    zd = ComplexFrequencyDescriptor.from_array(frq_w + 1.j * eta)
    if world.size > 1:
        nblocks = 2
    else:
        nblocks = 1

    # ---------- Script ---------- #

    # geet GS from GPW files
    calc = GPAW(gpw_files['h2_afm_wfs'])

    # Magnetic response calculation
    nbands = 2 * (1 + 0)  # 1s + 0 empty shell bands
    fxckwargs = {'calculator': {'method': 'old',
                                'rshelmax': rshelmax,
                                'rshewmin': rshewmin},
                 'fxc_scaling': fxc_scaling}
    gs = ResponseGroundStateAdapter(calc)
    chiks_calc = ChiKSCalculator(gs,
                                 nbands=nbands,
                                 ecut=ecut,
                                 gammacentered=True,
                                 nblocks=nblocks)
    chi_factory = ChiFactory(chiks_calc)
                  
    for q, q_c in enumerate(q_qc):
        filename = 'h-chain_macro_tms_q%d.csv' % q
        txt = 'h-chain_macro_tms_q%d.txt' % q
        chi = chi_factory('+-', q_c, zd,
                          fxc=fxc,
                          fxckwargs=fxckwargs,
                          txt=txt)
        chi.write_macroscopic_component(filename)

    chi_factory.context.write_timer()
    world.barrier()

    # Part 3: Identify magnon peak in finite q scattering function
    w0_w, _, chi0_w = read_response_function('h-chain_macro_tms_q0.csv')
    w1_w, _, chi1_w = read_response_function('h-chain_macro_tms_q1.csv')
    w2_w, _, chi2_w = read_response_function('h-chain_macro_tms_q2.csv')

    wpeak1, Ipeak1 = findpeak(w1_w, -chi1_w.imag / np.pi)
    wpeak2, Ipeak2 = findpeak(w2_w, -chi2_w.imag / np.pi)
    mw1 = calculate_afm_mw(wpeak1, eta) * 1000  # to meV
    mw2 = calculate_afm_mw(wpeak2, eta) * 1000  # to meV

    # Part 4: compare new results to test values
    test_fxcs = 1.04744
    test_mw1 = 285.  # meV
    test_mw2 = 494.  # meV  # Remark that mw2 < 2 * mw1 (linear dispersion)
    test_Ipeak1 = 0.0131  # a.u.
    test_Ipeak2 = 0.0290

    # Test fxc_scaling:
    fxcs = fxc_scaling.get_scaling()
    assert abs(fxcs - test_fxcs) < 0.005

    # Magnon peak at q=1/3 q_X:
    assert abs(mw1 - test_mw1) < 10.

    # Magnon peak at q=2/3 q_X:
    assert abs(mw2 - test_mw2) < 10.

    # Scattering function intensity:
    assert abs(Ipeak1 - test_Ipeak1) < 0.005
    assert abs(Ipeak2 - test_Ipeak2) < 0.005

    # Check that spectrum at q=0 vanishes
    chitol = 1e-3 * np.abs(chi1_w.imag).max()
    assert np.abs(chi0_w.imag).max() < chitol

    # Check that the spectrum is antisymmetric around q=0
    assert np.allclose(w0_w[19::-1] + w0_w[21:], 0.)
    assert np.allclose(chi0_w.imag[19::-1] + chi0_w.imag[21:], 0.,
                       atol=0.01 * chitol)
    assert np.allclose(chi1_w.imag[19::-1] + chi1_w.imag[21:], 0.,
                       atol=chitol)
    assert np.allclose(chi2_w.imag[19::-1] + chi2_w.imag[21:], 0.,
                       atol=chitol)


# ---------- Script functionality ---------- #


def calculate_afm_mw(peak_frequency, eta):
    """Assume lorentzian lineshape to compute the magnon frequency
    from the afm magnon lineshape peak position."""
    return np.sqrt(2 * np.sqrt(peak_frequency**4 + eta**2 * peak_frequency**2)
                   - peak_frequency**2 - eta**2)

import pytest

import numpy as np

from gpaw import GPAW
from gpaw.response import ResponseGroundStateAdapter, ResponseContext
from gpaw.response.frequencies import ComplexFrequencyDescriptor
from gpaw.response.chiks import ChiKSCalculator, SelfEnhancementCalculator
from gpaw.response.dyson import DysonEnhancer
from gpaw.response.susceptibility import (spectral_decomposition,
                                          read_eigenmode_lineshapes)

from gpaw.test import findpeak
from gpaw.test.gpwfile import response_band_cutoff


@pytest.mark.kspair
@pytest.mark.response
def test_response_cobalt_sf_gsspawALDA(in_tmp_dir, gpw_files):
    # ---------- Inputs ---------- #

    q_qc = [[0.0, 0.0, 0.0], [1. / 4., 0.0, 0.0]]  # Two q-points along G-M
    frq_w = np.linspace(-0.5, 2.0, 101)
    eta = 0.2

    rshelmax = 0
    ecut = 150
    pos_eigs = 5
    nmodes = 2  # majority modes
    nblocks = 'max'

    # ---------- Script ---------- #

    # Read ground state data
    context = ResponseContext(txt='cobalt_susceptibility.txt')
    calc = GPAW(gpw_files['co_pw'], parallel=dict(domain=1))
    nbands = response_band_cutoff['co_pw']
    gs = ResponseGroundStateAdapter(calc)

    # Set up response calculators
    calc_args = (gs,)
    calc_kwargs = dict(context=context,
                       nbands=nbands,
                       ecut=ecut,
                       gammacentered=True,
                       bandsummation='pairwise',
                       nblocks=nblocks)
    chiks_calc = ChiKSCalculator(*calc_args, **calc_kwargs)
    xi_calc = SelfEnhancementCalculator(*calc_args,
                                        rshelmax=rshelmax,
                                        **calc_kwargs)
    dyson_enhancer = DysonEnhancer(context)

    for q, q_c in enumerate(q_qc):
        # Calculate χ_KS^+-(q,z) and Ξ^++(q,z)
        zd = ComplexFrequencyDescriptor.from_array(frq_w + 1j * eta)
        chiks = chiks_calc.calculate('+-', q_c, zd)
        xi = xi_calc.calculate('+-', q_c, zd)

        # Distribute frequencies and invert dyson equation
        chiks = chiks.copy_with_global_frequency_distribution()
        xi = xi.copy_with_global_frequency_distribution()
        chi = dyson_enhancer(chiks, xi)

        # Calculate majority spectral function
        Amaj, _ = spectral_decomposition(chi, pos_eigs=pos_eigs)
        Amaj.write_eigenmode_lineshapes(f'cobalt_Amaj_q{q}.csv', nmodes=nmodes)

        # plot_enhancement(chiks, xi, Amaj, nmodes=nmodes)

    context.write_timer()

    # Compare magnon peaks to reference data
    refs_mq = [
        # Acoustic
        [
            # (wpeak, Apeak)
            (0.085, 7.895),
            (0.320, 5.828),
        ],
        # Optical
        [
            # (wpeak, Apeak)
            (0.904, 3.493),
            (0.857, 2.988),
        ],
    ]
    for q in range(len(q_qc)):
        w_w, a_wm = read_eigenmode_lineshapes(f'cobalt_Amaj_q{q}.csv')
        for m in range(nmodes):
            wpeak, Apeak = findpeak(w_w, a_wm[:, m])
            print(m, q, wpeak, Apeak)
            assert wpeak == pytest.approx(refs_mq[m][q][0], abs=0.01)  # eV
            assert Apeak == pytest.approx(refs_mq[m][q][1], abs=0.05)  # a.u.


def plot_enhancement(chiks, xi, Amaj, *, nmodes):
    import matplotlib.pyplot as plt
    from gpaw.mpi import world
    from ase.units import Ha

    # Project χ_KS^+-(q,z) and Ξ^++(q,z) onto the mode vectors
    wm = Amaj.get_eigenmode_frequency(nmodes=nmodes)
    a_mW = Amaj.get_eigenmodes_at_frequency(wm, nmodes=nmodes).T
    v_Gm = Amaj.get_eigenvectors_at_frequency(wm, nmodes=nmodes)
    chiks_wm = np.zeros((chiks.blocks1d.nlocal, nmodes), dtype=complex)
    xi_wm = np.zeros((xi.blocks1d.nlocal, nmodes), dtype=complex)
    for m, v_G in enumerate(v_Gm.T):
        chiks_wm[:, m] = np.conj(v_G) @ chiks.array @ v_G  # chiks_wGG
        xi_wm[:, m] = np.conj(v_G) @ xi.array @ v_G  # xi_wGG
    chiks_mW = chiks.blocks1d.all_gather(chiks_wm).T
    xi_mW = xi.blocks1d.all_gather(xi_wm).T

    for m in range(nmodes):
        plt.subplot(1, nmodes, m + 1)
        plt.plot(chiks.zd.omega_w * Ha, -chiks_mW[m].imag / np.pi)
        plt.plot(xi.zd.omega_w * Ha, xi_mW[m].real)
        plt.axhline(1., c='0.5')
        plt.plot(Amaj.omega_w, a_mW[m])
    if world.rank == 0:
        plt.show()

"""Simple magnons calculation in chromium (antiferromagnet)."""

# General modules
import pytest
import numpy as np

# Script modules
from ase.build import bulk
from ase.dft.kpoints import monkhorst_pack

from gpaw import PW, GPAW
from gpaw.mpi import world
from gpaw.test import findpeak, equal
from gpaw.response.tms import TransverseMagneticSusceptibility
from gpaw.response.susceptibility import read_macroscopic_component


@pytest.mark.response
def test_response_cr_sf_gssALDA(in_tmp_dir):
    # ---------- Inputs ---------- #

    # Part 1: Ground state calculation
    xc = 'LDA'
    kpts = 12
    nbands = 2 * (6 + 0)  # 4s + 3d + 0 empty shell bands
    pw = 300
    conv = {'bands': nbands}
    a = 2.884
    mm = 1.

    # Part 2: Magnetic response calculation
    q_qc = [[0., 0., 0.],
            [0., 1. / 6., 0.]]
    fxc = 'ALDA'
    fxc_scaling = [True, None, 'afm']
    rshelmax = -1
    rshewmin = 1e-8
    ecut = 150
    frq_w = np.linspace(-0.4, 1.0, 36)
    eta = 0.32
    if world.size > 1:
        nblocks = 2
    else:
        nblocks = 1

    # ---------- Script ---------- #

    # Part 1: Ground state calculation

    Crbcc = bulk('Cr', 'bcc', a=a, cubic=True)
    Crbcc.set_initial_magnetic_moments([mm, -mm])

    calc = GPAW(xc=xc,
                mode=PW(pw),
                kpts=monkhorst_pack((kpts, kpts, kpts)),
                nbands=nbands + 6,
                convergence=conv,
                symmetry={'point_group': True},
                idiotproof=False,
                parallel={'domain': 1})

    Crbcc.calc = calc
    Crbcc.get_potential_energy()

    # Part 2: Magnetic response calculation
    fxckwargs = {'rshelmax': rshelmax,
                 'rshewmin': rshewmin,
                 'fxc_scaling': fxc_scaling}
    tms = TransverseMagneticSusceptibility(calc,
                                           nbands=nbands,
                                           fxc=fxc,
                                           eta=eta,
                                           ecut=ecut,
                                           fxckwargs=fxckwargs,
                                           gammacentered=True,
                                           nblocks=nblocks)
    for q, q_c in enumerate(q_qc):
        tms.get_macroscopic_component('+-', q_c, frq_w,
                                      filename='cr_macro_tms' + '_q%d.csv' % q)

    tms.write_timer()
    world.barrier()

    # Part 3: Identify magnon peak in finite q scattering function
    w0_w, chiks0_w, chi0_w = read_macroscopic_component('cr_macro_tms_q0.csv')
    w1_w, chiks1_w, chi1_w = read_macroscopic_component('cr_macro_tms_q1.csv')

    wpeak, Ipeak = findpeak(w1_w, -chi1_w.imag / np.pi)
    mw = wpeak * 1000  # meV

    # Part 4: compare new results to test values
    test_fxcs = 0.933
    test_mw = 782.  # meV
    test_Ipeak = 0.034  # a.u.

    # Test fxc_scaling:
    equal(fxc_scaling[1], test_fxcs, 0.005)

    # Magnon peak at q=1/3 q_X:
    equal(mw, test_mw, 15.)

    # Scattering function intensity:
    equal(Ipeak, test_Ipeak, 0.005)

    # Check that spectrum at q=0 vanishes
    chitol = 1e-3 * np.abs(chi1_w.imag).max()
    assert np.abs(chi0_w.imag).max() < chitol

    # Check that the spectrum is antisymmetric around q=0
    assert np.allclose(w0_w[9::-1] + w0_w[11:21], 0.)
    assert np.allclose(chi0_w.imag[9::-1] + chi0_w.imag[11:21], 0.,
                       atol=0.01 * chitol)
    assert np.allclose(chi1_w.imag[9::-1] + chi1_w.imag[11:21], 0.,
                       atol=chitol)

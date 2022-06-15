"""Test self-consistency of the extend_head flag for a semi-conductor."""

# General modules
import pytest
import numpy as np

# Script modules
from ase import Atoms
from ase.dft.kpoints import monkhorst_pack

from gpaw import GPAW, PW
from gpaw.mpi import world
from gpaw.response.chi0 import Chi0


@pytest.mark.response
def test_he_chi0_extend_head(in_tmp_dir):
    # ---------- Inputs ---------- #

    # Part 1: Ground state calculation
    a = 3.0
    xc = 'LDA'
    kpts = 2
    nbands = 1 + 1 + 3  # 1s + 2s + 2p empty shell bands
    ebands = 1  # Include also 3s bands for numerical consistency
    pw = 250
    conv = {'bands': nbands}

    # Part 2: Chi0 calculation
    ecut = 50
    frequencies = np.linspace(0., 30., 11)
    eta = 0.05
    hilbert = False
    timeordered = False
    threshold = 1
    real_space_derivatives = False
    if world.size > 1:
        nblocks = 2
    else:
        nblocks = 1

    # ---------- Script ---------- #

    # Part 1: Ground state calculation
    atoms = Atoms('He', cell=[a, a, a], pbc=True)

    calc = GPAW(xc=xc,
                mode=PW(pw),
                kpts=monkhorst_pack((kpts, kpts, kpts)),
                nbands=nbands + ebands,
                convergence=conv,
                symmetry={'point_group': True},
                idiotproof=False,
                parallel={'domain': 1})

    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write('He', 'all')

    # Part 2: Chi0 calculation
    chi0fac = Chi0('He', ecut=ecut,
                   frequencies=frequencies, eta=eta, hilbert=hilbert,
                   nbands=nbands,
                   timeordered=timeordered,
                   threshold=threshold,
                   real_space_derivatives=real_space_derivatives,
                   nblocks=nblocks)

    chi0_f = calculate_optical_limit(chi0fac, extend_head=False)
    chi0_t = calculate_optical_limit(chi0fac, extend_head=True)

    is_equal, err_msg = chi0_data_is_equal(chi0_f, chi0_t)
    assert is_equal, err_msg


# ---------- Script functionality ---------- #


def calculate_optical_limit(chi0_factory, extend_head=True):
    """Use the update_chi0 method to calculate chi0 for q=0."""
    # Create Chi0Data object
    chi0 = chi0_factory.create_chi0([0., 0., 0.], extend_head=extend_head)

    # Prepare update_chi0
    spins = range(chi0_factory.calc.wfs.nspins)
    m1 = chi0_factory.nocc1
    m2 = chi0_factory.nbands
    chi0_factory.plasmafreq_vv = np.zeros((3, 3), complex)

    chi0 = chi0_factory.update_chi0(chi0, m1, m2, spins)

    # Distribute over frequencies
    chi0.chi0_wGG = chi0.blockdist.distribute_frequencies(chi0.chi0_wGG)

    return chi0


def chi0_data_is_equal(chi0_f, chi0_t):
    """Compare Chi0Data objects without and with extend_head=True."""
    err_msg = None
    if not np.allclose(chi0_f.chi0_wGG, chi0_t.chi0_wGG):
        err_msg = 'body mismatch'
    elif not np.allclose(chi0_f.chi0_wxvG, chi0_t.chi0_wxvG):
        err_msg = 'wings mismatch'
    elif not np.allclose(chi0_f.chi0_wvv, chi0_t.chi0_wvv):
        err_msg = 'head mismatch'

    is_equal = err_msg is None

    return is_equal, err_msg

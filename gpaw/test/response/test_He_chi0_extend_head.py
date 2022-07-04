"""Test self-consistency of the extend_head flag for a semi-conductor."""

# General modules
import pytest
import numpy as np
from pathlib import Path

# Script modules
from ase import Atoms
from ase.dft.kpoints import monkhorst_pack

from gpaw import GPAW, PW
from gpaw.mpi import world
from gpaw.response.chi0 import Chi0


# ---------- Chi0 parametrization ---------- #


def generate_He_chi0_params():
    """Check the following options for a semi-conductor:
    * threshold
    * hilbert
    * timeordered
    * nbands
    * eta=0.
    * real_space_derivatives
    """
    chi0kwargs = dict(
        frequencies=np.linspace(0., 30., 11),
        eta=0.05,
        hilbert=False,
        timeordered=False,
        threshold=1,
        real_space_derivatives=False)
    # Check different chi0 parameter combinations
    chi0_params = [chi0kwargs]
    ck1 = chi0kwargs.copy()  # Check k.p threshold
    ck1['threshold'] = 0.5
    chi0_params.append(ck1)
    ck2 = chi0kwargs.copy()  # Check hilbert transform
    ck2['hilbert'] = True
    ck2['frequencies'] = None
    chi0_params.append(ck2)
    ck3 = chi0kwargs.copy()  # Check timeordering
    ck3['timeordered'] = True
    chi0_params.append(ck3)
    ck4 = chi0kwargs.copy()  # Check nbands
    ck4['nbands'] = None
    chi0_params.append(ck4)
    ck5 = chi0kwargs.copy()  # Check eta=0.
    ck5['frequencies'] = 1.j * ck5['frequencies']
    ck5['eta'] = 0.
    chi0_params.append(ck5)
    ck6 = chi0kwargs.copy()  # Check real space derivs.
    ck6['real_space_derivatives'] = True
    chi0_params.append(ck6)

    return chi0_params


@pytest.fixture(scope='module', params=generate_He_chi0_params())
def chi0kwargs(request, He_gs):
    # Fill in nbands parameter, if not already specified
    my_chi0kwargs = request.param
    if 'nbands' not in my_chi0kwargs.keys():
        _, nbands = He_gs
        my_chi0kwargs['nbands'] = nbands

    return my_chi0kwargs


# ---------- Actual tests ---------- #


@pytest.mark.response
def test_he_chi0_extend_head(in_tmp_dir, He_gs, chi0kwargs):
    # ---------- Inputs ---------- #
    gpw, nbands = He_gs

    ecut = 50

    if world.size > 1:
        nblocks = 2
    else:
        nblocks = 1

    # ---------- Script ---------- #

    chi0fac = Chi0(gpw, ecut=ecut,
                   nblocks=nblocks,
                   **chi0kwargs)

    chi0_f = calculate_optical_limit(chi0fac, extend_head=False)
    chi0_t = calculate_optical_limit(chi0fac, extend_head=True)

    is_equal, err_msg = chi0_data_is_equal(chi0_f, chi0_t)
    assert is_equal, err_msg


# ---------- System ground states ---------- #


@pytest.fixture(scope='module')
def He_gs(module_tmp_path):
    # ---------- Inputs ---------- #

    a = 3.0
    xc = 'LDA'
    kpts = 2
    nbands = 1 + 1 + 3  # 1s + 2s + 2p empty shell bands
    ebands = 1  # Include also 3s bands for numerical consistency
    pw = 250
    conv = {'bands': nbands}
    gpw = 'He'

    # ---------- Script ---------- #

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
    calc.write(gpw, 'all')

    return tuple((Path(gpw).resolve(), nbands))


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
    chi0.chi0_wGG = chi0.distribute_frequencies()

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

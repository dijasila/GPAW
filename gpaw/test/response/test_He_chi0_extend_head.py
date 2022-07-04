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


@pytest.mark.response
def test_he_chi0_extend_head(in_tmp_dir, He_gs):
    # ---------- Inputs ---------- #
    gpw, nbands = He_gs

    ecut = 50

    rparams = dict(
        frequencies=np.linspace(0., 30., 11),
        eta=0.05,
        nbands=nbands,
        hilbert=False,
        timeordered=False,
        threshold=1,
        real_space_derivatives=False)
    # Check different response parameter settings
    rp_settings = [rparams]
    rp1 = rparams.copy()  # Check k.p threshold
    rp1['threshold'] = 0.5
    rp_settings.append(rp1)
    rp2 = rparams.copy()  # Check hilbert transform
    rp2['hilbert'] = True
    rp2['frequencies'] = None
    rp_settings.append(rp2)
    rp3 = rparams.copy()  # Check timeordering
    rp3['timeordered'] = True
    rp_settings.append(rp3)
    rp4 = rparams.copy()
    rp4['nbands'] = None
    rp_settings.append(rp4)
    rp5 = rparams.copy()  # Check eta=0.
    rp5['frequencies'] = 1.j * rp5['frequencies']
    rp5['eta'] = 0.
    rp_settings.append(rp5)
    rp6 = rparams.copy()  # Check real space derivs.
    rp6['real_space_derivatives'] = True
    rp_settings.append(rp6)

    if world.size > 1:
        nblocks = 2
    else:
        nblocks = 1

    # ---------- Script ---------- #

    for kwargs in rp_settings:
        chi0fac = Chi0(gpw, ecut=ecut,
                       nblocks=nblocks,
                       **kwargs)

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

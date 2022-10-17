"""Test self-consistency of the extend_head flag for a semi-conductor."""

# General modules
import pytest
import numpy as np
from pathlib import Path

# Script modules
from ase import Atoms
from ase.build import bulk
from ase.lattice.hexagonal import Graphene

from gpaw import GPAW, PW
from gpaw.mpi import world
from gpaw.response.chi0 import Chi0


# ---------- Chi0 parametrization ---------- #


def generate_semic_chi0_params():
    """Check the following options for a semi-conductor:
    * threshold
    * hilbert
    * timeordered
    * nbands
    * eta=0.
    * real_space_derivatives
    """
    # Define default parameters
    chi0kwargs = dict(
        frequencies=np.linspace(0., 30., 11),
        eta=0.05,
        hilbert=False,
        timeordered=False,
        threshold=1,
        real_space_derivatives=False)
    chi0_params = [chi0kwargs]

    # Check different chi0 parameter combinations
    ck1 = chi0kwargs.copy()  # Check k.p threshold
    ck1['threshold'] = 0.5
    chi0_params.append(ck1)

    ck2 = chi0kwargs.copy()  # Check hilbert transform
    ck2['hilbert'] = True
    ck2['frequencies'] = None
    chi0_params.append(ck2)

    ck3 = chi0kwargs.copy()  # Check time ordering wo. hilbert
    ck3['timeordered'] = True
    chi0_params.append(ck3)
    ck4 = ck2.copy()  # Check time ordering w. hilbert
    ck4['timeordered'] = True
    chi0_params.append(ck4)

    ck5 = chi0kwargs.copy()  # Check nbands
    ck5['nbands'] = None
    chi0_params.append(ck5)

    ck6 = chi0kwargs.copy()  # Check eta=0.
    ck6['frequencies'] = 1.j * ck6['frequencies'][1:]
    ck6['eta'] = 0.
    chi0_params.append(ck6)

    ck7 = chi0kwargs.copy()  # Check real space derivs.
    ck7['real_space_derivatives'] = True
    chi0_params.append(ck7)

    return chi0_params


def generate_metal_chi0_params():
    """In addition to the semi-conductor parameters, test also:
    * integrationmode
    * intraband (test all other settings with and without intraband)
    """
    # Get semi-conductor defaults
    chi0_params = generate_semic_chi0_params()
    chi0kwargs = chi0_params[0]

    ck7 = chi0kwargs.copy()  # Check tetrahedron integration
    ck7['integrationmode'] = 'tetrahedron integration'
    ck7['hilbert'] = True
    ck7['frequencies'] = None
    chi0_params.append(ck7)

    # Run all test settings without intraband
    nointra_chi0_params = []
    for ck in chi0_params:
        ck['intraband'] = True  # This is the default, but be specific
        ck['rate'] = 0.1  # We need a finite scattering rate, not to blow up
        nointra_ck = ck.copy()
        nointra_ck['intraband'] = False
        nointra_ck['rate'] = 0.0
        nointra_chi0_params.append(nointra_ck)
    chi0_params += nointra_chi0_params

    return chi0_params


def generate_magnetic_metal_chi0_params():
    """In addition to the magnetic parametrization, test also:
    * spin
    """
    # Get metallic defaults
    chi0_params = generate_metal_chi0_params()
    chi0kwargs = chi0_params[0]

    cks0 = chi0kwargs.copy()  # Check spin up
    cks0['spins'] = [0]
    cks0['intraband'] = False
    chi0_params.append(cks0)

    cks1 = chi0kwargs.copy()  # Check spin down
    cks1['spins'] = [1]
    cks1['intraband'] = False
    chi0_params.append(cks1)

    return chi0_params


def generate_graphene_chi0_params():
    """Try to mimic the exact parameters of the qeh test."""
    # Get metallic defaults
    chi0_params = generate_metal_chi0_params()

    chi0kwargs = dict(rate=0.001,
                      hilbert=True,
                      frequencies={'type': 'nonlinear',
                                   'omegamax': 10,
                                   'domega0': 0.2,
                                   'omega2': 0.6})

    chi0_params.append(chi0kwargs)

    return chi0_params


@pytest.fixture(scope='module', params=generate_semic_chi0_params())
def He_chi0kwargs(request, He_gs):
    chi0kwargs = request.param
    assure_nbands(chi0kwargs, He_gs)

    return chi0kwargs


@pytest.fixture(scope='module', params=generate_metal_chi0_params())
def Li_chi0kwargs(request, Li_gs):
    chi0kwargs = request.param
    assure_nbands(chi0kwargs, Li_gs)

    return chi0kwargs


@pytest.fixture(scope='module', params=generate_magnetic_metal_chi0_params())
def Ni_chi0kwargs(request, Ni_gs):
    chi0kwargs = request.param
    assure_nbands(chi0kwargs, Ni_gs)

    return chi0kwargs


@pytest.fixture(scope='module', params=generate_graphene_chi0_params())
def graphene_chi0kwargs(request, graphene_gs):
    chi0kwargs = request.param
    assure_nbands(chi0kwargs, graphene_gs)

    return chi0kwargs


def assure_nbands(chi0kwargs, my_gs):
    # Fill in nbands parameter, if not already specified
    if 'nbands' not in chi0kwargs:
        _, nbands = my_gs
        chi0kwargs['nbands'] = nbands


# ---------- Actual tests ---------- #


@pytest.mark.response
def test_he_chi0_extend_head(in_tmp_dir, He_gs, He_chi0kwargs, request):
    mark_hilbert_xfail(He_chi0kwargs, request)
    chi0_extend_head_test(He_gs, He_chi0kwargs)


@pytest.mark.response
def test_li_chi0_extend_head(in_tmp_dir, Li_gs, Li_chi0kwargs):
    chi0_extend_head_test(Li_gs, Li_chi0kwargs)


@pytest.mark.response
def test_ni_chi0_extend_head(in_tmp_dir, Ni_gs, Ni_chi0kwargs):
    chi0_extend_head_test(Ni_gs, Ni_chi0kwargs)


@pytest.mark.response
def test_graphene_chi0_extend_head(in_tmp_dir, graphene_gs,
                                   graphene_chi0kwargs, request):
    mark_hilbert_xfail(graphene_chi0kwargs, request,
                       custom_frequencies=True)
    chi0_extend_head_test(graphene_gs, graphene_chi0kwargs)


def mark_hilbert_xfail(chi0kwargs, request, custom_frequencies=False):
    if chi0kwargs['hilbert']:
        if not custom_frequencies \
           or isinstance(chi0kwargs['frequencies'], dict):
            request.node.add_marker(pytest.mark.xfail)


def chi0_extend_head_test(my_gs, chi0kwargs):
    # ---------- Inputs ---------- #
    gpw, nbands = my_gs

    ecut = 50

    if world.size > 1:
        nblocks = 2
    else:
        nblocks = 1

    # ---------- Script ---------- #

    if 'spins' in chi0kwargs:
        spins = chi0kwargs.pop('spins')
    else:
        spins = 'all'

    chi0fac = Chi0(gpw, ecut=ecut,
                   nblocks=nblocks,
                   **chi0kwargs)

    chi0_f = calculate_optical_limit(chi0fac, spins=spins, extend_head=False)
    chi0_t = calculate_optical_limit(chi0fac, spins=spins, extend_head=True)

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
    convergence = {'bands': nbands}
    gpw = Path('He.gpw').resolve()

    # ---------- Script ---------- #

    atoms = Atoms('He', cell=[a, a, a], pbc=True)
    calculate_gs(atoms, gpw, pw, kpts, nbands, ebands,
                 xc=xc, convergence=convergence)

    return gpw, nbands


@pytest.fixture(scope='module')
def Li_gs(module_tmp_path):
    # ---------- Inputs ---------- #

    a = 3.49
    xc = 'LDA'
    kpts = 3
    nbands = 1 + 3 + 1  # 2s + 2p + 3s empty shell bands
    ebands = 3  # Include also 3p bands for numerical consistency
    pw = 250
    convergence = {'bands': nbands}
    gpw = Path('Li.gpw').resolve()

    # ---------- Script ---------- #

    atoms = bulk('Li', 'bcc', a=a)
    calculate_gs(atoms, gpw, pw, kpts, nbands, ebands,
                 xc=xc, convergence=convergence)

    return gpw, nbands


@pytest.fixture(scope='module')
def Ni_gs(module_tmp_path):
    # ---------- Inputs ---------- #

    a = 3.524
    xc = 'LDA'
    kpts = 5
    setups = {'Ni': '10'}
    nbands = 1 + 5  # 4s + 3d valence bands
    ebands = 3 + 1  # Include also 4p and 5s for numerical consistency
    pw = 250
    convergence = {'bands': nbands}
    gpw = Path('Ni.gpw').resolve()

    # ---------- Script ---------- #

    atoms = bulk('Ni', 'fcc', a=a)
    atoms.set_initial_magnetic_moments([0.6])
    calculate_gs(atoms, gpw, pw, kpts, nbands, ebands,
                 setups=setups, xc=xc, convergence=convergence)

    return gpw, nbands


@pytest.fixture(scope='module')
def graphene_gs(module_tmp_path):
    # ---------- Inputs ---------- #

    a = 2.45
    pbc = (1, 1, 0)
    vacuum = 4.0
    xc = 'LDA'
    kpts = 6
    nbands = 2 * (1 + 3)  # 2s + 2p valence bands
    ebands = 2 * (1 + 3)  # Include also 3s and 3p for numerical consistency
    pw = 250
    convergence = {'bands': nbands}
    gpw = Path('graphene.gpw').resolve()

    # ---------- Script ---------- #

    atoms = Graphene(symbol='C',
                     latticeconstant={'a': a, 'c': 1.0},
                     size=(1, 1, 1))
    atoms.pbc = pbc
    atoms.center(axis=2, vacuum=vacuum)

    calculate_gs(atoms, gpw, pw, kpts, nbands, ebands,
                 xc=xc, convergence=convergence)

    return gpw, nbands


# ---------- Script functionality ---------- #


def calculate_gs(atoms, gpw, pw, kpts, nbands, ebands,
                 **kwargs):
    calc = GPAW(mode=PW(pw),
                kpts={'size': tuple([kpts * periodic + (1 - periodic)
                                     for periodic in atoms.pbc]),
                      'gamma': True},
                nbands=nbands + ebands,
                symmetry={'point_group': True},
                parallel={'domain': 1},
                **kwargs)

    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write(gpw, 'all')


def calculate_optical_limit(chi0_factory, spins='all', extend_head=True):
    """Use the update_chi0 method to calculate chi0 for q=0."""
    # Create Chi0Data object
    chi0 = chi0_factory.create_chi0([0., 0., 0.], extend_head=extend_head)

    # Prepare update_chi0
    if spins == 'all':
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

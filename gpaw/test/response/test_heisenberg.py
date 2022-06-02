"""Test the Heisenberg model based methodology of the response code."""

# General modules
import numpy as np

# Script modules
from gpaw.response.heisenberg import calculate_single_site_magnon_energies,\
    calculate_FM_magnon_energies


# ---------- Main test ---------- #


def test_heisenberg():
    magnon_dispersion_tests()


# ---------- Actual tests ---------- #


def magnon_dispersion_tests():
    single_site_magnons_test()
    single_site_magnons_consistency_test()
    FM_random_magnons_test()


def single_site_magnons_test():
    """Check the single site magnon dispersion functionality."""
    # ---------- Inputs ---------- #

    # Magnetic moment
    mm = 1.
    # q-point grid
    nq = 11
    q_qc = get_randomized_qpoints(nq)

    # Random J_q, with J=0 at q=0
    J_q = np.random.rand(q_qc.shape[0])
    J_q[list(q_qc[:, 2]).index(0.)] = 0.

    # Cosine J_qD with different spin wave stiffnesses D
    D_D = np.linspace(400., 800., 5)
    J_qD = D_D[np.newaxis, :] * np.cos(q_qc[:, 2])[:, np.newaxis]

    # ---------- Script ---------- #

    # Calculate magnon energies
    E_q = calculate_single_site_magnon_energies(J_q, q_qc, mm)
    E_qD = calculate_single_site_magnon_energies(J_qD, q_qc, mm)

    # Check dimensions of arrays
    assert E_q.shape == (q_qc.shape[0],)
    assert E_qD.shape == J_qD.shape

    # Check versus formulas
    assert np.allclose(E_q, -2. / mm * J_q)  # Remember: J(0) = 0
    assert np.allclose(E_qD, 2. / mm * D_D[np.newaxis, :]
                       * (1. - np.cos(q_qc[:, 2]))[:, np.newaxis])


def single_site_magnons_consistency_test():
    """Check that the generalized magnon dispersion calculation is consistent
    for a single site system with the simple analytical formula valid in that
    case."""
    # ---------- Inputs ---------- #

    # Magnetic moment
    mm = 1.
    # q-point grid
    nq = 11
    q_qc = get_randomized_qpoints(nq)

    # Random isotropic exchange constants
    J_q = np.random.rand(q_qc.shape[0])

    # ---------- Script ---------- #

    # Calculate assuming a single site
    E_q = calculate_single_site_magnon_energies(J_q, q_qc, mm)

    # Calcualte using generalized functionality
    E_qn = calculate_FM_magnon_energies(J_q[:, np.newaxis, np.newaxis],
                                        q_qc, np.array([mm]))

    # Test self-consistency
    assert E_qn.shape[0] == len(E_q)
    assert E_qn.shape[1] == 1
    assert np.allclose(E_qn[:, 0], E_q, atol=1e-8)


def FM_random_magnons_test():
    """Check that the functionality to calculate the magnon dispersion of a
    ferromagnetic system with multiple sites works for a randomized system with
    three sites."""
    # ---------- Inputs ---------- #

    # Magnetic moments
    nsites = 3
    mm_a = 5. * np.random.rand(nsites)
    # q-point grid
    nq = 11
    q_qc = get_randomized_qpoints(nq)

    # Random isotropic exchange constants
    J_qab = 1.j * np.random.rand(q_qc.shape[0], nsites, nsites)
    J_qab += np.random.rand(q_qc.shape[0], nsites, nsites)
    # Take the Hermitian part of random tensor
    J_qab = (J_qab + np.transpose(np.conjugate(J_qab), (0, 2, 1))) / 2.
    # The q=0 component should furthermore be real
    J_qab[list(q_qc[:, 2]).index(0.)].imag = 0.

    # ---------- Script ---------- #

    # Calculate magnon energies
    E_qn = calculate_FM_magnon_energies(J_qab, q_qc, mm_a)

    # Calculate the magnon energies manually
    mm_inv_ab = 2. / np.sqrt(np.outer(mm_a, mm_a))
    J0_ab = np.diag(np.sum(J_qab[list(q_qc[:, 2]).index(0.)], axis=1))
    H_qab = mm_inv_ab[np.newaxis, ...] * (J0_ab[np.newaxis, ...] - J_qab)
    test_E_qn, _ = np.linalg.eig(H_qab)

    assert E_qn.shape == (q_qc.shape[0], nsites)
    assert np.allclose(test_E_qn.imag, 0.)
    assert np.allclose(E_qn, test_E_qn.real)


# ---------- Test functionality ---------- #


def get_randomized_qpoints(nq):
    """Make a simple, but shuffled, q-point array."""
    q_qc = np.zeros((nq, 3), dtype=np.float)
    q_qc[:, 2] = np.linspace(0., np.pi, nq)
    np.random.shuffle(q_qc[:, 2])

    return q_qc

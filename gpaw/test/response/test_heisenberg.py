"""Test the Heisenberg model based methodology of the response code."""

# General modules
import numpy as np

# Script modules
from gpaw.response.heisenberg import compute_magnon_energy_simple,\
    compute_magnon_energy_FM


# ---------- Main test functionality ---------- #


def test_heisenberg():
    magnon_dispersion_tests()


def magnon_dispersion_tests():
    single_site_magnons_test()
    multiple_sites_magnons_test()


def single_site_magnons_test():
    """Check that the generalized magnon dispersion calculation is consistent
    for a single site system with the simple analytical formula valid in that
    case."""
    # ---------- Inputs ---------- #

    # Magnetic moment
    mm = 1.
    # q-point grid
    q_qc = np.zeros((11, 3), dtype=np.float)
    q_qc[:, 2] = np.linspace(0., 1., 11)

    # ---------- Script ---------- #

    # Generate random isotropic exchange constants
    J_q = np.random.rand(q_qc.shape[0])

    # Calculate assuming a single site
    E_q = compute_magnon_energy_simple(J_q, q_qc, mm)

    # Calcualte using generalized functionality
    E_nq = compute_magnon_energy_FM(J_q[np.newaxis, np.newaxis, :], q_qc, mm)

    # Test self-consistency
    assert E_nq.shape[0] == 1
    assert E_nq.shape[1] == len(E_q)
    assert np.allclose(E_nq[0, :], E_q, atol=1e-8)


def multiple_sites_magnons_test():
    pass

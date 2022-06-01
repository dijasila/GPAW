"""Methods to calculate material properties in the Heisenberg model.
Primarily based on the magnon dispersion relations."""

import numpy as np


def compute_magnon_energy_simple(J_q, q_qc, mm):
    """Compute magnon energy with single atom in magnetic unit cell"""
    # Check if J_mnq was passed instead of J_q
    if len(J_q.shape) == 3:
        J_q = J_q[0, 0, :]

    # Find index of Gamma point (q=0), i.e. row with all zeros
    zeroIndex = np.argwhere(np.all(q_qc == 0, axis=1))
    zeroIndex = int(zeroIndex[0])

    # Compute energies
    J0 = J_q[zeroIndex]
    E_q = 2 / mm * (J0 - J_q)

    # Imaginary part should be zero
    assert np.all(np.isclose(np.imag(E_q), 0))
    E_q = np.real(E_q)

    return E_q


def compute_magnon_energy_FM(J_mnq, q_qc, mm, return_H=False):
    """Compute magnon energy for ferromagnet with multiple sublattices
    Gamma point (q=0) must be included in dataset.
    """

    import numpy as np
    from numpy.linalg import eigvalsh

    N_sites, N_sites, Nq = J_mnq.shape

    # Reformat magnetisation
    if type(mm) in {float, int}:
        mm = np.ones(N_sites) * mm

    # Find rows where all components of q_c are zero (Gamma point)
    zeroIndex = np.argwhere([np.all(np.isclose(q_qc[q, :], 0))
                             for q in range(Nq)])
    try:
        zeroIndex = int(zeroIndex[0])
    except IndexError:
        zeroIndex = 0
    J0_mn = J_mnq[:, :, zeroIndex]   # Get J_mn(0)

    # Set up Hamiltonian matrix
    mm_inv_mn = np.diag(1 / mm)  # 1/M_mu * delta_mu,nu
    # 2/M_mu * sum_nu' J_mu,nu'(0) delta_mu,nu
    firstTerm_mn = 2 * mm_inv_mn * np.sum(J0_mn, axis=-1, keepdims=True)
    firstTerm_mnq = np.tile(firstTerm_mn[..., np.newaxis], [1, 1, Nq])
    mmProd_mn = np.outer(mm, mm)  # M_mu * M_nu
    mmProd_mnq = np.tile(mmProd_mn[:, :, np.newaxis],
                         [1, 1, Nq])  # Match dimension of J_mnq
    J_nmq = np.transpose(J_mnq, axes=[1, 0, 2])  # J^\nu\mu
    # -2J^nu,mu(q) / sqrt(M_mu * M_nu)
    secondTerm_mnq = -2 * J_nmq / np.sqrt(mmProd_mnq)
    H_mnq = firstTerm_mnq + secondTerm_mnq

    # Diagonalise Hamiltonian for all q values
    E_mq = np.zeros([N_sites, Nq])
    for q in range(Nq):
        H_mn = H_mnq[:, :, q]
        assert np.all(np.isclose(np.conj(H_mn.T), H_mn, atol=np.inf,
                                 rtol=1e-02))  # Check if Hermitian
        # 'eigvalsh' takes the lower triangluar part, then assumes
        #   Hermiticity to fill the rest of the matrix
        # This is faster than 'eigvals' and guarantees that the computed
        #   eigenvalues are real.
        E_mq[:, q] = eigvalsh(H_mn, UPLO='L')

    if return_H:
        return E_mq, H_mnq
    else:
        return E_mq

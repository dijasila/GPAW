"""Methods to calculate material properties in the Heisenberg model.
Primarily based on the magnon dispersion relations."""

import numpy as np


def calculate_single_site_magnon_energies(J_qx, q_qc, mm):
    """Compute the magnon energies from the isotropic exchange constants of a
    system with a single magnetic site in the unit cell, as a function of the
    wave vector q:

    ħω(q) = g μ_B / M [J(0) - J(q)]

    Parameters
    ----------
    J_qx : np.ndarray
        Isotropic exchange constants as a function of q.
        J_qx can have any number of additional dimensions x, which will be
        treated independently.
    q_qc : np.ndarray
        q-vectors in relative coordinates. Has to include q=0.
    mm : float
        Magnetic moment of the site in μ_B.

    Returns
    -------
    E_qx : np.ndarray
        Magnon energies as a function of q and x. Same shape as input J_qx.
    """
    assert J_qx.shape[0] == q_qc.shape[0]

    q0 = get_q0_index(q_qc)
    J0_x = J_qx[q0]

    # Compute energies
    E_qx = 2. / mm * (J0_x[np.newaxis, ...] - J_qx)

    # Imaginary part should be zero
    assert np.allclose(E_qx.imag, 0.)

    return E_qx.real


def calculate_FM_magnon_energies(J_qabx, q_qc, mm_ax, return_H=False):
    """Compute the magnon eigenmode energies from the isotropic exchange
    constants of a ferromagnetic system with an arbitrary number of magnetic
    sites in the unit cell, as a function of the wave vector q.

    The eigenmodes are calculated as the eigenvalues to the dynamic spin wave
    matrix:

    H^ab(q) = g μ_B / sqrt(M_a M_b) [Σ_c J^ac(0) δ_ab - J^ab(q)]

    Parameters
    ----------
    J_qabx : np.ndarray
        Isotropic exchange constants as a function of q and sublattice indices
        a and b. J_qabx can have any number of additional dimensions x, which
        will be treated independently.
    q_qc : np.ndarray
        q-vectors in relative coordinates. Has to include q=0.
    mm_ax : np.ndarray
        Magnetic moments of the sublattice sites in μ_B.
    return_H : bool
        Return also the dynamic spin wave matrix.

    Returns
    -------
    E_qnx : np.ndarray
        Magnon eigenmode energies as a function of q, mode index n and x.
    H_qabx : np.ndarray (Optional)
        Dynamic spin wave matrix. Has the same shape as the input J_qabx
    """
    assert len(J_qabx.shape) >= 3
    assert J_qabx.shape[1] == J_qabx.shape[2]
    assert J_qabx.shape[1] == mm_ax.shape[0]
    assert J_qabx.shape[0] == q_qc.shape[0]
    assert J_qabx.shape[3:] == mm_ax.shape[1:]

    # Get J^ab(0)
    q0 = get_q0_index(q_qc)
    J0_abx = J_qabx[q0]

    H_qabx = generate_FM_dynamic_spin_wave_matrix(J_qabx, J0_abx, mm_ax)

    # Move magnetic site axes in order to prepare for np.linalg.eig
    H_qbxa = np.moveaxis(H_qabx, 1, -1)
    H_qxab = np.moveaxis(H_qbxa, 1, -1)

    # Diagonalize the matrix
    E_qxn, _ = np.linalg.eig(H_qxab)

    # Transpose to output format
    E_qnx = np.moveaxis(E_qxn, -1, 1)

    # Eigenvalues should be real, otherwise input J_qabx has been invalid
    assert np.allclose(E_qnx.imag, 0.)
    E_qnx = E_qnx.real
    
    """
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
        E_mq[:, q] = np.linalg.eigvalsh(H_mn, UPLO='L')
    """

    if return_H:
        return E_qnx, H_qabx
    else:
        return E_qnx


def generate_FM_dynamic_spin_wave_matrix(J_qabx, J0_acx, mm_ax):
    """Generate the dynamic spin wave matrix from the isotropic exchange
    constants of a ferromagnet:

    H^ab(q) = g μ_B / sqrt(M_a M_b) [Σ_c J^ac(0) δ_ab - J^ab(q)]
    """
    na = mm_ax.shape[0]

    # Set up magnetic moment prefactor as outer product
    mm_inv_abx = 2. / np.sqrt(mm_ax[:, np.newaxis, ...]
                              * mm_ax[np.newaxis, ...])

    # Calculate diagonal component Σ_c J^ac(0) δ_ab
    J0_ax = np.sum(J0_acx, axis=1)
    diagonal_mapping = np.zeros((na, na, na))
    np.fill_diagonal(diagonal_mapping, 1)
    J0_abx = np.tensordot(diagonal_mapping, J0_ax, axes=(2, 0))

    # Calculate the dynamic spin wave matrix
    H_qabx = mm_inv_abx[np.newaxis, ...] * (J0_abx[np.newaxis, ...] -
                                            J_qabx)
    """
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
    """

    return H_qabx


def get_q0_index(q_qc):
    """Find index corresponding q=0 in q-vector array."""
    q0_indices = np.argwhere(np.all(q_qc == 0, axis=1))

    if len(q0_indices) >= 1:
        return int(q0_indices[0])
    else:
        raise ValueError('q_qc has to include q=0, i.e. q_c = [0., 0., 0.]')

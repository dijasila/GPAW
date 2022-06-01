"""Methods to calculate material properties in the Heisenberg model.
Primarily based on the magnon dispersion relations."""

import numpy as np


def calculate_single_site_magnon_energies(J_qx, q_qc, mm):
    """Compute the magnon energies from the isotropic exchange constants of a
    system with a single magnetic site in the unit cell, as a function of the
    wave vector q:

    hw(q) = g mu_B / M [J(0) - J(q)]

    Parameters
    ----------
    J_qx : np.ndarray
        Isotropic exchange constants as a function of q.
        J_qx can have any number of additional dimensions x, which are treated
        independently.
    q_qc : np.ndarray
        q-vectors in relative coordinates. Has to include q=0.
    mm : float
        Magnetic moment of the site in Bohr magnetons.

    Returns
    -------
    E_qx : np.ndarray
        Magnon energies as a function of q and x. Same shape as input J_qx.
    """
    assert J_qx.shape[0] == q_qc.shape[0]

    # Find index of Gamma point (q=0), i.e. row with all zeros
    zeroIndex = int(np.argwhere(np.all(q_qc == 0, axis=1))[0])

    # Compute energies
    J0_x = J_qx[zeroIndex]
    E_qx = 2. / mm * (J0_x[np.newaxis, ...] - J_qx)

    # Imaginary part should be zero
    assert np.allclose(E_qx.imag, 0.)

    return E_qx.real


def calculate_FM_magnon_energies(J_mnq, q_qc, mm, return_H=False):
    """Compute the magnon eigenmode energies from the isotropic exchange
    constants of a ferromagnetic system with an arbitrary number of magnetic
    sites in the unit cell, as a function of the wave vector q.

    The eigenmodes are calculated as the eigenvalues to the dynamic spin wave
    matrix:

    H^ab(q) = g mu_B
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

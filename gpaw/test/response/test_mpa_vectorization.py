import numpy as np
from gpaw.response.mpa_interpolation import fit_residue, RESolver
from .mpa_interpolation_from_fortran import mpa_R_fit as fit_residue_fortran, mpa_RE_solver


def test_residues():

    def Xeval(Omega_p, residues_p, omega_w):
        X_pw = residues_p[:, np.newaxis] * 2 * Omega_p[:, np.newaxis] / (omega_w[np.newaxis, :]**2 - Omega_p[:, np.newaxis]**2)
        return np.sum(X_pw, axis=0)

    nG = 2
    npols = 10
    Omega_GGp = np.empty((nG,nG,npols), dtype=np.complex128)
    residues_GGp = np.empty((nG,nG,npols), dtype=np.complex128)
    X_GGw = np.empty((nG,nG,2*npols), dtype=np.complex128)
    R_GGp = np.empty((nG,nG,npols), dtype=np.complex128)
    R_fortran_GGp = np.empty((nG,nG,npols), dtype=np.complex128)
    omega_w = np.linspace(0., 5., 2*npols) + 0.1j
    for g1 in range(nG):
        for g2 in range(nG):
            Omega_GGp[g1,g2] = np.random.randn(npols)*0.05 + 5.5 - 0.01j
            residues_GGp[g1,g2] = np.random.rand(npols)
            X_GGw[g1,g2] = Xeval(Omega_GGp[g1,g2], residues_GGp[g1,g2], omega_w)
            R_fortran_GGp[g1,g2] = fit_residue_fortran(npols, npols, omega_w, X_GGw[g1,g2], Omega_GGp[g1,g2])

    R_pGG = fit_residue(np.ones((nG,nG))*npols, omega_w, X_GGw.transpose(2,0,1), Omega_GGp.transpose(2,0,1))

    R_GGp = R_pGG.transpose(1,2,0)

    print('R_GGp',R_GGp)
    print('R_fortran_GGp',R_fortran_GGp)

    from matplotlib import pyplot as plt
    X_fit = Xeval(Omega_GGp[0,0], R_GGp[0,0],omega_w)
    X_fortran_fit = Xeval(Omega_GGp[0,0], R_fortran_GGp[0,0],omega_w)
    plt.plot(omega_w, X_GGw[0,0].real, 'k')
    plt.plot(omega_w, X_GGw[0,0].imag, 'gray')

    plt.plot(omega_w, X_fit.real,ls='--')
    plt.plot(omega_w, X_fit.imag,ls='--')
    plt.plot(omega_w, X_fortran_fit.real,ls=':')
    plt.plot(omega_w, X_fortran_fit.imag,ls=':')
    plt.show()

    assert np.allclose(R_GGp,R_fortran_GGp, atol=1e-6)


def test_poles():

    def Xeval(Omega_p, residues_p, omega_w):
        X_pw = residues_p[:, np.newaxis] * 2 * Omega_p[:, np.newaxis] / (omega_w[np.newaxis, :]**2 - Omega_p[:, np.newaxis]**2)
        return np.sum(X_pw, axis=0)

    nG = 2
    npols = 100
    Omega_GGp = np.empty((nG,nG,npols), dtype=np.complex128)
    residues_GGp = np.empty((nG,nG,npols), dtype=np.complex128)
    wmax = 2 

    npols_mpa = 6
    omega_p = np.linspace(0,wmax, npols_mpa)
    omega_w = np.concatenate((omega_p + 0.1j, omega_p +1.j))
    print(omega_w)

    X_GGw = np.empty((nG,nG,2*npols_mpa), dtype=np.complex128)
    E_GGp = np.empty((nG,nG,npols_mpa), dtype=np.complex128)
    R_GGp = np.empty((nG,nG,npols_mpa), dtype=np.complex128)
    E_fortran_GGp = np.empty((nG,nG,npols_mpa), dtype=np.complex128)
    R_fortran_GGp = np.empty((nG,nG,npols_mpa), dtype=np.complex128)

    for g1 in range(nG):
        for g2 in range(nG):
            Omega_GGp[g1,g2] = np.random.normal(1, 0.5, npols)  - 0.05j
            residues_GGp[g1,g2] = np.random.rand(npols)
            X_GGw[g1,g2] = Xeval(Omega_GGp[g1,g2], residues_GGp[g1,g2], omega_w)

            R_fortran_GGp[g1,g2], E_fortran_GGp[g1,g2], _, _ = mpa_RE_solver(npols_mpa, omega_w, X_GGw[g1,g2])

    E_pGG, R_pGG = RESolver(omega_w).solve(X_GGw.transpose(2,0,1))

    E_GGp = E_pGG.transpose(1,2,0)
    R_GGp = R_pGG.transpose(1,2,0)

    #print('R_GGp',R_GGp)
    #print('R_fortran_GGp',R_fortran_GGp)

    from matplotlib import pyplot as plt

    g1, g2 = 1, 1
    omega_plot = np.linspace(0,wmax, 500)
    X_num = Xeval(Omega_GGp[g1,g2], residues_GGp[g1,g2], omega_plot)
    X_new = Xeval(E_GGp[g1,g2], R_GGp[g1,g2], omega_plot)
    X_old = Xeval(E_fortran_GGp[g1,g2], R_fortran_GGp[g1,g2], omega_plot)

    plt.plot(omega_plot, X_num.real, 'r')
    plt.plot(omega_plot, X_num.imag, 'gray')

    plt.plot(omega_plot, X_new.real)
    plt.plot(omega_plot, X_new.imag)
    plt.plot(omega_plot, X_old.real,'k--')
    plt.plot(omega_plot, X_old.imag,'k--')
    plt.show()

    #assert np.allclose(R_GGp,R_fortran_GGp, atol=1e-6)

import numpy as np
#from gpaw.response.mpa_interpolation import mpa_R_fit, mpa_RE_solver
from gpaw.test.response.mpa_interpolation_from_fortran import *
from gpaw.response.mpa_interpolation import * 
from gpaw.response.mpa_sampling import mpa_frequency_sampling

def test_poles():
    npols = 1
    Omega_p = np.random.randn(npols)*0.05 + 5.5 - 0.01j
    residues_p = np.random.rand(npols)

    npol_fit = 2
    omega_w = mpa_frequency_sampling(npol_fit, [0, 0.8], varpi=1,
                                     eta0=0.1, eta_rest=0.1, parallel_lines=2)

    def Xeval(Omega_p, residues_p, omega_w):
        X_pw = residues_p[:, np.newaxis] * 2 * Omega_p[:, np.newaxis] / (omega_w[np.newaxis, :]**2 - Omega_p[:, np.newaxis]**2)
        return np.sum(X_pw, axis=0)

    X_w = Xeval(Omega_p, residues_p, omega_w)
    
    w_x = np.linspace(0, 1, 1000)# - 0.01j
    X_x = Xeval(Omega_p, residues_p, w_x)
    from matplotlib import pyplot as plt
    plt.plot(w_x, X_x.real, 'r')
    plt.plot(w_x, X_x.imag, 'g')

    plt.plot(omega_w[:npol_fit].real, X_w[:npol_fit].real, 'x')
    plt.plot(omega_w[:npol_fit].real, X_w[:npol_fit].imag, 'x')
    #plt.plot(omega_w, X_x.imag, 'g')

    R_p, E_p, MPres, PPcond_rate = mpa_RE_solver(npol_fit, omega_w, X_w)
    #print(E_p, 'E_p')
    #print(R_p, 'R_p')
    fit_x = Xeval(E_p, R_p, w_x)
    plt.plot(w_x, fit_x.real, 'r--')
    plt.plot(w_x, fit_x.imag, 'g--')
    
    E_pGG, R_pGG = RESolver(omega_w).solve(X_w.reshape((-1, 1, 1)))
    fit_x = Xeval(E_pGG[:,0,0], R_pGG[:,0,0], w_x)
    plt.plot(w_x, fit_x.real, 'k:')
    plt.plot(w_x, fit_x.imag, 'k:')
    #print("Pole positions", E_p, E_pGG[:,0,0])
    #print("Residuals", R_p, R_pGG[:,0,0])
    plt.show()

test_poles() 

def test_residue_fit_1pole():
    npols, npr, w, x, E = (1, 1, np.array([0.  +0.j, 0.  +1.j]), np.array([ 0.13034393-0.40439649j,  0.36642415-0.67018998j]), np.array([1.654376 -0.25302243j]))
    R = mpa_R_fit(npols, npr, w, x, E)
    Rnew = fit_residue(np.array([[npols]]), w, x.reshape((-1, 1, 1)), E.reshape((-1, 1, 1)))
    assert np.allclose(R, Rnew)

def test_residue_fit():
    npols, npr, w, x, E = (2, 2, np.array([0.  +0.j, 0.  +1.j, 2.33+0.j, 2.33+1.j]), np.array([ 0.13034393-0.40439649j,  0.36642415-0.67018998j,  0.77096523+0.85578656j,  -0.38002124-0.20843867j]), np.array([1.654376 -0.25302243j, 2.4306366-0.15733093j]))
    R = mpa_R_fit(npols, npr, w, x, E)
    Rnew = fit_residue(np.array([[npols]]), w, x.reshape((-1, 1, 1)), E.reshape((-1, 1, 1)))
    #print(R,'R')
    #print(Rnew[:,0,0],'Rew')
    assert np.allclose(R, Rnew[:,0,0])


def test_mpa():
    NG = 20 # XXX Back to 20
    nw = 4
    X_wGG = 2*(np.random.rand(nw, NG, NG) - 0.5) + 1j * (np.random.rand(nw, NG, NG) - 0.5)*2
    omega_w = np.array([0, 1j, 2.33, 2.33+1j])
    #omega_p = np.linspace(0,1,nw // 2)
    #omega_w = np.concatenate((omega_p, omega_p +1.j))

    E_pGG, R_pGG = RESolver(omega_w).solve(X_wGG)

    for i in range(8):
        for j in range(8):
            R_p, E_p, MPres, PPcond_rate = mpa_RE_solver(nw // 2, omega_w, X_wGG[:, i, j])
            print('old',E_p[0]) 
            print('new', E_pGG[0, i,j])
            assert np.allclose(E_pGG[0, i,j], E_p[0])#, atol=1e-5)
            #assert np.allclose(R_pGG[:, i,j], R_p)#, atol=1e-5)


def test_ppa():
    NG = 120
    X_wGG = 2*(np.random.rand(2, NG, NG) - 0.5) + 1j * (np.random.rand(2, NG, NG) - 0.5)*2
    omega_w = np.array([0, 2j])
    from time import time
    start = time()
    E_GG, R_GG = RESolver(omega_w).solve(X_wGG)
    stop = time()
    fail = False
    for i in range(NG):
        for j in range(NG):
            R, E, MPres, PPcond_rate = mpa_RE_solver(1, omega_w, X_wGG[:, i, j])
            assert np.allclose(E, E_GG[i, j])
            if not np.allclose(R, R_GG[i, j]):
                print(R, R_GG[i,j], 'ratio:', R_GG[i,j] / R, 'mismatch')
                fail = True
    assert not fail
    vectorized_time = stop - start
    print('Vectorized', vectorized_time)
    start = time()
    for i in range(NG):
        for j in range(NG):
            R, E, MPres, PPcond_rate = mpa_RE_solver(1, omega_w, X_wGG[:, i, j])
        #assert np.allclose(E, E_G[i])
        #print('old', E, 'new', E_G[i])
    stop = time()
    serial_time = stop - start
    print('Not vectorized', serial_time)
    print('speedup', serial_time / vectorized_time)

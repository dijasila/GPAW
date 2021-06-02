import pytest
from ase.build import molecule, bulk

from gpaw import GPAW
from gpaw.fd_operators import Gradient


@pytest.fixture
def calc():
    atoms = bulk('Li', orthorhombic=True) #molecule('H2O', vacuum=2.0)
    atoms.rattle(stdev=0.15)
    calc = GPAW(mode='lcao', basis='dzp', txt='gpaw.txt', h=.2,
                kpts=[[0.1, 0.3, 0.4]])
    atoms.calc = calc
    def stopcalc():
        calc.scf.converged = True
    calc.attach(stopcalc, 1)
    atoms.get_potential_energy()
    return calc


def test_nabla_matrix(calc):
    wfs = calc.wfs
    gd = wfs.gd

    Mstart = wfs.ksl.Mstart
    Mstop = wfs.ksl.Mstop

    dThetadR_qvMM, dTdR_qvMM = wfs.manytci.O_qMM_T_qMM(
        gd.comm, Mstart, Mstop, False, derivative=True)

    # We want C^dagger · dTheta/dR · C
    dThetadRz_MM = dThetadR_qvMM[0, 2]

    kpt = wfs.kpt_u[0]
    C_nM = kpt.C_nM
    import numpy as np
    np.set_printoptions(precision=5, suppress=1, linewidth=120)

    v = 2
    nabla_nn = -(C_nM.conj() @ dThetadRz_MM.conj() @ C_nM.T)
    print('NABLA_NN')
    print(nabla_nn)
    print('biggest', np.abs(nabla_nn).max())

    mynbands = wfs.bd.mynbands
    bfs = wfs.basis_functions
    psit_nG = gd.zeros(mynbands, dtype=wfs.dtype)
    gradpsit_nG = gd.zeros(mynbands, dtype=wfs.dtype)
    bfs.lcao_to_grid(C_nM, psit_nG, q=kpt.q)

    k_c = wfs.kd.ibzk_kc[kpt.k]
    #the_planewave = gd.plane_wave(-k_c)
    #for psit_G in psit_nG:
    #    psit_G *= the_planewave


    grad = Gradient(gd, v, n=2, dtype=wfs.dtype)
    grad.apply(psit_nG, gradpsit_nG, phase_cd=kpt.phase_cd)

    #nabla_fd_nn = np.zeros((mynbands, mynbands), dtype=wfs.dtype)
    #for n1 in range(mynbands):
    #    for n2 in range(mynbands):
    #        nabla_fd_nn[n1, n2] = (psit_nG[n1]
    #                               * gradpsit_nG[n2].conj()).sum() * gd.dv

    nabla_fd_nn = gd.integrate(psit_nG, gradpsit_nG)

    print('NABLA_FD_NN')
    print(nabla_fd_nn)
    err = abs(nabla_fd_nn - nabla_nn)
    print('ERR')
    print(err)
    maxerr = np.abs(err).max()
    print('MAXERR', maxerr)
    assert maxerr < 2e-4

import pytest
import numpy as np
from ase.build import molecule, bulk

from gpaw import GPAW
from gpaw.fd_operators import Gradient

DIR = 2  # z direction

@pytest.fixture
def calc():
    atoms = bulk('Li', orthorhombic=True)
    atoms.rattle(stdev=0.15)
    calc = GPAW(mode='lcao', basis='dzp', txt='gpaw.txt', h=.2,
                kpts=[[0.1, 0.3, 0.4]])
    atoms.calc = calc
    def stopcalc():
        calc.scf.converged = True
    calc.attach(stopcalc, 1)
    atoms.get_potential_energy()
    return calc


def get_nabla_fd(gd, kpt, psit_nG):
    gradpsit_nG = gd.zeros(len(psit_nG), dtype=psit_nG.dtype)
    grad = Gradient(gd, DIR, n=2, dtype=psit_nG.dtype)
    grad.apply(psit_nG, gradpsit_nG, phase_cd=kpt.phase_cd)
    return gd.integrate(psit_nG, gradpsit_nG)


def test_nabla_matrix(calc):
    wfs = calc.wfs
    gd = wfs.gd

    Mstart = wfs.ksl.Mstart
    Mstop = wfs.ksl.Mstop

    dThetadR_qvMM, dTdR_qvMM = wfs.manytci.O_qMM_T_qMM(
        gd.comm, Mstart, Mstop, False, derivative=True)

    # We want C^dagger · dTheta/dR · C
    dThetadRz_MM = dThetadR_qvMM[0, DIR]

    kpt = wfs.kpt_u[0]
    C_nM = kpt.C_nM
    # import numpy as np
    # np.set_printoptions(precision=5, suppress=1, linewidth=120)

    nabla_nn = -(C_nM.conj() @ dThetadRz_MM.conj() @ C_nM.T)
    print('NABLA_NN')
    print(nabla_nn)
    print('biggest', np.abs(nabla_nn).max())

    mynbands = wfs.bd.mynbands
    bfs = wfs.basis_functions
    psit_nG = gd.zeros(mynbands, dtype=wfs.dtype)
    bfs.lcao_to_grid(C_nM, psit_nG, q=kpt.q)

    k_c = wfs.kd.ibzk_kc[kpt.k]

    nabla_fd_nn = get_nabla_fd(gd, kpt, psit_nG)

    print('NABLA_FD_NN')
    print(nabla_fd_nn)
    err = abs(nabla_fd_nn - nabla_nn)
    print('ERR')
    print(err)
    maxerr = np.abs(err).max()
    print('MAXERR', maxerr)
    assert maxerr < 2e-4

import pytest
import numpy as np
from ase.build import bulk

from gpaw import GPAW
from gpaw.fd_operators import Gradient


DIR = 2  # z direction


@pytest.fixture
def calc():
    atoms = bulk('Li', orthorhombic=True)
    atoms.rattle(stdev=0.15)
    calc = GPAW(mode='lcao', basis='dzp', txt='gpaw.txt', h=.2,
                # parallel=dict(sl_auto=True),
                kpts=[[0.1, 0.3, 0.4]])
    atoms.calc = calc

    def stopcalc():
        calc.scf.converged = True

    calc.attach(stopcalc, 1)
    atoms.get_potential_energy()
    return calc


def get_nabla_fd(gd, kpt, psit_nG):
    gradpsit_nG = gd.zeros(len(psit_nG), dtype=psit_nG.dtype)
    grad = Gradient(gd, DIR, n=4, dtype=psit_nG.dtype)
    grad.apply(psit_nG, gradpsit_nG, phase_cd=kpt.phase_cd)
    return gd.integrate(psit_nG, gradpsit_nG)


def get_nabla_lcao(wfs):
    gd = wfs.gd
    kpt = wfs.kpt_u[0]

    dThetadR_qvMM, dTdR_qvMM = wfs._get_overlap_derivatives()
    print(dThetadR_qvMM.shape)
    # If using BLACS initialize lower triangle.
    #if wfs.ksl.using_blacs:
    #    for i in range(dThetadR_qvMM.shape[-1]):
    #        dThetadR_qvMM[:, :, i, i:] = -dThetadR_qvMM[:, :, i:, i]

    nabla_skvnm = np.zeros((wfs.nspins, wfs.kd.nibzkpts, 3, wfs.bd.nbands,
        wfs.bd.nbands), dtype=wfs.dtype)

    for kpt in wfs.kpt_u:
        C_nM = kpt.C_nM

        for v in range(3):
            # We want C^dagger · dTheta/dR · C
            dThetadRv_MM = dThetadR_qvMM[kpt.q, v]

        # import numpy as np
        # np.set_printoptions(precision=5, suppress=1, linewidth=120)

        nabla_nn_lcao = np.ascontiguousarray((C_nM @ dThetadRz_MM @ C_nM.T.conj()).T)
        gd.comm.sum(nabla_nn_lcao)
    
        nabla_skvnm[kpt.s, kpt.k, v] = nabla_nn_lcao

    return wfs.kd.comm.sum(nabla_skvnm)


def test_nabla_matrix(calc):
    wfs = calc.wfs
    gd = wfs.gd
    kpt = wfs.kpt_u[0]

    mynbands = wfs.bd.mynbands
    bfs = wfs.basis_functions
    psit_nG = gd.zeros(mynbands, dtype=wfs.dtype)
    bfs.lcao_to_grid(kpt.C_nM, psit_nG, q=kpt.q)

    nabla_nn_lcao = get_nabla_lcao(wfs)

    def print0(*args, **kwargs):
        if wfs.world.rank == 0:
            print(*args, **kwargs)

    print0('NABLA_NN_LCAO')
    print0(nabla_nn_lcao)
    print0('biggest', np.abs(nabla_nn_lcao).max())

    nabla_fd_nn = get_nabla_fd(gd, kpt, psit_nG)

    print0('NABLA_FD_NN')
    print0(nabla_fd_nn)
    err = abs(nabla_fd_nn - nabla_nn_lcao)
    print0('ERR')
    print0(err)
    maxerr = np.abs(err).max()
    print0('MAXERR', maxerr)
    assert maxerr < 1e-4

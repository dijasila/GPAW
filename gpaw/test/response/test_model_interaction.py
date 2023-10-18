import numpy as np
import pytest
from gpaw import GPAW
from gpaw.response.modelinteraction import initialize_w_model
from gpaw.response.chi0 import Chi0
from gpaw.wannier90 import Wannier90
import os
from gpaw.mpi import world, serial_comm


@pytest.mark.parametrize('symm', [True, False])
def test_w90(in_tmp_dir, gpw_files, symm):

    if symm:
        gpwfile = gpw_files['gaas_pw']
    else:
        gpwfile = gpw_files['gaas_pw_nosym']

    calc = GPAW(gpwfile, communicator=serial_comm)
    seed = 'GaAs'

    if world.rank == 0:
        w90 = Wannier90(calc, orbitals_ai=[[], [0, 1, 2, 3]],
                        bands=range(4),
                        seed=seed)
        w90.write_input(num_iter=1000,
                        plot=True,
                        write_u_matrices=True)
        w90.write_wavefunctions()
        os.system('wannier90.x -pp ' + seed)
        w90.write_projections()
        w90.write_eigenvalues()
        w90.write_overlaps()
        os.system('wannier90.x ' + seed)

    world.barrier()

    omega = np.array([0])
    chi0calc = Chi0(gpwfile, frequencies=omega, hilbert=False, ecut=100,
                    txt='test.log', intraband=False)
    Wm = initialize_w_model(chi0calc)
    Wwann = Wm.calc_in_Wannier(chi0calc, Uwan=seed, bandrange=[0, 4])

    assert Wwann[0, 0, 0, 0, 0] == pytest.approx(2.537, abs=0.003)
    assert Wwann[0, 1, 1, 1, 1] == pytest.approx(1.855, abs=0.003)
    assert Wwann[0, 2, 2, 2, 2] == pytest.approx(1.855, abs=0.003)
    assert Wwann[0, 3, 3, 3, 3] == pytest.approx(1.855, abs=0.003)
    assert Wwann[0, 3, 3, 0, 0] == pytest.approx(0.972, abs=0.003)
    assert Wwann[0, 3, 0, 3, 0].real == pytest.approx(1.808, abs=0.003)
    assert np.abs(Wwann[0, 3, 0, 3, 0].imag) < 0.005

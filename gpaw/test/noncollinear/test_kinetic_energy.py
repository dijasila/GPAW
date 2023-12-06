import numpy as np
import pytest

from gpaw.mpi import world
from gpaw.new.ase_interface import GPAW
from gpaw.utilities import pack


@pytest.mark.soc
@pytest.mark.skipif(world.size > 1, reason='No parallelisation implemented')
def test_kinetic_energy(gpw_files):
    # Test that we calculate the kinetic energy correctly in noncollinear mode.

    # Thallium atom in a simulation box, we use a heavy atom with large SOC.
    calc = GPAW(gpw_files['Tl_box_pw'])
    state = calc.calculation.state
    setup = calc.calculation.setups[0]

    # Kinetic energy calculated from sum of bands (the standard way)
    Ekin1 = state.ibzwfs.energies['band'] + state.potential.energies['kinetic']

    # Kinetic energy calculated directly from second-order derivative
    wfs = state.ibzwfs.wfs_qs[0][0]

    psit = wfs.psit_nX
    occ_n = wfs.occ_n

    psit_nsG = psit.data
    G_plus_k_Gv = psit.desc.G_plus_k_Gv
    ucvol = psit.desc.volume

    laplacian_on_psit_nsv = np.abs(psit_nsG)**2 @ G_plus_k_Gv**2
    laplacian_on_psit_n = - np.sum(np.sum(laplacian_on_psit_nsv, axis=1),
                                   axis=1) * ucvol
    Ekin_pseudo = -0.5 * (laplacian_on_psit_n @ occ_n)

    D_p = pack(state.density.D_asii[0][0].real)
    Ekin_PAW = setup.K_p @ D_p + setup.Kc

    Ekin2 = Ekin_pseudo + Ekin_PAW

    # We should get the same value for the kinetic
    # energy irrespective of the method used.
    assert Ekin1 == pytest.approx(Ekin2, abs=1e-5)

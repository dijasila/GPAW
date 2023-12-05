import numpy as np
import pytest

from gpaw.new.ase_interface import GPAW
from gpaw.utilities import pack

@pytest.mark.soc
def test_kinetic_energy(gpw_files):
    calc = GPAW(gpw_files['Tl_box_pw'],
                parallel={'domain': 1, 'band': 1})

    state = calc.calculation.state
    setup = calc.calculation.setups[0]

    # Kinetic energy calculated from sum of bands
    Ekin1 = state.ibzwfs.energies['band'] + state.potential.energies['kinetic']

    # Kinetic energy calculated from second-order derivative
    wfs = state.ibzwfs.wfs_qs[0][0]

    psit = wfs.psit_nX
    occ_n = wfs.occ_n

    psit_nsG = psit.data
    G_plus_k_Gv = psit.desc.G_plus_k_Gv
    ucvol = psit.desc.volume

    laplacian_on_psit_nsv = (np.abs(psit_nsG)**2) @ (G_plus_k_Gv)**2
    laplacian_on_psit_n = - np.sum(np.sum(laplacian_on_psit_nsv, axis=1),
                                   axis=1) * ucvol
    Ekin_pseudo = - 0.5 * (laplacian_on_psit_n @ occ_n)

    D_p = pack(state.density.D_asii[0][0].real)

    Ekin_PAW = setup.K_p @ D_p + setup.Kc

    Ekin2 = Ekin_pseudo + Ekin_PAW

    print(Ekin1)
    print(Ekin2)

    crash

    crash

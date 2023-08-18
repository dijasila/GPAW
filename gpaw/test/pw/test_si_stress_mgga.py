import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.test import numeric_force
from ase.parallel import parprint

from gpaw import GPAW, PW, Mixer, FermiDirac
from gpaw.mpi import world


@pytest.mark.mgga
def test_pw_si_stress_mgga(in_tmp_dir, gpaw_new):
    xc = 'TPSS'
    si = bulk('Si')
    k = 3
    si.calc = GPAW(mode=PW(250),
                   mixer=Mixer(0.7, 5, 50.0),
                   xc=xc,
                   occupations=FermiDirac(0.01),
                   kpts=(k, k, k),
                   convergence={'energy': 1e-8},
                   parallel={'domain': min(2, world.size)},
                   txt='si.txt')

    si.set_cell(np.dot(si.cell,
                       [[1.02, 0, 0.03],
                        [0, 0.99, -0.02],
                        [0.2, -0.01, 1.03]]),
                scale_atoms=True)

    si.get_potential_energy()

    # Trigger nasty bug (fixed in !486):
    if not gpaw_new:
        si.calc.wfs.pt.blocksize = si.calc.wfs.pd.maxmyng - 1
    else:
        for wfs in si.calc.calculation.state.ibzwfs:
            wfs._pt_aiX.blocksize = wfs._pt_aiX.pw.maxmysize

    s_analytical = si.get_stress()
    # Calculated numerical stress once, store here to speed up test
    s_numerical = np.array([-0.01140242, -0.04084746, -0.0401058,
                            -0.02119629, 0.13584242, 0.00911572])
    if 0:
        s_numerical = si.calc.calculate_numerical_stress(si, 1e-5)

    s_err = s_numerical - s_analytical

    parprint('Analytical stress:\n', s_analytical)
    parprint('Numerical stress:\n', s_numerical)
    parprint('Error in stress:\n', s_err)
    assert np.all(abs(s_err) < 1e-5)

    # Check y-component of second atom:
    f = si.get_forces()[1, 1]
    fref = -2.066952082010687
    if 0:
        fref = numeric_force(si, 1, 1)
    print(f, fref, f - fref)
    assert abs(f - fref) < 0.0005

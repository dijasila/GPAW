import numpy as np
from ase.build import bulk
from gpaw import GPAW, PW, Mixer
from gpaw.mpi import world
from ase.calculators.test import numeric_force


def test_pw_si_stress(in_tmp_dir):
    xc = 'PBE'
    si = bulk('Si')
    k = 2
    si.calc = GPAW(mode=PW(200),
                   mixer=Mixer(0.7, 5, 50.0),
                   xc=xc,
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
    si.calc.wfs.pt.blocksize = si.calc.wfs.pd.maxmyng - 1
    
    # compute error in stress as numerical - analytical
    s_analytical = si.get_stress()
    s_numerical = si.calc.calculate_numerical_stress(si, 1e-5)
    s_err = s_numerical - s_analytical

    assert np.all(abs(s_err) < 1e-4)


def test_pw_si_force(in_tmp_dir):
    xc = 'PBE'
    si = bulk('Si')
    k = 1
    si.calc = GPAW(mode=PW(200),
                   mixer=Mixer(0.7, 5, 50.0),
                   xc=xc,
                   kpts=(k, k, k),
                   convergence={'energy': 1e-8},
                   parallel={'domain': min(2, world.size)},
                   symmetry='off',  # needed for set_positions
                   txt='si.txt')

    si.get_potential_energy()

    # compute error in force as numerical-analytical
    f_analytical = si.get_forces()
    f_numerical = numeric_force(si, 1, 0, 0.001)

    assert abs(f_analytical[1, 0] - f_numerical) < 0.0005

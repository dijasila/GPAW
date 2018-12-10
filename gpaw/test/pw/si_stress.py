import numpy as np
from ase.build import bulk
from gpaw import GPAW, PW, Mixer

for xc in ['PBE', 'LDA']:
    si = bulk('Si')
    k = 3
    si.calc = GPAW(mode=PW(250),
                   mixer=Mixer(0.7, 5, 50.0),
                   xc=xc,
                   kpts=(k, k, k),
                   convergence={'energy': 1e-8},
                   txt='si.txt')

    si.set_cell(np.dot(si.cell,
                       [[1.02, 0, 0.03],
                        [0, 0.99, -0.02],
                        [0.2, -0.01, 1.03]]),
                scale_atoms=True)

    s_analytical = si.get_stress()
    s_numerical = si.calc.calculate_numerical_stress(si, 1e-5)
    s_err = s_numerical - s_analytical

    print('Analytical stress:\n', s_analytical)
    print('Numerical stress:\n', s_numerical)
    print('Error in stress:\n', s_err)
    assert np.all(abs(s_err) < 1e-4)

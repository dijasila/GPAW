import pytest
import numpy as np
from gpaw.response.g0w0 import G0W0
from ase.units import Hartree as Ha
from ase.build import bulk
from gpaw import GPAW, PW


refdata = eval("""{'f': np.array([[[1., 0.], [1., 0.],
 [1., 0.]]]),
 'eps': np.array([[[5.44216081, 7.83160495], [2.50723384, 5.84572768],
 [4.21468461, 6.77480385]]]),
 'vxc': np.array([[[-13.58093414, -11.68590091],
 [-12.43891431, -10.03810567],
 [-13.22609235, -12.56251973]]]),
 'exx': np.array([[[-15.92147417, -6.54714333],
 [-15.25022165,  -5.30715967],
 [-15.71928859,  -7.38947814]]]),
 'sigma': np.array([[[ 2.72328455, -3.82381496],
 [ 2.98815893, -3.6791976 ],
 [ 2.81412097, -3.7469274 ]]]),
 'dsigma': np.array([[[-0.27037353, -0.255096  ],
 [-0.29191701, -0.24547265],
 [-0.28314139, -0.25535679]]]),
 'Z': np.array([[[0.78717006, 0.7967518 ],
 [0.77404353, 0.80290804],
 [0.77933734, 0.79658628]]]),
 'qp': np.array([[[5.74344584, 8.87928785],
 [2.64412467, 6.69018492],
 [4.46479323, 7.91082685]]]),
 'sigma_eskn': np.array([[[[ 2.72328455, -3.82381496],
  [ 2.98815893, -3.6791976 ],
  [ 2.81412097, -3.7469274]]]]),
  'dsigma_eskn': np.array([[[[-0.27037353, -0.255096  ],
  [-0.29191701, -0.24547265], [-0.28314139, -0.25535679]]]]),
  'sigma_eskwn':
  np.array([[[[[0.93787012+0.00026737j,  1.06800255+0.00093579j],
   [0.97348861+0.00028235j, 1.11557913+0.00100937j]],
  [[0.94401979+0.00025306j, 1.02102979+0.00087934j],
   [0.9796042 +0.00026619j, 1.06621614+0.00095093j]],
  [[0.94620462+0.00025838j, 1.07205132+0.00092491j],
   [0.98204915+0.00027201j, 1.11947418+0.00100002j]]]]])}""")


@pytest.mark.response
def test_mpa(in_tmp_dir, gpw_files, scalapack):
    atoms = bulk('Si')
    calc = GPAW(mode=PW(ecut=200), kpts={'size': (2, 2, 2), 'gamma': True},
                nbands=30, convergence={'bands': 20}, occupations={'width': 0})
    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write('bands.gpw', mode='all')

    mpa_dict = {'npoles': 8, 'eta0': 0.01 * Ha, 'eta_rest': 0.1 * Ha,
                'varpi': 1 * Ha,
                'wrange': np.array([0, 1 * Ha]), 'parallel_lines': 2,
                'alpha': 1}

    w_w = np.array([5.80, 5.60])
    gw = G0W0('bands.gpw', 'mpa',
              bands=(3, 5),
              nbands=20,
              nblocks=1,
              ecut=50,
              evaluate_sigma=w_w,
              mpa=mpa_dict)

    results = gw.calculate()
    for key, value in refdata.items():
        assert np.allclose(value, results[key], rtol=1e-4, atol=1e-4)

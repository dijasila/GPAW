import pytest
import numpy as np
from gpaw.response.g0w0 import G0W0
from ase.units import Hartree as Ha
from ase.build import bulk
from gpaw import GPAW, PW


@pytest.mark.response
def test_mpa(in_tmp_dir, gpw_files, scalapack):
    atoms = bulk('Si')
    calc = GPAW(mode=PW(ecut=400), kpts={'size': (2, 2, 2), 'gamma': True},
                nbands=40, convergence={'bands': 20}, occupations={'width': 0})
    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write('bands.gpw', mode='all')

    mpa_dict = {'npoles': 8, 'eta0': 0.01 * Ha, 'eta_rest': 0.1 * Ha,
                'varpi': 1 * Ha,
                'wrange': np.array([0, 1 * Ha]), 'parallel_lines': 2,
                'alpha': 1}

    w_w = np.linspace(-1, 1, 250)

    gw = G0W0('bands.gpw', 'ff',
              bands=(3, 5),
              nbands=20,
              nblocks=1,
              evaluate_sigma=w_w,
              ecut=50)

    results = gw.calculate()
    gw = G0W0('bands.gpw', 'mpa',
              bands=(3, 5),
              nbands=20,
              nblocks=1,
              ecut=50,
              evaluate_sigma=w_w,
              ppa=False,
              mpa=mpa_dict)

    results2 = gw.calculate()

    print(results)
    print(results2)

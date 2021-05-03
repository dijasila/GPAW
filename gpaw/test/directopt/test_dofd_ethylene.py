from gpaw import GPAW, FD
from gpaw.directmin.fdpw.directmin import DirectMin
import numpy as np
from gpaw.test import equal
from ase import Atoms


def test_dofd_ethylene(in_tmp_dir):

    atoms = Atoms('CCHHHH',
                  positions=[
                      [-0.66874198, -0.00001714, -0.00001504],
                      [ 0.66874210, 0.00001699, 0.00001504],
                      [-1.24409879, 0.00000108, -0.93244784],
                      [-1.24406253, 0.00000112, 0.93242153],
                      [ 1.24406282, -0.93242148, 0.00000108],
                      [ 1.24409838, 0.93244792, 0.00000112]
                  ]
                  )
    atoms.center(vacuum=4.0)
    atoms.set_pbc(False)

    calc = GPAW(mode=FD(),
                h=0.3,
                xc='PBE',
                occupations={'name': 'fixed-occ-zero-width'},
                eigensolver=DirectMin(
                    convergelumo=True
                ),
                mixer={'name': 'dummy'},
                spinpol=True,
                symmetry='off',
                nbands=-5,
                convergence={'eigenstates': 4.0e-6},
                )
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    fsaved = [[9.23321, -0.01615, -0.00169],
              [-9.23057, 0.00276,  0.01506],
              [-3.42781, -2.65716, -2.17586],
              [-3.43347, 2.64956, 2.17609],
              [3.43284, -2.17649, -2.65107],
              [3.42732, 2.17634, 2.66001]]

    assert (np.abs(forces - fsaved) < 1.0e-3).all()
    equal(energy, -24.789097 , 1.0e-6)
    assert calc.wfs.kpt_u[0].eps_n[5] > calc.wfs.kpt_u[0].eps_n[6]

    calc.write('ethylene.gpw', mode='all')
    from gpaw import restart
    atoms, calc = restart('ethylene.gpw', txt='-')
    atoms.positions += 1.0e-6
    f3 = atoms.get_forces()
    niter = calc.get_number_of_iterations()
    equal(niter, 3, 1)
    equal(fsaved, f3, 1e-2)

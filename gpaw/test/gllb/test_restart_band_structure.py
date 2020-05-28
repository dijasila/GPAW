from gpaw import GPAW, restart, FermiDirac
from gpaw.test import equal
import os
from gpaw.mpi import world
from ase.build import bulk


def test_gllb_restart_band_structure(in_tmp_dir):
    e = {}

    energy_tolerance = 0.001

    e_ref = {'LDA': {'restart': -1.0695887848881105},
             'GLLBSC': {'restart': -0.9886252429853254}}

    for xc in ['LDA', 'GLLBSC']:
        atoms = bulk('Si')
        calc = GPAW(h=0.25,
                    nbands=8,
                    occupations=FermiDirac(width=0.01),
                    kpts=(3, 3, 3),
                    convergence={'eigenstates': 9.2e-11,
                                 'bands': 8},
                    xc=xc,
                    eigensolver='cg')

        atoms.calc = calc
        e[xc] = {'direct': atoms.get_potential_energy()}
        print(calc.get_ibz_k_points())
        old_eigs = calc.get_eigenvalues(kpt=3)
        calc.write('Si_gs.gpw')
        del atoms
        del calc
        atoms, calc = restart('Si_gs.gpw',
                              fixdensity=True,
                              kpts=[[0, 0, 0], [1.0 / 3, 1.0 / 3, 1.0 / 3]])
        e[xc] = {'restart': atoms.get_potential_energy()}

        if world.rank == 0:
            os.remove('Si_gs.gpw')
        diff = calc.get_eigenvalues(kpt=1)[:6] - old_eigs[:6]
        if world.rank == 0:
            print("occ. eig. diff.", diff)
            error = max(abs(diff))
            assert error < 5e-6

        for mode in e[xc].keys():
            equal(e[xc][mode], e_ref[xc][mode], energy_tolerance)

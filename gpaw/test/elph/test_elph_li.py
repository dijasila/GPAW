"""
Calculate the electron-phonon matrix in lithium using ALDA.

Tests whether the spin-paired and spin-polarised results are identical.
"""
import numpy as np
import os

from ase.build import bulk

from gpaw import GPAW
from gpaw.elph.electronphonon import ElectronPhononCoupling

@pytest.mark.elph
def test_elph_li(in_tmp_dir):
    # 2 atoms with one 1 valence electron each
    atoms = bulk('Li', crystalstructure='bcc', a=3.51, cubic=True)

    for spinpol in (0, 1):
        # Part 1: ground state calculation

        # sz:  1 orbital per atom
        # szp: 4 orbitals per atom
        calc = GPAW(mode='lcao',
                    basis='szp(dzp)',
                    kpts=(2, 2, 2),
                    symmetry={'point_group': False},
                    convergence={'bands': 'nao'},
                    spinpol=spinpol
                    )
        atoms.calc = calc

        # szp, k222: 4s/8s on 1 core
        atoms.get_potential_energy()

        # Part 2: compute displacements

        # szp, k222: 40s (80s) on 1 core
        # this step produces pickle files. do we need to clean them manually?
        if spinpol:
            name = 'elph_spinpol'
        else:
            name = 'elph_spinpared'
        elph = ElectronPhononCoupling(atoms=atoms, calc=atoms.calc,
                                      supercell=(1, 1, 1), name=name,
                                      calculate_forces=False)
        elph.run()

        # Part 3: contruct coefficient matrix

        # szp, k222: 4s on 1 core
        elph.set_lcao_calculator(atoms.calc)
        elph.calculate_supercell_matrix(dump=0, include_pseudo=True)
        if spinpol:
            g_xsMM = elph.g_xsNNMM[:, :, 0, 0]
        else:
            g_xMM = elph.g_xsNNMM[:, 0, 0, 0]

        # Part 4:  analyse matrix
        for s in range(spinpol+1):
            for x in range(6):  # 2 atoms * 2 directions
                # gMM is symmetric
                assert (np.allclose(elph.g_xsNNMM[x,s,0,0],
                                    elph.g_xsNNMM[x,s,0,0].T))
                # in this case both atoms and all 6 displacements are equivalent
                # all six gMM have same entries, but in different places
                assert (abs(np.max(elph.g_xsNNMM[x,s,0,0]) -
                            np.max(elph.g_xsNNMM[0,0,0,0])) < 5e-5)

    # Part 4: compare spin-paired and spin-polarised
    assert np.allclose(g_xsMM[:,0], g_xsMM[:,1])
    assert np.allclose(g_xMM, g_xsMM[:,0])

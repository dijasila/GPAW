"""Basic test of elph/supercell/

"""
import numpy as np
import pytest

from ase.build import bulk

from gpaw import GPAW
from gpaw.elph import DisplacementRunner
from gpaw.elph import Supercell

pytestmark = pytest.mark.usefixtures('module_tmp_path')
SUPERCELL = (2, 1, 1)

g00 = np.array([[[[-1.16418518e-06, 1.26395317e-02],
                  [1.26395317e-02, 4.05925161e-02]],
                 [[3.48680871e-02, 1.26402319e-02],
                  [1.37904135e-02, 5.31904081e-07]]],
                [[[3.48680871e-02, 1.37904135e-02],
                  [1.26402319e-02, 5.31904081e-07]],
                 [[3.59539410e-07, -1.37902297e-02],
                  [-1.37902297e-02, -4.05933006e-02]]]])


@pytest.fixture()
def elph_cache():
    """Minimum elph cache for Li
    
    Uses 1x2x2 k-points and 2x1x1 SC to allow for parallelisaiton
    test.
    Takes 6s on 4 cores.
    """
    atoms = bulk('Li', crystalstructure='bcc', a=3.51, cubic=True)
    calc = GPAW(mode='lcao',
                basis='sz(dzp)',
                kpts={'size': (1, 2, 2), 'gamma': False},
                symmetry={'point_group': False},
                convergence={'density': 1.5e-1, 'energy': 1},
                txt='elph_li.txt'
                )
    atoms.calc = calc
    elph = DisplacementRunner(atoms, calc,
                              supercell=SUPERCELL, name='elph',
                              calculate_forces=False)
    elph.run()
    return elph
    

@pytest.mark.elph
def test_supercell(in_tmp_dir, elph_cache):
    atoms = bulk('Li', crystalstructure='bcc', a=3.51, cubic=True)
    atoms_N = atoms * SUPERCELL
    elph_cache
    calc = GPAW(mode='lcao', basis='sz(dzp)',
                kpts={'size': (1, 2, 2), 'gamma': False},
                symmetry={'point_group': False},
                convergence={'density': 1.5e-1, 'energy': 1},
                parallel={'domain': 1},
                txt='gs_li.txt')
    atoms_N.calc = calc
    atoms_N.get_potential_energy()
    
    # create supercell cache
    sc = Supercell(atoms, supercell=SUPERCELL)
    sc.calculate_supercell_matrix(calc)
    
    # read supercell matrix
    g_xsNNMM, basis_info = Supercell.load_supercell_matrix()
    assert g_xsNNMM.shape == (6, 1, 2, 2, 2, 2)
    print(g_xsNNMM[0, 0])
    assert g_xsNNMM[0, 0] == pytest.approx(g00, abs=1e-4)

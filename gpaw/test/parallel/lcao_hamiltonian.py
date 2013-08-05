import cPickle as pickle
import numpy as np
from ase import Atoms
from gpaw import GPAW, restart, setup_paths
from gpaw.lcao.tools import get_lcao_hamiltonian
from gpaw.mpi import world
from gpaw.atom.basis import BasisMaker
from gpaw.test import equal

a = 2.7
bulk = Atoms('Li', pbc=True, cell=[a, a, a])
calc = GPAW(gpts=(8, 8, 8),
            kpts=(4, 4, 4),
            mode='lcao',
            nbands=7,
            basis='szp(dzp)')
bulk.set_calculator(calc)
e = bulk.get_potential_energy()
calc.write('temp.gpw')

atoms, calc = restart('temp.gpw')
H_skMM, S_kMM = get_lcao_hamiltonian(calc)
eigs = calc.get_eigenvalues(kpt=2)

if world.rank == 0:
    eigs2 = np.linalg.eigvals(np.linalg.solve(S_kMM[2], H_skMM[0, 2])).real
    eigs2.sort()
    assert abs(sum(eigs - eigs2)) < 1e-8

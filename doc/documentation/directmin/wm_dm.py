import time
import numpy as np
import tools_testing
from ase import Atoms
from gpaw import GPAW, LCAO, ConvergenceError
from ase.parallel import paropen
from gpaw.directmin.etdm import ETDM
from gpaw.mpi import world
from gpaw.atom.basis import BasisMaker
from gpaw import setup_paths
setup_paths.insert(0, '.')

for symbol in ['H', 'O']:
    bm = BasisMaker(symbol, xc='PBE')
    basis = bm.generate(zetacount=3, polarizationcount=2)
    basis.write_xml()

positions = tools_testing.positions

L = 9.8553729
r = [[1, 1, 1],
     [2, 1, 1],
     [2, 2, 1]]

with paropen('dm-water-results.txt', 'w') as fd:
    for x in r:
        atoms = Atoms('32(OH2)', positions=positions)
        atoms.set_cell((L, L, L))
        atoms.set_pbc(1)
        atoms = atoms.repeat(x)
        calc = GPAW(xc='PBE', h=0.2,
                    convergence={'density': 1.0e-6},
                    maxiter=333,
                    basis='tzdp',
                    mode=LCAO(),
                    txt=str(len(atoms) // 3) + '_H2Omlcls_scf.txt',
                    eigensolver=ETDM(matrix_exp='egdecomp-u-invar',
                                     representation='u-invar'),
                    mixer={'backend': 'no-mixing'},
                    nbands='nao',
                    occupations={'name': 'fixed-uniform'},
                    symmetry='off',
                    parallel={'domain': world.size})

        atoms.set_calculator(calc)

        try:
            t1 = time.time()
            e = atoms.get_potential_energy()
            t2 = time.time()
            steps = atoms.calc.get_number_of_iterations()
            iters = atoms.calc.wfs.eigensolver.eg_count
            print("{}\t{}\t{}\t{}\t{}".format(
                len(atoms), steps, e, iters, t2 - t1),
                flush=True, file=fd)  # s
        except ConvergenceError:
            print("{}\t{}\t{}\t{}\t{}\t{}".format(
                None, None, None, None, None, None),
                flush=True, file=fd)


output = \
    """
96	13	-449.29433888653887	15	13.834846258163452	417.422
192	13	-899.8689779482846	15	47.11155915260315	893.156
384	13	-1802.1980642103324	15	243.71619248390198	2633.445
"""

output.splitlines()

# this is saved data
saved_data = {}
for i in output.splitlines():
    if i == '':
        continue
    mol = i.split()
    # ignore last two columns which are memory and elapsed time
    saved_data[mol[0]] = np.array([float(_) for _ in mol[1:-2]])

with open('dm-water-results.txt', 'r') as fd:
    calculated_data_string = fd.read().split('\n')

# this is data calculated, we would like to coompare it to saved
# compare number of iteration, energy and gradient evaluation,
# and energy

calculated_data = {}
for i in calculated_data_string:
    if i == '':
        continue
    mol = i.split()
    # ignore last column (elapsed time)sure
    calculated_data[mol[0]] = np.array([float(_) for _ in mol[1:-1]])

error = np.array([3, 1.0e-2, 3])

assert len(calculated_data) == len(saved_data)
for k in saved_data.keys():
    assert (abs(saved_data[k] - calculated_data[k]) < error).all()

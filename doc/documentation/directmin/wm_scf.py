import numpy as np
from ase import Atoms
from gpaw import GPAW, FermiDirac, LCAO, ConvergenceError
from ase.parallel import parprint
from gpaw.utilities.memory import maxrss
import time

from gpaw.mpi import world

from gpaw import setup_paths
setup_paths.insert(0, '.')
from gpaw.atom.basis import BasisMaker

for symbol in ['H', 'O']:
    bm = BasisMaker(symbol, xc='PBE')
    basis = bm.generate(zetacount=3, polarizationcount=2)
    basis.write_xml()

positions = [
    (-0.069, 0.824, -1.295), (0.786, 0.943, -0.752), (-0.414, -0.001, -0.865),
    (-0.282, -0.674, -3.822), (0.018, -0.147, -4.624),
    (-0.113, -0.080, -3.034),
    (2.253, 1.261, 0.151), (2.606, 0.638, -0.539), (2.455, 0.790, 1.019),
    (3.106, -0.276, -1.795), (2.914, 0.459, -2.386), (2.447, -1.053, -1.919),
    (6.257, -0.625, -0.626), (7.107, -1.002, -0.317), (5.526, -1.129, -0.131),
    (5.451, -1.261, -2.937), (4.585, -0.957, -2.503), (6.079, -0.919, -2.200),
    (-0.515, 3.689, 0.482), (-0.218, 3.020, -0.189), (0.046, 3.568, 1.382),
    (-0.205, 2.640, -3.337), (-1.083, 2.576, -3.771), (-0.213, 1.885, -2.680),
    (0.132, 6.301, -0.278), (1.104, 6.366, -0.068), (-0.148, 5.363, -0.112),
    (-0.505, 6.680, -3.285), (-0.674, 7.677, -3.447), (-0.965, 6.278, -2.517),
    (4.063, 3.342, -0.474), (4.950, 2.912, -0.663), (3.484, 2.619, -0.125),
    (2.575, 2.404, -3.170), (1.694, 2.841, -3.296), (3.049, 2.956, -2.503),
    (6.666, 2.030, -0.815), (7.476, 2.277, -0.316), (6.473, 1.064, -0.651),
    (6.860, 2.591, -3.584), (6.928, 3.530, -3.176), (6.978, 2.097, -2.754),
    (2.931, 6.022, -0.243), (3.732, 6.562, -0.004), (3.226, 5.115, -0.404),
    (2.291, 7.140, -2.455), (1.317, 6.937, -2.532), (2.586, 6.574, -1.669),
    (6.843, 5.460, 1.065), (7.803, 5.290, 0.852), (6.727, 5.424, 2.062),
    (6.896, 4.784, -2.130), (6.191, 5.238, -2.702), (6.463, 4.665, -1.259),
    (0.398, 0.691, 4.098), (0.047, 1.567, 3.807), (1.268, 0.490, 3.632),
    (2.687, 0.272, 2.641), (3.078, 1.126, 3.027), (3.376, -0.501, 2.793),
    (6.002, -0.525, 4.002), (6.152, 0.405, 3.660), (5.987, -0.447, 4.980),
    (0.649, 3.541, 2.897), (0.245, 4.301, 3.459), (1.638, 3.457, 3.084),
    (-0.075, 5.662, 4.233), (-0.182, 6.512, 3.776), (-0.241, 5.961, 5.212),
    (3.243, 2.585, 3.878), (3.110, 2.343, 4.817), (4.262, 2.718, 3.780),
    (5.942, 2.582, 3.712), (6.250, 3.500, 3.566), (6.379, 2.564, 4.636),
    (2.686, 5.638, 5.164), (1.781, 5.472, 4.698), (2.454, 6.286, 5.887),
    (6.744, 5.276, 3.826), (6.238, 5.608, 4.632), (7.707, 5.258, 4.110),
    (8.573, 8.472, 0.407), (9.069, 7.656, 0.067), (8.472, 8.425, 1.397),
    (8.758, 8.245, 2.989), (9.294, 9.091, 3.172), (7.906, 8.527, 3.373),
    (4.006, 7.734, 3.021), (4.685, 8.238, 3.547), (3.468, 7.158, 3.624),
    (5.281, 6.089, 6.035), (5.131, 7.033, 6.378), (4.428, 5.704, 5.720),
    (5.067, 7.323, 0.662), (5.785, 6.667, 0.703), (4.718, 7.252, 1.585)]

L = 9.8553729
r = [[1, 1, 1],
     [2, 1, 1],
     [2, 2, 1],
     # [2, 2, 2],
     # [3, 2, 2],
     # [3, 3, 2]
     ]

file2write = open('scf-water-results.txt', 'w')

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
                occupations=FermiDirac(width=0.0, fixmagmom=True),
                parallel={'domain': world.size})

    atoms.set_calculator(calc)

    try:
        t1 = time.time()
        e = atoms.get_potential_energy()
        t2 = time.time()
        steps = atoms.calc.get_number_of_iterations()
        iters = steps
        memory = maxrss() / 1024.0 ** 2
        parprint("{}\t{}\t{}\t{}\t{}\t{:.3f}".format(
            len(atoms), steps, e, iters, t2 - t1, memory),
            flush=True, file=file2write)  # s,MB
    except ConvergenceError:
        parprint("{}\t{}\t{}\t{}\t{}\t{}".format(
                 None, None, None, None, None, None),
                 flush=True, file=file2write)

output = \
    """
96	22	-449.2501666690716	22	22.129411935806274	381.734
192	21	-899.7732083940263	21	91.08534455299377	773.613
384	21	-1802.1232238298205	21	659.9850523471832	2085.758
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

file2read = open('scf-water-results.txt', 'r')
calculated_data_string = file2read.read().split('\n')
file2read.close()

# this is data calculated, we would like to coompare it to saved
# compare number of iteration, energy and gradient evaluation,
# and energy

calculated_data = {}
for i in calculated_data_string:
    if i == '':
        continue
    mol = i.split()
    # ignore last two columns which are memory and elapsed time
    calculated_data[mol[0]] = np.array([float(_) for _ in mol[1:-2]])

error = np.array([3, 1.0e-2, 3])

assert len(calculated_data) == len(saved_data)
for k in saved_data.keys():
    assert (abs(saved_data[k] - calculated_data[k]) < error).all()

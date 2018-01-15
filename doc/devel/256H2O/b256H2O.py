from ase import Atoms

from gpaw import GPAW, ConvergenceError, RMMDIIS, setup_paths

# Use setups from the $PWD and $PWD/.. first
setup_paths.insert(0, '.')
setup_paths.insert(0, '../')

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

prefix = 'b256H2O'
L = 9.8553729
atoms = Atoms('32(OH2)',
              positions=positions)
atoms.set_cell((L, L, L), scale_atoms=False)
atoms.set_pbc(1)
r = [2, 2, 2]
atoms = atoms.repeat(r)
n = [56 * ri for ri in r]
# nbands (>=128) is the number of bands per 32 water molecules
nbands = 2 * 6 * 11  # 132
for ri in r:
    nbands = nbands * ri
# the next line decreases memory usage
es = RMMDIIS(keep_htpsit=False)
calc = GPAW(nbands=nbands,
            # uncomment next two lines to use lcao/sz
            # mode='lcao',
            # basis='sz',
            gpts=tuple(n),
            maxiter=5,
            width=0.01,
            eigensolver=es,
            txt=prefix + '.txt')

atoms.set_calculator(calc)
try:
    pot = atoms.get_potential_energy()
except ConvergenceError:
    pass

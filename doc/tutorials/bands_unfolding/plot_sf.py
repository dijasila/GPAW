import numpy as np
import pickle

from ase.units import Hartree
from ase.dft.kpoints import get_bandpath

from gpaw import GPAW


a = 3.184
L = 15.
PC = a * np.array([(1., 0., 0),
                 (-1. / 2, np.sqrt(3.) / 2., 0.),
                 (0, 0, L)])
G = np.array([0., 0., 0.])
M = np.array([1 / 2., 0., 0.])
K = np.array([1 / 3., 1 / 3., 0])
path = [M, K, G]
kpts, x, X = get_bandpath(path, PC, 48)

from gpaw.unfold import plot_spectral_function
calc = GPAW('gs_3x3_defect.gpw', txt=None)
ef = calc.get_fermi_level()

e, A_ke = pickle.load(open('sf_3x3_defect.pckl'))
e = e * Hartree - ef
plot_spectral_function(e, A_ke, kpts, x, X, path,
                       ['M', 'K', 'G'], 'blue', -3, 3)

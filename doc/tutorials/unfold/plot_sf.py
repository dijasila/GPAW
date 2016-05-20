import pickle
from ase.units import Hartree
from gpaw import GPAW
from gpaw.unfold import plot_spectral_function

calc = GPAW('gs_3x3_defect.gpw', txt=None)
ef = calc.get_fermi_level()

e, A_ke = pickle.load(open('sf_3x3_defect.pckl', 'rb'))
x, X = pickle.load(open('x.pckl', 'rb'))
e = e * Hartree - ef
plot_spectral_function(e, A_ke, x, X,
                       ['M', 'K', 'G'], 'blue', -3, 3)

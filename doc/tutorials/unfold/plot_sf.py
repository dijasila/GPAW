import pickle
from ase.build import mx2
from ase.dft.kpoints import get_bandpath, special_points
from ase.units import Hartree
from gpaw import GPAW
from gpaw.unfold import plot_spectral_function

a = 3.184
PC = mx2(a=a).cell
path = [special_points['hexagonal'][k] for k in 'MKG']
kpts, x, X = get_bandpath(path, PC, 48)

calc = GPAW('gs_3x3_defect.gpw', txt=None)
ef = calc.get_fermi_level()

e, A_ke = pickle.load(open('sf_3x3_defect.pckl', 'rb'))
e = e * Hartree - ef
plot_spectral_function(e, A_ke, x, X,
                       ['M', 'K', 'G'], 'blue', -3, 3)

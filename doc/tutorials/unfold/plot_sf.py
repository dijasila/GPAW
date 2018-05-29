# Creates: sf_3x3_defect.png
import pickle
from ase.units import Hartree
from gpaw import GPAW
from gpaw.unfold import plot_spectral_function

calc = GPAW('gs_3x3_defect.gpw', txt=None)
ef = calc.get_fermi_level()

plot_spectral_function(filename='sf_3x3_defect',
                       color='blue',
                       eref=ef,
                       emin=-3,
                       emax=3)

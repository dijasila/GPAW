import numpy as np

from ase.units import _hplanck, _c, J
from gpaw import GPAW
from gpaw.raman.raman import (calculate_raman, calculate_raman_intensity,
                              plot_raman)

# The Raman spectrum of three different excitation energies will be evaluated
wavelengths = np.array([488, 532, 633])  # nm
w_ls = _hplanck * _c * J / (wavelengths * 10**(-9))  # to eV

# Load pre-computed calculation
calc = GPAW("gs.gpw")
atoms = calc.atoms

# And the three Raman spectra are calculated
# Here choose xx direction via d_i=0, d_o=0
d_i = 0
d_o = 0
for i, w_l in enumerate(w_ls):
    name = "{}nm".format(wavelengths[i])
    # Calculate mode resolved Raman tensor for given direction
    calculate_raman(atoms, calc, w_l, d_i, d_o, resonant_only=False,
                    ramanname=name)
    # Calculate Raman intensity
    calculate_raman_intensity(d_i, d_o, ramanname=name)
    # And plot
    fname = "{}{}_{}".format('xyz'[d_i], 'xyz'[d_o], name)
    plot_raman(figname="Raman_{}.png".format(fname), ramanname=fname)

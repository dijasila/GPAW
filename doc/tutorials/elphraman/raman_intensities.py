import numpy as np

from gpaw import GPAW
from gpaw.raman.raman import (calculate_raman, calculate_raman_intensity,
                              plot_raman)

# Load pre-computed calculation
calc = GPAW("gs.gpw")
atoms = calc.atoms

# laser frequence633 nm approx 1.958676 eV
w_l = 1.958676

# Scan through all polarisations
pollist = []
for d_i in (0, 1, 2):
    for d_o in (0, 1, 2):
        # Calculate mode resolved Raman tensor for given direction
        calculate_raman(atoms, calc, w_l, d_i, d_o, resonant_only=True)
        # Calculate Raman intensity
        calculate_raman_intensity(d_i, d_o)
        pollist.append("{}{}".format('xyz'[d_i], 'xyz'[d_o]))

# And plot
plot_raman(figname="Raman_all.png", ramanname=pollist)


# web-page: Raman_all.png
import numpy as np
from gpaw import GPAW
from gpaw.raman.raman import (calculate_raman, calculate_raman_intensity,
                              plot_raman)

# Load pre-computed calculation
calc = GPAW('gs.gpw', parallel={'domain': 1, 'band': 1})
atoms = calc.atoms

# laser frequency 633 nm approx 1.958676 eV
w_l = 1.958676
suffix = '632nm'

# use previously saved phonon frequencies
w_ph = np.load('vib_frequencies.npy')

# Scan through all polarisations
pollist = []
for d_i in (0, 1, 2):
    for d_o in (0, 1, 2):
        # Calculate mode resolved Raman tensor for given direction
        calculate_raman(calc, w_ph, w_l, d_i, d_o, resonant_only=True,
                        suffix=suffix)
        if calc.wfs.kd.comm.rank == 0:
            # Calculate Raman intensity
            calculate_raman_intensity(w_ph, d_i, d_o, suffix=suffix)
            pollist.append('{}{}_{}'.format('xyz'[d_i], 'xyz'[d_o], suffix))

# And plot
if calc.wfs.kd.comm.rank == 0:
    plot_raman(figname='Raman_all.png', RIsuffix=pollist)

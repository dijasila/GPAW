import numpy as np
from gpaw.test import equal

results = np.load('formation_energies.npz')
repeats = results['repeats']
idx_222 = np.where(repeats == 2)[0, 0]
diff_222 = (results['corrected'] - results['uncorrected'])[idx_222]
equal(diff_222, 21.78, 0.01)

potfile = open('model_potentials.dat', 'r')
El = float(potfile.readline()[12:])

equal(El, -1.28, 0.01)

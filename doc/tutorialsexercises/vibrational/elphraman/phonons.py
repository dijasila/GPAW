import numpy as np
from ase.io import read
from ase.phonons import Phonons
from ase.units import invcm


atoms = read("MoS2_2H_relaxed_PBE.json")

# Phonon calculation
ph = Phonons(atoms, supercell=(3, 3, 2), name="elph", center_refcell=True)

ph.read(method='frederiksen', acoustic=True, )
frequencies = ph.band_structure([[0, 0, 0], ])[0]  # Get frequencies at Gamma

# save phonon frequencies for later use
np.save("vib_frequencies.npy", frequencies)
print('  i    cm^-1')
print('------------')
for i, fr in enumerate(frequencies):
    print('{:3d}  {:4.2f}'.format(i, fr / invcm))

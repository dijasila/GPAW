import numpy as np
from ase.build import bulk
from ase.phonons import Phonons
from ase.units import invcm
from gpaw import GPAW
from gpaw.mpi import world

atoms = bulk('C', 'diamond', a=3.567)
calc = GPAW(mode={'name': 'pw', 'ecut': 500},
            kpts=(5, 5, 5), xc='PBE',
            symmetry={'point_group': False},
            convergence={'density': 0.5e-5},
            txt='phonons.txt')

# Phonon calculation
ph = Phonons(atoms, calc, supercell=(1, 1, 1), delta=0.01)
ph.run()

# To display results (optional)
ph.read(method='frederiksen', acoustic=True)
frequencies = ph.band_structure([[0, 0, 0], ])[0]  # Get frequencies at Gamma
if world.rank == 0:
    # save phonon frequencies for later use
    np.save("vib_frequencies.npy", frequencies)
    print('  i    cm^-1')
    print('------------')
    for i, fr in enumerate(frequencies):
        print('{:3d}  {:4.2f}'.format(i, fr / invcm))

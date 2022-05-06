import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from ase.spectrum.band_structure import BandStructure

energies = []
magmoms = []
for i in range(31):
    atoms = read(f'gs-{i:02}.txt', format='gpaw-out')
    energy = atoms.get_potential_energy()
    energies.append(energy)

energies = np.array(energies) * 1000
energies -= energies[0]

path = atoms.cell.bandpath('GXW', npoints=31)

bs = BandStructure(path, energies[np.newaxis, :, np.newaxis])
bs.plot(emin=0, emax=31, label='q', ylabel='Energy [meV per atom]')
plt.savefig('espiral.png')

bs = BandStructure(path, magmoms[np.newaxis, :, np.newaxis])
bs.plot(emin=0, emax=1.5, label='q', ylabel=r'Total magnetic moment [$\mu_B$]')
plt.savefig('mspiral.png')

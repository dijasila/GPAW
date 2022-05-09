# web-page: e-spiral.png, m-spiral.png
import numpy as np
import matplotlib.pyplot as plt
from ase.spectrum.band_structure import BandStructure
from gpaw import GPAW

energies = []
magmoms = []
for i in range(31):
    atoms = GPAW(f'gs-{i:02}.gpw').get_atoms()
    energy = atoms.get_potential_energy()
    magmom = atoms.calc.calculation.magmoms()[0]
    energies.append(energy)
    magmoms.append(np.linalg.norm(magmom))

energies = np.array(energies) * 1000
energies -= energies[0]
magmoms = np.array(magmoms)

path = atoms.cell.bandpath('GXW', npoints=31)

bs = BandStructure(path, energies[np.newaxis, :, np.newaxis])
bs.plot(emin=-25, emax=5, label='q', ylabel='Energy [meV per atom]')
plt.savefig('e-spiral.png')

bs = BandStructure(path, magmoms[np.newaxis, :, np.newaxis])
bs.plot(emin=-0.1, emax=1.5, label='q',
        ylabel=r'Total magnetic moment [$\mu_B$]')
plt.savefig('m-spiral.png')

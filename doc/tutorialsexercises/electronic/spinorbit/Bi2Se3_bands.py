from ase.io import read
from ase.parallel import paropen
from gpaw import GPAW

atoms = read('gs_Bi2Se3.gpw')

G = [0.0, 0.0, 0.0]
L = [0.5, 0.0, 0.0]
F = [0.5, 0.5, 0.0]
Z = [0.5, 0.5, 0.5]
bandpath = atoms.cell.bandpath([G, Z, F, G, L],
                               npoints=200)
with paropen('bandpath.json', 'w') as fd:
    bandpath.write(fd)

calc = GPAW('gs_Bi2Se3.gpw').fixed_density(
    kpts=bandpath,
    symmetry='off',
    txt='Bi2Se3_bands.txt')
calc.write('Bi2Se3_bands.gpw')

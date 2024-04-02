from ase.io import read
from ase.parallel import paropen
from gpaw import GPAW

atoms = read('gs_Bi2Se3.gpw')

bandpath = atoms.cell.bandpath('GZFGL', npoints=200)
with paropen('bandpath.json', 'w') as fd:
    bandpath.write(fd)

calc = GPAW('gs_Bi2Se3.gpw').fixed_density(
    kpts=bandpath,
    symmetry='off',
    txt='Bi2Se3_bands.txt')
calc.write('Bi2Se3_bands.gpw')

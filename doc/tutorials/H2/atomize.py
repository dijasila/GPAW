# web-page: atomization.txt

from ase import Atoms
from ase.parallel import paropen as open
from gpaw import GPAW, PW, FermiDirac

a = 10.  # Size of unit cell (Angstrom)
c = a / 2

# Hydrogen atom:
atom = Atoms('H',
             positions=[(c, c, c)],
             magmoms=[0],
             cell=(a, a + 0.0001, a + 0.0002))  # Break cell symmetry

# gpaw calculator:
calc = GPAW(mode=PW(),
            xc='PBE',
            hund=True,
            eigensolver='rmm-diis',  # This solver can parallelize over bands
            occupations=FermiDirac(0.0, fixmagmom=True),
            txt='H.out',
            )
atom.calc = calc

e1 = atom.get_potential_energy()
calc.write('H.gpw')

# Hydrogen molecule:
d = 0.74  # Experimental bond length
molecule = Atoms('H2',
                 positions=([c - d / 2, c, c],
                            [c + d / 2, c, c]),
                 cell=(a, a, a))

calc.set(txt='H2.out')
calc.set(hund=False)  # No hund rule for molecules

molecule.calc = calc
e2 = molecule.get_potential_energy()
calc.write('H2.gpw')

fd = open('atomization.txt', 'w')
print(f'  hydrogen atom energy:     {e1:5.2f} eV', file=fd)
print(f'  hydrogen molecule energy: {e2:5.2f} eV', file=fd)
print(f'  atomization energy:       {2 * e1 - e2:5.2f} eV', file=fd)
fd.close()

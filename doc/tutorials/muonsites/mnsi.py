from ase import Atoms
from ase.dft.kpoints import *
from gpaw import GPAW, PW,MethfesselPaxton
from ase.io import write,read
from ase.units import Hartree
from gpaw import restart
from ase.spacegroup import crystal

a=4.55643
mnsi = crystal(['Mn','Si'], [(0.1380,0.1380,0.1380),(0.84620,0.84620,0.84620)], spacegroup=198, cellpar=[a, a, a, 90, 90, 90])


for atom in mnsi:
	if atom.symbol == 'Mn':
		atom.magmom=0.5

mnsi.set_calculator(GPAW(xc='PBE',kpts=(2, 2, 2), mode=PW(800),
    occupations=MethfesselPaxton(width=0.005), txt='mnsi.txt'))

mnsi.get_potential_energy()
mnsi.get_magnetic_moments()
mnsi.get_forces()
mnsi.calc.write('mnsi.gpw', mode='all')

mnsi1,calc = restart('mnsi.gpw', txt='mnsi.txt')
calc.get_fermi_level()
v = calc.get_electrostatic_potential()/Hartree
write('mnsi2.cube', mnsi1, data=v)


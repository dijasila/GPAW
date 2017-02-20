from gpaw import GPAW, PW, MethfesselPaxton
from ase.units import Hartree
from ase.spacegroup import crystal

a = 4.55643
mnsi = crystal(['Mn', 'Si'],
               [(0.1380, 0.1380, 0.1380), (0.84620, 0.84620, 0.84620)],
               spacegroup=198,
               cellpar=[a, a, a, 90, 90, 90])


for atom in mnsi:
    if atom.symbol == 'Mn':
        atom.magmom = 0.5

mnsi.calc = GPAW(xc='PBE',
                 kpts=(2, 2, 2),
                 mode=PW(800),
                 occupations=MethfesselPaxton(width=0.005),
                 txt='mnsi.txt')

mnsi.get_potential_energy()
mnsi.calc.write('mnsi.gpw')

calc = GPAW('mnsi.gpw', txt=None)
v = calc.get_electrostatic_potential() / Hartree
write('mnsi.cube', calc.get_atoms(), data=v)

from ase.build import molecule
from gpaw import GPAW, PW
ncpus = 1
atoms = molecule('Na2')
atoms.cell = [12, 12.01, 12.02]
atoms.center()
atoms.calc = GPAW(mode=PW(400),
                  xc='PBE')

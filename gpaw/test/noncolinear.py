from ase import Atoms
from gpaw import GPAW, FermiDirac
from gpaw.xc.noncolinear import NonColinearLDAKernel, \
     NonColinearLCAOEigensolver, NonColinearMixer
from gpaw.xc import XC

h = Atoms('H', magmoms=[1])
h.center(vacuum=2)
xc = XC(NonColinearLDAKernel())
c = GPAW(txt='nc.txt',
         mode='lcao',
         #basis='sz',
         #setups='ncpp',
         h=0.25,
         #occupations=FermiDirac(0.01),
         xc=xc,
         mixer=NonColinearMixer(),
         noncolinear=[(0, 0, 1)],
         eigensolver=NonColinearLCAOEigensolver())
c.set(nbands=1)
h.calc = c
h.get_potential_energy()

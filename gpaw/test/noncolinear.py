from ase import Atoms
from gpaw import GPAW
from gpaw.xc.noncolinear import NonColinearLDAKernel, \
     NonColinearLCAOEigensolver, NonColinearMixer
from gpaw.xc import XC

h = Atoms('H', magmoms=[1])
h.center(vacuum=2)
print h.cell
xc = XC(NonColinearLDAKernel())
c = GPAW(txt='H.txt',
         mode='lcao',
         #basis='sz',
         setups='ncpp',
         h=0.25,
         xc=xc,
         mixer=NonColinearMixer(),
         noncolinear=[(0,2,0)],
         eigensolver=NonColinearLCAOEigensolver())
c.set(nbands=1)
h.calc = c
h.get_potential_energy()

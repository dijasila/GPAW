from ase import Atoms
from gpaw import GPAW
from gpaw.xc.noncolinear import NonColinearLDA, NonColinearLCAOEigensolver, \
     NonColinearMixer

h = Atoms('H', magmoms=[1])
h.center(vacuum=2)
print h.cell
xc = NonColinearLDA()
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

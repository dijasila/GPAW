from ase import Atoms
from gpaw import GPAW
#from gpaw.xc.noncolinear import NonColinearLDA, NonColinearLCAOEigensolver, \
#     NonColinearMixer

h = Atoms('H', magmoms=[0])
h.center(vacuum=2)
xc = 'LDA'
c = GPAW(#txt='H.txt',
         mode='lcao',
         #basis='sz',
         #setups='ncpp',
         h=0.25,
         xc=xc,
         #mixer=NonColinearMixer(),
         #noncolinear=[(2,0,0)],
         )#eigensolver=NonColinearLCAOEigensolver())
c.set(nbands=1)
h.calc = c
h.get_potential_energy()

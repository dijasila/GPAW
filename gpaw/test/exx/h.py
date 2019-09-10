from ase import Atoms
from gpaw import GPAW, PW, Davidson
from gpaw.xc.hf import Hybrid

L = 5.5
a = Atoms('H',
          magmoms=[1],
          cell=[L, L, L],
          pbc=1)
# xc = Hybrid(xc='LDA', exx_fraction=0)
# xc.name = 'LDA'
xc = Hybrid('EXX')
es = Davidson(1)
es.keep_htpsit = False
a.calc = GPAW(mode=PW(400, force_complex_dtype=True),
              # setups='ae',
              eigensolver=es,
              xc=xc)
a.get_potential_energy()

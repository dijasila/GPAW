from ase import Atoms
from gpaw import GPAW, PW
from gpaw.xc.hf import Hybrid

L = 2.5
a = Atoms('H',
          magmoms=[1],
          cell=[L, L, L],
          pbc=1)
a.calc = GPAW(mode=PW(200, force_complex_dtype=True),
              setups='ae',
              )#xc=Hybrid())
a.get_potential_energy()

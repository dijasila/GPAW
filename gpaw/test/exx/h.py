import numpy as np
from ase import Atoms
from gpaw import GPAW, PW, Davidson
from gpaw.xc.hf import Hybrid
from gpaw.xc.hybrid import HybridXC as H1

L = 5.5
a = Atoms('HH',
          #magmoms=[1],
          cell=[L, L, L],
          pbc=1)
# xc = Hybrid(xc='LDA', exx_fraction=0)
# xc.name = 'LDA'
xc = Hybrid('EXX')
# xc = Hybrid(None, 'LDA', 0.0, 0.0)
# xc = H1('EXX')
es = Davidson(1)
es.keep_htpsit = False

D = np.linspace(0.65, 0.85, 11)
E = []
for d in D:
    a.positions[1, 0] = d
    a.calc = GPAW(
        mode=PW(400, force_complex_dtype=True),
        setups='ae',
        nbands=1,
        eigensolver=es,
        # eigensolver='rmm-diis',
        xc=xc)
    e = a.get_potential_energy()
    E.append(e)

if 1:
    import matplotlib.pyplot as plt
    plt.plot(D, E)
    plt.show()

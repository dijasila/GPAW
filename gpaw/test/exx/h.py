import numpy as np
from ase import Atoms
from gpaw import GPAW, PW, Davidson
from gpaw.xc.hf import Hybrid
from gpaw.xc.hybrid import HybridXC as H1

L = 6.5
a = Atoms('H',
          magmoms=[1],
          cell=[L, L, L],
          pbc=1)

# xc = Hybrid(xc='LDA', exx_fraction=0)
# xc.name = 'LDA'
# xc = Hybrid(None, 'LDA', 0.0, 0.0)

D = np.linspace(0.7, 0.8, 11)
E = []
for d in D:
    #a.positions[1, 0] = d
    a.center()#######################
    es = Davidson(1)
    es.keep_htpsit = False
    xc = Hybrid('EXX')
    # xc = H1('EXX')
    a.calc = GPAW(
        mode=PW(400, force_complex_dtype=True),
        # h=0.12,
        setups='p1',
        #setups='ae',
        nbands=1,
        eigensolver=es,
        # eigensolver='rmm-diis',
        # txt='h2.txt',
        #xc='PBE',
        xc=xc)
    e = a.get_potential_energy()
    raise SystemExit
    a.calc.set(xc=xc)
    e = a.get_potential_energy()
    E.append(e)
    print(d, e)

a = a[:1]
a[0].magmom = 1
es = Davidson(1)
es.keep_htpsit = False
# xc = Hybrid('EXX')
xc = H1('EXX')
a.calc = GPAW(
    # mode=PW(500, force_complex_dtype=True),
    h=0.12,
    setups='ae',
    nbands=1,
    # eigensolver=es,
    eigensolver='rmm-diis',
    txt='h.txt',
    xc=xc)
e = a.get_potential_energy()
print(2 * e - min(E))
if 1:
    import matplotlib.pyplot as plt
    plt.plot(D, E)
    plt.show()

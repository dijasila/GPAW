from ase import Atoms, Atom
from ase.vibrations.resonant_raman import ResonantRaman
from ase.vibrations.albrecht import Albrecht
from ase.parallel import world, parprint, DummyMPI

from gpaw import GPAW
from gpaw.lrtddft.kssingle import KSSingles
from gpaw.analyse.overlap import Overlap
from gpaw.test import equal

txt = '-'
txt = None
load = True
load = False
xc = 'LDA'

R = 0.7  # approx. experimental bond length
a = 4.0
c = 5.0
H2 = Atoms([Atom('H', (a / 2, a / 2, (c - R) / 2)),
            Atom('H', (a / 2, a / 2, (c + R) / 2))],
           cell=(a, a, c))

calc = GPAW(xc=xc, nbands=3, spinpol=False, eigensolver='rmm-diis', txt=txt)
H2.set_calculator(calc)
H2.get_potential_energy()

gsname = exname = 'rraman'
exkwargs={'eps':0.0, 'jend':1}
pz = ResonantRaman(H2, KSSingles, gsname=gsname, exname=exname,
                    exkwargs=exkwargs,
                   overlap=lambda x, y: Overlap(x).pseudo(y),
)
pz.run()

# check size
kss = KSSingles('rraman-d0.010.eq.ex.gz')
assert(len(kss) == 1)

om = 5
pz = Albrecht(H2, KSSingles, gsname=gsname, exname=exname,
              approximation='Albrecht A',
                   verbose=True, overlap=True,)
ai = pz.absolute_intensity(omega=om)[-1]
#equal(ai, 299.955946107, 1e-3) # earlier obtained value
i = pz.intensity(omega=om)[-1]
#equal(i, 7.82174195159e-05, 1e-11) # earlier obtained value
pz.summary(omega=5, method='frederiksen')

# parallel ------------------------

if world.size > 1 and world.rank == 0:
    # single core
    comm = DummyMPI()
    pzsi = Albrecht(H2, KSSingles, gsname=gsname, exname=exname,
                   comm=comm, verbose=True,)
    isi = pzsi.intensity(omega=om)[-1]
    equal(isi, i, 1e-11)

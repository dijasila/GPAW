import numpy as np

from ase.build import molecule

from gpaw import GPAW
from gpaw.lrtddft2 import LrTDDFT2
from gpaw.test import equal

name = 'H2O-lcao'
atoms = molecule('H2O')
atoms.center(vacuum=4)

# Ground state
calc = GPAW(h=0.4, mode='lcao', basis='dzp', txt='%s-gs.out' % name, nbands=8, xc='LDA')
atoms.set_calculator(calc)
atoms.get_potential_energy()
calc.write('%s.gpw' % name, mode='all')

# LrTDDFT2
calc = GPAW('%s.gpw' % name, txt='%s-lr.out' % name)
lr = LrTDDFT2(name, calc, fxc='LDA')
lr.calculate()
results = lr.get_transitions()[0:2]

if 0:
    np.set_printoptions(precision=13)
    print repr(results)

ref = (
np.array([  6.083223465439 ,   8.8741672507967,  13.5935056665245,
           14.291607433661 ,  15.9923770524209,  16.9925552100814,
           17.6504895168016,  17.6925534088714,  24.0929126532317,
           25.0027483575083,  25.6208990273861,  26.964991429833 ,
           29.5294793981159,  29.8440800287539]),
np.array([  3.6912275734538e-02,   1.6097104081322e-23,   3.0995162927818e-01,
            2.1338465380482e-02,   1.4257298380513e-22,   6.3876242246948e-02,
            1.0288210519694e-01,   1.7216431909253e-01,   2.8906903841875e-02,
            3.9353952343807e-01,   2.1927514220542e-02,   6.7747041558824e-01,
            7.9560508308094e-02,   1.0626657179089e-02]))

tol = 1e-12
for r0, r1 in zip(results, ref):
    equal(r0, r1, tol)

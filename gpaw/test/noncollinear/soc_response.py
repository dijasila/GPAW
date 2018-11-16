"""non-selfconsistent SOC with response."""

from unittest import SkipTest

import numpy as np
from ase.build import mx2

from gpaw import GPAW
from gpaw.spinorbit import get_nonsc_spinorbit_calc
from gpaw.mpi import size
from gpaw.test import equal
from gpaw.response.df import DielectricFunction

if size > 2:
    raise SkipTest()


def readcalc(calc):
    assert isinstance(calc, str)
    calc = GPAW(calc, txt=None)

    return calc


a = mx2('MoS2')
a.center(vacuum=3, axis=2)

params = dict(mode='pw',
              kpts={'size': (3, 3, 1),
                    'gamma': True},
              txt=None)

# Selfconsistent:
calc = GPAW(convergence={'bands': 28},
            xc='PBE',
            **params)
a.calc = calc
a.get_potential_energy()
calc.write('mos2_coll_wfs.gpw', mode='all')

socalc = get_nonsc_spinorbit_calc('mos2_coll_wfs.gpw', withsoc=False)
socalc.write('mos2_ncoll_wfs.gpw', mode='all')
socalc2 = readcalc('mos2_ncoll_wfs.gpw')
calc = readcalc('mos2_coll_wfs.gpw')

# Check that these values are the same
for kpt, sokpt, sokpt1 in zip(calc.wfs.kpt_u,
                              socalc.wfs.kpt_u,
                              socalc2.wfs.kpt_u):

    f0_n = kpt.f_n
    f_n = sokpt.f_n
    f1_n = sokpt1.f_n
    eps0_n = kpt.eps_n
    eps_n = sokpt.eps_n
    eps1_n = sokpt1.eps_n

    equal(np.sum(f0_n / kpt.weight), np.sum(f_n / sokpt.weight),
          tolerance=1e-4, msg='Occupations wrong in non-collinear calc.')
    equal(eps0_n, eps_n[::2], tolerance=1e-4, msg='Difference in eigenvalues')

df = DielectricFunction('mos2_ncoll_wfs.gpw', intraband=False,
                        txt='mos2_ncoll_response.txt')
socdf1, socdf2 = df.get_dielectric_function()
socomega_w = df.get_frequencies()

df = DielectricFunction('mos2_coll_wfs.gpw', intraband=False,
                        txt='mos2_coll_response.txt')
df1, df2 = df.get_dielectric_function()
omega_w = df.get_frequencies()

err1 = np.max(np.abs(df1 - socdf1))
err2 = np.max(np.abs(df2 - socdf2))

if 0:
    # If you want to see the calculated dielectric functions
    from matplotlib import pyplot as plt
    plt.figure()
    plt.plot(socomega_w, socdf2.real, label='real')
    plt.plot(socomega_w, socdf2.imag, label='imag')
    plt.plot(omega_w, df2.real, '--', label='real')
    plt.plot(omega_w, df2.imag, '--', label='imag')
    plt.legend()
    plt.xlim(0, 10)
    plt.ylim(-5, 10)
    plt.show()

msg = ('Too large difference between non-collinear '
       'response and spinpaired response of MoS2. Since '
       'soc=False in this test the calculated dielectric function '
       'should be identical')
equal(err1, 0, tolerance=1e-2, msg=msg)
equal(err2, 0, tolerance=1e-2, msg=msg)

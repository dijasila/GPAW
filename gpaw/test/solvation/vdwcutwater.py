from ase.structure import molecule
from gpaw import GPAW
from gpaw.solvation.calculator import SolvationGPAW
from gpaw.solvation.contributions import (
    MODE_RADII_CUTOFF, MODE_SURFACE_TENSION
    )
from gpaw.test import equal
from ase.units import Debye, kcal, mol, m
from ase.parallel import parprint
#from ase.optimize import FIRE
from ase.data.vdw import vdw_radii
import numpy

def norm(x):
    return numpy.sqrt(numpy.dot(x, x))

h = .3
vac = 3.
exponent = 3.4
xc = 'PBE'

water = molecule('H2O')
water.center(vac)

if False:
    water.calc = GPAW(xc=xc, h=h)
    #optimizer = FIRE(water)
    #optimizer.run(fmax=.05)
    E0 = water.get_potential_energy()
    p0 = norm(water.get_dipole_moment())
    parprint(E0)
    parprint(p0)
else:
    #E0 = -14.3331416248 #h=.2
    #p0 = 0.384069024983 #h=.2
    E0 = -14.5179032599 #h=.3
    p0 = 0.392950490281 #h=.3


water.calc = SolvationGPAW(
    xc=xc, h=h,
    solvation={
        'el':{
            'mode':MODE_RADII_CUTOFF,
            'radii':[vdw_radii[a.number] for a in water],
            'exponent':exponent,
            'epsilon_r':80.
            },
        'cav':{
            'mode':MODE_SURFACE_TENSION,
            'surface_tension': 0.0045
            }
        }
    )
#optimizer = FIRE(water)
#optimizer.run(fmax=.05)
E1 = water.get_potential_energy()
p1 = norm(water.get_dipole_moment())
Gsol = E1 - E0
Gcav = water.calc.get_cavity_formation_energy()
Acav = water.calc.get_cavity_surface_area()
Vcav = water.calc.get_cavity_volume()

parprint('G_cavity = %.6f eV = %.6f kcal / mol' % (Gcav, Gcav * mol / kcal))
parprint('G_sol = %.6f eV = %.6f kcal / mol' % (Gsol, Gsol * mol / kcal))
parprint('p_vac = %.6f e * Ang = %.6f D' % (p0, p0 / Debye))
parprint('p_sol = %.6f e * Ang = %.6f D' % (p1, p1 / Debye))
parprint('A_cavity = %.6f Ang ** 2' % (Acav, ))
parprint('V_cavity = %.6f Ang ** 3' % (Vcav, ))

assert Gsol < 0
assert  Gcav > 0
assert p1 > p0
equal(Gsol, -6.3 * kcal / mol, 1.0 * kcal / mol) #[1]
equal(Gcav, +5.7 * kcal / mol, 1.0 * kcal / mol) #[1]
equal(p1, 2.95 * Debye, .6 * Debye) #[2]
#equal(p1, 2.95 * Debye, .5 * Debye) #[2]
M = 18. / 1000. # kg / mol
rho = 1000. # kg / m ** 3
equal(Vcav, M / rho / mol * m ** 3, 3.)
#equal(Vcav, M / rho / mol * m ** 3, 1.)

#[1] D. A. Scherlis, J. F. Fattebert, F. Gygi, M. Cococcioni and N. Marzari
#    J. Chem. Phys. 124, 074103 (2006)

#[2] A. V. Gubskaya and P. G. Kusalik
#    J. Chem. Phys., Vol. 117, No. 11, 15 September 2002

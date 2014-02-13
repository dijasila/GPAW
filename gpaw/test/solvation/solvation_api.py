# XXX This test is a use case/acceptance test to help rewrite the api
# XXX and not included in the test suite.
# XXX TODO: make an example/documentation out of this test
# XXX       when the api has changed and the test passes

from ase.structure import molecule
from ase.units import Pascal, m, Bohr
from ase.data.vdw import vdw_radii
from gpaw.solvation import (
    # calculator
    SolvationGPAW,
    # cavities
    EffectivePotentialCavity,
    DensityCavity,
    # custom classes for the cavities
    Power12Potential,
    ADM12SmoothStep,
    ElDensity,
    SSS09Density,
    FG02SmoothStep,
    # dielectric
    LinearDielectric,
    CMDielectric,  # not used any more
    # non-el interactions
    SurfaceInteraction,
    VolumeInteraction,
    LeakedDensityInteraction,
    # surface and volume calculators
    GradientSurface,
    KB51Volume,
)
# poisson solver
from gpaw.solvation.poisson import ADM12PoissonSolver


# references for custom classes:
# KB51 = J. G. Kirkwood and F. P. Buff,
#        The Journal of Chemical Physics, vol. 19, no. 6, pp. 774--777, 1951
# FG02 = J.-L. Fattebert and F. Gygi,
#        Journal of Computational Chemistry, vol. 23, no. 6, pp. 662--666, 2002
# SSS09 = V. M. Sanchez, M. Sued, and D. A. Scherlis,
#         The Journal of Chemical Physics, vol. 131, no. 17, p. 174108, 2009
# ADM12 = O. Andreussi, I. Dabo, and N. Marzari,
#         The Journal of Chemical Physics, vol. 136, no. 6, p. 064102, 2012


# define some useful units (all api user units are ASE units!)
dyn_per_cm = 1e-3 * Pascal * m
Giga_Pascal = 1e9 * Pascal

# GPAW params (examples)
# ----------------------
xc = 'PBE'
h = 0.24
vac = 4.0

# general solvation params (examples)
# -----------------------------------
# electrostatic
epsinf = 78.36
# other interactions
gamma = 72. * dyn_per_cm  # surface tension
p = -0.1 * Giga_Pascal  # pressure
V_leak = 1.0  # V (interaction energy E = V_leak * [charge outside cavity])
# only for volume calculations respecting compressibility
T = 298.15  # K  (also used for Boltzmann distribution)
kappa_T = 4.53e-10 / Pascal

# effective potential cavity params (examples)
# --------------------------------------------
u0 = 0.180  # eV
vdw_radii = vdw_radii[:]
vdw_radii[1] = 1.09

# density cavity params (examples)
# --------------------------------
# ADM12
rhomin = 0.0001 / Bohr ** 3
rhomax = 0.0050 / Bohr ** 3
# FG02, SSS09
rho0 = 1.0 / Bohr ** 3
beta = 2.4


atoms = molecule('H2O')
atoms.center(vacuum=vac)
atomic_radii = [vdw_radii[n] for n in atoms.numbers]


# Cavity from 1 / r ** 12 effective potential
atoms.calc = SolvationGPAW(
    xc=xc, h=h,
    cavity=EffectivePotentialCavity(
        effective_potential=Power12Potential(atomic_radii=atomic_radii, u0=u0),
        temperature=T,
        surface_calculator=GradientSurface(),
        volume_calculator=KB51Volume(compressibility=kappa_T, temperature=T)
        ),
    dielectric=LinearDielectric(epsinf=epsinf),
    interactions=[
        SurfaceInteraction(surface_tension=gamma),
        VolumeInteraction(pressure=p),
        LeakedDensityInteraction(voltage=V_leak)
        ]
    )
print 'E', atoms.get_potential_energy()
print 'V', atoms.calc.get_cavity_volume()
print 'A', atoms.calc.get_cavity_surface()
print 'F', atoms.get_forces()


# Cavity from electron density a la ADM12
atoms.calc = SolvationGPAW(
    xc=xc, h=h,
    poissonsolver=ADM12PoissonSolver(),
    cavity=DensityCavity(
        density=ElDensity(),
        smooth_step=ADM12SmoothStep(rhomin, rhomax, epsinf),
        surface_calculator=GradientSurface(),
        volume_calculator=KB51Volume(compressibility=kappa_T, temperature=T)
        ),
    dielectric=LinearDielectric(epsinf=epsinf),
    interactions=[
        SurfaceInteraction(surface_tension=gamma),
        VolumeInteraction(pressure=p),
        LeakedDensityInteraction(voltage=V_leak)
        ]
    )
print 'E', atoms.get_potential_energy()
print 'V', atoms.calc.get_cavity_volume()
print 'A', atoms.calc.get_cavity_surface()
print 'F', atoms.get_forces()


# Cavity from fake electron density a la SSS09
atoms.calc = SolvationGPAW(
    xc=xc, h=h,
    cavity=DensityCavity(
        density=SSS09Density(atomic_radii=atomic_radii),
        smooth_step=FG02SmoothStep(rho0, beta),
        surface_calculator=GradientSurface(),
        volume_calculator=KB51Volume(compressibility=kappa_T, temperature=T)
        ),
    dielectric=LinearDielectric(epsinf=epsinf),
    interactions=[
        SurfaceInteraction(surface_tension=gamma),
        VolumeInteraction(pressure=p),
        LeakedDensityInteraction(voltage=V_leak)
        ]
    )
print 'E', atoms.get_potential_energy()
print 'V', atoms.calc.get_cavity_volume()
print 'A', atoms.calc.get_cavity_surface()
print 'F', atoms.get_forces()

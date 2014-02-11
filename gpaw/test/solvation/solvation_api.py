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
    BoltzmannDistributionFunction,
    # dielectric
    LinearDielectric,
    CMDielectric,  # not used any more
    # non-el interactions
    SurfaceInteraction,
    VolumeInteraction,
    LeakedDensityInteraction,
    # surface and volume calculators
    GeneralizedQuantumSurface,
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
E_leak = 1.0  # eV (energy for one electron to leak outside the cavity)
# only for volume calculations respecting compressibility
T = 298.15  # K  (also used for Boltzmann distribution)
kappa_T = 4.53e-10 / Pascal

# effective potential cavity params (examples)
# --------------------------------------------
u0 = 0.180  # eV
vdw_radii = vdw_radii[:]
vdw_radii[1] = 1.09
delta_pot = 1e-6  # eV in this case, numerical param for surface

# density cavity params (examples)
# --------------------------------
delta_dens = 1e-6 / Bohr ** 3  # numerical param for surface
# ADM12
rhomin = 0.0001 / Bohr ** 3
rhomax = 0.0050 / Bohr ** 3
# FG02, SSS09
rho0 = 1.0 / Bohr ** 3
beta = 2.4


atoms = molecule('H2O')
atoms.center(vacuum=vac)


# Cavity from 1 / r ** 12 effective potential
atoms.calc = SolvationGPAW(
    xc=xc, h=h,
    cavity=EffectivePotentialCavity(
        effective_potential=Power12Potential(vdw_radii=vdw_radii, u0=u0),
        distribution_function=BoltzmannDistributionFunction(temperature=T),
        surface_calculator=GeneralizedQuantumSurface(delta=delta_pot),
        volume_calculator=KB51Volume(compressibility=kappa_T, temperature=T)
        ),
    dielectric=LinearDielectric(epsinf=epsinf),
    interactions=[
        SurfaceInteraction(surface_tension=gamma),
        VolumeInteraction(pressure=p),
        LeakedDensityInteraction(charging_energy=E_leak)
        ]
    )
print atoms.get_potential_energy()
print atoms.get_forces()


# Cavity from electron density a la ADM12
atoms.calc = SolvationGPAW(
    xc=xc, h=h,
    poissonsolver=ADM12PoissonSolver(),
    cavity=DensityCavity(
        density=ElDensity(),
        smooth_step=ADM12SmoothStep(rhomin, rhomax, epsinf),
        surface_calculator=GeneralizedQuantumSurface(delta=delta_dens),
        volume_calculator=KB51Volume(compressibility=kappa_T, temperature=T)
        ),
    dielectric=LinearDielectric(epsinf=epsinf),
    interactions=[
        SurfaceInteraction(surface_tension=gamma),
        VolumeInteraction(pressure=p),
        LeakedDensityInteraction(charging_energy=E_leak)
        ]
    )
print atoms.get_potential_energy()
print atoms.get_forces()


# Cavity from fake electron density a la SSS09
atoms.calc = SolvationGPAW(
    xc=xc, h=h,
    cavity=DensityCavity(
        density=SSS09Density(),
        smooth_step=FG02SmoothStep(rho0, beta),
        surface_calculator=GeneralizedQuantumSurface(delta=delta_dens),
        volume_calculator=KB51Volume(compressibility=kappa_T, temperature=T)
        ),
    dielectric=LinearDielectric(epsinf=epsinf),
    interactions=[
        SurfaceInteraction(surface_tension=gamma),
        VolumeInteraction(pressure=p),
        LeakedDensityInteraction(charging_energy=E_leak)
        ]
    )
print atoms.get_potential_energy()
print atoms.get_forces()

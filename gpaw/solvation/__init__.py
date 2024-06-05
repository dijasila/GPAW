"""The gpaw.solvation package.

This packages extends GPAW to be used with different
continuum solvent models.
"""
import numpy as np
from ase.data.vdw import vdw_radii
from ase.data.vdw_alvarez import vdw_radii as vdw_radii_alvarez

from gpaw.solvation.calculator import SolvationGPAW
from gpaw.solvation.cavity import (EffectivePotentialCavity,
                                   Power12Potential,
                                   ElDensity,
                                   SSS09Density,
                                   ADM12SmoothStepCavity,
                                   FG02SmoothStepCavity,
                                   GradientSurface,
                                   KB51Volume)
from gpaw.solvation.dielectric import (LinearDielectric,
                                       CMDielectric)
from gpaw.solvation.interactions import (SurfaceInteraction,
                                         VolumeInteraction,
                                         LeakedDensityInteraction)


def default_vdw_radii(atoms):
    mod_radii = vdw_radii.copy()
    mod_radii[1] = 1.09

    radii = []
    for n in atoms.numbers:
        if np.isfinite(mod_radii[n]):
            radii.append(mod_radii[n])
        else:
            radii.append(vdw_radii_alvarez[n])
    return radii


def get_HW14_water_kwargs():
    """Return kwargs for initializing a SolvationGPAW instance.

    Parameters for water as a solvent as in
    A. Held and M. Walter, J. Chem. Phys. 141, 174108 (2014).
    """
    from ase.units import Pascal, m
    u0 = 0.180
    epsinf = 78.36
    st = 18.4 * 1e-3 * Pascal * m
    T = 298.15

    kwargs = {
        'cavity': EffectivePotentialCavity(
            effective_potential=Power12Potential(default_vdw_radii, u0),
            temperature=T,
            surface_calculator=GradientSurface()
        ),
        'dielectric': LinearDielectric(epsinf=epsinf),
        'interactions': [SurfaceInteraction(surface_tension=st)]
    }
    return kwargs


__all__ = ['SolvationGPAW',
           'EffectivePotentialCavity',
           'Power12Potential',
           'ElDensity',
           'SSS09Density',
           'ADM12SmoothStepCavity',
           'FG02SmoothStepCavity',
           'GradientSurface',
           'KB51Volume',
           'LinearDielectric',
           'CMDielectric',
           'SurfaceInteraction',
           'VolumeInteraction',
           'LeakedDensityInteraction',
           'get_HW14_water_kwargs']

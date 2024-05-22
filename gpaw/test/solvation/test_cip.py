from gpaw.solvation.sjm import SJM, SJMPower12Potential
import numpy as np
from ase.build import fcc111
from gpaw import FermiDirac

# Import solvation modules
from ase.data.vdw import vdw_radii
from gpaw.solvation import (
    EffectivePotentialCavity,
    LinearDielectric,
    GradientSurface,
    SurfaceInteraction)


def test_cip():
    # Solvent parameters
    u0 = 0.180  # eV
    epsinf = 78.36  # Dielectric constant of water at 298 K
    gamma = 0.00114843767916  # 18.4*1e-3 * Pascal* m
    T = 298.15   # K

    def atomic_radii(atoms):
        return [vdw_radii[n] for n in atoms.numbers]

    # Structure is created
    atoms = fcc111('Au', size=(1, 1, 4))
    atoms.cell[2][2] = 18
    atoms.translate([0, 0, 6 - min(atoms.positions[:, 2])])

    # SJM parameters
    potential = 4.5
    tol = 0.02
    sj_fermi = {'excess_electrons': -0.,
                'jelliumregion': {'top': 17.9},
                'tol': tol,
                'method': 'Fermi',
                'dirichlet': True,
                'cip': {'filter': 10,
                        'autoinner': {'nlayers': 4, 
                                      'threshold': 0.01}}
                }

    convergence = {
        'energy': 0.05 / 8.,
        'density': 1e-4,
        'eigenstates': 1e-4}

    # Calculator
    calc = SJM(mode='lcao',
               sj=sj_fermi,
               txt='sjm.test',
               gpts=(8, 8, 48),
               kpts=(2, 2, 1),
               xc='PBE',
               convergence=convergence,
               occupations=FermiDirac(0.1),
               cavity=EffectivePotentialCavity(
                   effective_potential=SJMPower12Potential(atomic_radii, u0),
                   temperature=T,
                   surface_calculator=GradientSurface()),
               dielectric=LinearDielectric(epsinf=epsinf),
               interactions=[SurfaceInteraction(surface_tension=gamma)])

    # Run the calculation using the Fermi level as the electrode potential
    atoms.calc = calc
    atoms.get_potential_energy()
    phi_pzc = calc.get_inner_potential(calc.atoms)
    mu_pzc = -calc.get_electrode_potential(sj_fermi['method'])
    print(mu_pzc)
    assert np.isclose(mu_pzc, -4.39, 1e-1, 1e-1)

    # Reset for CIP using computed potential of zero charge and fermi level 
    potential = 4.5
    tol = 0.02
    sj_cip = {'target_potential': potential,
              'excess_electrons': -0.1,
              'jelliumregion': {'top': 17.9},
              'tol': tol,
              'method': 'CIP',
              'dirichlet': True,
              'cip': {'autoinner': {'nlayers': 4, 
                      'threshold': 0.01},
                      'inner_region': None,
                      'mu_pzc': mu_pzc,
                      'phi_pzc': phi_pzc,
                      'filter': 10}
              }

    calc = SJM(mode='lcao',
               sj=sj_cip,
               txt='sjm.test',
               gpts=(8, 8, 48),
               kpts=(2, 2, 1),
               xc='PBE',
               convergence=convergence,
               occupations=FermiDirac(0.1),
               cavity=EffectivePotentialCavity(
                   effective_potential=SJMPower12Potential(atomic_radii, u0),
                   temperature=T,
                   surface_calculator=GradientSurface()),
               dielectric=LinearDielectric(epsinf=epsinf),
               interactions=[SurfaceInteraction(surface_tension=gamma)])

    atoms.calc = calc
    atoms.get_potential_energy()
    assert abs(calc.get_electrode_potential(sj_cip['method'], calc.atoms) - potential) < tol

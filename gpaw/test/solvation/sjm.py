import sys,os
from gpaw.solvation.sjm import  SJM, SJM_Power12Potential

from ase.visualize import view
from gpaw import FermiDirac
from ase.io import read
#Import solvation modules
from ase.data.vdw import vdw_radii
from gpaw.solvation import (
    SolvationGPAW,
    EffectivePotentialCavity,
    LinearDielectric,
    GradientSurface,
    SurfaceInteraction
)
from ase.build import fcc111
#Solvent parameters
u0=0.180 #eV
epsinf = 78.36 #dielectric constant of water at 298 K
gamma =0.00114843767916 #     18.4*1e-3 * Pascal*m
T=298.15   #K
atomic_radii = lambda atoms: [vdw_radii[n] for n in atoms.numbers]

#atoms=read('test_system.traj')
atoms=fcc111('Au',size=(1,1,3))
atoms.center(axis=2,vacuum=8)
atoms.translate([0,0,-2])

potential=3.0
ne=-0.01
dpot=0.01

calc= SJM(  symmetry={'do_not_symmetrize_the_density': True},
                 doublelayer={'upper_limit':19.5},
                 potential=potential,
                 dpot=dpot,
                 ne=ne,
                 verbose=True,

	         gpts =  (8, 8, 64),
                 #h=0.3,
                 poissonsolver={'dipolelayer':'xy'},
                 kpts = (1,1,1),
	         xc = 'PBE',
                 mode='lcao',
                 basis='szp(dzp)',
                 txt='out_no_wat.txt',
	         spinpol = False,
                 maxiter=1000,
	         occupations = FermiDirac(0.1),
                 cavity = EffectivePotentialCavity (
                     effective_potential = SJM_Power12Potential (atomic_radii, u0),#,H2O_layer=True),
                     temperature=T,
                     surface_calculator=GradientSurface ()),
                 #convergence={'energy': 0.005,  # eV / electron
                 #     'density': 1.0e-3,
                 #      'eigenstates': 4.0e-5,  # eV^2 / electron
                 #       'bands': 'occupied',
                 #        'forces': float('inf') # eV / Ang Max
                 #        },
                 dielectric = LinearDielectric (epsinf=epsinf),
                 interactions = [SurfaceInteraction (surface_tension=gamma)])

atoms.set_calculator(calc)
#atoms.calc.atoms=atoms
atoms.get_potential_energy()
assert abs(calc.get_electrode_potential() - potential) < dpot
elpot = calc.get_electrostatic_potential().mean(0).mean(0)
assert abs(elpot[0] - elpot[5]) < 1e-6
#atoms.get_forces()


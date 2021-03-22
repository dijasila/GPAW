import os
from gpaw.solvation.sjm import SJM,SJMPower12Potential
from gpaw import FermiDirac
from ase.io import read
from ase.data.vdw import vdw_radii
from gpaw.solvation import (
    EffectivePotentialCavity,
    LinearDielectric,
    GradientSurface,
    SurfaceInteraction
)

#Solvent parameters
u0=0.180 #eV
epsinf = 78.36 #dielectric constant of water at 298 K
gamma =0.00114843767916 #     18.4*1e-3 * Pascal*m
T=298.15   #K

def atomic_radii(atoms):
        return [vdw_radii[n] for n in atoms.numbers]

atoms=read(os.getcwd()+'/test_system.traj')


sj = {'target_potential': 4.5,
      'excess_electrons': 0.124,
      'jelliumregion': {'top': 14.},
      'tol': 0.005}

calc = SJM(sj=sj,
           gpts =  (48, 32, 88),
           #gpts =  (24, 16, 48),
           kpts = (2,2,1),
           xc ='PBE',
           txt='sjm.txt',
           occupations = FermiDirac(0.1),
           cavity = EffectivePotentialCavity (
               effective_potential = SJMPower12Potential (atomic_radii, u0,
                                                          H2O_layer=True),
               temperature=T,
               surface_calculator=GradientSurface ()),
           dielectric = LinearDielectric (epsinf=epsinf),
           interactions = [SurfaceInteraction (surface_tension=gamma)])

atoms.set_calculator(calc)
E=[]

for pot in [4.5,None,4.3,4.5]:
    if pot is None:
        calc.set(sj={'excess_electrons': 0.2,'target_potential':None})
    else:
        calc.set(sj={'target_potential': pot})
    E.append(atoms.get_potential_energy())

    if pot is None:
        assert abs(calc.wfs.nvalence - calc.setups.nvalence - 0.2) < 1e-4
    else:
        assert abs(calc.get_electrode_potential() - pot) < 0.005

assert abs(E[0]-E[-1])  < 1e-2

calc.write('sjm.gpw')
from gpaw import restart
atoms,calc=restart('sjm.gpw', Class=SJM)
calc.set(txt='sjm2.txt')
assert abs(calc.get_electrode_potential() - 4.5) < 0.002

calc.set(sj={'tol':0.002})
atoms.get_potential_energy()
assert abs(calc.get_electrode_potential() - 4.5) < 0.002

calc.set(sj={'jelliumregion':{'top':13},'tol':0.01})
atoms.get_potential_energy()
assert abs(calc.get_electrode_potential() - 4.5) < 0.01

calc.set(sj={'jelliumregion':{'thickness':2}})
atoms.get_potential_energy()
assert abs(calc.get_electrode_potential() - 4.5) < 0.01

from ase.optimize import BFGS
qn=BFGS(atoms,logfile='relax_sim_set.log',maxstep=0.05)
qn.run(fmax=0.05,steps=2)
assert abs(calc.get_electrode_potential() - 4.5) < 0.01


from ase import Atoms
from gpaw import GPAW
from gpaw.fdtd import FDTDPoissonSolver, PermittivityPlus, PolarizableMaterial, PolarizableSphere
from gpaw.mpi import world
from gpaw.tddft import TDDFT, photoabsorption_spectrum, units
from gpaw.test import equal
import numpy as np

# Whole simulation cell (Angstroms)
large_cell = [20, 20, 30];

# Quantum subsystem
atom_center = np.array([10.0, 10.0, 20.0]);
atoms = Atoms('Na2', [atom_center + [0.0, 0.0, -1.50],
                      atom_center + [0.0, 0.0, +1.50]]);

# Permittivity file
if world.rank==0:
    fo = open("ed.txt", "wb")
    fo.writelines(["1.20 0.20 25.0"])
    fo.close()
world.barrier()

# Classical subsystem
classical_material = PolarizableMaterial()
sphere_center = np.array([10.0, 10.0, 10.0]);
classical_material.add_component(PolarizableSphere(vector1 = sphere_center,
                                                   radius1 = 5.0,
                                                   permittivity = PermittivityPlus('ed.txt')))

# Combined Poisson solver
poissonsolver = FDTDPoissonSolver(classical_material  = classical_material,
                                  qm_spacing          = 0.40,
                                  cl_spacing          = 0.40*4,
                                  cell                = large_cell,
                                  remove_moments      = (4, 1),
                                  communicator        = world,
                                  debug_plots         = 0,
                                  potential_coupler   = 'Refiner',
                                  coupling_level      = 'both',
                                  tag                 = 'test')
poissonsolver.set_calculation_mode('iterate')

# Combined system
atoms.set_cell(large_cell)
atoms, qm_spacing, gpts = poissonsolver.cut_cell(atoms,
                                                 vacuum=2.50)
# Initialize GPAW
gs_calc = GPAW(gpts          = gpts,
               eigensolver   = 'cg',
               nbands        = -1,
               poissonsolver = poissonsolver);
atoms.set_calculator(gs_calc)

# Ground state
energy = atoms.get_potential_energy()

# Save state
gs_calc.write('gs.gpw', 'all')

# Initialize TDDFT and FDTD
kick = [0.0, 0.0, 1.0e-3]
time_step = 10.0
max_time = 100 # 0.1 fs

td_calc = TDDFT('gs.gpw')
td_calc.initialize_FDTD(poissonsolver, 'dmCl.dat')
td_calc.absorption_kick(kick_strength=kick)

# Propagate TDDFT and FDTD
td_calc.propagate(time_step,
                  max_time/time_step,
                  'dm.dat',
                  'td.gpw')

# Finalize
poissonsolver.finalize_propagation()

# Test
ref_dipole_moment_z = -0.0494828968091
tol = 0.0001
equal(td_calc.get_dipole_moment()[2], ref_dipole_moment_z, tol)


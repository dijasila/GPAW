from ase import Atoms
from ase.units import Hartree
from gpaw import GPAW
from gpaw.fdtd import FDTDPoissonSolver, PermittivityPlus, PolarizableMaterial, PolarizableSphere, _eps0_au
from gpaw.tddft import TDDFT, photoabsorption_spectrum
import numpy as np
import pylab as plt

# Nanosphere radius (Angstroms)
radius = 50.0

# Whole simulation cell (Angstroms)
large_cell = np.array([3.*radius, 3.*radius, 3.*radius])

# Permittivity of Gold (from  J. Chem. Phys. 137, 074113 (2012))
gold = [[0.2350, 0.1551,  95.62],
        [0.4411, 0.1480, -12.55],
        [0.7603,  1.946, -40.89],
        [1.161,   1.396,  17.22],
        [2.946,   1.183,  15.76],
        [4.161,   1.964,  36.63],
        [5.747,   1.958,  22.55],
        [7.912,   1.361,  81.04]]

# Initialize classical material
classical_material = PolarizableMaterial()

# Classical nanosphere
classical_material.add_component(
        PolarizableSphere(center = 0.5*large_cell,
                          radius=radius,
                          permittivity = PermittivityPlus(data=gold))
        )

# Combined Poisson solver
poissonsolver = FDTDPoissonSolver(classical_material  = classical_material,
                                  qm_spacing          = 1.00,
                                  cl_spacing          = 7.50,
                                  remove_moments      = (1, 4),
                                  cell                = large_cell)
poissonsolver.set_calculation_mode('iterate')

# Dummy quantum system
atoms = Atoms('H', positions=[0.5*large_cell], cell=large_cell)
atoms, qm_spacing, gpts = poissonsolver.cut_cell(atoms, vacuum=6.0)
del atoms[:]

# Initialize GPAW
gs_calc = GPAW(gpts          = gpts,
               eigensolver   = 'cg',
               nbands        = 1,
               poissonsolver = poissonsolver)
atoms.set_calculator(gs_calc)

# Get the ground state
energy = atoms.get_potential_energy()

# Save the ground state
gs_calc.write('gs.gpw', 'all')

# Initialize TDDFT and QSFDTD
td_calc = TDDFT('gs.gpw')
td_calc.absorption_kick(kick_strength=[0.001, 0.000, 0.000])
td_calc.hamiltonian.poisson.set_kick( [0.001, 0.000, 0.000])

# Propagate TDDFT and FDTD
td_calc.propagate(10.0, 1000, 'dm.dat')

# Spectrum
photoabsorption_spectrum('dmCl.dat', 'specCl.dat', width=0.0)


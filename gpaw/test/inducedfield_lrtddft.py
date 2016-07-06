import numpy as np
from ase import Atoms
from gpaw import GPAW
from gpaw.lrtddft import LrTDDFT
from gpaw.inducedfield.inducedfield_lrtddft import LrTDDFTInducedField
from gpaw.poisson import PoissonSolver
from gpaw.test import equal

# 0) PoissonSolver
poissonsolver = PoissonSolver(eps=1e-20)

# 1) Ground state calculation with empty states
atoms = Atoms(symbols='Na2',
              positions=[(0, 0, 0), (3.0, 0, 0)],
              pbc=False)
atoms.center(vacuum=3.0)

calc = GPAW(nbands=20, h=0.6, setups={'Na': '1'}, poissonsolver=poissonsolver)
atoms.set_calculator(calc)
energy = atoms.get_potential_energy()
calc.write('na2_gs_casida.gpw', mode='all')

# 2) Casida calculation
calc = GPAW('na2_gs_casida.gpw', poissonsolver=poissonsolver)
istart = 0
jend = 20
lr = LrTDDFT(calc, xc='LDA', istart=istart, jend=jend)
lr.diagonalize()
lr.write('na2_lr.dat.gz')

# 3) Calculate induced field
frequencies = [1.0, 2.08]  # Frequencies of interest in eV
folding = 'Gauss'          # Folding function
width = 0.1                # Line width for folding in eV
kickdir = 0                # Kick field direction 0, 1, 2 for x, y, z
ind = LrTDDFTInducedField(paw=calc, lr=lr, frequencies=frequencies,
                          folding=folding, width=width, kickdir=kickdir)
ind.calculate_induced_field(gridrefinement=2, from_density='comp',
                            poisson_eps=1e-20)

# Test
tol = 1e-4
val1 = ind.fieldgd.integrate(ind.Ffe_wg[0])
val2 = ind.fieldgd.integrate(np.abs(ind.Fef_wvg[0][0]))
val3 = ind.fieldgd.integrate(np.abs(ind.Fef_wvg[0][1]))
val4 = ind.fieldgd.integrate(np.abs(ind.Fef_wvg[0][2]))
val5 = ind.fieldgd.integrate(ind.Ffe_wg[1])
val6 = ind.fieldgd.integrate(np.abs(ind.Fef_wvg[1][0]))
val7 = ind.fieldgd.integrate(np.abs(ind.Fef_wvg[1][1]))
val8 = ind.fieldgd.integrate(np.abs(ind.Fef_wvg[1][2]))

def relative_equal(x, y, tol):
    return equal(x/y, 1.0, tol)

relative_equal(val1, 3175.7735779847876, tol)
relative_equal(val2, 1700.4802379802722, tol)
relative_equal(val3, 1187.2584619412758, tol)
relative_equal(val4, 1187.2584619407216, tol)
relative_equal(val5, 10957.19614721598, tol)
relative_equal(val6, 6574.846859393127, tol)
relative_equal(val7, 4589.770191548621, tol)
relative_equal(val8, 4589.770191545197, tol)

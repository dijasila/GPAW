import numpy as np
from ase import Atoms
from gpaw.wavefunctions.pw import PW
from ase.units import Bohr, Hartree
from gpaw import GPAW, FermiDirac, Mixer
from gpaw.mpi import world
from gpaw.xc.kernel import XCNull
from gpaw.xc import XC

sym = 'Ne'
nelectrons = 16

# We need lots of ghost atoms, because if we were to add too few,
# the occupation initialization procedure will recurse infinitely.
# And there's no way to override it!  *grumble* *grumble*
system = Atoms('%d%s' % (nelectrons // 2, sym),
               cell=(1.7, 1.7, 2.), pbc=1)
system.rattle(stdev=1.0)

class NoInteractionPoissonSolver:
    description = 'No interaction'
    relax_method = 0
    nn = 1
    def get_stencil(self):
        return self.nn
    def solve(self, phi, rho, charge):
        return 0
    def set_grid_descriptor(self, gd):
        pass
    def initialize(self):
        pass


calc = GPAW(mode=PW(250),
            nbands=24,
            convergence=dict(density=1e-5),
            occupations=FermiDirac(0.1),
            xc=XC(XCNull()),
            charge=-nelectrons,
            kpts=(4,1,1),
            dtype=complex, # XXXXXXXXX must be complex
            # There's an error right now if it's real
            mixer=Mixer(0.1, 5, 10.0),
            setups={sym: 'ghost'},
            usesymm=False,
            basis='sz(dzp)',
            # Can also use JelliumPoissonSolver
            poissonsolver=NoInteractionPoissonSolver())

system.set_calculator(calc)
system.get_potential_energy()
sigma_cv = calc.wfs.get_kinetic_stress()
E1 = calc.hamiltonian.Etot

deps = 1e-5

cell = system.cell.copy()
dEdx_fd_v = np.zeros(3)
calc.set(txt=None)
for v in range(3):
    system.cell[:, :] = cell
    # if adding atoms, remember to scale atomic positions as well
    system.cell[v, v] *= (1.0 + deps)
    system.get_potential_energy()
    E2 = calc.hamiltonian.Etot
    dEdx_fd_v[v] = (E2 - E1) / deps

dEdx_calc_cv = sigma_cv # * Hartree
maxerr = np.abs(dEdx_calc_cv.diagonal() - dEdx_fd_v).max()

world.barrier()
if world.rank == 0:
    print 'dEdx fd', dEdx_fd_v
    print 'dEdx calc'
    print np.array2string(dEdx_calc_cv,
                          precision=4, suppress_small=1)
    print 'Maxerr', maxerr
world.barrier()

assert maxerr < 5e-4, maxerr

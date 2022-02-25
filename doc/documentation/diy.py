from ase import Atoms
from ase.units import Bohr
from gpaw.setup import Setups
from gpaw.new.xc import XCFunctional
from gpaw.new.density import Density
from gpaw.core import UniformGrid
from gpaw.new.basis import create_basis
from gpaw.new.symmetry import create_symmetries_object
from gpaw.new.brillouin import BZPoints
from gpaw.core.atom_arrays import AtomDistribution
from gpaw.mpi import world

atoms = Atoms('H2',
              positions=[(0, 0, 0), (0, 0, 0.75)],
              cell=[2, 2, 3],
              pbc=True)

xc = XCFunctional('LDA')
setups = Setups([1, 1], {}, {}, xc)
grid = UniformGrid(cell=atoms.cell / Bohr,
                   size=[10, 10, 15],
                   pbc=[True, True, True])
symmetries = create_symmetries_object(atoms)
ibz = symmetries.reduce(BZPoints([[0, 0, 0]]))
basis = create_basis(
    ibz,
    nspins=1,
    pbc_c=atoms.pbc,
    grid=grid,
    setups=setups,
    dtype=float,
    fracpos_ac=atoms.get_scaled_positions())
nct_R = grid.zeros()
atomdist = AtomDistribution([0, 0], world)
density = Density.from_superposition(grid, nct_R, atomdist, setups, basis)
pot_calc = ...
potential = ...
wfs = ...
ibzwfs = ...
state = ...
scf_loop = ...
for _ in scf_loop.iterate(state, pot_calc):
    ...

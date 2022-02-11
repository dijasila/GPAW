from ase import Atoms
from gpaw.setup import Setups
from gpaw.new.xc import XCFunctional
from gpaw.new.density import Density

atoms = Atoms('H2',
              positions=[(0, 0, 0), (0, 0, 0.75)],
              cell=[2, 2, 3],
              pbc=True)
if 0:
    atoms.calc = GPAW(mode='pw')
    e1 = atoms.get_potential_energy()

xc = XCFunctional()
setups = Setups([1, 1])
grid = UniformGrid()
density = Density.from_superposition()
pot_calc = ...
potential = ...
wfs =
ibzwfs =
state = ...
scf_loop = ...
for ctx in scf_loop.iterate(state, pot_calc):
    print(ctx)
e2 = ...
assert abs(e2 - e1) < 1e-6

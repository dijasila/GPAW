from gpaw.new.density import Density
from gpaw.new.wave_functions import IBZWaveFunctions
from gpaw.new.configuration import CalculationConfiguration
from gpaw.new.scf import SCFLoop
from gpaw.new.davidson import Davidson


def calculate_ground_state(atoms, params, log):
    from gpaw.new.potential import PotentialCalculator
    mode = params.mode
    cfg = CalculationConfiguration.from_parameters(atoms, params)
    setups = cfg.setups

    density = Density.from_superposition(cfg, params.charge, params.hund)

    poisson_solver = mode.create_poisson_solver(cfg.grid2,
                                                params.poissonsolver)
    if mode.name == 'pw':
        pw = mode.create_plane_waves(cfg.grid)
        basis = pw
    else:
        basis = cfg.grid

    pot_calc = PotentialCalculator(basis, cfg, poisson_solver)
    potential = pot_calc.calculate(density)

    nbands = params.nbands(setups, density.charge, cfg.magmoms,
                           mode.name == 'lcao')

    if params.random:
        ibzwfs = IBZWaveFunctions.from_random_numbers(cfg, nbands)
    else:
        ...

    hamiltonian = mode.create_hamiltonian_operator(basis)
    eigensolver = Davidson(nbands, basis, cfg.band_comm, **params.eigensolver)
    mixer = ...

    scf = SCFLoop(ibzwfs, density, potential, hamiltonian, pot_calc,
                  eigensolver, mixer)

    for _ in scf.iconverge(params.convergence):
        ...

    return


class DrasticChangesError(Exception):
    """Atoms have changed so much that a fresh start is needed."""


def compare_atoms(a1, a2):
    if len(a1.numbers) != len(a2.numbers) or (a1.numbers != a2.numbers).any():
        raise DrasticChangesError
    if (a1.cell - a2.cell).abs().max() > 0.0:
        raise DrasticChangesError
    if (a1.pbc != a2.pbc).any():
        raise DrasticChangesError
    if (a1.positions - a2.positions).abs().max() > 0.0:
        return {'positions'}
    return set()


class Calculation:
    def __init__(self, atoms, parameters):
        self.atoms = atoms
        # self.parameters = parameters
        self.results = {}

    @classmethod
    def read(self, filename, log, parallel):
        ...

    def calculate_property(self, atoms, prop):
        changes = compare_atoms(self.atoms, atoms)
        if changes:
            self.recompute_ground_state()

        if prop in self.results:
            return self.results[prop]

        if prop == 'forces':
            self.calculate_forces()
        else:
            1 / 0

        return self.results[prop]

from typing import Dict, Any
from gpaw.mpi import MPIComm, world, Parallelization
from pathlib import Path
from contextlib import contextmanager


class Logger:
    def __init__(self, filename='-', comm=None):
        comm = comm or world

        if comm.rank > 0 or filename is None:
            self.fd = open(os.devnull, 'w')
            self.close_fd = True
        elif filename == '-':
            self.fd = sys.stdout
            self.close_fd = False
        elif isinstance(filename, (str, Path)):
            self.fd = open(filename, 'w')
            self.close_fd = True
        else:
            self.fd = filename
            self.close_fd = False

        self._indent = ''

    def __del__(self):
        if self.close_fd:
            self.fd.close()

    def __call__(self, *args):
        self.fd.write(self._indent)
        print(*args, file=self.fd)

    @contextmanager
    def indent(self, text):
        self(text)
        self._indent += '  '
        yield
        self._indent = self._indent[:-2]


def GPAW(filename=None,
         *,
         txt='?',
         communicator=None,
         parallel=None,
         **parameters):

    if txt == '?' and filename:
        txt = None
    else:
        txt = '-'

    log = Logger(txt, communicator)

    parallel = {**parallel, 'world': communicator}

    if filename:
        assert not parameters
        calculation = Calculation.read(filename, log, parallel)
        return calculation.ase_calculator()

    return ASECalculator(parameters, parallel, log)


class ASECalculator:
    """This is the ASE-calculator frontend for doing a GPAW calculation."""
    def __init__(self,
                 parameters,
                 parallel,
                 log,
                 calculation=None):
        self.parameters = parameters
        self.calculation = calculation

    def calculate_property(self, atoms, prop: str):
        if self.calculation is None:
            self.calculation = calculate_ground_state(
                atoms, self.parameters, self.parallel, self.log)
        return self.calculation.calculate_property(atoms, prop)

    def get_potential_energy(self, atoms):
        return self.calculate_property('energy')


def calculate_ground_state(atoms, parameters, parallel, log):
    params = InputParameters(**paramaters)

    world = parallel['world']

    xc = params.xc

    d = parallel['domain']
    k = parallel['kpt']
    b = parallel['band']

    if isinstance(xc, HybridXC):
        d = world.size

    communicators = create_communicators(d, k, b, world)

    mode = params.mode

    grid = mode.create_uniform_grid(params.h, params.gpts, communicators['d'])

    if mode.name == 'pw':
        pw = mode.create_plane_waves(grid)
        filter = None
    else:
        filter = mode.create_filter(grid)

    setups = Setups(atoms.numbers,
                    params.setups,
                    params.basis,
                    xc, filter, self.world)


class Calculation:
    def __init__(self, atoms, parameters):
        self.atoms = atoms
        self.parameters = parameters
        self.results = {}


    def calculate_property(self, atoms, prop):
        changes = compare_atoms(self.atoms, atoms)
        if not changes:
            if prop in self.results:
                return self.results[prop]
            self.


default_parameters: Dict[str, Any] = {
    'mode': 'fd',
    'h': None,  # Angstrom
    'gpts': None,
    'kpts': [(0.0, 0.0, 0.0)],
    'nbands': None,
    'charge': 0,
    'magmoms': None,
    'symmetry': {'point_group': True,
                 'time_reversal': True,
                 'symmorphic': True,
                 'tolerance': 1e-7,
                 'do_not_symmetrize_the_density': None},  # deprecated
    'soc': None,
    'background_charge': None,
    'setups': {},
    'basis': {},
    'spinpol': None,
    'xc': 'LDA',

    'occupations': None,
    'poissonsolver': None,
    'mixer': None,
    'eigensolver': None,
    'reuse_wfs_method': 'paw',
    'external': None,
    'random': False,
    'hund': False,
    'maxiter': 333,
    'idiotproof': True,
    'convergence': {'energy': 0.0005,  # eV / electron
                    'density': 1.0e-4,  # electrons / electron
                    'eigenstates': 4.0e-8,  # eV^2 / electron
                    'bands': 'occupied'},
    'verbose': 0}  # deprecated


default_parallel: Dict[str, Any] = {
    'kpt': None,
    'domain': None,
    'band': None,
    'order': 'kdb',
    'stridebands': False,
    'augment_grids': False,
    'sl_auto': False,
    'sl_default': None,
    'sl_diagonalize': None,
    'sl_inverse_cholesky': None,
    'sl_lcao': None,
    'sl_lrtddft': None,
    'use_elpa': False,
    'elpasolver': '2stage',
    'buffer_size': None}



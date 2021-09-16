def GPAW(filename=None,
         *,
         txt='?',
         comunicator=world,
         parallel=None,
         **parameters):

    comm = communicator
    logger = Logger(txt, comm)
    if filename:
        assert not parameters
        calculation = Calculation.read(filename, logger, comm, parallel)
    else:
        calculation = None

    return ASECalculator(calculation, logger, comm,
                         parallel, parameters)


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


class ASECalculator:
    """This is the ASE-calculator frontend for doing a GPAW calculation."""
    def __init__(self):
        ...

    def get_potential_energy(self, atoms):
        ...

    def calculate(self):
        ...

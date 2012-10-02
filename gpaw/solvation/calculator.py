from gpaw import GPAW
from gpaw.solvation.hamiltonian import SolvationRealSpaceHamiltonian
from gpaw.solvation.parameters import SolvationInputParameters
from gpaw.solvation.io import read
from gpaw.solvation.parameters import update_parameters
from ase.units import Hartree, Bohr
from gpaw.occupations import MethfesselPaxton
from gpaw.wavefunctions.base import EmptyWaveFunctions


class SolvationGPAW(GPAW):
    """
    Subclass of gpaw.GPAW calculator with continuum solvent model

    There is one additional parameter 'solvation' compared to gpaw.GPAW,
    which is a dict mapping the energy contribution name to a dict of
    parameters for the contribution.

    The five possible contributions are 'el', 'rep', 'dis', 'cav' and 'tm',
    which stand for electronic, repulsion, dispersion, cavity
    and thermal motion (compare J. Tomasi, B. Mennucci, R. Cammi:
    Chem. Rev. 2005, 105, 2999-3093) and are all set up to behave like an
    unmodified gpaw.GPAW calculation by default.

    Each contribution has a 'mode' parameter.
    For possible modes and additional parameters see
    gpaw/solvation/contributions.py

    New modes for any of the contributions can be added to
    gpaw.solvation.contributions.CONTRIBUTIONS

    The energy contributions can be retrieved with the following methods:
    get_electrostatic_energy
    get_repulsion_energy
    get_dispersion_energy
    get_cavity_formation_energy
    get_thermal_motion_energy

    In addition, the cavity surface area and volume can be accessed via
    get_cavity_surface_area
    get_cavity_volume

    For usage examples refer to gpaw/test/solvation
    """

    input_parameters_class = SolvationInputParameters
    contribtext = {
        'el' : 'Electronic',
        'rep': 'Repulsion',
        'dis': 'Dispersion',
        'cav': 'Cavity',
        'tm' : 'Thermal Motion'
        }

    def initialize(self, atoms=None):
        sparams = self.input_parameters['solvation']
        def real_space_factory(*args, **kwargs):
            return SolvationRealSpaceHamiltonian(sparams, *args, **kwargs)

        def reciprocal_space_factory(*args, **kwargs):
            raise NotImplementedError(
                'SolvationGPAW does not support '
                'calculations in reciprocal space yet.'
                )

        self.real_space_hamiltonian_class = real_space_factory
        self.reciprocal_space_hamiltonian_class = reciprocal_space_factory
        return GPAW.initialize(self, atoms)

    def initialize_positions(self, atoms=None):
        spos_ac = GPAW.initialize_positions(self, atoms)
        self.hamiltonian.set_atoms(self.atoms)
        return spos_ac

    def set(self, **kwargs):
        update = kwargs.pop('solvation', None)
        GPAW.set(self, **kwargs)
        old = self.input_parameters['solvation']
        if update is not None:
            modified = update_parameters(old, update)
            if len(modified) > 0:
                #XXX TODO: differentiated reset
                self.initialized = False
                self.wfs.set_orthonormalized(False)
                self.scf = None
                self.density = None
                self.occupations = None
                self.hamiltonian = None
                self.wfs = EmptyWaveFunctions()

    def read(self, reader):
        return read(self, reader)

    def get_electrostatic_energy(self, atoms=None, force_consistent=False):
        self.calculate(atoms, converge=True)
        if force_consistent:
            # Free energy:
            return Hartree * self.hamiltonian.Eel
        else:
            # Energy extrapolated to zero Kelvin:
            if isinstance(self.occupations, MethfesselPaxton) and \
                   self.occupations.iter > 0:
                raise NotImplementedError(
                    'Extrapolation to zero width not implemeted for '
                    'Methfessel-Paxton distribution with order > 0.'
                    )
            return Hartree * (self.hamiltonian.Eel + 0.5 * self.hamiltonian.S)

    def get_repulsion_energy(self, atoms=None):
        return self._get_energy('Erep', atoms)

    def get_dispersion_energy(self, atoms=None):
        return self._get_energy('Edis', atoms)

    def get_cavity_formation_energy(self, atoms=None):
        return self._get_energy('Ecav', atoms)

    def get_thermal_motion_energy(self, atoms=None):
        return self._get_energy('Etm', atoms)

    def _get_energy(self, name, atoms):
        self.calculate(atoms, converge=True)
        return Hartree * getattr(self.hamiltonian, name)

    def get_cavity_surface_area(self):
        """
        returns cavity surface area in Ang ** 2
        """
        return self.hamiltonian.Acav * Bohr ** 2

    def get_cavity_volume(self):
        """
        returns cavity volume in Ang ** 3
        """
        return self.hamiltonian.Vcav * Bohr ** 3

    def print_parameters(self):
        GPAW.print_parameters(self)
        t = self.text
        p = self.input_parameters
        t()
        t('Solvation Parameters:')
        for contrib in 'el rep dis cav tm'.split():
            params = p['solvation'][contrib]
            t(self.contribtext[contrib])
            spc = str(max([len(key) for key in params]))
            for key, value in params.iteritems():
                t(('  %-' + spc + 's: %s') % (key, repr(value)))
        t()

    def print_all_information(self):
        t = self.text
        t('Solvation Cavity:')
        t('Volume      : %11.6f' % (self.hamiltonian.Vcav * Bohr ** 3, ))
        t('Surface Area: %11.6f' % (self.hamiltonian.Acav * Bohr ** 2, ))
        t()
        t('Solvation Energy Contributions:')
        contribs = 'rep dis cav tm el'.split()
        for contrib in contribs:
            E = Hartree * getattr(self.hamiltonian, 'E' + contrib)
            t('%-14s: %+11.6f' % (self.contribtext[contrib], E))
        GPAW.print_all_information(self)

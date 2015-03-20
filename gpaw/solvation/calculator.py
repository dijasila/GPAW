from gpaw import GPAW
from gpaw.solvation.hamiltonian import SolvationRealSpaceHamiltonian
from ase.units import Hartree, Bohr


class NIReciprocalSpaceHamiltonian:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            'SolvationGPAW does not support '
            'calculations in reciprocal space yet.'
        )


class SolvationGPAW(GPAW):
    """Subclass of gpaw.GPAW calculator with continuum solvent model.

    See also Section III of
    A. Held and M. Walter, J. Chem. Phys. 141, 174108 (2014).
    """

    reciprocal_space_hamiltonian_class = NIReciprocalSpaceHamiltonian

    def __init__(self, cavity, dielectric, interactions=None,
                 **gpaw_kwargs):
        """Constructor for SolvationGPAW class.

        Additional arguments not present in GPAW class:
        cavity       -- A Cavity instance.
        dielectric   -- A Dielectric instance.
        interactions -- A list of Interaction instances.
        """
        if interactions is None:
            interactions = []

        def real_space_hamiltonian_factory(*args, **kwargs):
            return SolvationRealSpaceHamiltonian(
                cavity, dielectric, interactions,
                *args, **kwargs
            )

        self.real_space_hamiltonian_class = real_space_hamiltonian_factory
        GPAW.__init__(self, **gpaw_kwargs)

    def initialize_positions(self, atoms=None):
        spos_ac = GPAW.initialize_positions(self, atoms)
        self.hamiltonian.update_atoms(self.atoms)
        return spos_ac

    def get_electrostatic_energy(self, atoms=None, force_consistent=False):
        """Return electrostatic part of the total energy.

        The electrostatic part consists of everything except
        the short-range interactions defined in the interactions list.

        See the get_potential_energy method for the meaning
        of the force_consistent option.
        """
        self.calculate(atoms, converge=True)
        if force_consistent:
            # Free energy:
            return Hartree * self.hamiltonian.Eel
        else:
            # Energy extrapolated to zero width:
            return Hartree * self._extrapolate_energy_to_zero_width(
                self.hamiltonian.Eel
            )

    def get_solvation_interaction_energy(self, subscript, atoms=None):
        """Return a specific part of the solvation interaction energy.

        The subscript parameter defines which part is to be returned.
        It has to match the value of a subscript attribute of one of
        the interactions in the interactions list.
        """
        self.calculate(atoms, converge=True)
        return Hartree * getattr(self.hamiltonian, 'E_' + subscript)

    def get_cavity_volume(self, atoms=None):
        """Return the cavity volume in Angstrom ** 3.

        In case no volume calculator has been set for the cavity, None
        is returned.
        """
        self.calculate(atoms, converge=True)
        V = self.hamiltonian.cavity.V
        return V and V * Bohr ** 3

    def get_cavity_surface(self, atoms=None):
        """Return the cavity surface area in Angstrom ** 2.

        In case no surface calculator has been set for the cavity,
        None is returned.
        """
        self.calculate(atoms, converge=True)
        A = self.hamiltonian.cavity.A
        return A and A * Bohr ** 2

    def print_parameters(self):
        GPAW.print_parameters(self)
        t = self.text
        t()

        def ul(s, l):
            t(s)
            t(l * len(s))

        ul('Solvation Parameters:', '=')
        ul('Cavity:', '-')
        t('type: %s' % (self.hamiltonian.cavity.__class__, ))
        self.hamiltonian.cavity.print_parameters(t)
        t()
        ul('Dielectric:', '-')
        t('type: %s' % (self.hamiltonian.dielectric.__class__, ))
        self.hamiltonian.dielectric.print_parameters(t)
        t()
        for ia in self.hamiltonian.interactions:
            ul('Interaction:', '-')
            t('type: %s' % (ia.__class__, ))
            t('subscript: %s' % (ia.subscript, ))
            ia.print_parameters(t)
            t()

    def print_all_information(self):
        t = self.text
        t()
        t('Solvation Energy Contributions:')
        for ia in self.hamiltonian.interactions:
            E = Hartree * getattr(self.hamiltonian, 'E_' + ia.subscript)
            t('%-14s: %+11.6f' % (ia.subscript, E))
        Eel = Hartree * getattr(self.hamiltonian, 'Eel')
        t('%-14s: %+11.6f' % ('el', Eel))
        GPAW.print_all_information(self)

    def write(self, *args, **kwargs):
        raise NotImplementedError(
            'IO is not implemented yet for SolvationGPAW!'
        )

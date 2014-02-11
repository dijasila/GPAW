from gpaw import GPAW
from gpaw.solvation.hamiltonian import SolvationRealSpaceHamiltonian
from ase.units import Hartree
from gpaw.occupations import MethfesselPaxton


class NIReciprocalSpaceHamiltonian:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            'SolvationGPAW does not support '
            'calculations in reciprocal space yet.'
            )


class SolvationGPAW(GPAW):
    """Subclass of gpaw.GPAW calculator with continuum solvent model."""

    reciprocal_space_hamiltonian_class = NIReciprocalSpaceHamiltonian

    def __init__(self, cavity, dielectric, interactions=None,
                 **gpaw_kwargs):
        raise NotImplementedError('refactoring')
        if interactions is None:
            interactions = []

        def real_space_hamiltonian_factory(*args, **kwargs):
            return SolvationRealSpaceHamiltonian(
                cavdens, smoothedstep, dielectric, interactions,
                *args, **kwargs
                )

        self.real_space_hamiltonian_class = real_space_hamiltonian_factory
        GPAW.__init__(self, **gpaw_kwargs)

    def initialize_positions(self, atoms=None):
        spos_ac = GPAW.initialize_positions(self, atoms)
        self.hamiltonian.set_atoms(self.atoms)
        return spos_ac

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

    def get_solvation_interaction_energy(self, subscript, atoms=None):
        self.calculate(atoms, converge=True)
        return Hartree * getattr(self.hamiltonian, 'E_' + subscript)

    def print_parameters(self):
        GPAW.print_parameters(self)
        t = self.text
        t()
        t('Solvation Parameters:')
        fixed = [self.hamiltonian.cavdens,
                 self.hamiltonian.smoothedstep,
                 self.hamiltonian.dielectric]
        txt = ['Cavity Density:', 'Smoothed Step:', 'Dielectric:']
        for txt, fixed in zip(txt, fixed):
            t(txt)
            t('type: %s' % (fixed.name))
            fixed.print_parameters(t)
        for ia in self.hamiltonian.interactions:
            t('%s:' % (ia.name))
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

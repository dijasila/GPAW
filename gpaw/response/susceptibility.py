import sys

from ase.units import Hartree

import gpaw.mpi as mpi
from gpaw.response.chiks import ChiKS


class FourComponentSusceptibilityTensor:
    """Class calculating the full four-component susceptibility tensor"""

    def __init__(self, gs, frequencies=None,  # frequencies to __call__ XXX
                 fxc='ALDA', fxckwargs={},
                 eta=0.2, ecut=50, gammacentered=False,
                 disable_point_group=True, disable_time_reversal=True,
                 bandsummation='pairwise', nbands=None,
                 world=mpi.world, nblocks=1, txt=sys.stdout):
        """
        Currently, everything is in plane wave mode.
        If additional modes are implemented, maybe look to fxc to see how
        multiple modes can be supported.

        Parameters
        ----------
        gs, frequencies : SHOULD BE LOADED/INITIATED IN THIS SCRIPT XXX
            for now see gpaw.response.chiks, gpaw.response.kslrf

        fxc, fxckwargs : see gpaw.response.fxc

        eta, ecut, gammacentered
        disable_point_group,
        disable_time_reversal,
        bandsummation, nbands : see gpaw.response.chiks, gpaw.response.kslrf

        world, nblocks, txt : SHOULD BE LOADED/INITIATED IN THIS SCRIPT XXX
            for now see gpaw.response.chiks, gpaw.response.kslrf
        """
        self.chiks = ChiKS(gs, frequencies=frequencies, eta=eta, ecut=ecut,
                           gammacentered=gammacentered,
                           disable_point_group=disable_point_group,
                           disable_time_reversal=disable_time_reversal,
                           bandsummation=bandsummation, nbands=nbands,
                           world=world, nblocks=nblocks, txt=txt)
        self.calc = self.chiks.calc  # calc should be loaded here XXX
        self.fxc = create_fxc(self, fxc, fxckwargs)  # Write me XXX

    def get_macroscopic_component(self, spincomponent, q_c, filename=None):
        """Calculates the spatially averaged (macroscopic) component of the
        susceptibility tensor and writes it to a file.

        Parameters
        ----------
        spincomponent, q_c : see gpaw.response.chiks, gpaw.response.kslrf
        filename : str
            Save chiks_w and chi_w to file of given name.
            Defaults to:
            'chi%s_q«%+d-%+d-%+d».csv' % (spincomponent,
                                          *tuple((q_c * kd.N_c).round()))

        Returns
        -------
        see calculate_macroscopic_component
        """

        if filename is None:
            tup = (spincomponent, *tuple((q_c * kd.N_c).round()))  # fix me XXX
            filename = 'chi%s_q«%+d-%+d-%+d».csv' % tup

        (omega_w,
         chiks_w,
         chi_w) = self.calculate_macroscopic_component(spincomponent, q_c)

        write_macroscopic_component(omega_w, chiks_w, chi_w, filename)  # write me XXX

        return omega_w, chiks_w, chi_w

    def calculate_macroscopic_component(self, spincomponent, q_c):
        """Calculates the spatially averaged (macroscopic) component of the
        susceptibility tensor.

        Parameters
        ----------
        spincomponent, q_c : see gpaw.response.chiks, gpaw.response.kslrf

        Returns
        -------
        omega_w, chiks_w, chi_w : nd.array, nd.array, nd.array
            omega_w: frequencies in eV
            chiks_w: macroscopic dynamic susceptibility (Kohn-Sham system)
            chi_w: macroscopic(to be generalized?) dynamic susceptibility
        """
        (pd, omega_w,
         chiks_wGG, chi_wGG) = self.calculate_component(spincomponent, q_c)

        # Macroscopic component
        chiks_w = chiks_wGG[:, 0, 0]
        chi_w = chi_wGG[:, 0, 0]

        return omega_w, chiks_w, chi_w

    def calculate_component(self, spincomponent, q_c):
        # Some documentation needed XXX
        pd, omega_w, chiks_wGG = self.calculate_ks_component(spincomponent, q_c)
        Kxc_GG = self._calculate_Kxc(spincomponent, pd, chiks_wGG=chiks_wGG)

        # Initiate chi_wGG
        # Invert Dyson

        return pd, omega_w, chiks_wGG, chi_wGG

    def _calculate_ks_component(self, spincomponent, q_c):
        # Some documentation needed XXX
        # see also df.py XXX
        omega_w = self.chiks.omega_w * Hartree
        pd, chiks_wGG = self.chiks.calculate(q_c, spincomponent=spincomponent)

        return pd, omega_w, chiks_wGG

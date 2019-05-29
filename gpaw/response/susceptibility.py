import sys

from ase.units import Hartree

import gpaw.mpi as mpi
from gpaw.response.chiks import ChiKS


class FourComponentSusceptibilityTensor:
    """Class calculating the full four-component susceptibility tensor"""

    def __init__(self, gs, frequencies=None, eta=0.2,
                 ecut=50, gammacentered=False,
                 disable_point_group=True, disable_time_reversal=True,
                 bandsummation='pairwise', nbands=None,
                 world=mpi.world, nblocks=1, txt=sys.stdout):
        """
        Parameters
        ----------
        see gpaw.response.chiks, gpaw.response.kslrf
        """
        self.chiks = ChiKS(gs, frequencies=frequencies, eta=eta, ecut=ecut,
                           gammacentered=gammacentered,
                           disable_point_group=disable_point_group,
                           disable_time_reversal=disable_time_reversal,
                           bandsummation=bandsummation, nbands=nbands,
                           world=world, nblocks=nblocks, txt=txt)

        def calculate_component(self, spincomponent, q_c, fxc='ALDA',
                                filename=None, **fxckwargs):
            """Calculates the given component of the tensor and writes it.

            Parameters
            ----------
            spincomponent, q_c : see gpaw.response.chiks, gpaw.response.kslrf
            fxc, fxckwargs : see gpaw.response.fxc
            filename : str
                Save chiKS_w and chi_w to file of given name.
                Defaults to:
                'chi%s_q«%+d-%+d-%+d».csv' % (spincomponent,
                                              *tuple((q_c * kd.N_c).round()))

            Returns
            -------
            omega_w, chiKS_w, chi_w : nd.array, nd.array, nd.array
                omega_w: frequencies in eV
                chiKS_w: macroscopic dynamic susceptibility (Kohn-Sham system)
                chi_w: macroscopic(to be generalized?) dynamic susceptibility
            """
            omega_w = self.chiks.omega_w * Hartree
            (pd, chiKS_wGG,
             chi_wGG) = self._calculate_chi_component(spincomponent, q_c,
                                                      fxc=fxc, **fxckwargs)

            # Macroscopic component
            chiKS_w = chiKS_wGG[:, 0, 0]
            chi_w = chi_wGG[:, 0, 0]

            self.write_component(omega_w, chiKS_w, chi_w, filename=filename)

            return omega_w, chiKS_w, chi_w

        def _calculate_chi_component(self, spincomponent, q_c, fxc='ALDA',
                                     **fxckwargs):
            raise NotImplementedError()

        def _calculate_chiKS_component(self, spincomponent, q_c):
            raise NotImplementedError()

        def write_component(self, omega_w, chiKS_w, chi_w, filename=None):
            """Write macroscopic dynamic susceptibility component to a file
            with an appropriate filename"""
            raise NotImplementedError

import sys
from time import ctime
from pathlib import Path

import numpy as np

from ase.units import Hartree
from ase.utils import convert_string_to_fd
from ase.utils.timing import Timer, timer

import gpaw.mpi as mpi
from gpaw.blacs import (BlacsGrid, BlacsDescriptor,
                        Redistributor, DryRunBlacsGrid)
from gpaw.response.chiks import ChiKS
from gpaw.response.kxc import get_fxc


class FourComponentSusceptibilityTensor:
    """Class calculating the full four-component susceptibility tensor"""

    def __init__(self, gs, fxc='ALDA', fxckwargs={},
                 eta=0.2, ecut=50, gammacentered=False,
                 disable_point_group=True, disable_time_reversal=True,
                 bandsummation='pairwise', nbands=None, memory_safe=False,
                 world=mpi.world, nblocks=1, txt=sys.stdout):
        """
        Currently, everything is in plane wave mode.
        If additional modes are implemented, maybe look to fxc to see how
        multiple modes can be supported.

        Parameters
        ----------
        gs : SHOULD BE LOADED/INITIATED IN THIS SCRIPT XXX
            for now see gpaw.response.chiks, gpaw.response.kslrf

        fxc, fxckwargs : see gpaw.response.fxc

        eta, ecut, gammacentered
        disable_point_group,
        disable_time_reversal,
        bandsummation, nbands,
        memory_safe : see gpaw.response.chiks, gpaw.response.kslrf

        world, nblocks, txt : SHOULD BE LOADED/INITIATED IN THIS SCRIPT XXX
            for now see gpaw.response.chiks, gpaw.response.kslrf
        """
        # Initiate output file and timer
        self.world = world
        self.fd = convert_string_to_fd(txt, world)
        self.cfd = self.fd
        self.timer = Timer()
        
        self.chiks = ChiKS(gs, eta=eta, ecut=ecut, gammacentered=gammacentered,
                           disable_point_group=disable_point_group,
                           disable_time_reversal=disable_time_reversal,
                           bandsummation=bandsummation, nbands=nbands,
                           memory_safe=memory_safe, world=world,
                           nblocks=nblocks, txt=self.fd, timer=self.timer)
        self.calc = self.chiks.calc  # calc should be loaded here XXX
        self.fxc = get_fxc(self.calc, fxc,
                           response='susceptibility', mode='pw',
                           world=self.chiks.world, txt=self.chiks.fd,
                           timer=self.timer, **fxckwargs)

        # Parallelization over frequencies depends on the frequency input
        self.mynw = None
        self.w1 = None
        self.w2 = None

    def get_macroscopic_component(self, spincomponent, q_c, frequencies,
                                  filename=None, txt=None):
        """Calculates the spatially averaged (macroscopic) component of the
        susceptibility tensor and writes it to a file.

        Parameters
        ----------
        spincomponent, q_c,
        frequencies : see gpaw.response.chiks, gpaw.response.kslrf
        filename : str
            Save chiks_w and chi_w to file of given name.
            Defaults to:
            'chi%s_q«%+d-%+d-%+d».csv' % (spincomponent,
                                          *tuple((q_c * kd.N_c).round()))
        txt : str
            Save output of the calculation of this specific component into
            a file with the filename of the given input.

        Returns
        -------
        see calculate_macroscopic_component
        """

        if filename is None:
            tup = (spincomponent,
                   *tuple((q_c * self.calc.wfs.kd.N_c).round()))
            filename = 'chi%s_q«%+d-%+d-%+d».csv' % tup

        (omega_w,
         chiks_w,
         chi_w) = self.calculate_macroscopic_component(spincomponent, q_c,
                                                       frequencies,
                                                       txt=txt)

        write_macroscopic_component(omega_w / Hartree, chiks_w, chi_w,
                                    filename, self.world)

        return omega_w, chiks_w, chi_w

    def calculate_macroscopic_component(self, spincomponent,
                                        q_c, frequencies, txt=None):
        """Calculates the spatially averaged (macroscopic) component of the
        susceptibility tensor.

        Parameters
        ----------
        spincomponent, q_c,
        frequencies : see gpaw.response.chiks, gpaw.response.kslrf

        Returns
        -------
        omega_w, chiks_w, chi_w : nd.array, nd.array, nd.array
            omega_w: frequencies in eV
            chiks_w: macroscopic dynamic susceptibility (Kohn-Sham system)
            chi_w: macroscopic(to be generalized?) dynamic susceptibility
        """
        pd, chiks_wGG, chi_wGG = self.calculate_component(spincomponent, q_c,
                                                          frequencies, txt=txt)

        # Macroscopic component
        chiks_w = chiks_wGG[:, 0, 0]
        chi_w = chi_wGG[:, 0, 0]

        # Collect data for all frequencies
        omega_w = self.chiks.omega_w * Hartree
        chiks_w = self.collect(chiks_w)
        chi_w = self.collect(chi_w)

        return omega_w, chiks_w, chi_w

    def calculate_component(self, spincomponent, q_c, frequencies, txt=None):
        """Calculate a single component of the susceptibility tensor.

        Parameters
        ----------
        spincomponent, q_c,
        frequencies : see gpaw.response.chiks, gpaw.response.kslrf

        Returns
        -------
        pd : PWDescriptor
            Descriptor object for the plane wave basis
        chiks_wGG : ndarray
            The process' block of the Kohn-Sham susceptibility component
        chi_wGG : ndarray
            The process' block of the full susceptibility component
        """
        # Initiate new call-output file, if supplied
        if txt is not None:
            # Store timer and close old call-output file
            self.write_timer()
            if str(self.fd) != str(self.cfd):
                print('\nClosing, %s' % ctime(), file=self.cfd)
                self.cfd.close()
            # Initiate new output file
            self.cfd = convert_string_to_fd(txt, self.world)
        # Print to output file
        if str(self.fd) != str(self.cfd) or txt is not None:
            print('---------------', file=self.fd)
            print(f'Calculating susceptibility spincomponent={spincomponent}'
                  f'with q_c={q_c}', file=self.fd)

        # This should be initiated with G-parallelization, in this script! XXX
        nw = len(frequencies)
        self.mynw = (nw + self.world.size - 1) // self.world.size
        self.w1 = min(self.mynw * self.world.rank, nw)
        self.w2 = min(self.w1 + self.mynw, nw)

        return self._calculate_component(spincomponent, q_c, frequencies)

    def _calculate_component(self, spincomponent, q_c, frequencies):
        """In-place calculation of the given spin-component."""
        
        pd, chiks_wGG = self.calculate_ks_component(spincomponent, q_c,  # pd elsewhere XXX 
                                                    frequencies, txt=self.cfd)
        Kxc_GG = self.get_xc_kernel(spincomponent, pd, txt=self.cfd)  # move up, once pd is elsewhere XXX

        chi_wGG = self.invert_dyson(chiks_wGG, Kxc_GG)

        return pd, chiks_wGG, chi_wGG

    def write_timer(self):
        """Write timer to call-output file and initiate a new."""
        self.timer.write(self.cfd)
        self.timer = Timer()

        # Update all other class instance timers
        self.chiks.timer = self.timer
        self.chiks.integrator.timer = self.timer
        self.chiks.kspair.timer = self.timer
        self.chiks.pme.timer = self.timer
        self.fxc.timer = self.timer

    def calculate_ks_component(self, spincomponent, q_c,
                               frequencies, txt=None):
        """Calculate a single component of the Kohn-Sham susceptibility tensor.

        Parameters
        ----------
        spincomponent, q_c,
        frequencies : see gpaw.response.chiks, gpaw.response.kslrf

        Returns
        -------
        pd : PWDescriptor
            Descriptor object for the plane wave basis
        chiks_wGG : ndarray
            The process' block of the Kohn-Sham susceptibility component
        """
        # ChiKS calculates the susceptibility distributed over plane waves
        pd, chiks_wGG = self.chiks.calculate(q_c, frequencies,  # pd elsewhere XXX
                                             spincomponent=spincomponent, txt=txt)

        # Redistribute memory, so each block has its own frequencies, but all
        # plane waves (for easy invertion of the Dyson-like equation)
        chiks_wGG = self.distribute_frequencies(chiks_wGG)

        return pd, chiks_wGG

    def get_xc_kernel(self, spincomponent, pd, txt=None):
        """Check if the exchange correlation kernel has been calculated,
        if not, calculate it."""
        # Implement write/read/check functionality XXX
        if self.fxc.is_calculated(spincomponent, pd):
            Kxc_GG = self.fxc.read(spincomponent, pd)
        else:
            Kxc_GG = self.fxc(spincomponent, pd, txt=txt)
            self.fxc.write(Kxc_GG, spincomponent, pd)
        return Kxc_GG

    @timer('Invert dyson-like equation')
    def invert_dyson(self, chiks_wGG, Kxc_GG):
        """Invert the Dyson-like equation:

        chi = chi_ks - chi_ks Kxc chi

        # The sign convention in the Dyson equation needs to be examined XXX
        """
        chi_wGG = np.empty_like(chiks_wGG)
        for w, chiks_GG in enumerate(chiks_wGG):
            chi_GG = np.dot(np.linalg.inv(np.eye(len(chiks_GG)) +
                                          np.dot(chiks_GG, Kxc_GG)),
                            chiks_GG)

            chi_wGG[w] = chi_GG

        return chi_wGG

    def collect(self, a_w):
        """Collect frequencies from all blocks"""
        # More documentation is needed! XXX
        world = self.chiks.world
        b_w = np.zeros(self.mynw, a_w.dtype)
        b_w[:self.w2 - self.w1] = a_w
        nw = len(self.chiks.omega_w)
        A_w = np.empty(world.size * self.mynw, a_w.dtype)
        world.all_gather(b_w, A_w)
        return A_w[:nw]

    @timer('Distribute frequencies')
    def distribute_frequencies(self, chiks_wGG):
        """Distribute frequencies to all cores."""
        # More documentation is needed! XXX
        world = self.chiks.world
        comm = self.chiks.interblockcomm

        if world.size == 1:
            return chiks_wGG

        nw = len(self.chiks.omega_w)
        nG = chiks_wGG.shape[2]
        mynw = (nw + world.size - 1) // world.size
        mynG = (nG + comm.size - 1) // comm.size

        wa = min(world.rank * mynw, nw)
        wb = min(wa + mynw, nw)

        if self.chiks.interblockcomm.size == 1:
            return chiks_wGG[wa:wb].copy()

        if self.chiks.intrablockcomm.rank == 0:
            bg1 = BlacsGrid(comm, 1, comm.size)
            in_wGG = chiks_wGG.reshape((nw, -1))
        else:
            bg1 = DryRunBlacsGrid(mpi.serial_comm, 1, 1)
            in_wGG = np.zeros((0, 0), complex)
        md1 = BlacsDescriptor(bg1, nw, nG**2, nw, mynG * nG)

        bg2 = BlacsGrid(world, world.size, 1)
        md2 = BlacsDescriptor(bg2, nw, nG**2, mynw, nG**2)

        r = Redistributor(world, md1, md2)
        shape = (wb - wa, nG, nG)
        out_wGG = np.empty(shape, complex)
        r.redistribute(in_wGG, out_wGG.reshape((wb - wa, nG**2)))

        return out_wGG

    def close(self):
        self.timer.write(self.cfd)
        print('\nClosing, %s' % ctime(), file=self.cfd)
        self.cfd.close()
        print('\nClosing, %s' % ctime(), file=self.fd)
        self.fd.close()


def write_macroscopic_component(omega_w, chiks_w, chi_w, filename, world):
    """Write the spatially averaged dynamic susceptibility."""
    assert isinstance(filename, str)
    if world.rank == 0:
        with Path(filename).open('w') as fd:
            for omega, chiks, chi in zip(omega_w * Hartree, chiks_w, chi_w):
                print('%.6f, %.6f, %.6f, %.6f, %.6f' %
                      (omega, chiks.real, chiks.imag, chi.real, chi.imag),
                      file=fd)

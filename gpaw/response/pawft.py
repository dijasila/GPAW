"""Functionality to calculate the all-electron Fourier components of real space
quantities within the PAW formalism."""

from abc import ABC, abstractmethod

import numpy as np

import gpaw.mpi as mpi

from gpaw.utilities import convert_string_to_fd
from ase.utils.timing import Timer, timer

from gpaw.response.kspair import get_calc
from gpaw.response.groundstate import ResponseGroundStateAdapter


class LocalPAWFT(ABC):
    """Abstract base class for calculators of all-electron plane-wave
    components to some real space functional f[n](r) which can be written as a
    closed form function of the local ground state (spin-)density:

    f[n](r) = f(n(r)).

    Since n(r) is lattice periodic, so is f(r) and the plane-wave components
    can be calculated as (see [PRB 103, 245110 (2021)] for definitions)

           /
    f(G) = |dr f(r) e^(-iG.r),
           /
            V0

    where V0 is the unit-cell volume.
    """

    def __init__(self, gs,
                 world=mpi.world, txt='-', timer=None,
                 rshelmax=-1, rshewmin=None):
        """Constructor for the PAWFT

        Parameters
        ----------
        gs : str/obj
            Filename or GPAW calculator object of ground state calculation
        world : mpi.world
        txt : str or filehandle
            defines output file through gpaw.utilities.convert_string_to_fd
        timer : ase.utils.timing.Timer instance
        rshelmax : int or None
            Expand quantity in real spherical harmonics inside augmentation
            spheres. If None, the plane wave components will be calculated
            without augmentation. The value of rshelmax indicates the maximum
            index l to perform the expansion in (l < 6).
        rshewmin : float or None
            If None, the PAW correction will be fully expanded up to the chosen
            lmax. Given as a float (0 < rshewmin < 1), rshewmin indicates what
            coefficients to use in the expansion. If any (l,m) coefficient
            contributes with less than a fraction of rshewmin on average, it
            will not be included.
        """
        # Output .txt filehandle and timer
        self.world = world
        self.fd = convert_string_to_fd(txt, world)
        self.timer = timer or Timer()
        self.calc = get_calc(gs, fd=self.fd, timer=self.timer)
        self.gs = ResponseGroundStateAdapter(self.calc)

        # Do not carry out the expansion in real spherical harmonics, if lmax
        # is chosen as None
        self.rshe = rshelmax is not None

        if self.rshe:
            # Perform rshe up to l<=lmax(<=5)
            if rshelmax == -1:
                self.rshelmax = 5
            else:
                assert isinstance(rshelmax, int)
                assert rshelmax in range(6)
                self.rshelmax = rshelmax

            self.rshewmin = rshewmin if rshewmin is not None else 0.
            self.dfmask_g = None

    @abstractmethod
    def _add_real_space_functional(self, gd, n_sg, f_g):
        """Calculate the real-space quantity in question as a functional of
        the local (spin-)density on a real-space grid and add it to an array
        on the same real-space grid."""
        pass

    def print(self, *args, **kwargs):
        print(*args, file=self.fd, **kwargs)

    @timer('LocalPAWFT')
    def __call__(self, pd):
        self.print('Calculating f(G)', flush=True)
        f_G = self.calculate(pd)
        self.print('Finished calculating f(G)', flush=True)

        return f_G

    def calculate(self, pd):
        """Calculate the plane-wave components f(G) for the reciprocal lattice vectors
        defined by the plane-wave descriptor pd."""

        # Retrieve the (pseudo) spin-density and allocate f(r)
        n_sr = self.get_density_on_grid()
        f_r = np.zeros(np.shape(n_sr[0]))

        # FFT to reciprocal space
        f_G = self.ft_from_grid(f_r, pd)

        if self.rshe:  # Add PAW correction to Fourier transformed function
            fPAW_G = self.calculate_paw_correction(pd)
            f_G += fPAW_G

        return f_G

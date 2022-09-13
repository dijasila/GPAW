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
    def _add_f(self, gd, n_sr, f_r):
        """Calculate the real-space quantity in question as a function of the local
        (spin-)density on a given real-space grid and add it to a given output
        array."""
        pass

    def print(self, *args):
        print(*args, file=self.fd, flush=True)

    @timer('LocalPAWFT')
    def __call__(self, pd):
        self.print('Calculating f(G)')
        f_G = self.calculate(pd)
        self.print('Finished calculating f(G)')

        return f_G

    def calculate(self, pd):
        """Calculate the plane-wave components f(G) for the reciprocal lattice vectors
        defined by the plane-wave descriptor pd."""
        if self.rshe:
            return self._calculate_w_rshe(pd)
        else:
            return self._calculate_wo_rshe(pd)

    def _calculate_w_rshe(self, pd):
        """Calculate f(G) with an expansion of f(r) in real spherical harmonics inside
        the augmentation spheres."""
        # Retrieve the pseudo (spin-)density on the real-space grid
        nt_sr = self.get_pseudo_density(pd.gd)

        # Calculate ft(r) (t=tilde=pseudo)
        ft_r = np.zeros(np.shape(nt_sr[0]))
        self._add_f(pd.gd, nt_sr, ft_r)

        # FFT to reciprocal space
        ft_G = self.fft_from_grid(ft_r, pd)

        # Calculate PAW correction inside the augmentation spheres
        fPAW_G = self.calculate_paw_correction(pd)

        return ft_G + fPAW_G

    def _calculate_wo_rshe(self, pd):
        """Calculate f(G) directly from the all-electron density on a
        real-space grid."""
        # Retrieve the all-electron (spin-)density on the real-space grid
        n_sr = self.get_all_electron_density(pd.gd)

        # Calculate f(r)
        f_r = np.zeros(np.shape(n_sr[0]))
        self._add_f(pd.gd, n_sr, f_r)

        # FFT to reciprocal space
        f_G = self.fft_from_grid(f_r, pd)

        return f_G

    def get_pseudo_density(self, pd):
        """Return the pseudo (spin-)density on the coarse real-space grid of the
        ground state."""
        self.check_grid_equivalence(pd.gd, self.gs.gd)
        return self.gs.nt_sG  # nt=pseudo density, G=coarse grid

    @timer('Calculating the all-electron density')
    def get_all_electron_density(self, pd):
        """Calculate the all-electron (spin-)density on the coarse real-space
        grid of the ground state."""
        self.print('    Calculating the all-electron density')
        n_sG, gd1 = self.gs.all_electron_density(gridrefinement=1)
        self.check_grid_equivalence(pd.gd, gd1)

        return n_sG

    @staticmethod
    def check_grid_equivalence(gd1, gd2):
        assert gd1.comm.size == 1
        assert gd2.comm.size == 1
        assert (gd1.N_c == gd2.N_c).all()

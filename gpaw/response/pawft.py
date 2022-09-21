"""Functionality to calculate the all-electron Fourier components of real space
quantities within the PAW formalism."""

from abc import ABC, abstractmethod

import numpy as np
from scipy.special import spherical_jn

from ase.utils.timing import Timer, timer
from ase.units import Bohr

import gpaw.mpi as mpi
from gpaw.utilities import convert_string_to_fd
from gpaw.spherical_harmonics import Yarr
from gpaw.sphere.lebedev import weight_n, R_nv


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
        gs : ResponseGroundStateAdapter
            Adapter containing relevant information about the underlying DFT
            ground state
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
        self.world = world
        self.fd = convert_string_to_fd(txt, world)  # output filehandle
        self.timer = timer or Timer()
        self.gs = gs

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
    def _add_f(self, gd, n_sR, f_R):
        """Calculate the real-space quantity in question as a function of the local
        (spin-)density on a given real-space grid and add it to a given output
        array."""
        pass

    def print(self, *args, flush=True):
        print(*args, file=self.fd, flush=flush)

    @timer('LocalPAWFT')
    def __call__(self, pd):
        self.print('Calculating f(G)')
        f_G = self.calculate(pd)
        self.print('Finished calculating f(G)')

        return f_G

    def calculate(self, pd):
        """Calculate the plane-wave components f(G) for the reciprocal lattice
        vectors defined by the plane-wave descriptor pd."""
        if self.rshe:
            return self._calculate_w_rshe(pd)
        else:
            return self._calculate_wo_rshe(pd)

    def _calculate_w_rshe(self, pd):
        """Calculate f(G) with an expansion of f(r) in real spherical harmonics
        inside the augmentation spheres."""
        # Retrieve the pseudo (spin-)density on the coarse real-space grid
        nt_sR = self.get_pseudo_density(pd.gd)  # R = Coarse 3D real-space grid

        # Calculate ft(r) (t=tilde=pseudo)
        ft_R = np.zeros(np.shape(nt_sR[0]))
        self._add_f(pd.gd, nt_sR, ft_R)

        # FFT to reciprocal space
        ft_G = self.fft_from_grid(ft_R, pd)  # G = 1D grid of |G|^2/2 < ecut

        # Calculate PAW correction inside the augmentation spheres
        fPAW_G = self.calculate_paw_correction(pd)

        return ft_G + fPAW_G

    def _calculate_wo_rshe(self, pd):
        """Calculate f(G) directly from the all-electron density on a
        real-space grid."""
        # Retrieve the all-electron (spin-)density on the real-space grid
        # R = Coarse 3D real-space grid
        n_sR = self.get_all_electron_density(pd.gd)

        # Calculate f(r)
        f_R = np.zeros(np.shape(n_sR[0]))
        self._add_f(pd.gd, n_sR, f_R)

        # FFT to reciprocal space
        f_G = self.fft_from_grid(f_R, pd)  # G = 1D grid of |G|^2/2 < ecut

        return f_G

    def get_pseudo_density(self, gd):
        """Return the pseudo (spin-)density on the coarse real-space grid of the
        ground state."""
        self.check_grid_equivalence(gd, self.gs.gd)
        return self.gs.nt_sG  # nt=pseudo density, G=coarse grid

    @timer('Calculating the all-electron density')
    def get_all_electron_density(self, gd):
        """Calculate the all-electron (spin-)density on the coarse real-space
        grid of the ground state."""
        self.print('    Calculating the all-electron density')
        n_sR, gd1 = self.gs.all_electron_density(gridrefinement=1)
        self.check_grid_equivalence(gd, gd1)

        return n_sR

    @staticmethod
    def check_grid_equivalence(gd1, gd2):
        assert gd1.comm.size == 1
        assert gd2.comm.size == 1
        assert (gd1.N_c == gd2.N_c).all()

    def fft_from_grid(self, f_R, pd):
        """Perform a FFT to reciprocal space:
                                        __
               /                    V0  \
        f(G) = |dr f(r) e^(-iG.r) ≃ ‾‾  /  f(r) e^(-iG.r)
               /                    N   ‾‾
               V0                       r

        where N is the number of grid points."""
        Q_G = pd.Q_qG[0]

        # Perform the FFT
        N = np.prod(pd.gd.N_c)
        f_Q123 = pd.gd.volume / N * np.fft.fftn(f_R)  # Q123 = 3D grid in Q-rep

        # Change the view of the plane-wave components from the 3D grid in the
        # Q-representation that numpy spits out to the 1D grid in the
        # G-representation, that GPAW relies on internally
        f_G = f_Q123.ravel()[Q_G]

        return f_G

    @timer('Calculate PAW corrections to kernel')
    def calculate_paw_correction(self, pd):
        self.print('    Calculating PAW correction\n')

        # Extract reciprocal lattice vectors
        nG = pd.ngmax
        G_Gv = pd.get_reciprocal_vectors()
        assert G_Gv.shape[0] == nG

        # Allocate output array
        fPAW_G = np.zeros(nG, dtype=complex)

        # Distribute plane waves
        G_myG = self._distribute_correction(nG)
        G_myGv = G_Gv[G_myG]

        # Calculate PAW correction to each augmentation sphere (to each atom)
        R_av = self.gs.atoms.positions / Bohr
        for a, R_v in enumerate(R_av):
            # Calculate df on Lebedev quadrature and radial grid
            df_ng, Y_nL, rgd = self._calculate_df(a)

            # Calculate the surface norm square of df
            dfSns_g = self._ang_int(df_ng ** 2)
            # Reduce radial grid by excluding points where dfSns_g = 0
            df_ng, r_g, dv_g = self._reduce_radial_grid(df_ng, rgd, dfSns_g)

            # Expand correction in real spherical harmonics
            df_gL = self._perform_rshe(df_ng, Y_nL)
            # Reduce expansion by removing coefficients that do not contribute
            df_gM, L_M, l_M = self._reduce_rshe(a, df_gL, dfSns_g)

            # Expand the plane waves in real spherical harmonics (and spherical
            # Bessel functions)
            (ii_MmyG,
             j_gMmyG,
             Y_MmyG) = self._expand_plane_waves(G_myGv, r_g, L_M, l_M)

            # Calculate the PAW correction as an integral over the radial grid
            # and rshe coefficients
            with self.timer('Integrate PAW correction'):
                angular_coef_MmyG = ii_MmyG * Y_MmyG
                radial_coef_MmyG = np.tensordot(j_gMmyG * df_gL[:, L_M,
                                                                np.newaxis],
                                                dv_g, axes=([0, 0]))
                atomic_corr_myG = np.sum(angular_coef_MmyG * radial_coef_MmyG,
                                         axis=0)

                position_prefactor_myG = np.exp(-1j * np.inner(G_myGv, R_v))
                fPAW_G[G_myG] += position_prefactor_myG * atomic_corr_myG

        self.world.sum(fPAW_G)

        return fPAW_G

    def _distribute_correction(self, nG):
        """Distribute correction"""
        nGpr = (nG + self.world.size - 1) // self.world.size
        Ga = min(self.world.rank * nGpr, nG)
        Gb = min(Ga + nGpr, nG)

        return range(Ga, Gb)

    @timer('Calculate PAW correction inside augmentation spheres')
    def _calculate_df(self, a):
        """Calculate the difference between f(n(r)) (all-electron spin
        density) and f(ñ(r)) (pseudo spin density).

        Returns
        -------
        df_ng : nd.array
            (f_ng - ft_ng) where (n=Lebedev index, g=radial grid index)
        Y_nL : nd.array
            real spherical harmonics on Lebedev quadrature where L is a
            composit (l,m) index, L = l**2 + m
        rgd : GridDescriptor
            non-linear radial grid descriptor
        """
        # Extract spin densities from the ground state calculation
        n_sLg, nt_sLg = self.get_radial_densities(a)

        setup = self.gs.setups[a]
        Y_nL = setup.xc_correction.Y_nL
        rgd = setup.xc_correction.rgd
        f_g = rgd.zeros()
        ft_g = rgd.zeros()
        df_ng = np.array([rgd.zeros() for n in range(len(R_nv))])
        for n, Y_L in enumerate(Y_nL):
            f_g[:] = 0.
            n_sg = np.dot(Y_L, n_sLg)
            self._add_f(rgd, n_sg, f_g)

            ft_g[:] = 0.
            nt_sg = np.dot(Y_L, nt_sLg)
            self._add_f(rgd, nt_sg, ft_g)

            df_ng[n, :] = f_g - ft_g

        return df_ng, Y_nL, rgd

    def get_radial_densities(self, a):
        """Get the all-electron and pseudo spin densities inside
        augmentation sphere a.

        Returns
        -------
        n_sLg : nd.array
            all-electron density
        nt_sLg : nd.array
            pseudo density
        (s=spin, L=(l,m) spherical harmonic index, g=radial grid index)
        """
        setup = self.gs.setups[a]
        n_qg = setup.xc_correction.n_qg
        nt_qg = setup.xc_correction.nt_qg
        nc_g = setup.xc_correction.nc_g
        nct_g = setup.xc_correction.nct_g

        D_sp = self.gs.D_asp[a]
        B_pqL = setup.xc_correction.B_pqL
        D_sLq = np.inner(D_sp, B_pqL.T)
        nspins = len(D_sp)

        n_sLg = np.dot(D_sLq, n_qg)
        nt_sLg = np.dot(D_sLq, nt_qg)

        # Add core density
        n_sLg[:, 0] += np.sqrt(4. * np.pi) / nspins * nc_g
        nt_sLg[:, 0] += np.sqrt(4. * np.pi) / nspins * nct_g

        return n_sLg, nt_sLg

    @staticmethod
    def _ang_int(f_nx):
        """Perform the angular (spherical) surface integral of a function f(r)
        using the Lebedev quadrature (indexed by n)."""
        f_x = 4. * np.pi * np.tensordot(weight_n, f_nx, axes=([0], [0]))

        return f_x

    def _reduce_radial_grid(self, df_ng, rgd, dfSns_g):
        """Reduce the radial grid, by excluding points where dfSns_g = 0,
        in order to avoid excess computation. Only points after the outermost
        point where dfSns_g is non-zero will be excluded.

        Returns
        -------
        df_ng : nd.array
            df_ng on reduced radial grid
        r_g : nd.array
            radius of each point on the reduced radial grid
        dv_g : nd.array
            volume element of each point on the reduced radial grid
        """
        # Find PAW correction range
        self.dfmask_g = np.where(dfSns_g > 0.)
        ng = np.max(self.dfmask_g) + 1

        # Integrate only r-values inside augmentation sphere
        df_ng = df_ng[:, :ng]

        r_g = rgd.r_g[:ng]
        dv_g = rgd.dv_g[:ng]

        return df_ng, r_g, dv_g

    @timer('Expand PAW correction in real spherical harmonics')
    def _perform_rshe(self, df_ng, Y_nL):
        """Perform expansion of df in real spherical harmonics. Note that the
        Lebedev quadrature, which is used to calculate the expansion
        coefficients, is exact to order l=11. This implies that functions
        containing angular components l<=5 can be expanded exactly.
        Assumes df_ng to be a real function.

        Returns
        -------
        df_gL : nd.array
            df in g=radial grid index, L=(l,m) spherical harmonic index
        """
        lmax = min(int(np.sqrt(Y_nL.shape[1])) - 1, 36)
        nL = (lmax + 1)**2
        L_L = np.arange(nL)

        # Perform the real spherical harmonics expansion
        df_ngL = np.repeat(df_ng, nL, axis=1).reshape((*df_ng.shape, nL))
        Y_ngL = np.repeat(Y_nL[:, L_L], df_ng.shape[1],
                          axis=0).reshape((*df_ng.shape, nL))
        df_gL = self._ang_int(Y_ngL * df_ngL)

        return df_gL

    def _reduce_rshe(self, a, df_gL, dfSns_g):
        """Reduce the composite index L=(l,m) to M, which indexes coefficients
        contributing with a weight larger than rshewmin to the surface norm
        square on average.

        Returns
        -------
        df_gM : nd.array
            PAW correction in reduced rsh index
        L_M : nd.array
            L=(l,m) spherical harmonics indices in reduced rsh index
        l_M : list
            l spherical harmonics indices in reduced rsh index
        """
        # Create L_L and l_L array
        lmax = min(self.rshelmax, int(np.sqrt(df_gL.shape[1])) - 1)
        nL = (lmax + 1)**2
        L_L = np.arange(nL)
        l_L = []
        for l in range(lmax + 1):
            l_L += [l] * (2 * l + 1)

        # Filter away (l,m)-coefficients based on their average weight in
        # completing the surface norm square of df
        dfSw_gL = self._calculate_ns_weights(a, nL, df_gL, dfSns_g)
        rshew_L = np.average(dfSw_gL, axis=0)  # Average over the radial grid
        # Do the actual filtering
        L_M = np.where(rshew_L[L_L] > self.rshewmin)[0]
        l_M = [l_L[L] for L in L_M]
        df_gM = df_gL[:, L_M]

        # Print information about the final (reduced) expansion at atom a
        self.print_reduced_rshe_info(a, nL, dfSw_gL, rshew_L)

        return df_gM, L_M, l_M

    def _calculate_ns_weights(self, a, nL, df_gL, dfSns_g):
        """Calculate the weighted contribution of each rsh coefficient to the
        surface norm square of df as a function of radial grid index g."""
        nallL = df_gL.shape[1]
        dfSns_gL = np.repeat(dfSns_g, nallL).reshape(dfSns_g.shape[0], nallL)
        dfSw_gL = df_gL[self.dfmask_g] ** 2 / dfSns_gL[self.dfmask_g]

        return dfSw_gL

    def print_reduced_rshe_info(self, a, nL, dfSw_gL, rshew_L):
        """Print information about the reduced expansion in real spherical
        harmonics at atom (augmentation sphere) a."""
        self.print('    RSHE of atom', a, flush=False)
        self.print('      {0:6}  {1:10}  {2:10}  {3:8}'.format('(l,m)',
                                                               'max weight',
                                                               'avg weight',
                                                               'included'),
                   flush=False)
        for L, (dfSw_g, rshew) in enumerate(zip(dfSw_gL.T, rshew_L)):
            self.print_rshe_coef_info(L, nL, dfSw_g, rshew)

        tot_avg_cov = np.average(np.sum(dfSw_gL, axis=1))
        avg_cov = np.average(np.sum(dfSw_gL[:, :nL]
                                    [:, rshew_L[:nL] > self.rshewmin], axis=1))
        self.print(f'      In total: {avg_cov} of the dfSns is covered on'
                   ' average', flush=False)
        self.print(f'      In total: {tot_avg_cov} of the dfSns could be'
                   ' covered on average\n')

    def print_rshe_coef_info(self, L, nL, dfSw_g, rshew):
        """Print information about a specific rshe coefficient"""
        l = int(np.sqrt(L))
        m = L - l**2 - l
        included = 'yes' if (rshew > self.rshewmin and L < nL) else 'no'
        info = '      {0:6}  {1:1.8f}  {2:1.8f}  {3:8}'.format(f'({l},{m})',
                                                               np.max(dfSw_g),
                                                               rshew, included)
        self.print(info, flush=False)

    @timer('Expand plane waves')
    def _expand_plane_waves(self, G_myGv, r_g, L_M, l_M):
        r"""Expand plane waves in spherical Bessel functions and real spherical
        harmonics:
                        l
                    __  __
         -iG.r      \   \      l             ^     ^
        e      = 4π /   /  (-i)  j (|G|r) Y (G) Y (r)
                    ‾‾  ‾‾        l        lm    lm
                    l  m=-l

        Returns
        -------
        ii_MmyG : nd.array
            (-i)^l for used (l,m) coefficients M
        j_gMmyG : nd.array
            j_l(|G|r) for used (l,m) coefficients M
        Y_MmyG : nd.array
                 ^
            Y_lm(K) for used (l,m) coefficients M
        """
        nmyG = G_myGv.shape[0]
        Gnorm_myG, Gdir_myGv = self._calculate_norm_and_direction(G_myGv)

        # Setup arrays to fully vectorize computations
        nM = len(L_M)
        (r_gMmyG, l_gMmyG,
         Gnorm_gMmyG) = [a.reshape(len(r_g), nM, nmyG)
                         for a in np.meshgrid(r_g, l_M, Gnorm_myG,
                                              indexing='ij')]

        with self.timer('Compute spherical bessel functions'):
            # Slow step
            j_gMmyG = spherical_jn(l_gMmyG, Gnorm_gMmyG * r_gMmyG)

        Y_MmyG = Yarr(L_M, Gdir_myGv)
        ii_MmyG = (-1j) ** np.repeat(l_M, nmyG).reshape((nM, nmyG))

        return ii_MmyG, j_gMmyG, Y_MmyG

    @staticmethod
    def _calculate_norm_and_direction(G_myGv):
        """Calculate the length and direction of reciprocal lattice vectors."""
        Gnorm_myG = np.linalg.norm(G_myGv, axis=1)
        Gdir_myGv = np.zeros_like(G_myGv)
        mask0 = np.where(Gnorm_myG != 0.)
        Gdir_myGv[mask0] = G_myGv[mask0] / Gnorm_myG[mask0][:, np.newaxis]

        return Gnorm_myG, Gdir_myGv


class AllElectronDensityFT(LocalPAWFT):
    """Calculator class for the plane-wave components of the all-electron
    density n(G)."""

    def _add_f(self, gd, n_sR, f_R):
        """Calculate the real-space quantity in question as a function of the local
        (spin-)density on a given real-space grid and add it to a given output
        array."""
        f_R += np.sum(n_sR, axis=0)

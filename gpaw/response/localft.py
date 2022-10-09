"""Functionality to calculate the all-electron Fourier components of local
functions of the electon (spin-)density."""

from abc import ABC, abstractmethod

from functools import partial

import numpy as np
from scipy.special import spherical_jn

from ase.units import Bohr

from gpaw.response import ResponseGroundStateAdapter, ResponseContext, timer
from gpaw.spherical_harmonics import Yarr
from gpaw.sphere.lebedev import weight_n, R_nv
from gpaw.xc import XC


class LocalFTCalculator(ABC):
    r"""Calculator base class for calculators of all-electron plane-wave
    components to arbitrary real-valued real-space functionals f[n](r) which
    can be written as closed form functions of the local ground state
    (spin-)density:

    f[n](r) = f(n(r)).

    Since n(r) is lattice periodic, so is f(r) and the plane-wave components
    can be calculated as (see [PRB 103, 245110 (2021)] for definitions)

           /
    f(G) = |dr f(r) e^(-iG.r),
           /
            V0

    where V0 is the unit-cell volume.
    """

    def __init__(self, gs, context):
        """Constructor for the LocalFTCalculator

        Parameters
        ----------
        gs : ResponseGroundStateAdapter
            Adapter containing relevant information about the underlying DFT
            ground state
        context : ResponseContext
        """
        assert isinstance(gs, ResponseGroundStateAdapter)
        self.gs = gs
        assert isinstance(context, ResponseContext)
        self.context = context

    @staticmethod
    def from_rshe_parameters(gs, context,
                             rshelmax=-1, rshewmin=None):
        """Construct the LocalFTCalculator based on parameters for the
        expansion of the PAW correction in real spherical harmonics

        Parameters
        ----------
        rshelmax : int or None
            Expand f(r) in real spherical harmonics inside the augmentation
            spheres. If None, the plane-wave components will be calculated
            without augmentation. The value of rshelmax indicates the maximum
            index l to perform the expansion in (l < 6).
        rshewmin : float or None
            If None, the PAW correction will be fully expanded up to the chosen
            lmax. Given as a float (0 < rshewmin < 1), rshewmin indicates what
            coefficients to use in the expansion. If any (l,m) coefficient
            contributes with less than a fraction of rshewmin on average, it
            will not be included.
        """
        if rshelmax is None:
            return LocalGridFTCalculator(gs, context)
        else:
            return LocalPAWFTCalculator(gs, context,
                                        rshelmax=rshelmax, rshewmin=rshewmin)

    @timer('LocalFT')
    def __call__(self, pd, add_f):
        """Calculate the plane-wave components f(G).

        Parameters
        ----------
        pd : PlaneWaveDescriptor
            Defines the plane-wave basis to calculate the components in.
        add_f : method
            Defines the local function of the electron (spin-)density to
            Fourier transform. Should take arguments gd (GridDescriptor),
            n_sR (electron spin-density on the real space grid of gd) and
            f_R (output array) and add the function f(R) to the output array.
            Example:
            >>> def add_total_density(gd, n_sR, f_R):
            ...     f_R += np.sum(n_sR, axis=0)

        Returns
        -------
        f_G : np.array
            Plane-wave components of the function f, indexes by the reciprocal
            lattice vectors G.
        """
        self.context.print('Calculating f(G)')
        f_G = self.calculate(pd, add_f)
        self.context.print('Finished calculating f(G)')

        return f_G

    @abstractmethod
    def calculate(self, pd, add_f):
        pass

    @staticmethod
    def check_grid_equivalence(gd1, gd2):
        assert gd1.comm.size == 1
        assert gd2.comm.size == 1
        assert (gd1.N_c == gd2.N_c).all()


class LocalGridFTCalculator(LocalFTCalculator):

    def calculate(self, pd, add_f):
        """Calculate f(G) directly from the all-electron density on the cubic
        real-space grid."""
        n_sR = self.get_all_electron_density(pd.gd)
        f_G = self._calculate(pd, n_sR, add_f)

        return f_G

    def _calculate(self, pd, n_sR, add_f):
        """In-place calculation of the plane-wave components."""
        # Calculate f(r)
        gd = pd.gd
        f_R = gd.zeros()
        add_f(gd, n_sR, f_R)

        # FFT to reciprocal space
        f_G = fft_from_grid(f_R, pd)  # G = 1D grid of |G|^2/2 < ecut

        return f_G

    @timer('Calculate the all-electron density')
    def get_all_electron_density(self, gd):
        """Calculate the all-electron (spin-)density on the coarse real-space
        grid of the ground state."""
        self.context.print('    Calculating the all-electron density')
        n_sR, gd1 = self.gs.all_electron_density(gridrefinement=1)
        self.check_grid_equivalence(gd, gd1)

        return n_sR


class LocalPAWFTCalculator(LocalFTCalculator):

    def __init__(self, gs, context, rshelmax=-1, rshewmin=None):
        super().__init__(gs, context)

        self.engine = LocalPAWFTEngine(self.context, rshelmax, rshewmin)

    def calculate(self, pd, add_f):
        """Calculate f(G) with an expansion of f(r) in real spherical harmonics
        inside the augmentation spheres."""
        # Retrieve the pseudo (spin-)density on the coarse real-space grid
        nt_sR = self.get_pseudo_density(pd.gd)  # R = Coarse 3D real-space grid

        # Retrieve the pseudo and all-electron atomic centered densities inside
        # the augmentation spheres
        R_av, micro_setups = self.extract_atom_centered_quantities()

        # Let the engine perform the in-place calculation
        f_G = self.engine.calculate(pd, nt_sR, R_av, micro_setups, add_f)

        return f_G

    def get_pseudo_density(self, gd):
        """Return the pseudo (spin-)density on the coarse real-space grid of
        the ground state."""
        self.check_grid_equivalence(gd, self.gs.gd)
        return self.gs.nt_sR  # nt=pseudo density, R=coarse grid

    def extract_atom_centered_quantities(self):
        """Extract all relevant atom centered quantities that the engine needs
        in order to calculate PAW corrections. Most of the information is
        bundled as a list of MicroSetups for each atom."""
        R_av = self.gs.atoms.positions / Bohr

        micro_setups = [self.extract_micro_setup(a)
                        for a in range(len(self.gs.atoms))]

        return R_av, micro_setups

    def extract_micro_setup(self, a):
        """Extract the all-electron and pseudo spin-densities inside
        augmentation sphere a as well as the relevant radial and angular grid
        data.

        Returns
        -------
        micro_setup : MicroSetup
        """
        setup = self.gs.setups[a]
        # Radial grid descriptor:
        rgd = setup.xc_correction.rgd
        # Spherical harmonics on the Lebedev quadrature:
        Y_nL = setup.xc_correction.Y_nL

        D_sp = self.gs.D_asp[a]  # atomic density matrix
        n_sLg, nt_sLg = self.calculate_atom_centered_densities(setup, D_sp)

        return MicroSetup(rgd, Y_nL, n_sLg, nt_sLg)

    @staticmethod
    def calculate_atom_centered_densities(setup, D_sp):
        """Calculate the all-electron and pseudo densities inside the
        augmentation sphere.

        Returns
        -------
        n_sLg : nd.array
            all-electron density
        nt_sLg : nd.array
            pseudo density
        (s=spin, L=(l,m) spherical harmonic index, g=radial grid index)
        """
        n_qg = setup.xc_correction.n_qg
        nt_qg = setup.xc_correction.nt_qg
        nc_g = setup.xc_correction.nc_g
        nct_g = setup.xc_correction.nct_g

        B_pqL = setup.xc_correction.B_pqL
        D_sLq = np.inner(D_sp, B_pqL.T)
        nspins = len(D_sp)

        n_sLg = np.dot(D_sLq, n_qg)
        nt_sLg = np.dot(D_sLq, nt_qg)

        # Add core density
        n_sLg[:, 0] += np.sqrt(4. * np.pi) / nspins * nc_g
        nt_sLg[:, 0] += np.sqrt(4. * np.pi) / nspins * nct_g

        return n_sLg, nt_sLg


class MicroSetup:

    def __init__(self, rgd, Y_nL, n_sLg, nt_sLg):
        self.rgd = rgd
        self.Y_nL = Y_nL
        self.n_sLg = n_sLg
        self.nt_sLg = nt_sLg


class LocalPAWFTEngine:

    def __init__(self, context, rshelmax=-1, rshewmin=None):
        """Construct the engine."""
        self.context = context

        # Perform rshe up to l<=lmax(<=5)
        if rshelmax == -1:
            self.rshelmax = 5
        else:
            assert isinstance(rshelmax, int)
            assert rshelmax in range(6)
            self.rshelmax = rshelmax

        self.rshewmin = rshewmin if rshewmin is not None else 0.
        self.dfmask_g = None

        self._add_f = None

    def calculate(self, pd, nt_sR, R_av, micro_setups, add_f):
        r"""Calculate the Fourier transform f(G) by splitting up the
        calculation into a pseudo density contribution and a PAW correction
        accounting for the difference

        Δf[n,ñ](r) = f(n(r)) - f(ñ(r)),

        such that:

        f(G) = f[ñ](G) + Δf[n,ñ](G).

        See [PRB 103, 245110 (2021)] for definitions and notation details."""
        self._add_f = add_f

        ft_G = self.calculate_pseudo_contribution(pd, nt_sR)
        fPAW_G = self.calculate_paw_corrections(pd, R_av, micro_setups)

        return ft_G + fPAW_G

    def calculate_pseudo_contribution(self, pd, nt_sR):
        """Calculate the pseudo density contribution by performing a FFT of
        f(ñ(r)) on the cubic real-space grid.

        NB: This operation assumes that the function f is a slowly varrying
        function of the pseudo density ñ(r) everywhere in space, such that
        f(ñ(r)) is accurately described on the cubic real-space grid."""
        # Calculate ft(r) (t=tilde=pseudo)
        gd = pd.gd
        ft_R = gd.zeros()
        self._add_f(gd, nt_sR, ft_R)

        # FFT to reciprocal space
        ft_G = fft_from_grid(ft_R, pd)  # G = 1D grid of |G|^2/2 < ecut

        return ft_G

    @timer('Calculate PAW corrections')
    def calculate_paw_corrections(self, pd, R_av, micro_setups):
        r"""Calculate the PAW corrections to f(G), for each augmentation sphere
        at a time:
                      __
                      \   /
        Δf[n,ñ](G) =  /   |dr Δf_a[n_a,ñ_a](r - R_a) e^(-iG.r)
                      ‾‾  /
                      a    V0

        where Δf_a is the atom centered difference between the all electron
        and pseudo quantities inside augmentation sphere a:

        Δf_a[n_a,ñ_a](r) = f(n_a(r)) - f(ñ_a(r)).
        """
        self.context.print('    Calculating PAW corrections\n')

        # Extract reciprocal lattice vectors
        nG = pd.ngmax
        G_Gv = pd.get_reciprocal_vectors()
        assert G_Gv.shape[0] == nG

        # Allocate output array
        fPAW_G = np.zeros(nG, dtype=complex)

        # Distribute plane waves
        G_myG = self._distribute_correction(nG)
        G_myGv = G_Gv[G_myG]

        # Calculate and add the PAW corrections from each augmentation sphere
        for a, (R_v, micro_setup) in enumerate(zip(R_av, micro_setups)):
            self._add_paw_correction(a, R_v, micro_setup,
                                     G_myG, G_myGv, fPAW_G)

        self.context.world.sum(fPAW_G)

        return fPAW_G

    def _distribute_correction(self, nG):
        world = self.context.world
        nGpr = (nG + world.size - 1) // world.size
        Ga = min(world.rank * nGpr, nG)
        Gb = min(Ga + nGpr, nG)

        return range(Ga, Gb)

    def _add_paw_correction(self, a, R_v, micro_setup, G_myG, G_myGv, fPAW_G):
        r"""Calculate the PAW correction of augmentation sphere a,

                              /
        Δf_a(G) = e^(-iG.R_a) |dr Δf_a[n_a,ñ_a](r) e^(-iG.r),
                              /
                              V0

        by expanding both the atom centered correction and the plane wave in
        real spherical harmonics, see [PRB 103, 245110 (2021)]:

                                 l               a
                              __ __             R_c
                              \  \      l    ^  /                   a
        Δf_a(G) = e^(-iG.R_a) /  /  (-i)  Y (G) |4πr^2 dr j(|G|r) Δf (r)
                              ‾‾ ‾‾        lm   /          l        lm
                              l m=-l            0

        The calculated atomic correction is then added to the output array."""
        # Radial and angular grid information
        rgd, Y_nL = micro_setup.rgd, micro_setup.Y_nL

        # Calculate df on Lebedev quadrature (angular grid) and radial grid
        df_ng = self._calculate_df(micro_setup)

        # Calculate the surface norm square of df
        dfSns_g = self._ang_int(df_ng ** 2)
        # Reduce radial grid by excluding points where dfSns_g = 0
        df_ng, r_g, dv_g = self._reduce_radial_grid(df_ng, rgd, dfSns_g)

        # Expand correction in real spherical harmonics
        df_gL = self._perform_rshe(df_ng, Y_nL)
        # Reduce expansion by removing coefficients that contribute less than
        # rshewmin on average
        df_gM, L_M, l_M = self._reduce_rshe(a, df_gL, dfSns_g)

        # Expand the plane waves in real spherical harmonics (and spherical
        # Bessel functions)
        (ii_MmyG,
         j_gMmyG,
         Y_MmyG) = self._expand_plane_waves(G_myGv, r_g, L_M, l_M)

        # Calculate the PAW correction as an integral over the radial grid
        # and rshe coefficients
        with self.context.timer('Integrate PAW correction'):
            angular_coef_MmyG = ii_MmyG * Y_MmyG
            # Radial integral, dv = 4πr^2
            radial_coef_MmyG = np.tensordot(j_gMmyG * df_gL[:, L_M,
                                                            np.newaxis],
                                            dv_g, axes=([0, 0]))
            # Angular integral (sum over l,m)
            atomic_corr_myG = np.sum(angular_coef_MmyG * radial_coef_MmyG,
                                     axis=0)

            position_prefactor_myG = np.exp(-1j * np.inner(G_myGv, R_v))

            # Add to output array
            fPAW_G[G_myG] += position_prefactor_myG * atomic_corr_myG

    @timer('Calculate PAW correction inside augmentation spheres')
    def _calculate_df(self, micro_setup):
        r"""Calculate Δf_a[n_a,ñ_a](r).

        Returns
        -------
        df_ng : nd.array
            (f_ng - ft_ng) where (n=Lebedev index, g=radial grid index)
        """
        rgd, Y_nL = micro_setup.rgd, micro_setup.Y_nL

        f_g = rgd.zeros()
        ft_g = rgd.zeros()
        df_ng = np.array([rgd.zeros() for n in range(len(R_nv))])
        for n, Y_L in enumerate(Y_nL):
            f_g[:] = 0.
            n_sg = np.dot(Y_L, micro_setup.n_sLg)
            self._add_f(rgd, n_sg, f_g)

            ft_g[:] = 0.
            nt_sg = np.dot(Y_L, micro_setup.nt_sLg)
            self._add_f(rgd, nt_sg, ft_g)

            df_ng[n, :] = f_g - ft_g

        return df_ng

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
        r"""Expand the angular dependence of Δf_a[n_a,ñ_a](r) in real spherical
        harmonics.

          a      / ^    ^                ^
        Δf (r) = |dr Y (r) Δf_a[n_a,ñ_a](rr)
          lm     /    lm

        Note that the Lebedev quadrature, which is used to perform the angular
        integral above, is exact up to polynomial order l=11. This implies that
        corrections containing angular components l<=5 can be expanded exactly.

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
        dfSw_gL = self._calculate_ns_weights(nL, df_gL, dfSns_g)
        rshew_L = np.average(dfSw_gL, axis=0)  # Average over the radial grid
        # Do the actual filtering
        L_M = np.where(rshew_L[L_L] > self.rshewmin)[0]
        l_M = [l_L[L] for L in L_M]
        df_gM = df_gL[:, L_M]

        # Print information about the final (reduced) expansion at atom a
        self.print_reduced_rshe_info(a, nL, dfSw_gL, rshew_L)

        return df_gM, L_M, l_M

    def _calculate_ns_weights(self, nL, df_gL, dfSns_g):
        """Calculate the weighted contribution of each rsh coefficient to the
        surface norm square of df as a function of radial grid index g."""
        nallL = df_gL.shape[1]
        dfSns_gL = np.repeat(dfSns_g, nallL).reshape(dfSns_g.shape[0], nallL)
        dfSw_gL = df_gL[self.dfmask_g] ** 2 / dfSns_gL[self.dfmask_g]

        return dfSw_gL

    def print_reduced_rshe_info(self, a, nL, dfSw_gL, rshew_L):
        """Print information about the reduced expansion in real spherical
        harmonics at atom (augmentation sphere) a."""
        p = partial(self.context.print, flush=False)
        p('    RSHE of atom', a)
        p('      {0:6}  {1:10}  {2:10}  {3:8}'.format('(l,m)',
                                                      'max weight',
                                                      'avg weight',
                                                      'included'))
        for L, (dfSw_g, rshew) in enumerate(zip(dfSw_gL.T, rshew_L)):
            self.print_rshe_coef_info(L, nL, dfSw_g, rshew)

        tot_avg_cov = np.average(np.sum(dfSw_gL, axis=1))
        avg_cov = np.average(np.sum(dfSw_gL[:, :nL]
                                    [:, rshew_L[:nL] > self.rshewmin], axis=1))
        p(f'      In total: {avg_cov} of the dfSns is covered on average')
        p(f'      In total: {tot_avg_cov} of the dfSns could be covered on'
          ' average')
        self.context.print('')

    def print_rshe_coef_info(self, L, nL, dfSw_g, rshew):
        """Print information about a specific rshe coefficient"""
        l = int(np.sqrt(L))
        m = L - l**2 - l
        included = 'yes' if (rshew > self.rshewmin and L < nL) else 'no'
        info = '      {0:6}  {1:1.8f}  {2:1.8f}  {3:8}'.format(f'({l},{m})',
                                                               np.max(dfSw_g),
                                                               rshew, included)
        self.context.print(info, flush=False)

    @timer('Expand plane waves in real spherical harmonics')
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

        with self.context.timer('Compute spherical bessel functions'):
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


def fft_from_grid(f_R, pd):
    r"""Perform a FFT to reciprocal space:
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


# ---------- Local functions of the (spin-)density ---------- #


def add_total_density(gd, n_sR, n_R):
    n_R += np.sum(n_sR, axis=0)


def add_LSDA_Bxc(gd, n_sR, Bxc_R):
    """Calculate B^(xc) in the local spin-density approximation for a collinear
    system and add it to the output array Bxc_R:

                δE_xc[n,m]   1
    B^(xc)(r) = ‾‾‾‾‾‾‾‾‾‾ = ‾ [V_LSDA^↑(r) - V_LSDA^↓(r)]
                  δm(r)      2
    """
    # Allocate an array for the spin-dependent xc potential on the real
    # space grid
    v_sR = np.zeros(np.shape(n_sR))

    # Calculate the spin-dependent potential
    xc = XC('LDA')
    xc.calculate(gd, n_sR, v_sg=v_sR)

    # Add B^(xc) in real space to the output array
    Bxc_R += (v_sR[0] - v_sR[1]) / 2

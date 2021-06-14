"""Contains methods for calculating LR-TDDFT kernels.
Substitutes gpaw.response.fxc in the new format."""

from pathlib import Path

import numpy as np
from scipy.special import spherical_jn

from gpaw.utilities import convert_string_to_fd
from ase.utils.timing import Timer, timer
from ase.units import Bohr

import gpaw.mpi as mpi
from gpaw.xc import XC
from gpaw.spherical_harmonics import Yarr
from gpaw.sphere.lebedev import weight_n, R_nv
from gpaw.response.kspair import get_calc


def get_fxc(gs, fxc, response='susceptibility', mode='pw',
            world=mpi.world, txt='-', timer=None, **kwargs):
    """Factory function getting an initiated version of the fxc class."""
    functional = fxc

    if functional == 'RPA':
        # No exchange and correlation
        def dummy_fxc(*args, **kwargs):
            return None
        return dummy_fxc

    fxc = create_fxc(functional, response, mode)
    return fxc(gs, functional, world=world, txt=txt, timer=timer, **kwargs)


def create_fxc(functional, response, mode):
    """Creator component for the FXC classes."""
    # Only one kind of response and mode is supported for now
    if functional in ['ALDA_x', 'ALDA_X', 'ALDA']:
        if response == 'susceptibility' and mode == 'pw':
            return AdiabaticSusceptibilityFXC
    raise ValueError(functional, response, mode)


class FXC:
    """Base class to calculate exchange-correlation kernels."""

    def __init__(self, gs, world=mpi.world, txt='-', timer=None):
        """
        Parameters
        ----------
        gs : str/obj
            Filename or GPAW calculator object of ground state calculation
        world : mpi.world
        txt : str or filehandle
            defines output file through gpaw.utilities.convert_string_to_fd
        timer : ase.utils.timing.Timer instance
        """
        # Output .txt filehandle and timer
        self.world = world
        self.fd = convert_string_to_fd(txt, world)
        self.cfd = self.fd
        self.timer = timer or Timer()
        self.calc = get_calc(gs, fd=self.fd, timer=self.timer)

    def __call__(self, *args, txt=None, timer=None, **kwargs):
        # A specific output file can be supplied for each individual call
        if txt is not None:
            self.cfd = convert_string_to_fd(txt, self.world)
        else:
            self.cfd = self.fd

        if self.is_calculated():
            Kxc_GG = self.read(*args, **kwargs)
        else:
            # A specific timer may also be supplied
            if timer is not None:
                # Swap timers to use supplied one
                self.timer, timer = timer, self.timer

            if str(self.fd) != str(self.cfd):
                print('Calculating fxc', file=self.fd)

            Kxc_GG = self.calculate(*args, **kwargs)

            if timer is not None:
                # Swap timers back
                self.timer, timer = timer, self.timer

            self.write(Kxc_GG)

        return Kxc_GG

    def calculate(self, *args, **kwargs):
        raise NotImplementedError

    def is_calculated(self, *args, **kwargs):
        # Read/write has not been implemented
        return False

    def read(self, *args, **kwargs):
        raise NotImplementedError

    def write(self, Kxc_GG):
        # Not implemented
        pass


class PlaneWaveAdiabaticFXC(FXC):
    """Adiabatic exchange-correlation kernels in plane wave mode using PAW."""

    def __init__(self, gs, functional,
                 world=mpi.world, txt='-', timer=None,
                 rshelmax=-1, rshewmin=None, filename=None, **ignored):
        """
        Parameters
        ----------
        gs, world, txt, timer : see FXC
        functional : str
            xc-functional
        rshelmax : int or None
            Expand kernel in real spherical harmonics inside augmentation
            spheres. If None, the kernel will be calculated without
            augmentation. The value of rshelmax indicates the maximum index l
            to perform the expansion in (l < 6).
        rshewmin : float or None
            If None, the kernel correction will be fully expanded up to the
            chosen lmax. Given as a float, (0 < rshewmin < 1) indicates what
            coefficients to use in the expansion. If any coefficient
            contributes with less than a fraction of rshewmin on average,
            it will not be included.
        """
        FXC.__init__(self, gs, world=world, txt=txt, timer=timer)

        self.functional = functional

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

        self.filename = filename

    def is_calculated(self):
        if self.filename is None:
            return False
        return Path(self.filename).is_file()

    def write(self, Kxc_GG):
        if self.filename is not None:
            np.save(self.filename, Kxc_GG)

    def read(self, *unused, **ignored):
        return np.load(self.filename)

    @timer('Calculate XC kernel')
    def calculate(self, pd):
        print('Calculating fxc', file=self.cfd)
        # Get the spin density we need and allocate fxc
        n_sG = self.get_density_on_grid(pd.gd)
        fxc_G = np.zeros(np.shape(n_sG[0]))

        print('    Calculating fxc on real space grid', file=self.cfd)
        self._add_fxc(pd.gd, n_sG, fxc_G)

        # Fourier transform to reciprocal space
        Kxc_GG = self.ft_from_grid(fxc_G, pd)

        if self.rshe:  # Do PAW correction to Fourier transformed kernel
            KxcPAW_GG = self.calculate_kernel_paw_correction(pd)
            Kxc_GG += KxcPAW_GG

        print('', file=self.cfd)

        return Kxc_GG / pd.gd.volume

    def get_density_on_grid(self, gd):
        """Get the spin density on coarse real-space grid.

        Returns
        -------
        nt_sG or n_sG : nd.array
            Spin density on coarse real-space grid. If not self.rshe, use
            the PAW corrected all-electron spin density.
        """
        if self.rshe:
            return self.calc.density.nt_sG  # smooth density

        print('    Calculating all-electron density', file=self.cfd)
        with self.timer('Calculating all-electron density'):
            get_ae_density = self.calc.density.get_all_electron_density
            n_sG, gd1 = get_ae_density(atoms=self.calc.atoms, gridrefinement=1)
            assert gd1 is gd
            assert gd1.comm.size == 1

            return n_sG

    @timer('Fourier transform of kernel from real-space grid')
    def ft_from_grid(self, fxc_G, pd):
        print('    Fourier transforming kernel from real-space grid',
              file=self.cfd)
        nG = pd.gd.N_c
        nG0 = nG[0] * nG[1] * nG[2]

        tmp_g = np.fft.fftn(fxc_G) * pd.gd.volume / nG0

        # The unfolding procedure could use vectorization and parallelization.
        # This remains a slow step for now.
        Kxc_GG = np.zeros((pd.ngmax, pd.ngmax), dtype=complex)
        for iG, iQ in enumerate(pd.Q_qG[0]):
            iQ_c = (np.unravel_index(iQ, nG) + nG // 2) % nG - nG // 2
            for jG, jQ in enumerate(pd.Q_qG[0]):
                jQ_c = (np.unravel_index(jQ, nG) + nG // 2) % nG - nG // 2
                ijQ_c = (iQ_c - jQ_c)
                if (abs(ijQ_c) < nG // 2).all():
                    Kxc_GG[iG, jG] = tmp_g[tuple(ijQ_c)]

        return Kxc_GG

    @timer('Calculate PAW corrections to kernel')
    def calculate_kernel_paw_correction(self, pd):
        print("    Calculating PAW corrections to the kernel\n",
              file=self.cfd)

        # Calculate (G-G') reciprocal space vectors
        dG_GGv = self._calculate_dG(pd)

        # Reshape to composite K = (G, G') index
        dG_Kv = dG_GGv.reshape(-1, dG_GGv.shape[-1])

        # Find unique dG-vectors
        dG_dGv, dG_K = np.unique(dG_Kv, return_inverse=True, axis=0)
        ndG = len(dG_dGv)

        # Allocate array and distribute plane waves
        KxcPAW_dG = np.zeros(ndG, dtype=complex)
        dG_mydG = self._distribute_correction(ndG)
        dG_mydGv = dG_dGv[dG_mydG]

        # Calculate my (G-G') reciprocal space vector lengths and directions
        dGl_mydG, dGn_mydGv = self._normalize_by_length(dG_mydGv)

        # Calculate PAW correction to each augmentation sphere (to each atom)
        R_av = self.calc.atoms.positions / Bohr
        for a, R_v in enumerate(R_av):
            # Calculate dfxc on Lebedev quadrature and radial grid
            # Please note: Using the radial grid descriptor with _add_fxc
            # might give problems beyond ALDA
            df_ng, Y_nL, rgd = self._calculate_dfxc(a)

            # Calculate the surface norm square of df
            dfSns_g = self._ang_int(df_ng ** 2)
            # Reduce radial grid by excluding points where dfSns_g = 0
            df_ng, r_g, dv_g = self._reduce_radial_grid(df_ng, rgd, dfSns_g)

            # Expand correction in real spherical harmonics
            df_gL = self._perform_rshe(df_ng, Y_nL)
            # Reduce expansion by removing coefficients that do not contribute
            df_gM, L_M, l_M = self._reduce_rsh_expansion(a, df_gL, dfSns_g)

            # Expand plane wave differences (G-G')
            (ii_MmydG,
             j_gMmydG,
             Y_MmydG) = self._expand_plane_waves(dGl_mydG, dGn_mydGv,
                                                 r_g, L_M, l_M)

            # Perform integration
            with self.timer('Integrate PAW correction'):
                coefatomR_dG = np.exp(-1j * np.inner(dG_mydGv, R_v))
                coefatomang_MdG = ii_MmydG * Y_MmydG
                coefatomrad_MdG = np.tensordot(j_gMmydG * df_gL[:, L_M,
                                                                np.newaxis],
                                               dv_g, axes=([0, 0]))
                coefatom_dG = np.sum(coefatomang_MdG * coefatomrad_MdG, axis=0)
                KxcPAW_dG[dG_mydG] += coefatom_dG * coefatomR_dG

        self.world.sum(KxcPAW_dG)

        # Unfold PAW correction
        KxcPAW_GG = KxcPAW_dG[dG_K].reshape(dG_GGv.shape[:2])

        return KxcPAW_GG

    def _calculate_dG(self, pd):
        """Calculate (G-G') reciprocal space vectors"""
        npw = pd.ngmax
        G_Gv = pd.get_reciprocal_vectors()

        # Distribute dG to calculate
        nGpr = (npw + self.world.size - 1) // self.world.size
        Ga = min(self.world.rank * nGpr, npw)
        Gb = min(Ga + nGpr, npw)
        G_myG = range(Ga, Gb)

        # Calculate dG_v for every set of (G-G')
        dG_GGv = np.zeros((npw, npw, 3))
        for v in range(3):
            dG_GGv[Ga:Gb, :, v] = np.subtract.outer(G_Gv[G_myG, v], G_Gv[:, v])
        self.world.sum(dG_GGv)

        return dG_GGv

    def _distribute_correction(self, ndG):
        """Distribute correction"""
        ndGpr = (ndG + self.world.size - 1) // self.world.size
        dGa = min(self.world.rank * ndGpr, ndG)
        dGb = min(dGa + ndGpr, ndG)

        return range(dGa, dGb)

    @staticmethod
    def _normalize_by_length(dG_mydGv):
        """Find the length and direction of reciprocal lattice vectors."""
        dGl_mydG = np.linalg.norm(dG_mydGv, axis=1)
        dGn_mydGv = np.zeros_like(dG_mydGv)
        mask0 = np.where(dGl_mydG != 0.)
        dGn_mydGv[mask0] = dG_mydGv[mask0] / dGl_mydG[mask0][:, np.newaxis]

        return dGl_mydG, dGn_mydGv

    def _get_densities_in_augmentation_sphere(self, a):
        """Get the all-electron and smooth spin densities inside the
        augmentation spheres.

        Returns
        -------
        n_sLg : nd.array
            all-electron density
        nt_sLg : nd.array
            smooth density
        (s=spin, L=(l,m) spherical harmonic index, g=radial grid index)
        """
        setup = self.calc.wfs.setups[a]
        n_qg = setup.xc_correction.n_qg
        nt_qg = setup.xc_correction.nt_qg
        nc_g = setup.xc_correction.nc_g
        nct_g = setup.xc_correction.nct_g

        D_sp = self.calc.density.D_asp[a]
        B_pqL = setup.xc_correction.B_pqL
        D_sLq = np.inner(D_sp, B_pqL.T)
        nspins = len(D_sp)

        n_sLg = np.dot(D_sLq, n_qg)
        nt_sLg = np.dot(D_sLq, nt_qg)

        # Add core density
        n_sLg[:, 0] += np.sqrt(4. * np.pi) / nspins * nc_g
        nt_sLg[:, 0] += np.sqrt(4. * np.pi) / nspins * nct_g

        return n_sLg, nt_sLg

    @timer('Calculate PAW correction inside augmentation spheres')
    def _calculate_dfxc(self, a):
        """Calculate the difference between fxc of the all-electron spin
        density and fxc of the smooth spin density.

        Returns
        -------
        df_ng : nd.array
            (f_ng - ft_ng) where (n=Lebedev index, g=radial grid index)
        Y_nL : nd.array
            real spherical harmonics on Lebedev quadrature
        rgd : GridDescriptor
            non-linear radial grid descriptor
        """
        # Extract spin densities from ground state calculation
        n_sLg, nt_sLg = self._get_densities_in_augmentation_sphere(a)

        setup = self.calc.wfs.setups[a]
        Y_nL = setup.xc_correction.Y_nL
        rgd = setup.xc_correction.rgd
        f_g = rgd.zeros()
        ft_g = rgd.zeros()
        df_ng = np.array([rgd.zeros() for n in range(len(R_nv))])
        for n, Y_L in enumerate(Y_nL):
            f_g[:] = 0.
            n_sg = np.dot(Y_L, n_sLg)
            self._add_fxc(rgd, n_sg, f_g)

            ft_g[:] = 0.
            nt_sg = np.dot(Y_L, nt_sLg)
            self._add_fxc(rgd, nt_sg, ft_g)

            df_ng[n, :] = f_g - ft_g

        return df_ng, Y_nL, rgd

    @staticmethod
    def _ang_int(f_nA):
        """ Make surface integral on a sphere using Lebedev quadrature """
        return 4. * np.pi * np.tensordot(weight_n, f_nA, axes=([0], [0]))

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
        """Perform expansion of dfxc in real spherical harmonics. Note that the
        Lebedev quadrature, which is used to calculate the expansion
        coefficients, is exact to order l=11. This implies that functions
        containing angular components l<=5 can be expanded exactly.
        Assumes df_ng to be a real function.

        Returns
        -------
        df_gL : nd.array
            dfxc in g=radial grid index, L=(l,m) spherical harmonic index
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

    def _reduce_rsh_expansion(self, a, df_gL, dfSns_g):
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
        for l in range(int(np.sqrt(nL))):
            l_L += [l] * (2 * l + 1)

        # Filter away (l,m)-coefficients that do not contribute
        rshew_L = self._evaluate_rshe_coefficients(a, nL, df_gL, dfSns_g)
        L_M = np.where(rshew_L[L_L] > self.rshewmin)[0]
        l_M = [l_L[L] for L in L_M]
        df_gM = df_gL[:, L_M]

        return df_gM, L_M, l_M

    def _evaluate_rshe_coefficients(self, a, nL, df_gL, dfSns_g):
        """If some of the rshe coefficients are very small for all radii g,
        we may choose to exclude them from the kernel PAW correction.

        The "smallness" is evaluated from their average weight in
        evaluating the surface norm square for each radii g.
        """
        # Compute each coefficient's fraction of the surface norm square
        nallL = df_gL.shape[1]
        dfSns_gL = np.repeat(dfSns_g, nallL).reshape(dfSns_g.shape[0], nallL)
        dfSw_gL = df_gL[self.dfmask_g] ** 2 / dfSns_gL[self.dfmask_g]

        # The smallness is evaluated from the average
        rshew_L = np.average(dfSw_gL, axis=0)

        # Print information about the expansion
        print('    RSHE of atom', a, file=self.cfd)
        print('      {0:6}  {1:10}  {2:10}  {3:8}'.format('(l,m)',
                                                          'max weight',
                                                          'avg weight',
                                                          'included'),
              file=self.cfd)
        for L, (dfSw_g, rshew) in enumerate(zip(dfSw_gL.T, rshew_L)):
            self.print_rshe_info(L, nL, dfSw_g, rshew)

        tot_avg_cov = np.average(np.sum(dfSw_gL, axis=1))
        avg_cov = np.average(np.sum(dfSw_gL[:, :nL]
                                    [:, rshew_L[:nL] > self.rshewmin], axis=1))
        print(f'      In total: {avg_cov} of the dfSns is covered on average',
              file=self.cfd)
        print(f'      In total: {tot_avg_cov} of the dfSns could be covered',
              'on average\n', flush=True, file=self.cfd)

        return rshew_L

    def print_rshe_info(self, L, nL, dfSw_g, rshew):
        """Print information about the importance of the rshe coefficient"""
        l = int(np.sqrt(L))
        m = L - l**2 - l
        included = 'yes' if (rshew > self.rshewmin and L < nL) else 'no'
        print('      {0:6}  {1:1.8f}  {2:1.8f}  {3:8}'.format(f'({l},{m})',
                                                              np.max(dfSw_g),
                                                              rshew, included),
              file=self.cfd)

    @timer('Expand plane waves')
    def _expand_plane_waves(self, dG_mydG, dGn_mydGv, r_g, L_M, l_M):
        r"""Expand plane waves in spherical Bessel functions and real spherical
        harmonics:
                         l
                     __  __
         -iK.r       \   \      l             ^     ^
        e      = 4pi /   /  (-i)  j (|K|r) Y (K) Y (r)
                     ‾‾  ‾‾        l        lm    lm
                     l  m=-l

        Returns
        -------
        ii_MmydG : nd.array
            (-i)^l for used (l,m) coefficients M
        j_gMmydG : nd.array
            j_l(|dG|r) for used (l,m) coefficients M
        Y_MmydG : nd.array
                 ^
            Y_lm(K) for used (l,m) coefficients M
        """
        nmydG = len(dG_mydG)
        # Setup arrays to fully vectorize computations
        nM = len(L_M)
        (r_gMmydG, l_gMmydG,
         dG_gMmydG) = [a.reshape(len(r_g), nM, nmydG)
                       for a in np.meshgrid(r_g, l_M, dG_mydG, indexing='ij')]

        with self.timer('Compute spherical bessel functions'):
            # Slow step
            j_gMmydG = spherical_jn(l_gMmydG, dG_gMmydG * r_gMmydG)

        Y_MmydG = Yarr(L_M, dGn_mydGv)
        ii_X = (-1j) ** np.repeat(l_M, nmydG)
        ii_MmydG = ii_X.reshape((nM, nmydG))

        return ii_MmydG, j_gMmydG, Y_MmydG

    def _add_fxc(self, gd, n_sg, fxc_g):
        raise NotImplementedError


class AdiabaticSusceptibilityFXC(PlaneWaveAdiabaticFXC):
    """Adiabatic exchange-correlation kernel for susceptibility calculations in
    the plane wave mode"""

    def __init__(self, gs, functional,
                 world=mpi.world, txt='-', timer=None,
                 rshelmax=-1, rshewmin=None, filename=None,
                 density_cut=None, spinpol_cut=None, **ignored):
        """
        gs, world, txt, timer : see PlaneWaveAdiabaticFXC, FXC
        functional, rshelmax, rshewmin, filename : see PlaneWaveAdiabaticFXC
        density_cut : float
            cutoff density below which f_xc is set to zero
        spinpol_cut : float
            Cutoff spin polarization. Below, f_xc is evaluated in zeta=0 limit
            Note: only implemented for spincomponents '+-' and '-+'
        """
        assert functional in ['ALDA_x', 'ALDA_X', 'ALDA']

        PlaneWaveAdiabaticFXC.__init__(self, gs, functional,
                                       world=world, txt=txt, timer=timer,
                                       rshelmax=rshelmax, rshewmin=rshewmin,
                                       filename=filename)

        self.density_cut = density_cut
        self.spinpol_cut = spinpol_cut

    def calculate(self, spincomponent, pd):
        """Creator component to set up the right calculation."""
        if spincomponent in ['00', 'uu', 'dd']:
            assert self.spinpol_cut is None
            assert len(self.calc.density.nt_sG) == 1  # nspins, see XXX below

            self._calculate_fxc = self.calculate_dens_fxc
            self._calculate_unpol_fxc = None
        elif spincomponent in ['+-', '-+']:
            assert len(self.calc.density.nt_sG) == 2  # nspins

            self._calculate_fxc = self.calculate_trans_fxc
            self._calculate_unpol_fxc = self.calculate_trans_unpol_fxc
        else:
            raise ValueError(spincomponent)

        return PlaneWaveAdiabaticFXC.calculate(self, pd)

    def _add_fxc(self, gd, n_sG, fxc_G):
        """
        Calculate fxc, using the cutoffs from input above

        ALDA_x is an explicit algebraic version
        ALDA_X uses the libxc package
        """
        _calculate_fxc = self._calculate_fxc
        _calculate_unpol_fxc = self._calculate_unpol_fxc

        # Mask small zeta
        if self.spinpol_cut is not None:
            zetasmall_G = np.abs((n_sG[0] - n_sG[1]) /
                                 (n_sG[0] + n_sG[1])) < self.spinpol_cut
        else:
            zetasmall_G = np.full(np.shape(n_sG[0]), False,
                                  np.array(False).dtype)

        # Mask small n
        if self.density_cut:
            npos_G = np.abs(np.sum(n_sG, axis=0)) > self.density_cut
        else:
            npos_G = np.full(np.shape(n_sG[0]), True, np.array(True).dtype)

        # Don't use small zeta limit if n is small
        zetasmall_G = np.logical_and(zetasmall_G, npos_G)

        # In small zeta limit, use unpolarized fxc
        if zetasmall_G.any():
            fxc_G[zetasmall_G] += _calculate_unpol_fxc(gd, n_sG)[zetasmall_G]

        # Set fxc to zero if n is small
        allfine_G = np.logical_and(np.invert(zetasmall_G), npos_G)

        # Above both spinpol_cut and density_cut calculate polarized fxc
        fxc_G[allfine_G] += _calculate_fxc(gd, n_sG)[allfine_G]

    def calculate_dens_fxc(self, gd, n_sG):
        if self.functional == 'ALDA_x':
            n_G = np.sum(n_sG, axis=0)
            fx_G = -1. / 3. * (3. / np.pi)**(1. / 3.) * n_G**(-2. / 3.)
            return fx_G

        assert len(n_sG) == 1
        from gpaw.xc.libxc import LibXC
        kernel = LibXC(self.functional[1:])
        fxc_sG = np.zeros_like(n_sG)
        kernel.xc.calculate_fxc_spinpaired(n_sG.ravel(), fxc_sG)

        return fxc_sG[0]  # not tested for spin-polarized calculations XXX

    def calculate_trans_fxc(self, gd, n_sG):
        """Calculate polarized fxc of spincomponents '+-', '-+'."""
        m_G = n_sG[0] - n_sG[1]

        if self.functional == 'ALDA_x':
            fx_G = - (6. / np.pi)**(1. / 3.) \
                * (n_sG[0]**(1. / 3.) - n_sG[1]**(1. / 3.)) / m_G
            return fx_G
        else:
            v_sG = np.zeros(np.shape(n_sG))
            xc = XC(self.functional[1:])
            xc.calculate(gd, n_sG, v_sg=v_sG)

            return (v_sG[0] - v_sG[1]) / m_G

    def calculate_trans_unpol_fxc(self, gd, n_sG):
        """Calculate unpolarized fxc of spincomponents '+-', '-+'."""
        n_G = np.sum(n_sG, axis=0)
        fx_G = - (3. / np.pi)**(1. / 3.) * 2. / 3. * n_G**(-2. / 3.)
        if self.functional in ('ALDA_x', 'ALDA_X'):
            return fx_G
        else:
            # From Perdew & Wang 1992
            A = 0.016887
            a1 = 0.11125
            b1 = 10.357
            b2 = 3.6231
            b3 = 0.88026
            b4 = 0.49671

            rs_G = 3. / (4. * np.pi) * n_G**(-1. / 3.)
            X_G = 2. * A * (b1 * rs_G**(1. / 2.)
                            + b2 * rs_G + b3 * rs_G**(3. / 2.) + b4 * rs_G**2.)
            ac_G = 2. * A * (1 + a1 * rs_G) * np.log(1. + 1. / X_G)

            fc_G = 2. * ac_G / n_G

            return fx_G + fc_G

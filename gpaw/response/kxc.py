"""Contains methods for calculating LR-TDDFT kernels.
Substitutes gpaw.response.fxc in the new format."""

from pathlib import Path

import numpy as np
from scipy.special import spherical_jn

from ase.utils import convert_string_to_fd
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
            defines output file through ase.utils.convert_string_to_fd
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
                 rshe=0.99, filename=None, **ignored):
        """
        Parameters
        ----------
        gs, world, txt, timer : see FXC
        functional : str
            xc-functional
        rshe : float or None
            Expand kernel in real spherical harmonics inside augmentation
            spheres. If None, the kernel will be calculated without
            augmentation. The value of rshe (0<rshe<1) sets a convergence
            criteria for the expansion in real spherical harmonics.
        """
        FXC.__init__(self, gs, world=world, txt=txt, timer=timer)

        self.functional = functional

        self.rshe = rshe is not None

        if self.rshe:
            self.rshecc = rshe
            self.dfSns_g = None
            self.dfSns_gL = None
            self.dfmask_g = None
            self.rsheconvmin = None

            self.rsheL_M = None

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
        print("    Calculating PAW corrections to the kernel",
              file=self.cfd)

        # Allocate array and distribute plane waves
        npw = pd.ngmax
        KxcPAW_GG = np.zeros((npw, npw), dtype=complex)
        G_myG = self._distribute_correction(npw)

        # Calculate (G-G') reciprocal space vectors, their length and direction
        dG_myGGv, dG_myGG, dGn_myGGv = self._calculate_dG(pd, G_myG)

        # Calculate PAW correction to each augmentation sphere (to each atom)
        R_av = self.calc.atoms.positions / Bohr
        for a, R_v in enumerate(R_av):
            # Calculate dfxc on Lebedev quadrature and radial grid
            # Please note: Using the radial grid descriptor with _add_fxc
            # might give problems beyond ALDA
            df_ng, Y_nL, rgd = self._calculate_dfxc(a)

            # Calculate the surface norm square of df
            self.dfSns_g = self._ang_int(df_ng ** 2)
            # Reduce radial grid by excluding points where dfSns_g = 0
            df_ng, r_g, dv_g = self._reduce_radial_grid(df_ng, rgd)

            # Expand correction in real spherical harmonics
            df_gL = self._perform_rshe(a, df_ng, Y_nL)
            # Reduce expansion by removing coefficients that are zero
            df_gM, L_M, l_M = self._reduce_rsh_expansion(df_gL)

            # Expand plane wave differences (G-G')
            (ii_MmyGG,
             j_gMmyGG,
             Y_MmyGG) = self._expand_plane_waves(dG_myGG, dGn_myGGv,
                                                 r_g, L_M, l_M)

            # Perform integration
            with self.timer('Integrate PAW correction'):
                coefatomR_GG = np.exp(-1j * np.inner(dG_myGGv, R_v))
                coefatomang_MGG = ii_MmyGG * Y_MmyGG
                coefatomrad_MGG = np.tensordot(j_gMmyGG * df_gL[:, L_M,
                                                                np.newaxis,
                                                                np.newaxis],
                                               dv_g, axes=([0, 0]))
                coefatom_GG = np.sum(coefatomang_MGG * coefatomrad_MGG, axis=0)
                KxcPAW_GG[G_myG] += coefatom_GG * coefatomR_GG

        self.world.sum(KxcPAW_GG)

        return KxcPAW_GG

    def _distribute_correction(self, npw):
        """Distribute correction"""
        nGpr = (npw + self.world.size - 1) // self.world.size
        Ga = min(self.world.rank * nGpr, npw)
        Gb = min(Ga + nGpr, npw)

        return range(Ga, Gb)

    def _calculate_dG(self, pd, G_myG):
        """Calculate (G-G') reciprocal space vectors,
        their length and direction"""
        npw = pd.ngmax
        G_Gv = pd.get_reciprocal_vectors()

        # Calculate bare dG
        dG_myGGv = np.zeros((len(G_myG), npw, 3))
        for v in range(3):
            dG_myGGv[:, :, v] = np.subtract.outer(G_Gv[G_myG, v], G_Gv[:, v])

        # Find length of dG and the normalized dG
        dG_myGG = np.linalg.norm(dG_myGGv, axis=2)
        dGn_myGGv = np.zeros_like(dG_myGGv)
        mask0 = np.where(dG_myGG != 0.)
        dGn_myGGv[mask0] = dG_myGGv[mask0] / dG_myGG[mask0][:, np.newaxis]

        return dG_myGGv, dG_myGG, dGn_myGGv

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

    def _ang_int(self, f_nA):
        """ Make surface integral on a sphere using Lebedev quadrature """
        return 4. * np.pi * np.tensordot(weight_n, f_nA, axes=([0], [0]))

    def _reduce_radial_grid(self, df_ng, rgd):
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
        self.dfmask_g = np.where(self.dfSns_g > 0.)
        ng = np.max(self.dfmask_g) + 1

        # Integrate only r-values inside augmentation sphere
        df_ng = df_ng[:, :ng]

        r_g = rgd.r_g[:ng]
        dv_g = rgd.dv_g[:ng]

        return df_ng, r_g, dv_g

    @timer('Expand PAW correction in real spherical harmonics')
    def _perform_rshe(self, a, df_ng, Y_nL):
        """Perform expansion of dfxc in real spherical harmonics. Note that the
        Lebedev quadrature, which is used to calculate the expansion
        coefficients, is exact to order l=11. This implies that functions
        containing angular components l<=5 can be expanded exactly.

        Returns
        -------
        df_gL : nd.array
            dfxc in g=radial grid index, L=(l,m) spherical harmonic index
        """
        L_L = []
        l_L = []
        nL = min(Y_nL.shape[1], 36)
        # The convergence of the expansion is tracked through the
        # surface norm square of df
        self.dfSns_gL = np.repeat(self.dfSns_g,
                                  nL).reshape(self.dfSns_g.shape[0], nL)
        # Initialize convergence criteria
        self.rsheconvmin = 0.

        # Add real spherical harmonics to fulfill convergence criteria.
        df_gL = np.zeros((df_ng.shape[1], nL))
        l = 0
        while self.rsheconvmin < self.rshecc:
            if l > int(np.sqrt(nL) - 1):
                raise Exception('Could not expand %.f of' % self.rshecc
                                + ' PAW correction to atom %d in ' % a
                                + 'real spherical harmonics up to '
                                + 'order l=%d' % int(np.sqrt(nL) - 1))

            L_L += range(l**2, l**2 + 2 * l + 1)
            l_L += [l] * (2 * l + 1)

            self._add_rshe_coefficients(df_ng, df_gL, Y_nL, l)
            self._evaluate_rshe_convergence(df_gL)

            print('    At least a fraction of '
                  + '%.8f' % self.rsheconvmin
                  + ' of the PAW correction to atom %d could be ' % a
                  + 'expanded in spherical harmonics up to l=%d' % l,
                  file=self.cfd)
            l += 1

        return df_gL

    def _add_rshe_coefficients(self, df_ng, df_gL, Y_nL, l):
        """
        Adds the l-components in the real spherical harmonic expansion
        of df_ng to df_gL.
        Assumes df_ng to be a real function.
        """
        nm = 2 * l + 1
        L_L = np.arange(l**2, l**2 + nm)
        df_ngm = np.repeat(df_ng, nm, axis=1).reshape((*df_ng.shape, nm))
        Y_ngm = np.repeat(Y_nL[:, L_L],
                          df_ng.shape[1], axis=0).reshape((*df_ng.shape, nm))
        df_gL[:, L_L] = self._ang_int(Y_ngm * df_ngm)

    def _evaluate_rshe_coefficients(self, f_gL):
        """
        Checks weither some (l,m)-coefficients are very small for all g,
        in that case they can be excluded from the expansion.
        """
        fc_L = np.sum(f_gL[self.dfmask_g]**2 / self.dfSns_gL[self.dfmask_g],
                      axis=0)
        self.rsheL_M = fc_L > (1. - self.rshecc) * 1.e-3

    def _evaluate_rshe_convergence(self, f_gL):
        """ The convergence of the real spherical harmonics expansion is
        tracked by comparing the surface norm square calculated using the
        expansion and the full result.
        
        Find also the minimal fraction of f_ng captured by the expansion
        in real spherical harmonics f_gL.
        """
        self._evaluate_rshe_coefficients(f_gL)

        rsheconv_g = np.ones(f_gL.shape[0])
        dfSns_g = np.sum(f_gL[:, self.rsheL_M]**2, axis=1)
        rsheconv_g[self.dfmask_g] = dfSns_g[self.dfmask_g]
        rsheconv_g[self.dfmask_g] /= self.dfSns_g[self.dfmask_g]

        self.rsheconvmin = np.min(rsheconv_g)

    def _reduce_rsh_expansion(self, df_gL):
        """Reduce the composite index L=(l,m) to M, which indexes non-zero
        coefficients in the expansion only.

        Returns
        -------
        df_gM : nd.array
            PAW correction in reduced rsh index
        L_M : nd.array
            L=(l,m) spherical harmonics indices in reduced rsh index
        l_M : list
            l spherical harmonics indices in reduced rsh index
        """
        # Recreate l_L array
        nL = df_gL.shape[1]
        l_L = []
        for l in range(int(np.sqrt(nL))):
            l_L += [l] * (2 * l + 1)
            
        # Filter away unused (l,m)-coefficients
        L_M = np.where(self.rsheL_M)[0]
        l_M = [l_L[L] for L in L_M]
        df_gM = df_gL[:, L_M]

        return df_gM, L_M, l_M

    @timer('Expand plane waves')
    def _expand_plane_waves(self, dG_myGG, dGn_myGGv, r_g, L_M, l_M):
        """Expand plane waves in spherical Bessel functions and real spherical
        harmonics:
                         l
                     __  __
         -iK.r       \   \      l             ^     ^
        e      = 4pi /   /  (-i)  j (|K|r) Y (K) Y (r)
                     ‾‾  ‾‾        l        lm    lm
                     l  m=-l

        Returns
        -------
        ii_MmyGG : nd.array
            (-i)^l for used (l,m) coefficients M
        j_gMmyGG : nd.array
            j_l(|dG|r) for used (l,m) coefficients M
        Y_MmyGG : nd.array
                 ^
            Y_lm(K) for used (l,m) coefficients M
        """
        # Setup arrays to fully vectorize computations
        nM = len(L_M)
        (r_gMmyGG, l_gMmyGG,
         dG_gMmyGG) = [a.reshape(len(r_g), nM, *dG_myGG.shape)
                       for a in np.meshgrid(r_g, l_M, dG_myGG.flatten(),
                                            indexing='ij')]

        with self.timer('Compute spherical bessel functions'):
            # Slow step. If it ever gets too slow, one can use the same
            # philosophy as _ft_from_grid, where dG=(G-G') results are
            # "unfolded" from a fourier transform to all unique K=dG
            # reciprocal lattice vectors. It should be possible to vectorize
            # the unfolding procedure to make it fast.
            j_gMmyGG = spherical_jn(l_gMmyGG, dG_gMmyGG * r_gMmyGG)

        Y_MmyGG = Yarr(L_M, dGn_myGGv)
        ii_MK = (-1j) ** np.repeat(l_M,
                                   np.prod(dG_myGG.shape))
        ii_MmyGG = ii_MK.reshape((nM, *dG_myGG.shape))

        return ii_MmyGG, j_gMmyGG, Y_MmyGG

    def _add_fxc(self, gd, n_sg, fxc_g):
        raise NotImplementedError


class AdiabaticSusceptibilityFXC(PlaneWaveAdiabaticFXC):
    """Adiabatic exchange-correlation kernel for susceptibility calculations in
    the plane wave mode"""

    def __init__(self, gs, functional,
                 world=mpi.world, txt='-', timer=None,
                 rshe=0.99, filename=None,
                 density_cut=None, spinpol_cut=None, **ignored):
        """
        gs, world, txt, timer : see PlaneWaveAdiabaticFXC, FXC
        functional, rshe, filename : see PlaneWaveAdiabaticFXC
        density_cut : float
            cutoff density below which f_xc is set to zero
        spinpol_cut : float
            Cutoff spin polarization. Below, f_xc is evaluated in zeta=0 limit
            Note: only implemented for spincomponents '+-' and '-+'
        """
        assert functional in ['ALDA_x', 'ALDA_X', 'ALDA']

        PlaneWaveAdiabaticFXC.__init__(self, gs, functional,
                                       world=world, txt=txt, timer=timer,
                                       rshe=rshe, filename=filename)

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
        else:
            fxc_sG = np.zeros_like(n_sG)
            xc = XC(self.functional[1:])
            xc.calculate_fxc(gd, n_sG, fxc_sG)

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

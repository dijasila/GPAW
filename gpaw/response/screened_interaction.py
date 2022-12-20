import numpy as np
from math import pi
from gpaw.response.q0_correction import Q0Correction
from ase.units import Ha
from ase.dft.kpoints import monkhorst_pack

import gpaw.mpi as mpi
from gpaw.kpt_descriptor import KPointDescriptor

from gpaw.response import ResponseContext
from gpaw.response.pw_parallelization import Blocks1D
from gpaw.response.gamma_int import GammaIntegrator
from gpaw.response.temp import DielectricFunctionCalculator


def get_qdescriptor(kd, atoms):
    # Find q-vectors and weights in the IBZ:
    assert -1 not in kd.bz2bz_ks
    offset_c = 0.5 * ((kd.N_c + 1) % 2) / kd.N_c
    bzq_qc = monkhorst_pack(kd.N_c) + offset_c
    qd = KPointDescriptor(bzq_qc)
    qd.set_symmetry(atoms, kd.symmetry)
    return qd


def initialize_w_calculator(chi0calc, txt='w.txt', ppa=False, xc='RPA',
                            world=mpi.world, timer=None,
                            E0=Ha, Eg=None, fxc_mode='GW', *,
                            coulomb, integrate_gamma=0,
                            q0_correction=False):
    """A function to initialize a WCalculator with more readable inputs
    than the actual calculator.
    chi0calc: Chi0Calculator

    txt: str
         Text output file
    xc: str
         Kernel to use when including vertex corrections.
    world: MPI communicator
    timer: timer
    Eg: float
        Gap to apply in the 'JGMs' (simplified jellium-with-gap) kernel.
        If None the DFT gap is used.
    truncation: str or None
         Coulomb truncation scheme. Can be None, 'wigner-seitz', or '2D'.
    integrate_gamma: int
         Method to integrate the Coulomb interaction. 1 is a numerical
         integration at all q-points with G=[0,0,0] - this breaks the
         symmetry slightly. 0 is analytical integration at q=[0,0,0] only
         this conserves the symmetry. integrate_gamma=2 is the same as 1,
         but the average is only carried out in the non-periodic directions.
    Remaining arguments: See WCalculator
    """
    from gpaw.response.g0w0_kernels import G0W0Kernel
    gs = chi0calc.gs
    context = ResponseContext(txt=txt, timer=timer, world=world)
    if Eg is None and xc == 'JGMsx':
        Eg = gs.get_band_gap()
    elif Eg is not None:
        Eg /= Ha

    xckernel = G0W0Kernel(xc=xc, ecut=chi0calc.ecut,
                          gs=gs,
                          ns=gs.nspins,
                          wd=chi0calc.wd,
                          Eg=Eg,
                          context=context)

    wcalc = WCalculator(gs=gs, ppa=ppa,
                        xckernel=xckernel,
                        context=context,
                        E0=E0,
                        coulomb=coulomb,
                        integrate_gamma=integrate_gamma,
                        q0_correction=q0_correction)
    return wcalc


class WCalculator:
    def __init__(self, *,
                 gs,
                 ppa,
                 xckernel,
                 context,
                 E0,
                 coulomb,
                 integrate_gamma=0,
                 q0_correction=False):
        """
        W Calculator.

        Parameters
        ----------
        wd: FrequencyDescriptor
        pair: gpaw.response.pair.PairDensity instance
              Class for calculating matrix elements of pairs of wavefunctions.
        gs: calc.gs_adapter()
        ppa: bool
            Sets whether the Godby-Needs plasmon-pole approximation for the
            dielectric function should be used.
        xckernel: G0W0Kernel object
        context: ResponseContext object
        E0: float
            Energy (in eV) used for fitting the plasmon-pole approximation
        fxc_mode: str
            Where to include the vertex corrections; polarizability and/or
            self-energy. 'GWP': Polarizability only, 'GWS': Self-energy only,
            'GWG': Both.
        truncation: str
            Coulomb truncation scheme. Can be either wigner-seitz,
            2D, 1D, or 0D
        q0_correction: bool
            Analytic correction to the q=0 contribution applicable to 2D
            systems.

        Update documentation!                                                  XXX
        """
        self.ppa = ppa
        self.gs = gs
        self.coulomb = coulomb
        self.context = context
        self.integrate_gamma = integrate_gamma
        self.qd = get_qdescriptor(self.gs.kd, self.gs.atoms)
        self.xckernel = xckernel

        if q0_correction:
            assert self.coulomb.truncation == '2D'
            self.q0_corrector = Q0Correction(
                cell_cv=self.gs.gd.cell_cv,
                bzk_kc=self.gs.kd.bzk_kc,
                N_c=self.qd.N_c)

            npts_c = self.q0_corrector.npts_c
            self.context.print('Applying analytical 2D correction to W:',
                               flush=False)
            self.context.print('    Evaluating Gamma point contribution to W '
                               + 'on a %dx%dx%d grid' % tuple(npts_c))
        else:
            self.q0_corrector = None

        self.E0 = E0 / Ha

    def calculate(self, chi0,
                  fxc_mode='GW',
                  only_correlation=False,
                  out_dist='WgG'):
        """Calculate the screened interaction."""

        W_wGG = self._calculate(chi0, fxc_mode,
                                only_correlation=only_correlation)

        if out_dist == 'WgG':
            assert not self.ppa
            W_WgG = chi0.blockdist.distribute_as(W_wGG, chi0.nw, 'WgG')
            W_x = W_WgG
        elif out_dist == 'wGG':
            W_x = W_wGG
        else:
            raise ValueError(f'Invalid out_dist {out_dist}')

        return W_x

    def basic_dyson_arrays(self, pd, fxc_mode):
        delta_GG = np.eye(pd.ngmax)

        if fxc_mode == 'GW':
            fv = delta_GG
        else:
            fv = self.xckernel.calculate(pd)

        return delta_GG, fv

    def _calculate(self, chi0, fxc_mode,
                   only_correlation=False):
        """In-place calculation of the screened interaction."""
        # Unpack data
        pd = chi0.pd
        chi0_wGG = chi0.copy_array_with_distribution('wGG')
        chi0_Wvv = chi0.chi0_Wvv
        chi0_WxvG = chi0.chi0_WxvG

        q_c = pd.q_c
        kd = self.gs.kd
        wblocks1d = Blocks1D(chi0.blockdist.blockcomm, len(chi0.wd))

        delta_GG, fv = self.basic_dyson_arrays(pd, fxc_mode)

        if self.integrate_gamma != 0:
            reduced = (self.integrate_gamma == 2)
            V0, sqrtV0 = self.coulomb.integrated_kernel(pd=pd, reduced=reduced)
        elif self.integrate_gamma == 0 and np.allclose(q_c, 0):
            bzvol = (2 * np.pi)**3 / self.gs.volume / self.qd.nbzkpts
            Rq0 = (3 * bzvol / (4 * np.pi))**(1. / 3.)
            V0 = 16 * np.pi**2 * Rq0 / bzvol
            sqrtV0 = (4 * np.pi)**(1.5) * Rq0**2 / bzvol / 2

        if self.ppa:
            einv_wGG = []

        # Generate fine grid in vicinity of gamma
        # Use optical_limit check on chi0_data in the future XXX
        if np.allclose(q_c, 0) and len(chi0_wGG) > 0:
            gamma_int = GammaIntegrator(
                truncation=self.coulomb.truncation,
                kd=kd, pd=pd,
                chi0_wvv=chi0_Wvv[wblocks1d.myslice],
                chi0_wxvG=chi0_WxvG[wblocks1d.myslice])

        self.context.timer.start('Dyson eq.')
        for iw, chi0_GG in enumerate(chi0_wGG):
            if np.allclose(q_c, 0):
                einv_GG = np.zeros(delta_GG.shape, complex)
                for iqf in range(len(gamma_int.qf_qv)):
                    gamma_int.set_appendages(chi0_GG, iw, iqf)

                    sqrtV_G = self.coulomb.sqrtV(
                        pd=pd, q_v=gamma_int.qf_qv[iqf])

                    dfc = DielectricFunctionCalculator(
                        sqrtV_G, chi0_GG, mode=fxc_mode, fv_GG=fv)
                    einv_GG += dfc.get_einv_GG() * gamma_int.weight_q[iqf]
            else:
                sqrtV_G = self.coulomb.sqrtV(pd=pd, q_v=None)
                dfc = DielectricFunctionCalculator(
                    sqrtV_G, chi0_GG, mode=fxc_mode, fv_GG=fv)
                einv_GG = dfc.get_einv_GG()
            if self.ppa:
                einv_wGG.append(einv_GG - delta_GG)
            else:
                einv_GG_full = einv_GG.copy()
                if only_correlation:
                    einv_GG -= delta_GG
                W_GG = chi0_GG
                W_GG[:] = (einv_GG) * (sqrtV_G *
                                       sqrtV_G[:, np.newaxis])
                if self.q0_corrector is not None and np.allclose(q_c, 0):
                    W = wblocks1d.a + iw
                    self.q0_corrector.add_q0_correction(pd, W_GG,
                                                        einv_GG_full,
                                                        chi0_WxvG[W],
                                                        chi0_Wvv[W],
                                                        sqrtV_G)
                # XXX Is it to correct to have "or" here?
                elif np.allclose(q_c, 0) or self.integrate_gamma != 0:
                    W_GG[0, 0] = einv_GG[0, 0] * V0
                    W_GG[0, 1:] = einv_GG[0, 1:] * sqrtV_G[1:] * sqrtV0
                    W_GG[1:, 0] = einv_GG[1:, 0] * sqrtV0 * sqrtV_G[1:]

        if self.ppa:
            omegat_GG = self.E0 * np.sqrt(einv_wGG[1] /
                                          (einv_wGG[0] - einv_wGG[1]))
            R_GG = -0.5 * omegat_GG * einv_wGG[0]
            W_GG = pi * R_GG * sqrtV_G * sqrtV_G[:, np.newaxis]
            if np.allclose(q_c, 0) or self.integrate_gamma != 0:
                W_GG[0, 0] = pi * R_GG[0, 0] * V0
                W_GG[0, 1:] = pi * R_GG[0, 1:] * sqrtV_G[1:] * sqrtV0
                W_GG[1:, 0] = pi * R_GG[1:, 0] * sqrtV0 * sqrtV_G[1:]

            self.context.timer.stop('Dyson eq.')
            # This is very bad! The output data structure should not depend
            # on self.ppa! XXX
            return [W_GG, omegat_GG]

        self.context.timer.stop('Dyson eq.')
        return chi0_wGG

    def dyson_and_W_new(self, iq, q_c, chi0, ecut, coulomb):
        assert not self.ppa
        # assert not self.do_GW_too
        assert ecut == chi0.pd.ecut
        assert self.fxc_mode == 'GW'

        assert not np.allclose(q_c, 0)

        nW = len(self.wd)
        nG = chi0.pd.ngmax

        from gpaw.response.wgg import Grid

        WGG = (nW, nG, nG)
        WgG_grid = Grid(
            comm=self.blockcomm,
            shape=WGG,
            cpugrid=(1, self.blockcomm.size, 1))
        assert chi0.chi0_wGG.shape == WgG_grid.myshape

        my_gslice = WgG_grid.myslice[1]

        dielectric_WgG = chi0.chi0_wGG  # XXX
        for iw, chi0_GG in enumerate(chi0.chi0_wGG):
            sqrtV_G = coulomb.sqrtV(chi0.pd, q_v=None)
            e_GG = np.eye(nG) - chi0_GG * sqrtV_G * sqrtV_G[:, np.newaxis]
            e_gG = e_GG[my_gslice]

            dielectric_WgG[iw, :, :] = e_gG

        wgg_grid = Grid(comm=self.blockcomm, shape=WGG)

        dielectric_wgg = wgg_grid.zeros(dtype=complex)
        WgG_grid.redistribute(wgg_grid, dielectric_WgG, dielectric_wgg)

        assert np.allclose(dielectric_wgg, dielectric_WgG)

        wgg_grid.invert_inplace(dielectric_wgg)

        wgg_grid.redistribute(WgG_grid, dielectric_wgg, dielectric_WgG)
        inveps_WgG = dielectric_WgG

        self.context.timer.start('Dyson eq.')

        for iw, inveps_gG in enumerate(inveps_WgG):
            inveps_gG -= np.identity(nG)[my_gslice]
            thing_GG = sqrtV_G * sqrtV_G[:, np.newaxis]
            inveps_gG *= thing_GG[my_gslice]

        W_WgG = inveps_WgG
        Wp_wGG = W_WgG.copy()
        Wm_wGG = W_WgG.copy()
        return chi0.pd, Wm_wGG, Wp_wGG  # not Hilbert transformed yet

import numpy as np
from math import pi
from gpaw.response.q0_correction import Q0Correction
from ase.units import Ha
from ase.dft.kpoints import monkhorst_pack

from gpaw.kpt_descriptor import KPointDescriptor

from gpaw.response.pw_parallelization import Blocks1D
from gpaw.response.gamma_int import GammaIntegrator
from gpaw.response.temp import DielectricFunctionCalculator, EpsilonInverse


class QPointDescriptor(KPointDescriptor):

    @staticmethod
    def from_gs(gs):
        kd, atoms = gs.kd, gs.atoms
        # Find q-vectors and weights in the IBZ:
        assert -1 not in kd.bz2bz_ks
        offset_c = 0.5 * ((kd.N_c + 1) % 2) / kd.N_c
        bzq_qc = monkhorst_pack(kd.N_c) + offset_c
        qd = KPointDescriptor(bzq_qc)
        qd.set_symmetry(atoms, kd.symmetry)
        return qd


def initialize_w_calculator(chi0calc, context, *,
                            coulomb,
                            xc='RPA', Eg=None,  # G0W0Kernel arguments
                            ppa=False, E0=Ha,
                            integrate_gamma=0, q0_correction=False):
    """Initialize a WCalculator from a Chi0Calculator.

    Parameters
    ----------
    chi0calc : Chi0Calculator
    xc : str
        Kernel to use when including vertex corrections.
    Eg: float
        Gap to apply in the 'JGMs' (simplified jellium-with-gap) kernel.
        If None the DFT gap is used.

    Remaining arguments: See WCalculator
    """
    from gpaw.response.g0w0_kernels import G0W0Kernel

    gs = chi0calc.gs

    if Eg is None and xc == 'JGMsx':
        Eg = gs.get_band_gap()
    elif Eg is not None:
        Eg /= Ha

    qd = QPointDescriptor.from_gs(gs)

    xckernel = G0W0Kernel(xc=xc, ecut=chi0calc.ecut,
                          gs=gs, qd=qd,
                          ns=gs.nspins,
                          wd=chi0calc.wd,
                          Eg=Eg,
                          context=context)

    wcalc = WCalculator(gs, context, qd=qd,
                        coulomb=coulomb, xckernel=xckernel,
                        ppa=ppa, E0=E0,
                        integrate_gamma=integrate_gamma,
                        q0_correction=q0_correction)

    return wcalc


class WCalculator:

    def __init__(self, gs, context, *, qd,
                 coulomb, xckernel,
                 ppa, E0,
                 integrate_gamma=0, q0_correction=False):
        """
        W Calculator.

        Parameters
        ----------
        gs : ResponseGroundStateAdapter
        context : ResponseContext
        qd : QPointDescriptor
        coulomb : CoulombKernel
        xckernel : G0W0Kernel
        ppa : bool
            Sets whether the Godby-Needs plasmon-pole approximation for the
            dielectric function should be used.
        E0 : float
            Energy (in eV) used for fitting the plasmon-pole approximation
        integrate_gamma: int
             Method to integrate the Coulomb interaction. 1 is a numerical
             integration at all q-points with G=[0,0,0] - this breaks the
             symmetry slightly. 0 is analytical integration at q=[0,0,0] only
             this conserves the symmetry. integrate_gamma=2 is the same as 1,
             but the average is only carried out in the non-periodic directions
        q0_correction : bool
            Analytic correction to the q=0 contribution applicable to 2D
            systems.
        """
        self.gs = gs
        self.context = context
        self.qd = qd
        self.coulomb = coulomb
        self.xckernel = xckernel
        self.ppa = ppa
        self.E0 = E0 / Ha  # eV -> Hartree

        self.integrate_gamma = integrate_gamma

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

    def calculate(self, chi0,
                  fxc_mode='GW',
                  only_correlation=False,
                  out_dist='WgG'):
        """Calculate the screened interaction.

        Parameters
        ----------
        fxc_mode: str
            Where to include the vertex corrections; polarizability and/or
            self-energy. 'GWP': Polarizability only, 'GWS': Self-energy only,
            'GWG': Both.
        """

        if self.ppa:
            W_wGG = self._calculate_ppa(chi0, fxc_mode,
                                        only_correlation=only_correlation)
        else:
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


    def _calculate_ppa(self, chi0, fxc_mode,
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
        #einv = EpsilonInverse(chi0, self.gs.kd, self.gs.volume, fxc_mode)

        for iw, chi0_GG in enumerate(chi0_wGG):
            if np.allclose(q_c, 0):
                einv_GG = np.zeros(delta_GG.shape, complex)
                for iqf in range(len(gamma_int.qf_qv)):
                    gamma_int.set_appendages(chi0_GG, iw, iqf)

                    sqrtV_G = self.coulomb.sqrtV(
                        pd=pd, q_v=gamma_int.qf_qv[iqf])

                    dfc = DielectricFunctionCalculator(
                        sqrtV_G, chi0_GG, mode=fxc_mode, fv_GG=fv)
                    einv_GG += dfc.get_einv_GG() * gamma_int.weight_q
            else:
                sqrtV_G = self.coulomb.sqrtV(pd=pd, q_v=None)
                dfc = DielectricFunctionCalculator(
                    sqrtV_G, chi0_GG, mode=fxc_mode, fv_GG=fv)
                einv_GG = dfc.get_einv_GG()
            einv_wGG.append(einv_GG - delta_GG)

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


    def _calculate(self, chi0, fxc_mode,
                   only_correlation=False):
        """In-place calculation of the screened interaction."""
        
        self.context.timer.start('Epsilon inverse.')
        einv = EpsilonInverse(chi0, self.qd, self.coulomb, self.gs.kd, self.gs.volume, fxc_mode,
                integrate_gamma=self.integrate_gamma)
        self.context.timer.stop('Epsilon inverse.')
        self.context.timer.start('W')
        # inplace replacement of einv data with W_GG data
        W_wGG = einv.einv_wGG
        einv.einv_wGG = None  # Invalidate einv, because we stole its data
        for iw, W_GG in enumerate(W_wGG):
            einv_GG_full = W_GG.copy()
            if only_correlation:
                W_GG.ravel()[::len(W_GG)+1] -= 1 
            W_GG[:] = W_GG * (einv.sqrtV_G *
                              einv.sqrtV_G[:, np.newaxis])
            if self.q0_corrector is not None and np.allclose(einv.q_c, 0):
                W = einv.wblocks1d.a + iw
                self.q0_corrector.add_q0_correction(einv.pd, W_GG,
                                                    einv_GG_full,
                                                    chi0_WxvG[W],
                                                    chi0_Wvv[W],
                                                    einv.sqrtV_G)
            # XXX Is it to correct to have "or" here?
            elif np.allclose(einv.q_c, 0) or self.integrate_gamma != 0:
                W_GG[0, 0] = W_GG[0, 0] * einv.V0
                W_GG[0, 1:] = W_GG[0, 1:] * einv.sqrtV_G[1:] * einv.sqrtV0
                W_GG[1:, 0] = W_GG[1:, 0] * einv.sqrtV0 * einv.sqrtV_G[1:]

        self.context.timer.start('W')
        return W_wGG

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

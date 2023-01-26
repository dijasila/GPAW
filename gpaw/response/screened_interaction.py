import numpy as np
from math import pi
from gpaw.response.q0_correction import Q0Correction
from ase.units import Ha
from ase.dft.kpoints import monkhorst_pack
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.response.temp import DielectricFunctionCalculator


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
                            xc='RPA',  # G0W0Kernel arguments
                            ppa=False, E0=Ha,
                            integrate_gamma=0, q0_correction=False):
    """Initialize a WCalculator from a Chi0Calculator.

    Parameters
    ----------
    chi0calc : Chi0Calculator
    xc : str
        Kernel to use when including vertex corrections.

    Remaining arguments: See WCalculator
    """
    from gpaw.response.g0w0_kernels import G0W0Kernel

    gs = chi0calc.gs
    qd = QPointDescriptor.from_gs(gs)

    xckernel = G0W0Kernel(xc=xc, ecut=chi0calc.ecut,
                          gs=gs, qd=qd,
                          ns=gs.nspins,
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

    def get_V0sqrtV0(self, chi0):
        """
        Integrated Coulomb kernels.
        integrate_gamma = 0: Analytically integrated kernel
        in sphere around Gamma
        integrate_gamma > 0: Numerically  integrated kernel
        XXX: Understand and document Rq0, V0, sqrtV0
        """
        V0 = None
        sqrtV0 = None
        if self.integrate_gamma != 0:
            reduced = (self.integrate_gamma == 2)
            V0, sqrtV0 = self.coulomb.integrated_kernel(qpd=chi0.qpd,
                                                        reduced=reduced)
        elif self.integrate_gamma == 0 and chi0.optical_limit:
            bzvol = (2 * np.pi)**3 / self.gs.volume / self.qd.nbzkpts
            Rq0 = (3 * bzvol / (4 * np.pi))**(1. / 3.)
            V0 = 16 * np.pi**2 * Rq0 / bzvol
            sqrtV0 = (4 * np.pi)**(1.5) * Rq0**2 / bzvol / 2
        return V0, sqrtV0

    def apply_gamma_correction(self, W_GG, einv_GG, V0, sqrtV0, sqrtV_G):
        """
        Replacing q=0, (G,G')= (0,0), (0,:), (:,0) with corresponding
        matrix elements calculated with an average of the (diverging)
        Coulomb interaction.
        XXX: Understand and document exact expressions
        """
        W_GG[0, 0] = einv_GG[0, 0] * V0
        W_GG[0, 1:] = einv_GG[0, 1:] * sqrtV_G[1:] * sqrtV0
        W_GG[1:, 0] = einv_GG[1:, 0] * sqrtV0 * sqrtV_G[1:]

    def _calculate(self, chi0, fxc_mode,
                   only_correlation=False):
        """In-place calculation of the screened interaction."""
        chi0_wGG = chi0.copy_array_with_distribution('wGG')
        dfc = DielectricFunctionCalculator(chi0, self.coulomb,
                                           self.xckernel, fxc_mode)
        self.context.timer.start('Dyson eq.')
        
        V0, sqrtV0 = self.get_V0sqrtV0(chi0)
        for iw, chi0_GG in enumerate(chi0_wGG):
            # Note, at q=0 get_epsinv_GG modifies chi0_GG
            einv_GG = dfc.get_epsinv_GG(chi0_GG, iw)
            # Renaming the chi0_GG buffer since it will be used to store W
            W_GG = chi0_GG
            # If only_correlation = True function spits out
            # W^c = sqrt(V)(epsinv - delta_GG')sqrt(V). However, full epsinv
            # is still needed for q0_corrector.
            einvt_GG = (einv_GG - dfc.I_GG) if only_correlation else einv_GG
            W_GG[:] = einvt_GG * (dfc.sqrtV_G *
                                  dfc.sqrtV_G[:, np.newaxis])
            if self.q0_corrector is not None and chi0.optical_limit:
                W = dfc.wblocks1d.a + iw
                self.q0_corrector.add_q0_correction(chi0.qpd, W_GG,
                                                    einv_GG,
                                                    chi0.chi0_WxvG[W],
                                                    chi0.chi0_Wvv[W],
                                                    dfc.sqrtV_G)
                # XXX Is it to correct to have "or" here?
            elif chi0.optical_limit or self.integrate_gamma != 0:
                self.apply_gamma_correction(W_GG, einvt_GG,
                                            V0, sqrtV0, dfc.sqrtV_G)

        self.context.timer.stop('Dyson eq.')
        return chi0_wGG

    def _calculate_ppa(self, chi0, fxc_mode,
                       only_correlation=False):
        """In-place calculation of the screened interaction."""
        dfc = DielectricFunctionCalculator(chi0,
                                           self.coulomb,
                                           self.xckernel,
                                           fxc_mode)
        assert only_correlation

        V0, sqrtV0 = self.get_V0sqrtV0(chi0)
        self.context.timer.start('Dyson eq.')
        einv_wGG = dfc.get_epsinv_wGG(only_correlation=True)
        omegat_GG = self.E0 * np.sqrt(einv_wGG[1] /
                                      (einv_wGG[0] - einv_wGG[1]))
        R_GG = -0.5 * omegat_GG * einv_wGG[0]
        W_GG = pi * R_GG * dfc.sqrtV_G * dfc.sqrtV_G[:, np.newaxis]
        if chi0.optical_limit or self.integrate_gamma != 0:
            self.apply_gamma_correction(W_GG, pi * R_GG,
                                        V0, sqrtV0,
                                        dfc.sqrtV_G)

        self.context.timer.stop('Dyson eq.')
        # This is very bad! The output data structure should not depend
        # on self.ppa! XXX
        return [W_GG, omegat_GG]

    def dyson_and_W_new(self, iq, q_c, chi0, ecut, coulomb):
        assert not self.ppa
        # assert not self.do_GW_too
        assert ecut == chi0.qpd.ecut
        assert self.fxc_mode == 'GW'

        assert not np.allclose(q_c, 0)

        nW = len(self.wd)
        nG = chi0.qpd.ngmax

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
            sqrtV_G = coulomb.sqrtV(chi0.qpd, q_v=None)
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
        return chi0.qpd, Wm_wGG, Wp_wGG  # not Hilbert transformed yet

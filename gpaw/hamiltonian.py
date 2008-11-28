# -*- coding: utf-8 -*-
# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""This module defines a Hamiltonian."""

from math import pi, sqrt

import numpy as np

from gpaw.poisson import PoissonSolver
from gpaw.transformers import Transformer
from gpaw.xc_functional import XC3DGrid
from gpaw.lfc import LocalizedFunctionsCollection as LFC


class Hamiltonian:
    """Hamiltonian object.

    Attributes:
     =============== =====================================================
     ``xc``          ``XC3DGrid`` object.
     ``poisson``     ``PoissonSolver``.
     ``gd``          Grid descriptor for coarse grids.
     ``finegd``      Grid descriptor for fine grids.
     ``restrict``    Function for restricting the effective potential.
     =============== =====================================================

    Soft and smooth pseudo functions on uniform 3D grids:
     ========== =========================================
     ``vHt_g``  Hartree potential on the fine grid.
     ``vt_sG``  Effective potential on the coarse grid.
     ``vt_sg``  Effective potential on the fine grid.
     ========== =========================================
    """

    def __init__(self, gd, finegd, nspins, setups, stencil, timer, xcfunc,
                 psolver, vext_g):
        """Create the Hamiltonian."""
        self.gd = gd
        self.finegd = finegd
        self.nspins = nspins
        self.setups = setups
        self.timer = timer

        # Solver for the Poisson equation:
        if psolver is None:
            psolver = PoissonSolver(nn='M', relax='J')
        self.poisson = psolver

        # The external potential
        self.vext_g = vext_g

        self.vt_sG = None
        self.vHt_g = None
        self.vt_sg = None
        self.vbar_g = None

        # Restrictor function for the potential:
        self.restrict = Transformer(self.finegd, self.gd, stencil).apply

        # Exchange-correlation functional object:
        self.xc = XC3DGrid(xcfunc, finegd, nspins)

        self.vbar = LFC(self.finegd, [[setup.vbar] for setup in setups])

    def set_positions(self, spos_ac):
        self.vbar.set_positions(spos_ac)
        if self.vbar_g is None:
            self.vbar_g = self.finegd.empty()
        self.vbar_g[:] = 0.0
        self.vbar.add(self.vbar_g)

    def update(self, density):
        """Calculate effective potential.

        The XC-potential and the Hartree potential are evaluated on
        the fine grid, and the sum is then restricted to the coarse
        grid."""


        self.timer.start('Hamiltonian')

        if self.vt_sg is None:
            self.vt_sg = self.finegd.empty(2)
            self.vHt_g = self.finegd.zeros()
            self.vt_sG = self.gd.empty(2)
            self.poisson.initialize(self.finegd)

        Ebar = np.vdot(self.vbar_g, density.nt_g) * self.finegd.dv

        vt_g = self.vt_sg[0]
        vt_g[:] = self.vbar_g

        Eext = 0.0
        if self.vext_g is not None:
            vt_g += self.vext_g.get_potential(self.finegd)
            Eext = np.vdot(vt_g, density.nt_g) * self.finegd.dv - Ebar

        if self.nspins == 2:
            self.vt_sg[1] = vt_g

        if self.nspins == 2:
            Exc = self.xc.get_energy_and_potential(
                density.nt_sg[0], self.vt_sg[0],
                density.nt_sg[1], self.vt_sg[1])
        else:
            Exc = self.xc.get_energy_and_potential(
                density.nt_sg[0], self.vt_sg[0])

        if self.xc.xcfunc.is_gllb():
            self.timer.start('GLLB')
            Exc = self.xc.xcfunc.xc.update_xc_potential()
            self.timer.stop('GLLB')

        self.timer.start('Poisson')
        # npoisson is the number of iterations:
        self.npoisson = self.poisson.solve(self.vHt_g, density.rhot_g,
                                           charge=-density.charge)
        self.timer.stop('Poisson')

        Epot = 0.5 * np.vdot(self.vHt_g, density.rhot_g) * self.finegd.dv
        Ekin = 0.0
        for vt_g, vt_G, nt_G in zip(self.vt_sg, self.vt_sG, density.nt_sG):
            vt_g += self.vHt_g
            self.restrict(vt_g, vt_G)
            Ekin -= np.vdot(vt_G, nt_G - density.nct_G) * self.gd.dv

        # Calculate atomic hamiltonians:
        self.timer.start('Atomic Hamiltonians')
        W_aL = {}
        for a in density.D_asp:
            W_aL[a] = np.empty((self.setups[a].lmax + 1)**2)
        density.ghat.integrate(self.vHt_g, W_aL)
        self.dH_asp = {}
        for a, D_sp in density.D_asp.items():
            W_L = W_aL[a]
            setup = self.setups[a]

            D_p = D_sp.sum(0)
            dH_p = (setup.K_p + setup.M_p +
                    setup.MB_p + 2.0 * np.dot(setup.M_pp, D_p) +
                    np.dot(setup.Delta_pL, W_L))
            Ekin += np.dot(setup.K_p, D_p) + setup.Kc
            Ebar += setup.MB + np.dot(setup.MB_p, D_p)
            Epot += setup.M + np.dot(D_p, (setup.M_p + np.dot(setup.M_pp, D_p)))

            if setup.HubU is not None:
##                 print '-----'
                nspins = len(self.D_sp)
                i0 = setup.Hubi
                i1 = i0 + 2 * setup.Hubl + 1
                for D_p, H_p in zip(self.D_sp, self.H_sp):
                    N_mm = unpack2(D_p)[i0:i1, i0:i1] / 2 * nspins 
                    Eorb = setup.HubU/2. * (N_mm - np.dot(N_mm,N_mm)).trace()
                    Vorb = setup.HubU * (0.5 * np.eye(i1-i0) - N_mm)
##                     print '========='
##                     print 'occs:',np.diag(N_mm)
##                     print 'Eorb:',Eorb
##                     print 'Vorb:',np.diag(Vorb)
##                     print '========='
                    Exc += Eorb                    
                    Htemp = unpack(H_p)
                    Htemp[i0:i1,i0:i1] += Vorb
                    H_p[:] = pack2(Htemp)

            if 0:#vext is not None:
                # Tailor expansion to the zeroth order
                Eext += vext[0][0] * (sqrt(4 * pi) * self.Q_L[0] + setup.Z)
                dH_p += vext[0][0] * sqrt(4 * pi) * setup.Delta_pL[:, 0]
                if len(vext) > 1:
                    # Tailor expansion to the first order
                    Eext += sqrt(4 * pi / 3) * np.dot(vext[1], self.Q_L[1:4])
                    # there must be a better way XXXX
                    Delta_p1 = np.array([setup.Delta_pL[:, 1],
                                          setup.Delta_pL[:, 2],
                                          setup.Delta_pL[:, 3]])
                    dH_p += sqrt(4 * pi / 3) * np.dot(vext[1], Delta_p1)

            self.dH_asp[a] = dH_sp = np.zeros_like(D_sp)
            Exc += setup.xc_correction.calculate_energy_and_derivatives(
                D_sp, dH_sp)
            dH_sp += dH_p

            Ekin -= (D_sp * dH_sp).sum()

        self.timer.stop('Atomic Hamiltonians')

        comm = self.gd.comm
        self.Ekin = comm.sum(Ekin)
        self.Epot = comm.sum(Epot)
        self.Ebar = comm.sum(Ebar)
        self.Eext = comm.sum(Eext)
        self.Exc = comm.sum(Exc)

        self.timer.stop('Hamiltonian')

    def apply_local_potential(self, psit_nG, Htpsit_nG, s):
        """Apply the Hamiltonian operator to a set of vectors.

        Parameters
        ==========
        a_nG: ndarray
            Set of vectors to which the overlap operator is applied.
        b_nG: ndarray, output
            Resulting S times a_nG vectors.
        kpt: KPoint object
            k-point object defined in kpoint.py.
        calculate_P_uni: bool
            When True, the integrals of projector times vectors
            P_ni = <p_i | a_nG> are calculated.
            When False, existing P_uni are used
        
        """
        vt_G = self.vt_sG[s]
        if psit_nG.ndim == 3:
            Htpsit_nG += psit_nG * vt_G
        else:
            for psit_G, Htpsit_G in zip(psit_nG, Htpsit_nG):
                Htpsit_G += psit_G * vt_G

# -*- coding: utf-8 -*-
# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""This module defines a Hamiltonian."""

from math import pi, sqrt

import numpy as np

from gpaw.poisson import PoissonSolver
from gpaw.transformers import Transformer
from gpaw.lfc import LFC
from gpaw.utilities import pack2,unpack,unpack2
from gpaw.utilities.tools import tri2full
from gpaw import debug


class Hamiltonian:
    def __init__(self, gd, nspins, xc, setups, collinear=True,
                 vext_G=None, timer=None):
        """Hamiltonian object.
    
        gd: GridDescriptor object
            Grid descriptor for real-space grids.
        xc: XC object
            Exchange-correlation object.

        Energy contributions:
    
        :Ekin:    Kinetic energy.
        :Epot:    Potential energy.
        :Etot:    Total energy.
        :Exc:     Exchange-Correlation energy.
        :Eext:    Energy of external potential
        :Eref:    Reference energy for all-electron atoms.
        :S:       Entropy.
        :Ebar:    Should be close to zero!
        """

        self.gd = gd
        self.nspins = nspins
        self.setups = setups
        self.timer = timer
        self.xc = xc
        self.collinear = collinear
        self.ncomp = 2 - int(collinear)

        self.dH_asp = None

        # The external potential
        self.vext_G = vext_G

        self.vt_sG = None
        self.vbar_G = None

        self.rank_a = None

        self.Ekin0 = None
        self.Ekin = None
        self.Epot = None
        self.Ebar = None
        self.Eext = None
        self.Exc = None
        self.Etot = None
        self.S = None

        print 'ALLOCATE!'

    def set_positions(self, spos_ac, rank_a=None):
        self.spos_ac = spos_ac

        self.ghat.set_positions(spos_ac)

        self.vbar.set_positions(spos_ac)
        if self.vbar_G is None:
            self.vbar_G = self.finegd.empty()
        self.vbar_G[:] = 0.0
        self.vbar.add(self.vbar_G)

        self.xc.set_positions(spos_ac)
        
        # If both old and new atomic ranks are present, start a blank dict if
        # it previously didn't exist but it will needed for the new atoms.
        if (self.rank_a is not None and rank_a is not None and
            self.dH_asp is None and (rank_a == self.gd.comm.rank).any()):
            self.dH_asp = {}

        if self.rank_a is not None and self.dH_asp is not None:
            self.timer.start('Redistribute')
            requests = []
            flags = (self.rank_a != rank_a)
            my_incoming_atom_indices = np.argwhere(np.bitwise_and(flags, \
                rank_a == self.gd.comm.rank)).ravel()
            my_outgoing_atom_indices = np.argwhere(np.bitwise_and(flags, \
                self.rank_a == self.gd.comm.rank)).ravel()

            for a in my_incoming_atom_indices:
                # Get matrix from old domain:
                ni = self.setups[a].ni
                dH_sp = np.empty((self.nspins * self.ncomp**2,
                                  ni * (ni + 1) // 2))
                requests.append(self.gd.comm.receive(dH_sp, self.rank_a[a],
                                                     tag=a, block=False))
                assert a not in self.dH_asp
                self.dH_asp[a] = dH_sp

            for a in my_outgoing_atom_indices:
                # Send matrix to new domain:
                dH_sp = self.dH_asp.pop(a)
                requests.append(self.gd.comm.send(dH_sp, rank_a[a],
                                                  tag=a, block=False))
            self.gd.comm.waitall(requests)
            self.timer.stop('Redistribute')

        self.rank_a = rank_a

    def update(self, density):
        """Calculate effective potential.

        The XC-potential and the Hartree potential are evaluated on
        the fine grid, and the sum is then restricted to the coarse
        grid."""

        self.timer.start('Hamiltonian')

        if self.vt_sG is None:
            self.timer.start('Initialize Hamiltonian')
            self.vt_sG = self.gd.empty(self.nspins * self.ncomp**2)
            self.poissonsolver.initialize()
            self.timer.stop('Initialize Hamiltonian')

        Ebar, Exc, Epot, Ekin, W_aL = self.calculate_effective_potential(density)

        Eext = 0.0
                
        # Calculate atomic hamiltonians:
        self.timer.start('Atomic')
        self.dH_asp = {}
        for a, D_sp in density.D_asp.items():
            W_L = W_aL[a]
            setup = self.setups[a]

            D_p = D_sp[:self.nspins].sum(0)
            dH_p = (setup.K_p + setup.M_p +
                    setup.MB_p + 2.0 * np.dot(setup.M_pp, D_p) +
                    np.dot(setup.Delta_pL, W_L))
            Ekin += np.dot(setup.K_p, D_p) + setup.Kc
            Ebar += setup.MB + np.dot(setup.MB_p, D_p)
            Epot += setup.M + np.dot(D_p, (setup.M_p +
                                           np.dot(setup.M_pp, D_p)))

            self.dH_asp[a] = dH_sp = np.zeros_like(D_sp)
            self.timer.start('XC Correction')
            Exc += self.xc.calculate_paw_correction(setup, D_sp, dH_sp, a=a)
            self.timer.stop('XC Correction')

            dH_sp[:self.nspins] += dH_p

            Ekin -= (D_sp * dH_sp).sum()  # NCXXX

        self.timer.stop('Atomic')

        # Make corrections due to non-local xc:
        #xcfunc = self.xc.xcfunc
        self.Enlxc = 0.0#XXXxcfunc.get_non_local_energy()
        Ekin += self.xc.get_kinetic_energy_correction() / self.gd.comm.size

        energies = np.array([Ekin, Epot, Ebar, Eext, Exc])
        self.timer.start('Communicate energies')
        self.gd.comm.sum(energies)
        self.timer.stop('Communicate energies')
        (self.Ekin0, self.Epot, self.Ebar, self.Eext, self.Exc) = energies

        #self.Exc += self.Enlxc
        #self.Ekin0 += self.Enlkin

        self.timer.stop('Hamiltonian')

    def get_energy(self, occupations):
        self.Ekin = self.Ekin0 + occupations.e_band
        self.S = occupations.e_entropy

        # Total free energy:
        self.Etot = (self.Ekin + self.Epot + self.Eext +
                     self.Ebar + self.Exc - self.S)

        return self.Etot

    def external(self):
        if self.vext_G is not None:
                vext = self.vext_G.get_taylor(spos_c=self.spos_ac[a, :])
                # Tailor expansion to the zeroth order
                Eext += vext[0][0] * (sqrt(4 * pi) * density.Q_aL[a][0]
                                      + setup.Z)
                dH_p += vext[0][0] * sqrt(4 * pi) * setup.Delta_pL[:, 0]
                if len(vext) > 1:
                    # Tailor expansion to the first order
                    Eext += sqrt(4 * pi / 3) * np.dot(vext[1],
                                                      density.Q_aL[a][1:4])
                    # there must be a better way XXXX
                    Delta_p1 = np.array([setup.Delta_pL[:, 1],
                                          setup.Delta_pL[:, 2],
                                          setup.Delta_pL[:, 3]])
                    dH_p += sqrt(4 * pi / 3) * np.dot(vext[1], Delta_p1)

    def hubbardu(self):
        if setup.HubU is not None:
                assert self.collinear
                nspins = len(D_sp)
                
                l_j = setup.l_j
                l   = setup.Hubl
                nl  = np.where(np.equal(l_j,l))[0]
                nn  = (2*np.array(l_j)+1)[0:nl[0]].sum()
                
                for D_p, H_p in zip(D_sp, self.dH_asp[a]):
                    [N_mm,V] = aoom(setup, unpack2(D_p),a,l)
                    N_mm = N_mm / 2 * nspins
                     
                    Eorb = setup.HubU / 2. * (N_mm - np.dot(N_mm,N_mm)).trace()
                    Vorb = setup.HubU * (0.5 * np.eye(2*l+1) - N_mm)
                    Exc += Eorb
                    if nspins == 1:
                        # add contribution of other spin manyfold
                        Exc += Eorb
                    
                    if len(nl)==2:
                        mm  = (2*np.array(l_j)+1)[0:nl[1]].sum()
                        
                        V[nn:nn+2*l+1,nn:nn+2*l+1] *= Vorb
                        V[mm:mm+2*l+1,nn:nn+2*l+1] *= Vorb
                        V[nn:nn+2*l+1,mm:mm+2*l+1] *= Vorb
                        V[mm:mm+2*l+1,mm:mm+2*l+1] *= Vorb
                    else:
                        V[nn:nn+2*l+1,nn:nn+2*l+1] *= Vorb
                    
                    Htemp = unpack(H_p)
                    Htemp += V
                    H_p[:] = pack2(Htemp)

    def apply_local_potential(self, psit_nG, Htpsit_nG, s):
        """Apply the Hamiltonian operator to a set of vectors.

        XXX Parameter description is deprecated!
        
        Parameters:

        a_nG: ndarray
            Set of vectors to which the overlap operator is applied.
        b_nG: ndarray, output
            Resulting H times a_nG vectors.
        kpt: KPoint object
            k-point object defined in kpoint.py.
        calculate_projections: bool
            When True, the integrals of projector times vectors
            P_ni = <p_i | a_nG> are calculated.
            When False, existing P_uni are used
        local_part_only: bool
            When True, the non-local atomic parts of the Hamiltonian
            are not applied and calculate_projections is ignored.
        
        """
        vt_G = self.vt_sG[s]
        if psit_nG.ndim == 3:
            Htpsit_nG += psit_nG * vt_G
        else:
            for psit_G, Htpsit_G in zip(psit_nG, Htpsit_nG):
                Htpsit_G += psit_G * vt_G

    def apply(self, a_xG, b_xG, wfs, kpt, calculate_P_ani=True):
        """Apply the Hamiltonian operator to a set of vectors.

        Parameters:

        a_nG: ndarray
            Set of vectors to which the overlap operator is applied.
        b_nG: ndarray, output
            Resulting S times a_nG vectors.
        wfs: WaveFunctions
            Wave-function object defined in wavefunctions.py
        kpt: KPoint object
            k-point object defined in kpoint.py.
        calculate_P_ani: bool
            When True, the integrals of projector times vectors
            P_ni = <p_i | a_nG> are calculated.
            When False, existing P_ani are used
        
        """

        wfs.kin.apply(a_xG, b_xG, kpt.phase_cd)
        self.apply_local_potential(a_xG, b_xG, kpt.s)
        shape = a_xG.shape[:-3]
        P_axi = wfs.pt.dict(shape)

        if calculate_P_ani: #TODO calculate_P_ani=False is experimental
            wfs.pt.integrate(a_xG, P_axi, kpt.q)
        else:
            for a, P_ni in kpt.P_ani.items():
                P_axi[a][:] = P_ni

        for a, P_xi in P_axi.items():
            dH_ii = unpack(self.dH_asp[a][kpt.s])
            P_axi[a] = np.dot(P_xi, dH_ii)
        wfs.pt.add(b_xG, P_axi, kpt.q)

    def get_xc_difference(self, xc, density):
        """Calculate non-selfconsistent XC-energy difference."""
        if density.nt_sg is None:
            density.interpolate()
        nt_sg = density.nt_sg
        if hasattr(xc, 'hybrid'):
            xc.calculate_exx()
        Exc = xc.calculate(density.finegd, nt_sg) / self.gd.comm.size
        for a, D_sp in density.D_asp.items():
            setup = self.setups[a]
            Exc += xc.calculate_paw_correction(setup, D_sp)
        Exc = self.gd.comm.sum(Exc)
        return Exc - self.Exc

    def estimate_memory(self, mem):
        nbytes = self.gd.bytecount()
        nfinebytes = self.finegd.bytecount()
        arrays = mem.subnode('Arrays', 0)
        arrays.subnode('vHt_G', nfinebytes)
        arrays.subnode('vt_sG', self.nspins * nbytes)
        self.restrictor.estimate_memory(mem.subnode('Restrictor'))
        self.xc.estimate_memory(mem.subnode('XC'))
        self.poisson.estimate_memory(mem.subnode('Poisson'))
        self.vbar.estimate_memory(mem.subnode('vbar'))


class RealSpaceHamiltonian(Hamiltonian):
    def __init__(self, gd, nspins, xc, setups, collinear=True,
                 vext_G=None, timer=None, poissonsolver=None, stencil=3):
        Hamiltonian.__init__(self, gd, nspins, xc, setups, collinear,
                             vext_G, timer)

        self.finegd = self.gd.refine()

        # Solver for the Poisson equation:
        if poissonsolver is None:
            poissonsolver = PoissonSolver(nn=3, relax='J')
        self.poissonsolver = poissonsolver
        poissonsolver.set_grid_descriptor(self.finegd)

        # Restrictor function for the potential:
        self.restrictor = Transformer(self.finegd, self.gd, stencil)

        # Interpolation function for the density:
        self.interpolator = Transformer(self.gd, self.finegd, stencil)

        self.ghat = LFC(self.finegd, [setup.ghat_l for setup in setups],
                        integral=sqrt(4 * pi), forces=True)

        self.vbar = LFC(self.finegd, [[setup.vbar] for setup in self.setups],
                        forces=True)

        self.vHt_g = None

    def calculate_effective_potential(self, density):
        self.timer.start('vbar')
        
        Ebar = self.finegd.integrate(self.vbar_G,
                                 self.interpolator.apply(density.nt_sG),
                                 global_integral=False).sum()

        vt_G = self.vt_sG[0]
        vt_G[:] = self.restrictor.apply(self.vbar_G)
        self.timer.stop('vbar')


        self.vt_sG[1:self.nspins] = vt_G

        self.vt_sG[self.nspins:] = 0.0
            
        self.timer.start('XC 3D grid')
        if 0:
            Exc = self.xc.calculate(self.gd, density.nt_sG, self.vt_sG)
        else:
            nt_sg = self.finegd.empty(self.nspins * self.ncomp**2)
            self.interpolator.apply(density.nt_sG, nt_sg)
            vxct_sg = self.finegd.zeros(self.nspins * self.ncomp**2)
            Exc = self.xc.calculate(self.finegd, nt_sg, vxct_sg)
            vxct_sG = self.gd.zeros(self.nspins * self.ncomp**2)
            self.restrictor.apply(vxct_sg, vxct_sG)
            self.vt_sG += vxct_sG
        Exc /= self.gd.comm.size
        self.timer.stop('XC 3D grid')

        self.timer.start('Poisson')

        """Interpolate pseudo density to fine grid."""
        rhot_g = self.finegd.empty()
        nt_G = density.nt_sG[:self.nspins].sum(0)
        self.interpolator.apply(nt_G, rhot_g)

        # With periodic boundary conditions, the interpolation will
        # conserve the number of electrons.
        if not self.gd.pbc_c.all():
            # With zero-boundary conditions in one or more directions,
            # this is not the case.

            comp_charge = density.calculate_multipole_moments()

            pseudo_charge = -(density.charge + comp_charge)
            x = pseudo_charge / self.finegd.integrate(rhot_g)
            rhot_g *= x

        self.ghat.add(rhot_g, density.Q_aL)

        if debug:
            charge = self.finegd.integrate(rhot_g) + density.charge
            if abs(charge) > 1e-7:
                raise RuntimeError('Charge not conserved: excess=%.9f' %
                                   charge)

        if self.vHt_g is None:
            self.vHt_g = self.finegd.zeros()

        # npoisson is the number of iterations:
        self.npoisson = self.poissonsolver.solve(self.vHt_g, rhot_g,
                                                 charge=-density.charge)
        self.timer.stop('Poisson')

        self.timer.start('Hartree integrate/restrict')
        Epot = 0.5 * self.finegd.integrate(self.vHt_g, rhot_g,
                                           global_integral=False)

        vHt_G = self.gd.empty()
        self.restrictor.apply(self.vHt_g, vHt_G)
        
        W_aL = {}
        for a in density.D_asp:
            W_aL[a] = np.empty((self.setups[a].lmax + 1)**2)
        self.ghat.integrate(self.vHt_g, W_aL)

        self.timer.stop('Hartree integrate/restrict')
            
        self.vt_sG[:self.nspins] += vHt_G

        Ekin = 0.0
        s = 0
        for vt_G, nt_G in zip(self.vt_sG, density.nt_sG):
            Ekin -= self.gd.integrate(vt_G, nt_G, global_integral=False)
            if s < self.nspins:
                Ekin += self.gd.integrate(vt_G, density.nct_G,
                                          global_integral=False)
            s += 1

        return Ebar, Exc, Epot, Ekin, W_aL

# -*- coding: utf-8 -*-
# Copyright (C) 2003-2015  CAMP
# Please see the accompanying LICENSE file for further information.

"""This module defines a Hamiltonian."""

import numpy as np
from ase.units import Ha

from gpaw.atom_centered_functions import AtomCenteredFunctions as ACF
from gpaw.utilities.debug import frozen
from gpaw.external import create_external_potential
from gpaw.poisson import create_poisson_solver
from gpaw.transformers import Transformer
from gpaw.utilities import (pack, unpack,
                            unpack_atomic_matrices, pack_atomic_matrices)


ENERGY_NAMES = ['e_kinetic', 'e_coulomb', 'e_zero', 'e_external', 'e_xc',
                'e_entropy', 'e_total_free', 'e_total_extrapolated']


@frozen
class AtomBlockHamiltonian:
    def __init__(self, setups, spinpolarized, collinear):
        self.setups = setups
        self.spinpolarized = spinpolarized

        self.rank_a = None
        self.dH_asii = {}

    def set_ranks(self, rank_a):
        self.rank_a = rank_a

    def multiply(self, alpha, opa, P1_In, opb, beta, P2_In):
        assert opa == 'N'
        assert opb == 'N'
        assert beta == 0.0

        for a, I1, I2 in P2_In.indices:
            dH_ii = self.dH_asii[a][P2_In.spin]
            P2_In.array[I1:I2] = np.dot(dH_ii, P1_In.array[I1:I2])

        return P2_In

    def update(self, D_II, W_aL, xc, world, timer):
        # kinetic, coulomb, zero, external, xc:
        energies = np.zeros(5)
        for a in D_II.D_asii:
            W_L = W_aL[a]
            energies += self.update1(a, D_II, W_L, xc, timer)

        # Make corrections due to non-local xc:
        energies[0] += xc.get_kinetic_energy_correction() / world.size

        return energies

    def update1(self, a, D_II, W_L, xc, timer):
        setup = self.setups[a]
        D_sii = D_II.D_asii[a]
        D_sp = [pack(D_ii) for D_ii in D_sii]
        D_p = sum(D_sp)
        dH_sii = self.dH_asii[a] = np.empty_like(D_sii)

        dH_p = (setup.K_p + setup.M_p +
                setup.MB_p + 2.0 * np.dot(setup.M_pp, D_p) +
                np.dot(setup.Delta_pL, W_L))
        e_kinetic = np.dot(setup.K_p, D_p) + setup.Kc
        e_zero = setup.MB + np.dot(setup.MB_p, D_p)
        e_coulomb = setup.M + np.dot(D_p, (setup.M_p +
                                           np.dot(setup.M_pp, D_p)))

        dH_sii[:] = unpack(dH_p)

        if setup.HubU is not None:
            dH_sii += hubbard(setup, D_sii)

        # if self.ref_dH_asp:
        #     dH_sp += self.ref_dH_asp[a]

        with timer('XC Correction'):
            dH_sp = np.zeros_like(D_sp)
            e_xc = xc.calculate_paw_correction(setup, D_sp, dH_sp, a=a)
            dH_sii += [unpack(dH_p) for dH_p in dH_sp]

        e_kinetic -= (D_sii * dH_sii).sum()
        return [e_kinetic, e_coulomb, e_zero, 0.0, e_xc]


class Hamiltonian:
    def __init__(self, gd, finegd, spinpolarized, setups, timer, xc, world,
                 redistributor, vext=None):
        self.gd = gd
        self.finegd = finegd
        self.spinpolarized = spinpolarized
        self.nspins = 1 + spinpolarized
        self.collinear = True
        self.setups = setups
        self.timer = timer
        self.xc = xc
        self.world = world
        self.redistributor = redistributor

        self.dH_II = AtomBlockHamiltonian(setups, spinpolarized, True)
        self.vt_sR = None  # coarse grid
        self.vHt_r = None  # fine grid
        self.vt_sr = None  # fine grid

        self.vbar_a = None

        # Energy contributioons that sum up to e_total_free:
        self.e_kinetic = None
        self.e_coulomb = None
        self.e_zero = None
        self.e_external = None
        self.e_xc = None
        self.e_entropy = None

        self.e_total_free = None
        self.e_total_extrapolated = None
        self.e_kinetic0 = None

        self.ref_vt_sG = None
        self.ref_dH_asp = None

        if isinstance(vext, dict):
            vext = create_external_potential(**vext)
        self.vext = vext  # external potential

        self.positions_set = False

    @property
    def vt_sG(self):
        1 / 0
        return self.vt_sR

    @property
    def vt_sg(self):
        1 / 0
        return self.vt_sr

    def __str__(self):
        s = 'Hamiltonian:\n'
        s += ('  XC and Coulomb potentials evaluated on a {0}*{1}*{2} grid\n'
              .format(*self.finegd.N_c))
        s += '  Using the %s Exchange-Correlation functional\n' % self.xc.name
        # We would get the description of the XC functional here,
        # except the thing has probably not been fully initialized yet.
        if self.vext is not None:
            s += '  External potential:\n    {0}\n'.format(self.vext)
        return s

    def summary(self, fermilevel, log):
        log('Energy contributions relative to reference atoms:',
            '(reference = {0:.6f})\n'.format(self.setups.Eref * Ha))

        energies = [('Kinetic:      ', self.e_kinetic),
                    ('Potential:    ', self.e_coulomb),
                    ('External:     ', self.e_external),
                    ('XC:           ', self.e_xc),
                    ('Entropy (-ST):', self.e_entropy),
                    ('Local:        ', self.e_zero)]

        for name, e in energies:
            log('%-14s %+11.6f' % (name, Ha * e))

        log('--------------------------')
        log('Free energy:   %+11.6f' % (Ha * self.e_total_free))
        log('Extrapolated:  %+11.6f' % (Ha * self.e_total_extrapolated))
        log()
        self.xc.summary(log)

        try:
            correction = self.poisson.correction
        except AttributeError:
            pass
        else:
            c = self.poisson.c  # index of axis perpendicular to dipole-layer
            if not self.gd.pbc_c[c]:
                # zero boundary conditions
                vacuum = 0.0
            else:
                axes = (c, (c + 1) % 3, (c + 2) % 3)
                v_r = self.pd3.ifft(self.vHt_q).transpose(axes)
                vacuum = v_r[0].mean()

            wf1 = (vacuum - fermilevel + correction) * Ha
            wf2 = (vacuum - fermilevel - correction) * Ha
            log('Dipole-layer corrected work functions: {}, {} eV'
                .format(wf1, wf2))
            log()

    def set_positions(self, spos_ac, rank_a):
        self.vbar_a.set_positions(spos_ac)
        self.xc.set_positions(spos_ac)
        self.dH_II.set_ranks(rank_a)
        self.positions_set = True

    def initialize(self):
        self.vt_sr = self.finegd.empty(1 + self.spinpolarized)
        self.vt_sR = None
        self.vHt_r = self.finegd.zeros()
        self.poisson.initialize()

    def update(self, dens):
        """Calculate effective potential.

        The XC-potential and the Hartree potential are evaluated on
        the fine grid, and the sum is then restricted to the coarse
        grid."""

        self.timer.start('Hamiltonian')

        if self.vt_sR is None:
            with self.timer('Initialize Hamiltonian'):
                self.initialize()

        coarsegrid_e_kinetic, finegrid_energies = \
            self.update_pseudo_potential(dens)

        with self.timer('Calculate atomic Hamiltonians'):
            W_aL = self.calculate_atomic_hamiltonians(dens)

        atomic_energies = self.dH_II.update(dens.D_II, W_aL, self.xc,
                                            self.world, self.timer)

        # Make energy contributions summable over world:
        finegrid_energies *= self.finegd.comm.size / float(self.world.size)
        coarsegrid_e_kinetic *= self.gd.comm.size / float(self.world.size)
        # (careful with array orderings/contents)
        energies = atomic_energies  # kinetic, coulomb, zero, external, xc
        energies[1:] += finegrid_energies  # coulomb, zero, external, xc
        energies[0] += coarsegrid_e_kinetic  # kinetic
        with self.timer('Communicate'):
            self.world.sum(energies)

        (self.e_kinetic0, self.e_coulomb, self.e_zero,
         self.e_external, self.e_xc) = energies

        self.timer.stop('Hamiltonian')

    def get_energy(self, occ):
        self.e_kinetic = self.e_kinetic0 + occ.e_band
        print(self.e_kinetic0, occ.e_band)
        self.e_entropy = occ.e_entropy

        self.e_total_free = (self.e_kinetic + self.e_coulomb +
                             self.e_external + self.e_zero + self.e_xc +
                             self.e_entropy)
        self.e_total_extrapolated = occ.extrapolate_energy_to_zero_width(
            self.e_total_free)

        return self.e_total_free

    def linearize_to_xc(self, new_xc, dens):
        # Store old hamiltonian
        ref_vt_sG = self.vt_sG.copy()
        ref_dH_asp = self.dH_asp.copy()
        self.xc = new_xc
        self.xc.set_positions(self.spos_ac)
        self.update(dens)

        ref_vt_sG -= self.vt_sG
        for a, dH_sp in self.dH_asp.items():
            ref_dH_asp[a] -= dH_sp
        self.ref_vt_sG = ref_vt_sG
        self.ref_dH_asp = self.atomdist.to_work(ref_dH_asp)

    def calculate_forces(self, dens, F_av):
        ghat_aLv, nct_av, vbar_av = self.calculate_forces2(dens)

        # Force from compensation charges:
        for a, dF_Lv in ghat_aLv.items():
            F_av[a] += np.dot(dens.Q_aL[a], dF_Lv)

        # Force from smooth core charge:
        F_coarsegrid_av = np.zeros_like(F_av)
        for a, dF_xv in nct_av.items():
            F_coarsegrid_av[a] = dF_xv[0]

        # Force from zero potential:
        for a, dF_xv in vbar_av.items():
            F_av[a] += dF_xv[0]

        self.xc.add_forces(F_av)
        self.gd.comm.sum(F_coarsegrid_av, 0)
        self.finegd.comm.sum(F_av, 0)

        F_av += F_coarsegrid_av

    def apply_local_potential(self, psit_nR, Htpsit_nR, s):
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
            P_ni = <p_i | a_nR> are calculated.
            When False, existing P_uni are used
        local_part_only: bool
            When True, the non-local atomic parts of the Hamiltonian
            are not applied and calculate_projections is ignored.

        """
        assert self.collinear
        vt_R = self.vt_sR[s]
        for psit_R, Htpsit_R in zip(psit_nR, Htpsit_nR):
            Htpsit_R += psit_R * vt_R

    def apply(self, a_xR, b_xR, wfs, kpt, calculate_P_ani=True):
        """Apply the Hamiltonian operator to a set of vectors.

        Parameters:

        a_nR: ndarray
            Set of vectors to which the overlap operator is applied.
        b_nR: ndarray, output
            Resulting S times a_nR vectors.
        wfs: WaveFunctions
            Wave-function object defined in wavefunctions.py
        kpt: KPoint object
            k-point object defined in kpoint.py.
        calculate_P_ani: bool
            When True, the integrals of projector times vectors
            P_ni = <p_i | a_nR> are calculated.
            When False, existing P_ani are used

        """

        wfs.kin.apply(a_xR, b_xR, kpt.phase_cd)
        self.apply_local_potential(a_xR, b_xR, kpt.s)
        shape = a_xR.shape[:-3]
        P_axi = wfs.pt.dict(shape)

        if calculate_P_ani:  # TODO calculate_P_ani=False is experimental
            wfs.pt.integrate(a_xR, P_axi, kpt.q)
        else:
            for a, P_ni in kpt.P_ani.items():
                P_axi[a][:] = P_ni

        for a, P_xi in P_axi.items():
            dH_ii = unpack(self.dH_asp[a][kpt.s])
            P_axi[a] = np.dot(P_xi, dH_ii)
        wfs.pt.add(b_xR, P_axi, kpt.q)

    def get_xc_difference(self, xc, dens):
        """Calculate non-selfconsistent XC-energy difference."""
        if dens.nt_sr is None:
            dens.interpolate_pseudo_density()
        nt_sr = dens.nt_sr
        if hasattr(xc, 'hybrid'):
            xc.calculate_exx()
        finegd_e_xc = xc.calculate(dens.finegd, nt_sr)
        D_asp = self.atomdist.to_work(dens.D_asp)
        atomic_e_xc = 0.0
        for a, D_sp in D_asp.items():
            setup = self.setups[a]
            atomic_e_xc += xc.calculate_paw_correction(setup, D_sp)
        e_xc = finegd_e_xc + self.world.sum(atomic_e_xc)
        return e_xc - self.e_xc

    def estimate_memory(self, mem):
        nbytes = self.gd.bytecount()
        nfinebytes = self.finegd.bytecount()
        arrays = mem.subnode('Arrays', 0)
        arrays.subnode('vHt_r', nfinebytes)
        arrays.subnode('vt_sR', self.nspins * nbytes)
        arrays.subnode('vt_sr', self.nspins * nfinebytes)
        self.xc.estimate_memory(mem.subnode('XC'))
        self.poisson.estimate_memory(mem.subnode('Poisson'))
        self.vbar.estimate_memory(mem.subnode('vbar'))

    def write(self, writer):
        # Write all eneriges:
        for name in ENERGY_NAMES:
            energy = getattr(self, name)
            if energy is not None:
                energy *= Ha
            writer.write(name, energy)

        writer.write(
            potential=self.gd.collect(self.vt_sR) * Ha,
            atomic_hamiltonian_matrices=pack_atomic_matrices(self.dH_asp) * Ha)

        self.xc.write(writer.child('xc'))

        if hasattr(self.poisson, 'write'):
            self.poisson.write(writer.child('poisson'))

    def read(self, reader):
        h = reader.hamiltonian

        # Read all energies:
        for name in ENERGY_NAMES:
            energy = h.get(name)
            if energy is not None:
                energy /= reader.ha
            setattr(self, name, energy)

        # Read pseudo potential on the coarse grid
        # and broadcast on kpt/band comm:
        self.vt_sR = self.gd.empty(self.nspins)
        self.gd.distribute(h.potential / reader.ha, self.vt_sR)

        # self.atom_partition = AtomPartition(self.gd.comm,
        #                                     np.zeros(len(self.setups), int),
        #                                     name='hamiltonian-init-serial')

        # Read non-local part of hamiltonian
        self.dH_asp = {}
        dH_sP = h.atomic_hamiltonian_matrices / reader.ha

        if self.gd.comm.rank == 0:
            self.dH_asp = unpack_atomic_matrices(dH_sP, self.setups)

        if hasattr(self.poisson, 'read'):
            self.poisson.read(reader)
            self.poisson.set_grid_descriptor(self.finegd)


@frozen
class RealSpaceHamiltonian(Hamiltonian):
    def __init__(self, gd, finegd, spinpolarized, setups, timer, xc, world,
                 vext=None,
                 psolver=None, stencil=3, redistributor=None):
        Hamiltonian.__init__(self, gd, finegd, spinpolarized,
                             setups, timer, xc,
                             world, vext=vext,
                             redistributor=redistributor)

        # Solver for the Poisson equation:
        if psolver is None:
            psolver = {}
        if isinstance(psolver, dict):
            psolver = create_poisson_solver(**psolver)
        self.poisson = psolver
        self.poisson.set_grid_descriptor(self.finegd)

        # Restrictor function for the potential:
        self.restrictor = Transformer(self.finegd, self.redistributor.aux_gd,
                                      stencil)

        self.vbar_a = ACF(self.finegd, [[setup.vbar] for setup in setups])
        self.vbar_r = None

        self.npoisson = None

    def __str__(self):
        s = Hamiltonian.__str__(self)

        degree = self.restrictor.nn * 2 - 1
        name = ['linear', 'cubic', 'quintic', 'heptic'][degree // 2]
        s += ('  Interpolation: tri-%s ' % name +
              '(%d. degree polynomial)\n' % degree)
        s += '  Poisson solver: %s' % self.poisson.get_description()
        return s

    def set_positions(self, spos_ac, rank_a):
        Hamiltonian.set_positions(self, spos_ac, rank_a)
        self.vbar_r = self.finegd.zeros()
        self.vbar_a.add_to(self.vbar_r)

    def update_pseudo_potential(self, dens):
        self.timer.start('vbar')
        e_zero = self.finegd.integrate(self.vbar_r, dens.nt_sr,
                                       global_integral=False).sum()

        vt_r = self.vt_sr[0]
        vt_r[:] = self.vbar_r
        self.timer.stop('vbar')

        e_external = 0.0
        if self.vext is not None:
            vext_r = self.vext.get_potential(self.finegd)
            vt_r += vext_r
            e_external = self.finegd.integrate(vext_r, dens.rhot_r,
                                               global_integral=False)

        if self.nspins == 2:
            self.vt_sr[1] = vt_r

        self.timer.start('XC 3D grid')
        e_xc = self.xc.calculate(self.finegd, dens.nt_sr, self.vt_sr)
        e_xc /= self.finegd.comm.size
        self.timer.stop('XC 3D grid')

        self.timer.start('Poisson')
        # npoisson is the number of iterations:
        self.npoisson = self.poisson.solve(self.vHt_r, dens.rhot_r,
                                           charge=-dens.charge)
        self.timer.stop('Poisson')

        self.timer.start('Hartree integrate')
        e_coulomb = 0.5 * self.finegd.integrate(self.vHt_r, dens.rhot_r,
                                                global_integral=False)

        for vt_r in self.vt_sr:
            vt_r += self.vHt_r

        self.timer.stop('Hartree integrate')

        with self.timer('Hartree restrict'):
            self.vt_sR = self.restrictor.gdout.empty(1 + self.spinpolarized)
            for a, b in zip(self.vt_sr, self.vt_sR):
                if self.redistributor.enabled:
                    c = self.restrictor.apply(a)
                    self.redistributor.collect(c, b)
                else:
                    self.restrictor.apply(a, output=b)

        e_kinetic = 0.0
        s = 0
        for vt_R, nt_R in zip(self.vt_sR, dens.nt_sR):
            if self.ref_vt_sG is not None:
                vt_R += self.ref_vt_sG[s]

            e_kinetic -= self.gd.integrate(vt_R, nt_R - dens.nct_R,
                                           global_integral=False)
            assert self.collinear
            s += 1

        return e_kinetic, np.array([e_coulomb, e_zero, e_external, e_xc])

    def calculate_atomic_hamiltonians(self, dens):
        v_r = self.vHt_r
        if self.vext:
            v_r = v_r + self.vext.get_potential(self.finegd)
        return dens.ghat_aL.integrate(v_r)

    def calculate_forces2(self, dens):
        vbar_av = self.vbar_a.derivative(dens.nt_r)

        vHt_r = self.vHt_r
        if self.vext:
            vHt_r = vHt_r + self.vext.get_potential(self.finegd)
        ghat_aLv = dens.ghat_aL.derivative(self.vHt_r)

        vt_R = self.vt_sR.mean(0) if self.spinpolarized else self.vt_sR[0]
        nct_av = dens.nct_a.derivative(vt_R)

        return ghat_aLv, nct_av, vbar_av

    def get_electrostatic_potential(self, dens):
        self.update(dens)

        v_r = self.finegd.collect(self.vHt_r, broadcast=True)
        v_r = self.finegd.zero_pad(v_r)
        if hasattr(self.poisson, 'correction'):
            assert self.poisson.c == 2
            v_r[:, :, 0] = self.poisson.correction
        return v_r


@frozen
class ReciprocalSpaceHamiltonian(Hamiltonian):
    def __init__(self, gd, finegd, pd2, pd3, spinpolarized, setups, timer, xc,
                 world, vext=None,
                 psolver=None, redistributor=None, realpbc_c=None):

        assert gd.comm.size == 1
        assert finegd.comm.size == 1
        assert redistributor is not None  # XXX should not be like this
        Hamiltonian.__init__(self, gd, finegd, spinpolarized, setups,
                             timer, xc, world, vext=vext,
                             redistributor=redistributor)

        self.vbar_a = ACF(pd2, [[setup.vbar] for setup in setups])
        self.pd2 = pd2
        self.pd3 = pd3

        self.vbar_Q = None

        self.vHt_q = pd3.empty()

        from gpaw.poisson import ReciprocalSpacePoissonSolver
        if psolver is None:
            psolver = ReciprocalSpacePoissonSolver(pd3, realpbc_c)
        elif isinstance(psolver, dict):
            direction = psolver['dipolelayer']
            assert len(psolver) == 1
            from gpaw.dipole_correction import DipoleCorrection
            psolver = DipoleCorrection(
                ReciprocalSpacePoissonSolver(pd3, realpbc_c), direction)
        self.poisson = psolver
        self.npoisson = 0
        self.e_stress = None
        self.vt_Q = None

    def set_positions(self, spos_ac, rank_a):
        Hamiltonian.set_positions(self, spos_ac, rank_a)
        self.vbar_Q = self.pd2.zeros()
        self.vbar_a.add_to(self.vbar_Q)

    def update_pseudo_potential(self, dens):
        ebar = self.pd2.integrate(self.vbar_Q, dens.nt_Q)

        with self.timer('Poisson'):
            self.poisson.solve(self.vHt_q, dens)
            epot = 0.5 * self.pd3.integrate(self.vHt_q, dens.rhot_q)

        self.vt_sR = self.gd.empty(1 + self.spinpolarized)

        self.vt_Q = self.vbar_Q + self.vHt_q[dens.G3_G] / 8
        self.vt_sR[:] = self.pd2.ifft(self.vt_Q)

        self.timer.start('XC 3D grid')
        nt_dist_sr = dens.xc_redistributor.distribute(dens.nt_sr)
        vxct_dist_sr = dens.xc_redistributor.aux_gd.zeros(self.nspins)
        exc = self.xc.calculate(dens.xc_redistributor.aux_gd,
                                nt_dist_sr, vxct_dist_sr)
        vxct_sr = dens.xc_redistributor.collect(vxct_dist_sr)

        for vt_R, vxct_r in zip(self.vt_sR, vxct_sr):
            vxc_R, vxc_Q = self.pd3.restrict(vxct_r, self.pd2)
            vt_R += vxc_R
            self.vt_Q += vxc_Q / self.nspins
        self.timer.stop('XC 3D grid')

        eext = 0.0

        self.e_stress = ebar + epot

        ekin = 0.0
        for vt_R, nt_R in zip(self.vt_sR, dens.nt_sR):
            ekin -= self.gd.integrate(vt_R, nt_R)
        ekin += self.gd.integrate(self.vt_sR, dens.nct_R).sum()

        return ekin, np.array([epot, ebar, eext, exc])

    def calculate_atomic_hamiltonians(self, dens):
        return dens.ghat_aL.integrate(self.vHt_q)

    def restrict(self, in_xR, out_xR=None):
        """Restrict array."""
        if out_xR is None:
            out_xR = self.gd.empty(in_xR.shape[:-3])

        a_xR = in_xR.reshape((-1,) + in_xR.shape[-3:])
        b_xR = out_xR.reshape((-1,) + out_xR.shape[-3:])

        for in_R, out_R in zip(a_xR, b_xR):
            out_R[:] = self.pd3.restrict(in_R, self.pd2)[0]

        return out_xR

    restrict_and_collect = restrict

    def calculate_forces2(self, dens):
        ghat_aLv = dens.ghat_aL.derivative(self.vHt_q)
        nct_av = dens.nct_a.derivative(self.vt_Q)
        vbar_av = self.vbar_a.derivative(dens.nt_Q)
        return ghat_aLv, nct_av, vbar_av

    def get_electrostatic_potential(self, dens):
        self.poisson.solve(self.vHt_q, dens)
        return self.pd3.ifft(self.vHt_q)

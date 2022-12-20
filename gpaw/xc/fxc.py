import os
from time import time

import ase.io.ulm as ulm
import numpy as np
from ase.units import Ha
from gpaw.response import timer
from scipy.special import p_roots, sici


import gpaw.mpi as mpi
from gpaw.blacs import BlacsGrid, Redistributor
from gpaw.fd_operators import Gradient
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.pw.descriptor import PWDescriptor
from gpaw.utilities.blas import axpy, gemmdot
from gpaw.xc.rpa import RPACorrelation
from gpaw.heg import HEG


class FXCCorrelation:
    def __init__(self,
                 calc,
                 xc='RPA',
                 nlambda=8,
                 frequencies=None,
                 weights=None,
                 density_cut=1.e-6,
                 unit_cells=None,
                 tag=None,
                 range_rc=1.0,
                 av_scheme=None,
                 Eg=None,
                 **kwargs):

        self.rpa = RPACorrelation(
            calc,
            xc=xc,
            nlambda=nlambda,
            frequencies=frequencies,
            weights=weights,
            calculate_q=self.calculate_q_fxc,
            **kwargs)

        self.gs = self.rpa.gs
        self.context = self.rpa.context

        self.l_l, self.weight_l = p_roots(nlambda)
        self.l_l = (self.l_l + 1.0) * 0.5
        self.weight_l *= 0.5
        self.xc = xc
        self.density_cut = density_cut
        if unit_cells is None:
            unit_cells = self.gs.kd.N_c
        self.unit_cells = unit_cells
        self.range_rc = range_rc  # Range separation parameter in Bohr

        if Eg is not None:
            Eg /= Ha
        self.Eg = Eg  # Band gap in eV

        self.xcflags = XCFlags(self.xc)
        if self.xcflags.bandgap_dependent != (self.Eg is not None):
            raise RuntimeError(
                'Gap must be provided with and only with '
                f'the gap dependent functionals {self.xcflags._gapped}.')
        self.av_scheme = self.xcflags.choose_avg_scheme(av_scheme)

        if tag is None:

            tag = self.gs.atoms.get_chemical_formula(mode='hill')

            if self.av_scheme is not None:

                tag += '_' + self.av_scheme

        self.tag = tag

        self.omega_w = self.rpa.omega_w
        self.ibzq_qc = self.rpa.ibzq_qc
        self.nblocks = self.rpa.nblocks
        self.weight_w = self.rpa.weight_w

    @property
    def blockcomm(self):
        # Cannot be aliased as attribute
        # because rpa gets blockcomm during calculate
        return self.rpa.blockcomm

    @timer('FXC')
    def calculate(self, ecut, nbands=None):

        if self.xc not in ('RPA', 'range_RPA'):
            # kernel not required for RPA/range_sep RPA

            if isinstance(ecut, (float, int)):
                self.ecut_max = ecut
            else:
                self.ecut_max = max(ecut)

            # Find the first q vector to calculate kernel for
            # (density averaging scheme always calculates all q points anyway)

            q_empty = None

            for iq in reversed(range(len(self.ibzq_qc))):

                if not os.path.isfile('fhxc_%s_%s_%s_%s.ulm' %
                                      (self.tag, self.xc, self.ecut_max, iq)):
                    q_empty = iq

            kernelkwargs = dict(
                gs=self.gs,
                xc=self.xc,
                ibzq_qc=self.ibzq_qc,
                ecut=self.ecut_max,
                tag=self.tag,
                context=self.context)

            if q_empty is not None:

                if self.av_scheme == 'wavevector':

                    self.context.print('Calculating %s kernel starting from '
                                       'q point %s \n' % (self.xc, q_empty))

                    kernelkwargs.update(l_l=self.l_l,
                                        q_empty=q_empty,
                                        omega_w=self.omega_w,
                                        Eg=self.Eg)

                    if self.xcflags.linear_kernel:
                        kernelkwargs.update(l_l=None, omega_w=None)
                    elif not self.xcflags.dyn_kernel:
                        kernelkwargs.update(omega_w=None)

                    kernel = KernelWave(**kernelkwargs)

                else:
                    kernel = KernelDens(**kernelkwargs,
                                        unit_cells=self.unit_cells,
                                        density_cut=self.density_cut)

                kernel.calculate_fhxc()
                del kernel

            else:
                self.context.print('%s kernel already calculated\n' %
                                   self.xc)

        if self.xc in ('range_RPA', 'range_rALDA'):

            shortrange = range_separated(self.gs, self.context, self.omega_w,
                                         self.weight_w, self.l_l,
                                         self.weight_l, self.range_rc, self.xc)

            self.shortrange = shortrange.calculate()

        if self.gs.nspins == 1:
            spin = False
        else:
            spin = True

        e = self.rpa.calculate(ecut, spin=spin, nbands=nbands)

        return e

    @timer('Chi0(q)')
    def calculate_q_fxc(self, chi0calc, chi0_s, m1, m2, cut_G):
        for s, chi0 in enumerate(chi0_s):
            chi0calc.update_chi0(chi0,
                                 m1,
                                 m2, [s])
        self.context.print('E_c(q) = ', end='', flush=False)

        pd = chi0.pd
        nw = chi0.nw
        mynw = nw // self.nblocks
        assert nw % self.nblocks == 0
        nspins = len(chi0_s)
        nG = pd.ngmax
        chi0_swGG = np.empty((nspins, mynw, nG, nG), complex)
        for chi0_wGG, chi0 in zip(chi0_swGG, chi0_s):
            chi0_wGG[:] = chi0.distribute_as('wGG')
        if self.nblocks > 1:
            chi0_swGG = np.swapaxes(chi0_swGG, 2, 3)

        if not pd.kd.gamma:
            e = self.calculate_energy_fxc(pd, chi0_swGG, cut_G)
            self.context.print('%.3f eV' % (e * Ha))
        else:
            w1 = self.blockcomm.rank * mynw
            w2 = w1 + mynw
            e = 0.0
            for v in range(3):
                for chi0_wGG, chi0 in zip(chi0_swGG, chi0_s):
                    chi0_wGG[:, 0] = chi0.chi0_wxvG[w1:w2, 0, v]
                    chi0_wGG[:, :, 0] = chi0.chi0_wxvG[w1:w2, 1, v]
                    chi0_wGG[:, 0, 0] = chi0.chi0_wvv[w1:w2, v, v]
                ev = self.calculate_energy_fxc(pd, chi0_swGG, cut_G)
                e += ev
                self.context.print('%.3f' % (ev * Ha), end='', flush=False)
                if v < 2:
                    self.context.print('/', end='', flush=False)
                else:
                    self.context.print('eV')
            e /= 3

        return e

    @timer('Energy')
    def calculate_energy_fxc(self, pd, chi0_swGG, cut_G):
        """Evaluate correlation energy from chi0 and the kernel fhxc"""

        ibzq2_q = [
            np.dot(self.ibzq_qc[i] - pd.kd.bzk_kc[0],
                   self.ibzq_qc[i] - pd.kd.bzk_kc[0])
            for i in range(len(self.ibzq_qc))
        ]

        qi = np.argsort(ibzq2_q)[0]

        G_G = pd.G2_qG[0]**0.5  # |G+q|

        if cut_G is not None:
            G_G = G_G[cut_G]

        nG = len(G_G)
        ns = len(chi0_swGG)

        # There are three options to calculate the
        # energy depending on kernel and/or averaging scheme.

        # Option (1) - Spin-polarized form of kernel exists
        #              e.g. rALDA, rAPBE.
        #              Then, solve block diagonal form of Dyson
        #              equation (dimensions (ns*nG) * (ns*nG))
        #              (note this does not necessarily mean that
        #              the calculation is spin-polarized!)

        if self.xcflags.spin_kernel:
            with ulm.open('fhxc_%s_%s_%s_%s.ulm' %
                          (self.tag, self.xc, self.ecut_max, qi)) as r:
                fv = r.fhxc_sGsG

            if cut_G is not None:
                cut_sG = np.tile(cut_G, ns)
                cut_sG[len(cut_G):] += len(fv) // ns
                fv = fv.take(cut_sG, 0).take(cut_sG, 1)

            # the spin-polarized kernel constructed from wavevector average
            # is already multiplied by |q+G| |q+G'|/4pi, and doesn't require
            # special treatment of the head and wings.  However not true for
            # density average:

            if self.av_scheme == 'density':
                for s1 in range(ns):
                    for s2 in range(ns):
                        m1 = s1 * nG
                        n1 = (s1 + 1) * nG
                        m2 = s2 * nG
                        n2 = (s2 + 1) * nG
                        fv[m1:n1,
                           m2:n2] *= (G_G * G_G[:, np.newaxis] / (4 * np.pi))

                        if np.prod(self.unit_cells) > 1 and pd.kd.gamma:
                            m1 = s1 * nG
                            n1 = (s1 + 1) * nG
                            m2 = s2 * nG
                            n2 = (s2 + 1) * nG
                            fv[m1, m2:n2] = 0.0
                            fv[m1:n1, m2] = 0.0
                            fv[m1, m2] = 1.0

            if pd.kd.gamma:
                G_G[0] = 1.0

            e_w = []

            # Loop over frequencies
            for chi0_sGG in np.swapaxes(chi0_swGG, 0, 1):
                if cut_G is not None:
                    chi0_sGG = chi0_sGG.take(cut_G, 1).take(cut_G, 2)
                chi0v = np.zeros((ns * nG, ns * nG), dtype=complex)
                for s in range(ns):
                    m = s * nG
                    n = (s + 1) * nG
                    chi0v[m:n, m:n] = chi0_sGG[s] / G_G / G_G[:, np.newaxis]
                chi0v *= 4 * np.pi

                del chi0_sGG

                e = 0.0

                for l, weight in zip(self.l_l, self.weight_l):
                    chiv = np.linalg.solve(
                        np.eye(nG * ns) - l * np.dot(chi0v, fv),
                        chi0v).real  # this is SO slow
                    for s1 in range(ns):
                        for s2 in range(ns):
                            m1 = s1 * nG
                            n1 = (s1 + 1) * nG
                            m2 = s2 * nG
                            n2 = (s2 + 1) * nG
                            chiv_s1s2 = chiv[m1:n1, m2:n2]
                            e -= np.trace(chiv_s1s2) * weight

                e += np.trace(chi0v.real)

                e_w.append(e)

        else:
            # Or, if kernel does not have a spin polarized form,
            #
            # Option (2)  kernel does not scale linearly with lambda,
            #             so we solve nG*nG Dyson equation at each value
            #             of l.  Requires kernel to be constructed
            #             at individual values of lambda
            #
            # Option (3)  Divide correlation energy into
            #             long range part which can be integrated
            #             analytically w.r.t. lambda, and a short
            #             range part which again requires
            #             solving Dyson equation (hence no speedup,
            #             but the maths looks nice and
            #             fits with range-separated RPA)
            #
            #
            # Construct/read kernels

            if self.xc == 'RPA':

                fv = np.eye(nG)

            elif self.xc == 'range_RPA':

                fv = np.exp(-0.25 * (G_G * self.range_rc)**2.0)

            elif self.xcflags.linear_kernel:
                with ulm.open('fhxc_%s_%s_%s_%s.ulm' %
                              (self.tag, self.xc, self.ecut_max, qi)) as r:
                    fv = r.fhxc_sGsG

                if cut_G is not None:
                    fv = fv.take(cut_G, 0).take(cut_G, 1)

            elif not self.xcflags.dyn_kernel:
                # static kernel which does not scale with lambda

                with ulm.open('fhxc_%s_%s_%s_%s.ulm' %
                              (self.tag, self.xc, self.ecut_max, qi)) as r:
                    fv = r.fhxc_lGG

                if cut_G is not None:
                    fv = fv.take(cut_G, 1).take(cut_G, 2)

            else:  # dynamical kernel
                with ulm.open('fhxc_%s_%s_%s_%s.ulm' %
                              (self.tag, self.xc, self.ecut_max, qi)) as r:
                    fv = r.fhxc_lwGG

                if cut_G is not None:
                    fv = fv.take(cut_G, 2).take(cut_G, 3)

            if pd.kd.gamma:
                G_G[0] = 1.0

            # Loop over frequencies; since the kernel has no spin,
            # we work with spin-summed response function
            e_w = []
            iw = 0

            for chi0_sGG in np.swapaxes(chi0_swGG, 0, 1):
                if cut_G is not None:
                    chi0_sGG = chi0_sGG.take(cut_G, 1).take(cut_G, 2)
                chi0v = np.zeros((nG, nG), dtype=complex)
                for s in range(ns):
                    chi0v += chi0_sGG[s] / G_G / G_G[:, np.newaxis]
                chi0v *= 4 * np.pi
                del chi0_sGG

                e = 0.0

                if not self.xcflags.linear_kernel:

                    il = 0
                    for l, weight in zip(self.l_l, self.weight_l):

                        if not self.xcflags.dyn_kernel:
                            chiv = np.linalg.solve(
                                np.eye(nG) - np.dot(chi0v, fv[il]), chi0v).real
                        else:
                            chiv = np.linalg.solve(
                                np.eye(nG) - np.dot(chi0v, fv[il][iw]),
                                chi0v).real
                        e -= np.trace(chiv) * weight
                        il += 1

                    e += np.trace(chi0v.real)

                    e_w.append(e)

                    iw += 1

                else:

                    # Coupling constant integration
                    # for long-range part
                    # Do this analytically, except for the RPA
                    # simply since the analytical method is already
                    # implemented in rpa.py
                    if self.xc == 'range_RPA':
                        # way faster than np.dot for diagonal kernels
                        e_GG = np.eye(nG) - chi0v * fv
                    elif self.xc != 'RPA':
                        e_GG = np.eye(nG) - np.dot(chi0v, fv)

                    if self.xc == 'RPA':
                        # numerical RPA
                        elong = 0.0
                        for l, weight in zip(self.l_l, self.weight_l):

                            chiv = np.linalg.solve(
                                np.eye(nG) - l * np.dot(chi0v, fv), chi0v).real

                            elong -= np.trace(chiv) * weight

                        elong += np.trace(chi0v.real)

                    else:
                        # analytic everything else
                        elong = (np.log(np.linalg.det(e_GG)) + nG -
                                 np.trace(e_GG)).real
                    # Numerical integration for short-range part
                    eshort = 0.0
                    if self.xc not in ('RPA', 'range_RPA', 'range_rALDA'):
                        fxcv = fv - np.eye(nG)  # Subtract Hartree contribution

                        for l, weight in zip(self.l_l, self.weight_l):

                            chiv = np.linalg.solve(
                                np.eye(nG) - l * np.dot(chi0v, fv), chi0v)
                            eshort += (np.trace(np.dot(chiv, fxcv)).real *
                                       weight)

                        eshort -= np.trace(np.dot(chi0v, fxcv)).real

                    elif self.xcflags.is_ranged:
                        eshort = (2 * np.pi * self.shortrange /
                                  np.sum(self.weight_w))

                    e = eshort + elong
                    e_w.append(e)

        E_w = np.zeros_like(self.omega_w)
        self.blockcomm.all_gather(np.array(e_w), E_w)
        energy = np.dot(E_w, self.weight_w) / (2 * np.pi)
        return energy


class KernelWave:
    def __init__(self, gs, xc, ibzq_qc, l_l, q_empty, omega_w, Eg, ecut,
                 tag, context):

        self.gs = gs
        self.gd = gs.density.gd
        self.xc = xc
        self.xcflags = XCFlags(xc)
        self.ibzq_qc = ibzq_qc
        self.l_l = l_l
        self.ns = self.gs.nspins
        self.q_empty = q_empty
        self.omega_w = omega_w
        self.Eg = Eg
        self.ecut = ecut
        self.tag = tag
        self.context = context

        if l_l is None:  # -> Kernel is linear in coupling strength
            self.l_l = [1.0]

        # Density grid
        n_sg, finegd = self.gs.all_electron_density(gridrefinement=2)
        self.n_g = n_sg.sum(axis=0).flatten()

        #  For atoms with large vacuum regions
        #  this apparently can take negative values!
        mindens = np.amin(self.n_g)

        if mindens < 0:
            self.context.print('Negative densities found! (magnitude %s)' %
                               np.abs(mindens), flush=False)
            self.context.print('These will be reset to 1E-12 elec/bohr^3)')
            self.n_g[np.where(self.n_g < 0.0)] = 1.0E-12

        r_g = finegd.get_grid_point_coordinates()
        self.x_g = 1.0 * r_g[0].flatten()
        self.y_g = 1.0 * r_g[1].flatten()
        self.z_g = 1.0 * r_g[2].flatten()
        self.gridsize = len(self.x_g)
        assert len(self.n_g) == self.gridsize

        if self.omega_w is not None:
            self.context.print('Calculating dynamical kernel at %s '
                               'frequencies' % len(self.omega_w))

        if self.Eg is not None:
            self.context.print('Band gap of %s eV used to evaluate kernel'
                               % (self.Eg * Ha))

        # Enhancement factor for GGA
        if self.xcflags.is_apbe:
            nf_g = self.gs.hacky_all_electron_density(gridrefinement=4)
            gdf = self.gd.refine().refine()
            grad_v = [Gradient(gdf, v, n=1).apply for v in range(3)]
            gradnf_vg = gdf.empty(3)

            for v in range(3):
                grad_v[v](nf_g, gradnf_vg[v])

            self.s2_g = np.sqrt(np.sum(gradnf_vg[:, ::2, ::2, ::2]**2.0,
                                       0)).flatten()  # |\nabla\rho|
            self.s2_g *= 1.0 / (2.0 * (3.0 * np.pi**2.0)**(1.0 / 3.0) *
                                self.n_g**(4.0 / 3.0))
            # |\nabla\rho|/(2kF\rho) = s
            self.s2_g = self.s2_g**2  # s^2
            assert len(self.n_g) == len(self.s2_g)

            # Now we find all the regions where the
            # APBE kernel wants to be positive, and hack s to = 0,
            # so that we are really using the ALDA kernel
            # at these points
            apbe_g = self.get_PBE_fxc(self.n_g, self.s2_g)
            poskern_ind = np.where(apbe_g >= 0.0)
            if len(poskern_ind[0]) > 0:
                self.context.print(
                    'The APBE kernel takes positive values at '
                    + '%s grid points out of a total of %s (%3.2f%%).'
                    % (len(poskern_ind[0]), self.gridsize, 100.0 * len(
                        poskern_ind[0]) / self.gridsize), flush=False)
                self.context.print('The ALDA kernel will be used at these '
                                   'points')
                self.s2_g[poskern_ind] = 0.0

    def calculate_fhxc(self):

        self.context.print('Calculating %s kernel at %d eV cutoff'
                           % (self.xc, self.ecut), flush=False)

        for iq, q_c in enumerate(self.ibzq_qc):

            if iq < self.q_empty:  # don't recalculate q vectors
                continue

            thisqd = KPointDescriptor([q_c])
            pd = PWDescriptor(self.ecut / Ha, self.gd, complex, thisqd)

            nG = pd.ngmax
            G_G = pd.G2_qG[0]**0.5  # |G+q|
            Gv_G = pd.get_reciprocal_vectors(q=0, add_q=False)
            # G as a vector (note we are at a specific q point here so set q=0)

            # Distribute G vectors among processors
            # Later we calculate for iG' > iG,
            # so stagger allocation in order to balance load
            local_Gvec_grid_size = nG // mpi.world.size
            my_Gints = (mpi.world.rank + np.arange(
                0, local_Gvec_grid_size * mpi.world.size, mpi.world.size))

            if (mpi.world.rank + (local_Gvec_grid_size) * mpi.world.size) < nG:
                my_Gints = np.append(
                    my_Gints,
                    [mpi.world.rank + local_Gvec_grid_size * mpi.world.size])

            my_Gv_G = Gv_G[my_Gints]

            if (self.ns == 2) and (self.xc == 'rALDA' or self.xc == 'rAPBE'):

                assert len(self.l_l) == 1

                # Form spin-dependent kernel according to
                # PRB 88, 115131 (2013) equation 20
                # (note typo, should be \tilde{f^rALDA})
                # spincorr is just the ALDA exchange kernel
                # with a step function (\equiv \tilde{f^rALDA})
                # fHxc^{up up}     = fHxc^{down down} = fv_nospin + fv_spincorr
                # fHxc^{up down}   = fHxc^{down up}   = fv_nospin - fv_spincorr

                calc_spincorr = True
                fv_spincorr = np.zeros((nG, nG), dtype=complex)

            else:

                calc_spincorr = False

            if self.omega_w is None:
                fv_nospin = np.zeros((len(self.l_l), nG, nG), dtype=complex)
            else:
                fv_nospin = np.zeros(
                    (len(self.l_l), len(self.omega_w), nG, nG), dtype=complex)

            for il, l in enumerate(self.l_l):  # loop over coupling constant

                for iG, Gv in zip(my_Gints, my_Gv_G):  # loop over G vecs

                    # For all kernels except JGM we
                    # treat head and wings analytically
                    if G_G[iG] > 1.0E-5 or self.xc == 'JGMs':

                        # Symmetrised |q+G||q+G'|, where iG' >= iG
                        mod_Gpq = np.sqrt(G_G[iG] * G_G[iG:])

                        # Phase factor \vec{G}-\vec{G'}
                        deltaGv = Gv - Gv_G[iG:]

                        if (self.xc in ('rALDA', 'range_rALDA', 'rALDAns')):

                            # rALDA trick: the Hartree-XC kernel is exactly
                            # zero for densities below rho_min =
                            # min_Gpq^3/(24*pi^2),
                            # so we don't need to include these contributions
                            # in the Fourier transform

                            min_Gpq = np.amin(mod_Gpq)
                            rho_min = min_Gpq**3.0 / (24.0 * np.pi**2.0)
                            small_ind = np.where(self.n_g >= rho_min)

                        elif self.xcflags.is_apbe:

                            # rAPBE trick: the Hartree-XC kernel
                            # is exactly zero at grid points where
                            # min_Gpq > cutoff wavevector

                            min_Gpq = np.amin(mod_Gpq)
                            small_ind = np.where(min_Gpq <= np.sqrt(
                                -4.0 * np.pi /
                                self.get_PBE_fxc(self.n_g, self.s2_g)))

                        else:

                            small_ind = np.arange(self.gridsize)

                        phase_Gpq = np.exp(
                            -1.0j *
                            (deltaGv[:, 0, np.newaxis] * self.x_g[small_ind] +
                             deltaGv[:, 1, np.newaxis] * self.y_g[small_ind] +
                             deltaGv[:, 2, np.newaxis] * self.z_g[small_ind]))
                        if self.omega_w is None:
                            fv_nospin[il, iG, iG:] = self.get_scaled_fHxc_q(
                                q=mod_Gpq,
                                sel_points=small_ind,
                                Gphase=phase_Gpq,
                                l=l,
                                spincorr=False,
                                w=None)
                        else:
                            for iw, omega in enumerate(self.omega_w):
                                fv_nospin[il, iw, iG, iG:] = \
                                    self.get_scaled_fHxc_q(
                                        q=mod_Gpq,
                                        sel_points=small_ind,
                                        Gphase=phase_Gpq,
                                        l=l,
                                        spincorr=False,
                                        w=omega)

                        if calc_spincorr:
                            fv_spincorr[iG, iG:] = self.get_scaled_fHxc_q(
                                q=mod_Gpq,
                                sel_points=small_ind,
                                Gphase=phase_Gpq,
                                l=1.0,
                                spincorr=True,
                                w=None)

                    else:
                        # head and wings of q=0 are dominated by
                        # 1/q^2 divergence of scaled Coulomb interaction

                        assert iG == 0

                        if self.omega_w is None:
                            fv_nospin[il, 0, 0] = l
                            fv_nospin[il, 0, 1:] = 0.0
                        else:
                            fv_nospin[il, :, 0, 0] = l
                            fv_nospin[il, :, 0, 1:] = 0.0

                        if calc_spincorr:
                            fv_spincorr[0, :] = 0.0

                    # End loop over G vectors

                mpi.world.sum(fv_nospin[il])

                if self.omega_w is None:
                    # We've only got half the matrix here,
                    # so add the hermitian conjugate:
                    fv_nospin[il] += np.conj(fv_nospin[il].T)
                    # but now the diagonal's been doubled,
                    # so we multiply these elements by 0.5
                    fv_nospin[il][np.diag_indices(nG)] *= 0.5

                else:  # same procedure for dynamical kernels
                    for iw in range(len(self.omega_w)):
                        fv_nospin[il][iw] += np.conj(fv_nospin[il][iw].T)
                        fv_nospin[il][iw][np.diag_indices(nG)] *= 0.5

                # End of loop over coupling constant

            if calc_spincorr:
                mpi.world.sum(fv_spincorr)
                fv_spincorr += np.conj(fv_spincorr.T)
                fv_spincorr[np.diag_indices(nG)] *= 0.5

            # Write to disk
            if mpi.rank == 0:

                w = ulm.open(
                    'fhxc_%s_%s_%s_%s.ulm' %
                    (self.tag, self.xc, self.ecut, iq), 'w')

                if calc_spincorr:
                    # Form the block matrix kernel
                    fv_full = np.empty((2 * nG, 2 * nG), dtype=complex)
                    fv_full[:nG, :nG] = fv_nospin[0] + fv_spincorr
                    fv_full[:nG, nG:] = fv_nospin[0] - fv_spincorr
                    fv_full[nG:, :nG] = fv_nospin[0] - fv_spincorr
                    fv_full[nG:, nG:] = fv_nospin[0] + fv_spincorr
                    w.write(fhxc_sGsG=fv_full)

                elif len(self.l_l) == 1:
                    w.write(fhxc_sGsG=fv_nospin[0])

                elif self.omega_w is None:
                    w.write(fhxc_lGG=fv_nospin)

                else:
                    w.write(fhxc_lwGG=fv_nospin)
                w.close()

            self.context.print('q point %s complete' % iq)

            mpi.world.barrier()

    def get_scaled_fHxc_q(self, q, sel_points, Gphase, l, spincorr, w):
        # Given a coupling constant l, construct the Hartree-XC
        # kernel in q space a la Lein, Gross and Perdew,
        # Phys. Rev. B 61, 13431 (2000):
        #
        # f_{Hxc}^\lambda(q,\omega,r_s) = \frac{4\pi \lambda }{q^2}  +
        # \frac{1}{\lambda} f_{xc}(q/\lambda,\omega/\lambda^2,\lambda r_s)
        #
        # divided by the unscaled Coulomb interaction!!
        #
        # i.e. this subroutine returns f_{Hxc}^\lambda(q,\omega,r_s)
        #                              *  \frac{q^2}{4\pi}
        # = \lambda * [\frac{(q/lambda)^2}{4\pi}
        #              f_{Hxc}(q/\lambda,\omega/\lambda^2,\lambda r_s)]
        # = \lambda * [1/scaled_coulomb * fHxc computed with scaled quantities]

        # Apply scaling
        rho = self.n_g[sel_points]

        # GGA enhancement factor s is lambda independent,
        # but we might want to truncate it
        if self.xcflags.is_apbe:
            s2_g = self.s2_g[sel_points]
        else:
            s2_g = None

        scaled_q = q / l
        scaled_rho = rho / l**3.0
        scaled_rs = (3.0 / (4.0 * np.pi * scaled_rho))**(1.0 / 3.0
                                                         )  # Wigner radius
        if w is not None:
            scaled_w = w / (l**2.0)
        else:
            scaled_w = None

        if self.Eg is not None:
            scaled_Eg = self.Eg / (l**1.5)
        else:
            scaled_Eg = None

        if not spincorr:
            scaled_kernel = l * self.get_fHxc_q(scaled_rs, scaled_q, Gphase,
                                                s2_g, scaled_w, scaled_Eg)
        else:
            scaled_kernel = l * self.get_spinfHxc_q(scaled_rs, scaled_q,
                                                    Gphase, s2_g)

        return scaled_kernel

    def get_fHxc_q(self, rs, q, Gphase, s2_g, w, scaled_Eg):
        # Construct fHxc(q,G,:), divided by scaled Coulomb interaction

        heg = HEG(rs)
        qF = heg.qF

        if self.xc in ('rALDA', 'rALDAns', 'range_rALDA'):
            # rALDA (exchange only) kernel
            # Olsen and Thygesen, Phys. Rev. B 88, 115131 (2013)
            # ALDA up to 2*qF, -vc for q >2qF (such that fHxc vanishes)

            rxalda_A = 0.25
            rxalda_qcut = qF * np.sqrt(1.0 / rxalda_A)

            # construct fHxc(k,r)
            fHxc_Gr = (0.5 + 0.0j) * (
                (1.0 + np.sign(rxalda_qcut - q[:, np.newaxis])) *
                (1.0 + (-1.0) * rxalda_A * (q[:, np.newaxis] / qF)**2.0))

        elif self.xc == 'rALDAc':
            # rALDA exchange+correlation kernel
            # Olsen and Thygesen, Phys. Rev. B 88, 115131 (2013)
            # Cutoff wavevector ensures kernel is continuous

            rxcalda_A = self.get_heg_A(rs)
            rxcalda_qcut = qF * np.sqrt(1.0 / rxcalda_A)

            # construct fHxc(k,r)
            fHxc_Gr = (0.5 + 0.0j) * (
                (1.0 + np.sign(rxcalda_qcut - q[:, np.newaxis])) *
                (1.0 + (-1.0) * rxcalda_A * (q[:, np.newaxis] / qF)**2.0))

        elif self.xcflags.is_apbe:

            # Olsen and Thygesen, Phys. Rev. Lett. 112, 203001 (2014)
            # Exchange only part of the PBE XC kernel, neglecting the terms
            # arising from the variation of the density gradient
            # i.e. second functional derivative
            # d2/drho^2 -> \partial^2/\partial rho^2 at fixed \nabla \rho

            fxc_PBE = self.get_PBE_fxc(pbe_rho=3.0 / (4.0 * np.pi * rs**3.0),
                                       pbe_s2_g=s2_g)
            rxapbe_qcut = np.sqrt(-4.0 * np.pi / fxc_PBE)

            fHxc_Gr = (0.5 + 0.0j) * (
                (1.0 + np.sign(rxapbe_qcut - q[:, np.newaxis])) *
                (1.0 + fxc_PBE / (4.0 * np.pi) * (q[:, np.newaxis])**2.0))

        elif self.xc == 'CP':
            # Constantin & Pitarke, Phys. Rev. B 75, 245127 (2007)
            # equation 4, omega = 0
            # and equation 9 [note that fxc(q=0) = fxc^ALDA = -4piA/qf^2]
            # The simplest, static error-function kernel.
            # Produces correct q = 0 limit, but not q->oo

            cp_kappa = self.get_heg_A(rs) / (qF**2.0)
            fHxc_Gr = (1.0 + 0.0j) * np.exp(-cp_kappa * q[:, np.newaxis]**2.0)

        elif self.xc == 'CP_dyn':
            # CP kernel with frequency dependence

            cp_A = self.get_heg_A(rs)
            cp_D = self.get_heg_D(rs)
            cp_c = cp_D / cp_A
            cp_a = 6.0 * np.sqrt(cp_c)

            cp_kappa_w = cp_A / (qF**2.0)
            cp_kappa_w *= (1.0 + cp_a * w + cp_c * w**2.0) / (1.0 + w**2.0)

            fHxc_Gr = (1.0 + 0.0j) * np.exp(
                -cp_kappa_w * q[:, np.newaxis]**2.0)

        elif self.xc == 'CDOP' or self.xc == 'CDOPs':
            # Corradini, Del Sole, Onida and Palummo (CDOP),
            # Phys. Rev. B 57, 14569 (1998)
            # Reproduces exact q=0 and q -> oo limits

            cdop_Q = q[:, np.newaxis] / qF
            cdop_A = self.get_heg_A(rs)
            cdop_B = self.get_heg_B(rs)
            if self.xc == 'CDOPs':
                cdop_C = np.zeros(np.shape(cdop_B))
            else:
                cdop_C = self.get_heg_C(rs)
            cdop_g = cdop_B / (cdop_A - cdop_C)
            cdop_alpha = 1.5 / (rs**0.25) * cdop_A / (cdop_B * cdop_g)
            cdop_beta = 1.2 / (cdop_B * cdop_g)

            fHxc_Gr = -cdop_C * cdop_Q**2.0 * (1.0 + 0.0j)
            fHxc_Gr += -cdop_B * cdop_Q**2.0 * 1.0 / (cdop_g + cdop_Q**2.0)
            fHxc_Gr += -(cdop_alpha * cdop_Q**4.0 *
                         np.exp(-1.0 * cdop_beta * cdop_Q**2.0))

            # Hartree
            fHxc_Gr += 1.0

        elif self.xc == 'JGMs':
            # Trevisanutto et al.,
            # Phys. Rev. B 87, 205143 (2013)
            # equation 4,
            # so-called "jellium with gap" model, but simplified
            # so that it is similar to CP.

            jgm_kappa = self.get_heg_A(rs) / (qF**2.0)
            jgm_n = 3.0 / (4.0 * np.pi * rs**3.0)
            fHxc_Gr = (1.0 +
                       0.0j) * (np.exp(-jgm_kappa * q[:, np.newaxis]**2.0) *
                                np.exp(-scaled_Eg**2.0 /
                                       (4.0 * np.pi * jgm_n)))

        # Integrate over r with phase
        fHxc_Gr *= Gphase
        fHxc_GG = np.sum(fHxc_Gr, 1) / self.gridsize
        return fHxc_GG

    def get_spinfHxc_q(self, rs, q, Gphase, s2_g):
        qF = HEG(rs).qF

        if self.xc == 'rALDA':

            rxalda_A = 0.25
            rxalda_qcut = qF * np.sqrt(1.0 / rxalda_A)

            fspinHxc_Gr = ((0.5 + 0.0j) *
                           (1.0 + np.sign(rxalda_qcut - q[:, np.newaxis])) *
                           (-1.0) * rxalda_A * (q[:, np.newaxis] / qF)**2.0)

        elif self.xc == 'rAPBE':

            fxc_PBE = self.get_PBE_fxc(pbe_rho=(3.0 / (4.0 * np.pi * rs**3.0)),
                                       pbe_s2_g=s2_g)
            rxapbe_qcut = np.sqrt(-4.0 * np.pi / fxc_PBE)

            fspinHxc_Gr = ((0.5 + 0.0j) *
                           (1.0 + np.sign(rxapbe_qcut - q[:, np.newaxis])) *
                           fxc_PBE / (4.0 * np.pi) * q[:, np.newaxis]**2.0)

        fspinHxc_Gr *= Gphase
        fspinHxc_GG = np.sum(fspinHxc_Gr, 1) / self.gridsize
        return fspinHxc_GG

    def get_PBE_fxc(self, pbe_rho, pbe_s2_g):
        pbe_kappa = 0.804
        pbe_mu = 0.2195149727645171

        pbe_denom_g = 1.0 + pbe_mu * pbe_s2_g / pbe_kappa

        F_g = 1.0 + pbe_kappa - pbe_kappa / pbe_denom_g
        Fn_g = -8.0 / 3.0 * pbe_mu * pbe_s2_g / pbe_rho / pbe_denom_g**2.0
        Fnn_g = (-11.0 / 3.0 / pbe_rho * Fn_g -
                 2.0 / pbe_kappa * Fn_g**2.0 * pbe_denom_g)

        e_g = -3.0 / 4.0 * (3.0 / np.pi)**(1.0 / 3.0) * pbe_rho**(4.0 / 3.0)
        v_g = 4.0 / 3.0 * e_g / pbe_rho
        f_g = 1.0 / 3.0 * v_g / pbe_rho

        pbe_f_g = f_g * F_g + 2.0 * v_g * Fn_g + e_g * Fnn_g
        return pbe_f_g

    def get_heg_A(self, rs):
        # Returns the A coefficient, where the
        # q ->0 limiting value of static fxc
        # of the HEG = -\frac{4\pi A }{q_F^2} = f_{xc}^{ALDA}.
        # We need correlation energy per electron and first and second derivs
        # w.r.t. rs
        # See for instance Moroni, Ceperley and Senatore,
        # Phys. Rev. Lett. 75, 689 (1995)
        # (and also Kohn and Sham, Phys. Rev. 140, A1133 (1965) equation 2.7)

        # Exchange contribution
        heg_A = 0.25

        # Correlation contribution
        A_ec, A_dec, A_d2ec = self.get_pw_lda(rs)
        heg_A += (1.0 / 27.0 * rs**2.0 * (9.0 * np.pi / 4.0)**(2.0 / 3.0) *
                  (2 * A_dec - rs * A_d2ec))

        return heg_A

    def get_heg_B(self, rs):
        # Returns the B coefficient, where the
        # q -> oo limiting behaviour of static fxc
        # of the HEG is -\frac{4\pi B}{q^2} - \frac{4\pi C}{q_F^2}.
        # Use the parametrisation of Moroni, Ceperley and Senatore,
        # Phys. Rev. Lett. 75, 689 (1995)

        mcs_xs = np.sqrt(rs)

        mcs_a = (1.0, 2.15, 0.0, 0.435)
        mcs_b = (3.0, 1.57, 0.0, 0.409)

        mcs_num = 0

        for mcs_j, mcs_coeff in enumerate(mcs_a):
            mcs_num += mcs_coeff * mcs_xs**mcs_j

        mcs_denom = 0

        for mcs_j, mcs_coeff in enumerate(mcs_b):
            mcs_denom += mcs_coeff * mcs_xs**mcs_j

        heg_B = mcs_num / mcs_denom
        return heg_B

    def get_heg_C(self, rs):
        # Returns the C coefficient, where the
        # q -> oo limiting behaviour of static fxc
        # of the HEG is -\frac{4\pi B}{q^2} - \frac{4\pi C}{q_F^2}.
        # Again see Moroni, Ceperley and Senatore,
        # Phys. Rev. Lett. 75, 689 (1995)

        C_ec, C_dec, Cd2ec = self.get_pw_lda(rs)

        heg_C = ((-1.0) * np.pi**(2.0 / 3.0) * (1.0 / 18.0)**(1.0 / 3.0) *
                 (rs * C_ec + rs**2.0 * C_dec))

        return heg_C

    def get_heg_D(self, rs):
        # Returns a 'D' coefficient, where the
        # q->0 omega -> oo limiting behaviour
        # of the frequency dependent fxc is -\frac{4\pi D}{q_F^2}
        # see Constantin & Pitarke Phys. Rev. B 75, 245127 (2007) equation 7

        D_ec, D_dec, D_d2ec = self.get_pw_lda(rs)

        # Exchange contribution
        heg_D = 0.15
        # Correlation contribution
        heg_D += ((9.0 * np.pi / 4.0)**(2.0 / 3.0) * rs / 3.0 *
                  (22.0 / 15.0 * D_ec + 26.0 / 15.0 * rs * D_dec))
        return heg_D

    def get_pw_lda(self, rs):
        # Returns LDA correlation energy and its first and second
        # derivatives with respect to rs according to the parametrisation
        # of Perdew and Wang, Phys. Rev. B 45, 13244 (1992)

        pw_A = 0.031091
        pw_alp = 0.21370
        pw_beta = (7.5957, 3.5876, 1.6382, 0.49294)

        pw_logdenom = 2.0 * pw_A * (
            pw_beta[0] * rs**0.5 + pw_beta[1] * rs**1.0 +
            pw_beta[2] * rs**1.5 + pw_beta[3] * rs**2.0)

        pw_dlogdenom = 2.0 * pw_A * (0.5 * pw_beta[0] * rs**(-0.5) +
                                     1.0 * pw_beta[1] +
                                     1.5 * pw_beta[2] * rs**0.5 +
                                     2.0 * pw_beta[3] * rs)

        pw_d2logdenom = 2.0 * pw_A * (-0.25 * pw_beta[0] * rs**(-1.5) +
                                      0.75 * pw_beta[2] * rs**(-0.5) +
                                      2.0 * pw_beta[3])

        pw_logarg = 1.0 + 1.0 / pw_logdenom
        pw_dlogarg = (-1.0) / (pw_logdenom**2.0) * pw_dlogdenom
        pw_d2logarg = 2.0 / (pw_logdenom**3.0) * (pw_dlogdenom**2.0)
        pw_d2logarg += (-1.0) / (pw_logdenom**2.0) * pw_d2logdenom

        # pw_ec = the correlation energy (per electron)
        pw_ec = -2.0 * pw_A * (1 + pw_alp * rs) * np.log(pw_logarg)

        # pw_dec = first derivative

        pw_dec = -2.0 * pw_A * (1 + pw_alp * rs) * pw_dlogarg / pw_logarg
        pw_dec += (-2.0 * pw_A * pw_alp) * np.log(pw_logarg)

        # pw_d2ec = second derivative

        pw_d2ec = (-2.0) * pw_A * pw_alp * pw_dlogarg / pw_logarg
        pw_d2ec += (-2.0) * pw_A * ((1 + pw_alp * rs) *
                                    (pw_d2logarg / pw_logarg -
                                     (pw_dlogarg**2.0) / (pw_logarg**2.0)))
        pw_d2ec += (-2.0 * pw_A * pw_alp) * pw_dlogarg / pw_logarg

        return pw_ec, pw_dec, pw_d2ec


class range_separated:
    def __init__(self, gs, context, frequencies, freqweights, l_l, lweights,
                 range_rc, xc):

        self.gs = gs

        self.context = context
        self.frequencies = frequencies
        self.freqweights = freqweights
        self.l_l = l_l
        self.lweights = lweights
        self.range_rc = range_rc
        self.xc = xc

        self.cutoff_rs = 36.278317

        if self.xc == 'range_RPA':
            self.context.print(
                'Using range-separated RPA approach, with parameter %s Bohr' %
                self.range_rc, flush=False)

        nval_g = self.gs.hacky_all_electron_density(
            gridrefinement=4, skip_core=True).flatten()
        self.dv = self.gs.density.gd.dv / 64.0  # 64 = gridrefinement^3

        density_cut = 3.0 / (4.0 * np.pi * self.cutoff_rs**3.0)
        if (nval_g < 0.0).any():
            self.context.print('Warning, negative densities found! ('
                               'Magnitude %s)' % np.abs(np.amin(nval_g)),
                               flush=False)
            self.context.print('These will be ignored', flush=False)
        if (nval_g < density_cut).any():
            nval_g = nval_g[np.where(nval_g > density_cut)]
            self.context.print(
                'Not calculating correlation energy ',
                'contribution for densities < %3.2e elecs/Bohr ^ 3'
                % density_cut, flush=False)

        densitysum = np.sum(nval_g * self.dv)
        # XXX probably wrong for charged systems
        valence = self.gs.setups.nvalence

        self.context.print('Density integrates to %s electrons' % densitysum,
                           flush=False)

        self.context.print('Renormalized to %s electrons' % valence)

        nval_g *= valence / densitysum
        self.rs_g = (3.0 / (4.0 * np.pi * nval_g))**(1.0 / 3.0)

        self.rsmin = np.amin(self.rs_g)
        self.rsmax = np.amax(self.rs_g)

    def calculate(self):

        self.context.print('Generating tables of electron gas energies...',
                           flush=False)

        table_SR = self.generate_tables()

        self.context.print('...done', flush=False)
        # Now interpolate the table to calculate local density terms
        E_SR = np.sum(np.interp(self.rs_g, table_SR[:, 0],
                                table_SR[:, 1])) * self.dv

        # RPA energy minus long range correlation
        self.context.print('Short range correlation energy/unit cell = '
                           '%5.4f eV \n' % (E_SR * Ha))
        return E_SR

    def generate_tables(self):

        # Finite difference steps for density and k vec
        rs_step = 0.01
        k_step = 0.01

        rs_r = np.arange(self.rsmin - rs_step, self.rsmax + rs_step, rs_step)

        table_SR = np.empty((len(rs_r), 2))
        table_SR[:, 0] = rs_r
        for iR, Rs in enumerate(rs_r):

            qF = HEG(Rs).qF

            q_k = np.arange(k_step, 10.0 * qF, k_step)

            if self.xc == 'range_RPA':
                # Correlation energy per electron, in Hartree, per k
                Eeff_k, Erpa_k = self.RPA_corr_hole(q_k, Rs)
                ESR_k = Erpa_k - Eeff_k
            elif self.xc == 'range_rALDA':
                ESR_k = self.rALDA_corr_hole(q_k, Rs)

            # Integrate over k
            table_SR[iR, 1] = k_step * np.sum(ESR_k)

        return table_SR

    def RPA_corr_hole(self, q, rs):

        # Integrating this quantity over q, gives
        # correlation energy per unit volume
        # calcuated with a Coulomb-like effective
        # interaction, in Hartree
        # = 1/(2\pi) * \sum_{\vec{q}} \int_0^infty ds
        #   * [ ln (1 - v_eff \chi_0) + v_eff \chi_0]
        # = 1/(2\pi) * \int 4 \pi q^2 dq /((2\pi)^3)
        #   * \int_0^infty ds [ ln (1 - v_eff \chi_0) + v_eff \chi_0]
        # = 1/(4\pi^3) * \int q^2 dq  \int_0^infty ds
        #   * [ ln (1 - v_eff \chi_0) + v_eff \chi_0]

        veff = 4.0 * np.pi / (q * q) * np.exp(
            -0.25 * q * q * self.range_rc * self.range_rc)
        vc = 4.0 * np.pi / (q * q)

        # Do the integral over frequency using Gauss-Legendre

        eeff_q = np.zeros(len(q))
        erpa_q = np.zeros(len(q))

        for u, freqweight in zip(self.frequencies, self.freqweights):
            chi0 = HEG(rs).lindhard_function(q, u)

            eff_integrand = np.log(np.ones(len(q)) - veff * chi0) + veff * chi0
            eeff_q += eff_integrand * freqweight

            rpa_integrand = np.log(np.ones(len(q)) - vc * chi0) + vc * chi0
            erpa_q += rpa_integrand * freqweight

        # Per unit volume

        eeff_q *= 1.0 / (4.0 * np.pi**3.0) * q * q
        erpa_q *= 1.0 / (4.0 * np.pi**3.0) * q * q

        return (eeff_q, erpa_q)

    def rALDA_corr_hole(self, q, rs):
        qF = HEG(rs).qF

        veff = 4.0 * np.pi / (q * q) * ((1.0 - 0.25 * q * q /
                                         (qF * qF)) * 0.5 *
                                        (1.0 + np.sign(2.0 * qF - q)))
        fxc = veff - 4.0 * np.pi / (q * q)

        esr_q = np.zeros(len(q))

        for u, freqweight in zip(self.frequencies, self.freqweights):
            chi0 = HEG(rs).lindhard_function(q, u)
            esr_u = np.zeros(len(q))

            for l, lweight in zip(self.l_l, self.lweights):

                chil = chi0 / (1.0 - l * fxc * chi0)
                esr_u += lweight * (-fxc) * (chil - chi0)

            esr_q += freqweight * esr_u

        esr_q *= 1.0 / (4.0 * np.pi**3.0) * q * q
        return esr_q


class KernelDens:
    def __init__(self, gs, xc, ibzq_qc, unit_cells, density_cut, ecut,
                 tag, context):

        self.gs = gs
        self.gd = self.gs.density.gd
        self.xc = xc
        self.ibzq_qc = ibzq_qc
        self.unit_cells = unit_cells
        self.density_cut = density_cut
        self.ecut = ecut
        self.tag = tag
        self.context = context

        self.A_x = -(3 / 4.) * (3 / np.pi)**(1 / 3.)

        self.n_g = self.gs.hacky_all_electron_density(gridrefinement=1)

        if xc[-3:] == 'PBE':
            nf_g = self.gs.hacky_all_electron_density(gridrefinement=2)
            gdf = self.gd.refine()
            grad_v = [Gradient(gdf, v, n=1).apply for v in range(3)]
            gradnf_vg = gdf.empty(3)
            for v in range(3):
                grad_v[v](nf_g, gradnf_vg[v])
            self.gradn_vg = gradnf_vg[:, ::2, ::2, ::2]

        qd = KPointDescriptor(self.ibzq_qc)
        self.pd = PWDescriptor(ecut / Ha, self.gd, complex, qd)

    @timer('FHXC')
    def calculate_fhxc(self):

        self.context.print('Calculating %s kernel at %d eV cutoff' % (
            self.xc, self.ecut))
        if self.xc[0] == 'r':
            self.calculate_rkernel()
        else:
            assert self.xc[0] == 'A'
            self.calculate_local_kernel()

    def calculate_rkernel(self):

        gd = self.gd
        ng_c = gd.N_c
        cell_cv = gd.cell_cv
        icell_cv = 2 * np.pi * np.linalg.inv(cell_cv)
        vol = gd.volume

        ns = self.gs.nspins
        n_g = self.n_g  # density on rough grid

        fx_g = ns * self.get_fxc_g(n_g)  # local exchange kernel
        qc_g = (-4 * np.pi * ns / fx_g)**0.5  # cutoff functional
        flocal_g = qc_g**3 * fx_g / (6 * np.pi**2)  # ren. x-kernel for r=r'
        Vlocal_g = 2 * qc_g / np.pi  # ren. Hartree kernel for r=r'

        ng = np.prod(ng_c)  # number of grid points
        r_vg = gd.get_grid_point_coordinates()
        rx_g = r_vg[0].flatten()
        ry_g = r_vg[1].flatten()
        rz_g = r_vg[2].flatten()

        self.context.print('    %d grid points and %d plane waves at the '
                           'Gamma point' % (ng, self.pd.ngmax), flush=False)

        # Unit cells
        R_Rv = []
        weight_R = []
        nR_v = self.unit_cells
        nR = np.prod(nR_v)
        for i in range(-nR_v[0] + 1, nR_v[0]):
            for j in range(-nR_v[1] + 1, nR_v[1]):
                for h in range(-nR_v[2] + 1, nR_v[2]):
                    R_Rv.append(i * cell_cv[0] + j * cell_cv[1] +
                                h * cell_cv[2])
                    weight_R.append((nR_v[0] - abs(i)) * (nR_v[1] - abs(j)) *
                                    (nR_v[2] - abs(h)) / float(nR))
        if nR > 1:
            # with more than one unit cell only the exchange kernel is
            # calculated on the grid. The bare Coulomb kernel is added
            # in PW basis and Vlocal_g only the exchange part
            dv = self.gs.density.gd.dv
            gc = (3 * dv / 4 / np.pi)**(1 / 3.)
            Vlocal_g -= 2 * np.pi * gc**2 / dv
            self.context.print(
                '    Lattice point sampling: (%s x %s x %s)^2 '
                % (nR_v[0], nR_v[1], nR_v[2]) + ' Reduced to %s lattice points'
                % len(R_Rv), flush=False)

        l_g_size = -(-ng // mpi.world.size)
        l_g_range = range(mpi.world.rank * l_g_size,
                          min((mpi.world.rank + 1) * l_g_size, ng))

        fhxc_qsGr = {}
        for iq in range(len(self.ibzq_qc)):
            fhxc_qsGr[iq] = np.zeros(
                (ns, len(self.pd.G2_qG[iq]), len(l_g_range)), dtype=complex)

        inv_error = np.seterr()
        np.seterr(invalid='ignore')
        np.seterr(divide='ignore')

        t0 = time()
        # Loop over Lattice points
        for i, R_v in enumerate(R_Rv):
            # Loop over r'. f_rr and V_rr are functions of r (dim. as r_vg[0])
            if i == 1:
                self.context.print(
                    '      Finished 1 cell in %s seconds' % int(time() - t0) +
                    ' - estimated %s seconds left' % int((len(R_Rv) - 1) *
                                                         (time() - t0)))
            if len(R_Rv) > 5:
                if (i + 1) % (len(R_Rv) / 5 + 1) == 0:
                    self.context.print(
                        '      Finished %s cells in %s seconds'
                        % (i, int(time() - t0)) + ' - estimated '
                        '%s seconds left' % int((len(R_Rv) - i) * (time() -
                                                                   t0) / i))
            for g in l_g_range:
                rx = rx_g[g] + R_v[0]
                ry = ry_g[g] + R_v[1]
                rz = rz_g[g] + R_v[2]

                # |r-r'-R_i|
                rr = ((r_vg[0] - rx)**2 + (r_vg[1] - ry)**2 +
                      (r_vg[2] - rz)**2)**0.5

                n_av = (n_g + n_g.flatten()[g]) / 2.
                fx_g = ns * self.get_fxc_g(n_av, index=g)
                qc_g = (-4 * np.pi * ns / fx_g)**0.5
                x = qc_g * rr
                osc_x = np.sin(x) - x * np.cos(x)
                f_rr = fx_g * osc_x / (2 * np.pi**2 * rr**3)
                if nR > 1:  # include only exchange part of the kernel here
                    V_rr = (sici(x)[0] * 2 / np.pi - 1) / rr
                else:  # include the full kernel (also hartree part)
                    V_rr = (sici(x)[0] * 2 / np.pi) / rr

                # Terms with r = r'
                if (np.abs(R_v) < 0.001).all():
                    tmp_flat = f_rr.flatten()
                    tmp_flat[g] = flocal_g.flatten()[g]
                    f_rr = tmp_flat.reshape(ng_c)
                    tmp_flat = V_rr.flatten()
                    tmp_flat[g] = Vlocal_g.flatten()[g]
                    V_rr = tmp_flat.reshape(ng_c)
                    del tmp_flat

                f_rr[np.where(n_av < self.density_cut)] = 0.0
                V_rr[np.where(n_av < self.density_cut)] = 0.0

                f_rr *= weight_R[i]
                V_rr *= weight_R[i]

                # r-r'-R_i
                r_r = np.array([r_vg[0] - rx, r_vg[1] - ry, r_vg[2] - rz])

                # Fourier transform of r
                for iq, q in enumerate(self.ibzq_qc):
                    q_v = np.dot(q, icell_cv)
                    e_q = np.exp(-1j * gemmdot(q_v, r_r, beta=0.0))
                    f_q = self.pd.fft((f_rr + V_rr) * e_q, iq) * vol / ng
                    fhxc_qsGr[iq][0, :, g - l_g_range[0]] += f_q
                    if ns == 2:
                        f_q = self.pd.fft(V_rr * e_q, iq) * vol / ng
                        fhxc_qsGr[iq][1, :, g - l_g_range[0]] += f_q

        mpi.world.barrier()

        np.seterr(**inv_error)

        for iq, q in enumerate(self.ibzq_qc):
            npw = len(self.pd.G2_qG[iq])
            fhxc_sGsG = np.zeros((ns * npw, ns * npw), complex)
            l_pw_size = -(-npw // mpi.world.size)  # parallelize over PW below
            l_pw_range = range(mpi.world.rank * l_pw_size,
                               min((mpi.world.rank + 1) * l_pw_size, npw))

            if mpi.world.size > 1:
                # redistribute grid and plane waves in fhxc_qsGr[iq]
                bg1 = BlacsGrid(mpi.world, 1, mpi.world.size)
                bg2 = BlacsGrid(mpi.world, mpi.world.size, 1)
                bd1 = bg1.new_descriptor(npw, ng, npw,
                                         -(-ng // mpi.world.size))
                bd2 = bg2.new_descriptor(npw, ng, -(-npw // mpi.world.size),
                                         ng)

                fhxc_Glr = np.zeros((len(l_pw_range), ng), dtype=complex)
                if ns == 2:
                    Koff_Glr = np.zeros((len(l_pw_range), ng), dtype=complex)

                r = Redistributor(bg1.comm, bd1, bd2)
                r.redistribute(fhxc_qsGr[iq][0], fhxc_Glr, npw, ng)
                if ns == 2:
                    r.redistribute(fhxc_qsGr[iq][1], Koff_Glr, npw, ng)
            else:
                fhxc_Glr = fhxc_qsGr[iq][0]
                if ns == 2:
                    Koff_Glr = fhxc_qsGr[iq][1]

            # Fourier transform of r'
            for iG in range(len(l_pw_range)):
                f_g = fhxc_Glr[iG].reshape(ng_c)
                f_G = self.pd.fft(f_g.conj(), iq) * vol / ng
                fhxc_sGsG[l_pw_range[0] + iG, :npw] = f_G.conj()
                if ns == 2:
                    v_g = Koff_Glr[iG].reshape(ng_c)
                    v_G = self.pd.fft(v_g.conj(), iq) * vol / ng
                    fhxc_sGsG[npw + l_pw_range[0] + iG, :npw] = v_G.conj()

            if ns == 2:  # f_00 = f_11 and f_01 = f_10
                fhxc_sGsG[:npw, npw:] = fhxc_sGsG[npw:, :npw]
                fhxc_sGsG[npw:, npw:] = fhxc_sGsG[:npw, :npw]

            mpi.world.sum(fhxc_sGsG)
            fhxc_sGsG /= vol

            if mpi.rank == 0:
                w = ulm.open(
                    'fhxc_%s_%s_%s_%s.ulm' %
                    (self.tag, self.xc, self.ecut, iq), 'w')
                if nR > 1:  # add Hartree kernel evaluated in PW basis
                    Gq2_G = self.pd.G2_qG[iq]
                    if (q == 0).all():
                        Gq2_G = Gq2_G.copy()
                        Gq2_G[0] = 1.
                    vq_G = 4 * np.pi / Gq2_G
                    fhxc_sGsG += np.tile(np.eye(npw) * vq_G, (ns, ns))
                w.write(fhxc_sGsG=fhxc_sGsG)
                w.close()
            mpi.world.barrier()
        self.context.print('')

    def calculate_local_kernel(self):
        # Standard ALDA exchange kernel
        # Use with care. Results are very difficult to converge
        # Sensitive to density_cut
        ns = self.gs.nspins
        gd = self.gd
        pd = self.pd
        cell_cv = gd.cell_cv
        icell_cv = 2 * np.pi * np.linalg.inv(cell_cv)
        vol = gd.volume

        fxc_sg = ns * self.get_fxc_g(ns * self.n_g)
        fxc_sg[np.where(self.n_g < self.density_cut)] = 0.0

        r_vg = gd.get_grid_point_coordinates()

        for iq in range(len(self.ibzq_qc)):
            Gvec_Gc = np.dot(pd.get_reciprocal_vectors(q=iq, add_q=False),
                             cell_cv / (2 * np.pi))
            npw = len(Gvec_Gc)
            l_pw_size = -(-npw // mpi.world.size)
            l_pw_range = range(mpi.world.rank * l_pw_size,
                               min((mpi.world.rank + 1) * l_pw_size, npw))
            fhxc_sGsG = np.zeros((ns * npw, ns * npw), dtype=complex)
            for s in range(ns):
                for iG in l_pw_range:
                    for jG in range(npw):
                        fxc = fxc_sg[s].copy()
                        dG_c = Gvec_Gc[iG] - Gvec_Gc[jG]
                        dG_v = np.dot(dG_c, icell_cv)
                        dGr_g = gemmdot(dG_v, r_vg, beta=0.0)
                        ft_fxc = gd.integrate(np.exp(-1j * dGr_g) * fxc)
                        fhxc_sGsG[s * npw + iG, s * npw + jG] = ft_fxc

            mpi.world.sum(fhxc_sGsG)
            fhxc_sGsG /= vol

            Gq2_G = self.pd.G2_qG[iq]
            if (self.ibzq_qc[iq] == 0).all():
                Gq2_G[0] = 1.
            vq_G = 4 * np.pi / Gq2_G
            fhxc_sGsG += np.tile(np.eye(npw) * vq_G, (ns, ns))

            if mpi.rank == 0:
                w = ulm.open(
                    'fhxc_%s_%s_%s_%s.ulm' %
                    (self.tag, self.xc, self.ecut, iq), 'w')
                w.write(fhxc_sGsG=fhxc_sGsG)
                w.close()
            mpi.world.barrier()
        self.context.print('')

    def get_fxc_g(self, n_g, index=None):
        if self.xc[-3:] == 'LDA':
            return self.get_lda_g(n_g)
        elif self.xc[-3:] == 'PBE':
            return self.get_pbe_g(n_g, index=index)
        else:
            raise '%s kernel not recognized' % self.xc

    def get_lda_g(self, n_g):
        return (4. / 9.) * self.A_x * n_g**(-2. / 3.)

    def get_pbe_g(self, n_g, index=None):
        if index is None:
            gradn_vg = self.gradn_vg
        else:
            gradn_vg = self.gs.density.gd.empty(3)
            for v in range(3):
                gradn_vg[v] = (self.gradn_vg[v] +
                               self.gradn_vg[v].flatten()[index]) / 2

        kf_g = (3. * np.pi**2 * n_g)**(1 / 3.)
        s2_g = np.zeros_like(n_g)
        for v in range(3):
            axpy(1.0, gradn_vg[v]**2, s2_g)
        s2_g /= 4 * kf_g**2 * n_g**2

        e_g = self.A_x * n_g**(4 / 3.)
        v_g = (4 / 3.) * e_g / n_g
        f_g = (1 / 3.) * v_g / n_g

        kappa = 0.804
        mu = 0.2195149727645171

        denom_g = (1 + mu * s2_g / kappa)
        F_g = 1. + kappa - kappa / denom_g
        Fn_g = -mu / denom_g**2 * 8 * s2_g / (3 * n_g)
        Fnn_g = -11 * Fn_g / (3 * n_g) - 2 * Fn_g**2 / kappa

        fxc_g = f_g * F_g
        fxc_g += 2 * v_g * Fn_g
        fxc_g += e_g * Fnn_g
        return fxc_g


class XCFlags:
    _accepted_flags = {
        'RPA',
        'range_RPA',  # range separated RPA a la Bruneval
        'rALDA',  # renormalized kernels
        'rAPBE',
        'range_rALDA',
        'rALDAns',  # no spin (ns)
        'rAPBEns',
        'rALDAc',  # rALDA + correlation
        'CP',  # Constantin Pitarke
        'CP_dyn',  # Dynamical form of CP
        'CDOP',  # Corradini et al
        'CDOPs',  # CDOP without local term
        'JGMs',  # simplified jellium-with-gap kernel
        'JGMsx',  # simplified jellium-with-gap kernel,
        # constructed with exchange part only
        # so that it scales linearly with l
        'ALDA'}  # standard ALDA

    _spin_kernels = {'rALDA', 'rAPBE', 'ALDA'}

    _linear_kernels = {'rALDAns', 'rAPBEns', 'range_RPA', 'JGMsx', 'RPA',
                       'rALDA', 'rAPBE', 'range_rALDA', 'ALDA'}

    _gapped = {'JGMs', 'JGMsx'}

    def __init__(self, xc):
        if xc not in self._accepted_flags:
            raise RuntimeError('%s kernel not recognized' % self.xc)

        self.xc = xc

    @property
    def spin_kernel(self):
        # rALDA/rAPBE are the only kernels which have spin-dependent forms
        return self.xc in self._spin_kernels

    @property
    def linear_kernel(self):
        # Scales linearly with coupling constant
        return self.xc in self._linear_kernels

    @property
    def dyn_kernel(self):
        return self.xc == 'CP_dyn'

    @property
    def bandgap_dependent(self):
        return self.xc in self._gapped

    @property
    def is_ranged(self):
        return self.xc in {'range_RPA', 'range_rALDA'}

    @property
    def is_apbe(self):
        # If new GGA kernels are added, maybe there should be an
        # is_gga property.
        return self.xc in {'rAPBE', 'rAPBEns'}

    def choose_avg_scheme(self, avg_scheme=None):
        xc = self.xc

        if self.spin_kernel:
            if avg_scheme is None:
                avg_scheme = 'density'
                # Two-point scheme default for rALDA and rAPBE

        if avg_scheme == 'density':
            assert self.spin_kernel, ('Two-point density average '
                                      'only implemented for rALDA and rAPBE')

        elif xc not in ('RPA', 'range_RPA'):
            avg_scheme = 'wavevector'
        else:
            avg_scheme = None

        return avg_scheme

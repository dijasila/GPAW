# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""
See Kresse, Phys. Rev. B 54, 11169 (1996)
"""

import numpy as np
from numpy.fft import fftn, ifftn

import gpaw.mpi as mpi
from gpaw.utilities.blas import axpy
from gpaw.fd_operators import FDOperator
from gpaw.utilities.tools import construct_reciprocal

"""About mixing-related classes.

(FFT/Broyden)BaseMixer: These classes know how to mix one density
array and store history etc.  But they do not take care of complexity
like spin.

(SpinSum/etc.)MixerDriver: These combine one or more BaseMixers to
implement a full algorithm.  Think of them as stateless (immutable).
The user can give an object of these types as input, but they will generally
be constructed by a utility function so the interface is nice.

The density object always wraps the (X)MixerDriver with a
MixerWrapper.  The wrapper contains the common code for all mixers so
we don't have to implement it multiple times (estimate memory, etc.).

In the end, what the user provides is probably a dictionary anyway, and the
relevant objects are instantiated automatically."""


class BaseMixer:
    name = 'pulay'

    """Pulay density mixer."""
    def __init__(self, beta, nmaxold, weight):
        """Construct density-mixer object.

        Parameters:

        beta: float
            Mixing parameter between zero and one (one is most
            aggressive).
        nmaxold: int
            Maximum number of old densities.
        weight: float
            Weight parameter for special metric (for long wave-length
            changes).

        """

        self.beta = beta
        self.nmaxold = nmaxold
        self.weight = weight

    def initialize_metric(self, gd):
        self.gd = gd

        if self.weight == 1:
            self.metric = None

        else:
            a = 0.125 * (self.weight + 7)
            b = 0.0625 * (self.weight - 1)
            c = 0.03125 * (self.weight - 1)
            d = 0.015625 * (self.weight - 1)
            self.metric = FDOperator([a,
                                      b, b, b, b, b, b,
                                      c, c, c, c, c, c, c, c, c, c, c, c,
                                      d, d, d, d, d, d, d, d],
                                     [(0, 0, 0),  # a
                                      (-1, 0, 0), (1, 0, 0),  # b
                                      (0, -1, 0), (0, 1, 0),
                                      (0, 0, -1), (0, 0, 1),
                                      (1, 1, 0), (1, 0, 1), (0, 1, 1),  # c
                                      (1, -1, 0), (1, 0, -1), (0, 1, -1),
                                      (-1, 1, 0), (-1, 0, 1), (0, -1, 1),
                                      (-1, -1, 0), (-1, 0, -1), (0, -1, -1),
                                      (1, 1, 1), (1, 1, -1), (1, -1, 1),  # d
                                      (-1, 1, 1), (1, -1, -1), (-1, -1, 1),
                                      (-1, 1, -1), (-1, -1, -1)],
                                     gd, float).apply
            self.mR_G = gd.empty()
            self.mR_sG = [gd.empty(), gd.empty(), gd.empty(), gd.empty()]

    def reset(self):
        """Reset Density-history.

        Called at initialization and after each move of the atoms.

        my_nuclei:   All nuclei in local domain.
        """

        # History for Pulay mixing of densities:
        self.nt_iG = []  # Pseudo-electron densities
        self.R_iG = []  # Residuals
        self.A_ii = np.zeros((0, 0))

        self.D_iap = []
        self.dD_iap = []

    def calculate_charge_sloshing(self, R_G):
        return self.gd.integrate(np.fabs(R_G))

    def mix_single_density(self, nt_G, D_ap, blas=True):
        iold = len(self.nt_iG)

        dNt = np.inf
        if iold > 0:
            if iold > self.nmaxold:
                # Throw away too old stuff:
                del self.nt_iG[0]
                del self.R_iG[0]
                del self.D_iap[0]
                del self.dD_iap[0]
                # for D_p, D_ip, dD_ip in self.D_a:
                #     del D_ip[0]
                #     del dD_ip[0]
                iold = self.nmaxold

            # Calculate new residual (difference between input and
            # output density):
            R_G = nt_G - self.nt_iG[-1]
            dNt = self.calculate_charge_sloshing(R_G)
            self.R_iG.append(R_G)
            self.dD_iap.append([])
            for D_p, D_ip in zip(D_ap, self.D_iap[-1]):
                self.dD_iap[-1].append(D_p - D_ip)

            # Update matrix:
            A_ii = np.zeros((iold, iold))
            i2 = iold - 1

            if self.metric is None:
                mR_G = R_G
            else:
                mR_G = self.mR_G
                self.metric(R_G, mR_G)

            for i1, R_1G in enumerate(self.R_iG):
                a = self.gd.comm.sum(self.dotprod(R_1G, mR_G, self.dD_iap[i1],
                                                  self.dD_iap[-1]))
                A_ii[i1, i2] = a
                A_ii[i2, i1] = a
            A_ii[:i2, :i2] = self.A_ii[-i2:, -i2:]
            self.A_ii = A_ii
            try:
                B_ii = np.linalg.inv(A_ii)
            except np.linalg.LinAlgError:
                alpha_i = np.zeros(iold)
                alpha_i[-1] = 1.0
            else:
                alpha_i = B_ii.sum(1)
                try:
                    # Normalize:
                    alpha_i /= alpha_i.sum()
                except ZeroDivisionError:
                    alpha_i[:] = 0.0
                    alpha_i[-1] = 1.0

            # Calculate new input density:
            nt_G[:] = 0.0
            # for D_p, D_ip, dD_ip in self.D_a:
            for D in D_ap:
                D[:] = 0.0
            beta = self.beta
            for i, alpha in enumerate(alpha_i):
                if blas:
                    axpy(alpha, self.nt_iG[i], nt_G)
                    axpy(alpha * beta, self.R_iG[i], nt_G)
                else:
                    nt_G[:] = alpha * self.nt_iG[i] + nt_G
                    nt_G[:] = alpha * beta * self.R_iG[i] + nt_G
                for D_p, D_ip, dD_ip in zip(D_ap, self.D_iap[i],
                                            self.dD_iap[i]):
                    if blas:
                        axpy(alpha, D_ip, D_p)
                        axpy(alpha * beta, dD_ip, D_p)
                    else:
                        D_p[:] = alpha * D_ip + D_p
                        D_p[:] = alpha *beta * dD_ip + D_p

        # Store new input density (and new atomic density matrices):
        self.nt_iG.append(nt_G.copy())
        self.D_iap.append([])
        for D_p in D_ap:
            self.D_iap[-1].append(D_p.copy())
        return dNt

    def mix_density(self, nt_sG, D_asp, g_ss=None, blas=True):
        nt_isG = self.nt_iG
        R_isG = self.R_iG
        D_iasp = self.D_iap
        dD_iasp = self.dD_iap
        spin = len(nt_sG)
        iold = len(self.nt_iG)
        dNt = [np.inf]*spin
        if iold > 0:
            if iold > self.nmaxold:
                # Throw away too old stuff:
                del nt_isG[0]
                del R_isG[0]
                del D_iasp[0]
                del dD_iasp[0]
                # for D_p, D_ip, dD_ip in self.D_a:
                #     del D_ip[0]
                #     del dD_ip[0]
                iold = self.nmaxold

            # Calculate new residual (difference between input and
            # output density):
            R_sG = nt_sG - nt_isG[-1]
            dNt = self.calculate_charge_sloshing(R_sG)
            R_isG.append(R_sG)
            dD_iasp.append([])
            for D_sp, D_isp in zip(D_asp, D_iasp[-1]):
                dD_iasp[-1].append(D_sp - D_isp)            

            if self.metric is None:
                mR_sG = R_sG
            else:
                mR_sG = self.mR_sG[:spin]
                for s in range(spin):
                    self.metric(R_sG[s], mR_sG[s])
            mR_sG = np.tensordot(g_ss, mR_sG, axes=(1,0))

            # Update matrix:
            A_ii = np.zeros((iold, iold))
            i2 = iold - 1
            
            for i1, R_1sG in enumerate(R_isG):
                a = self.gd.comm.sum(self.dotprod(R_1sG, mR_sG, dD_iasp[i1],
                                                  dD_iasp[-1]))
                A_ii[i1, i2] = a
                A_ii[i2, i1] = a
            A_ii[:i2, :i2] = self.A_ii[-i2:, -i2:]
            self.A_ii = A_ii

            try:
                B_ii = np.linalg.pinv(A_ii)
            except np.linalg.LinAlgError:
                alpha_i = np.zeros(iold)
                alpha_i[-1] = 1.0
            else:
                alpha_i = B_ii.sum(1)
                try:
                    # Normalize:
                    alpha_i /= alpha_i.sum()
                except ZeroDivisionError:
                    alpha_i[:] = 0.0
                    alpha_i[-1] = 1.0

            # Calculate new input density:
            nt_sG[:] = 0.0
            # for D_p, D_ip, dD_ip in self.D_a:
            for D in D_asp:
                D[:] = 0.0
            beta = self.beta
            for i, alpha in enumerate(alpha_i):
                if blas:
                    axpy(alpha, nt_isG[i], nt_sG)
                    axpy(alpha * beta, R_isG[i], nt_sG)
                else:
                    nt_sG[:] += alpha * nt_isG[i]
                    nt_sG[:] += alpha * beta * R_isG[i]

                for D_sp, D_isp, dD_isp in zip(D_asp, D_iasp[i],
                                               dD_iasp[i]):
                    if blas:
                        axpy(alpha, D_isp, D_sp)
                        axpy(alpha * beta, dD_isp, D_sp)
                    else:
                        D_sp[:] += alpha * D_isp
                        D_sp[:] += alpha *beta * dD_isp
 
        # Store new input density (and new atomic density matrices):
        nt_isG.append(nt_sG.copy())
        D_iasp.append([])
        for D_sp in D_asp:
            D_iasp[-1].append(D_sp.copy())
        return dNt


    # may presently be overridden by passing argument in constructor
    def dotprod(self, R1_G, R2_G, dD1_ap, dD2_ap):
        return np.vdot(R1_G, R2_G).real

    def estimate_memory(self, mem, gd):
        gridbytes = gd.bytecount()
        mem.subnode('nt_iG, R_iG', 2 * self.nmaxold * gridbytes)

    def __repr__(self):
        classname = self.__class__.__name__
        template = '%s(beta=%f, nmaxold=%d, weight=%f)'
        string = template % (classname, self.beta, self.nmaxold, self.weight)
        return string


class ExperimentalDotProd:
    def __init__(self, calc):
        self.calc = calc

    def __call__(self, R1_G, R2_G, dD1_ap, dD2_ap):
        prod = np.vdot(R1_G, R2_G).real
        setups = self.calc.wfs.setups
        # okay, this is a bit nasty because it depends on dD1_ap
        # and its friend having come from D_asp.values() and the dictionaries
        # not having been modified.  This is probably true... for now.
        avalues = self.calc.density.D_asp.keys()
        for a, dD1_p, dD2_p in zip(avalues, dD1_ap, dD2_ap):
            I4_pp = setups[a].four_phi_integrals()
            dD4_pp = np.outer(dD1_p, dD2_p)  # not sure if corresponds quite
            prod += (I4_pp * dD4_pp).sum()
        return prod


class ReciprocalMetric:
    def __init__(self, weight, k2_Q):
        self.k2_Q = k2_Q
        k2_min = np.min(self.k2_Q)
        self.q1 = (weight - 1) * k2_min

    def __call__(self, R_Q, mR_Q):
        mR_Q[:] = R_Q * (1.0 + self.q1 / self.k2_Q)


class FFTBaseMixer(BaseMixer):
    name = 'fft'

    """Mix the density in Fourier space"""
    def __init__(self, beta, nmaxold, weight):
        BaseMixer.__init__(self, beta, nmaxold, weight)
        self.gd1 = None

    def initialize_metric(self, gd):
        self.gd = gd

        if gd.comm.rank == 0:
            self.gd1 = gd.new_descriptor(comm=mpi.serial_comm)
            k2_Q, _ = construct_reciprocal(self.gd1)
            self.metric = ReciprocalMetric(self.weight, k2_Q)
            self.mR_G = self.gd1.empty(dtype=complex)
        else:
            self.metric = lambda R_Q, mR_Q: None
            self.mR_G = np.empty((0, 0, 0), dtype=complex)

    def calculate_charge_sloshing(self, R_Q):
        if self.gd.comm.rank == 0:
            cs = self.gd1.integrate(np.fabs(ifftn(R_Q).real))
        else:
            cs = 0.0
        return self.gd.comm.sum(cs)

    def mix_single_density(self, nt_G, D_ap):
        # Transform real-space density to Fourier space
        nt1_G = self.gd.collect(nt_G)
        if self.gd.comm.rank == 0:
            nt_Q = np.ascontiguousarray(fftn(nt1_G))
        else:
            nt_Q = np.empty((0, 0, 0), dtype=complex)

        dNt = BaseMixer.mix_single_density(self, nt_Q, D_ap)

        # Return density in real space
        if self.gd.comm.rank == 0:
            nt1_G = ifftn(nt_Q).real
        self.gd.distribute(nt1_G, nt_G)

        return dNt


class BroydenBaseMixer:
    name = 'broyden'

    def __init__(self, beta, nmaxold, weight):
        self.verbose = False
        self.beta = beta
        self.nmaxold = nmaxold
        self.weight = 1.0  # XXX discards argument

    def initialize_metric(self, gd):
        self.gd = gd

    def reset(self):
        self.step = 0
        # self.d_nt_G = []
        # self.d_D_ap = []

        self.R_iG = []
        self.dD_iap = []

        self.nt_iG = []
        self.D_iap = []
        self.c_G = []
        self.v_G = []
        self.u_G = []
        self.u_D = []

    def mix_single_density(self, nt_G, D_ap):
        dNt = np.inf
        if self.step > 2:
            del self.R_iG[0]
            for d_Dp in self.dD_iap:
                del d_Dp[0]
        if self.step > 0:
            self.R_iG.append(nt_G - self.nt_iG[-1])
            for d_Dp, D_p, D_ip in zip(self.dD_iap, D_ap, self.D_iap):
                d_Dp.append(D_p - D_ip[-1])
            fmin_G = self.gd.integrate(self.R_iG[-1] * self.R_iG[-1])
            dNt = self.gd.integrate(np.fabs(self.R_iG[-1]))
            if self.verbose:
                print('Mixer: broyden: fmin_G = %f fmin_D = %f' % fmin_G)
        if self.step == 0:
            self.eta_G = np.empty(nt_G.shape)
            self.eta_D = []
            for D_p in D_ap:
                self.eta_D.append(0)
                self.u_D.append([])
                self.D_iap.append([])
                self.dD_iap.append([])
        else:
            if self.step >= 2:
                del self.c_G[:]
                if len(self.v_G) >= self.nmaxold:
                    del self.u_G[0]
                    del self.v_G[0]
                    for u_D in self.u_D:
                        del u_D[0]
                temp_nt_G = self.R_iG[1] - self.R_iG[0]
                self.v_G.append(temp_nt_G / self.gd.integrate(temp_nt_G *
                                                              temp_nt_G))
                if len(self.v_G) < self.nmaxold:
                    nstep = self.step - 1
                else:
                    nstep = self.nmaxold
                for i in range(nstep):
                    self.c_G.append(self.gd.integrate(self.v_G[i] *
                                                      self.R_iG[1]))
                self.u_G.append(self.beta * temp_nt_G + self.nt_iG[1] -
                                self.nt_iG[0])
                for d_Dp, u_D, D_ip in zip(self.dD_iap, self.u_D, self.D_iap):
                    temp_D_ap = d_Dp[1] - d_Dp[0]
                    u_D.append(self.beta * temp_D_ap + D_ip[1] - D_ip[0])
                usize = len(self.u_G)
                for i in range(usize - 1):
                    a_G = self.gd.integrate(self.v_G[i] * temp_nt_G)
                    axpy(-a_G, self.u_G[i], self.u_G[usize - 1])
                    for u_D in self.u_D:
                        axpy(-a_G, u_D[i], u_D[usize - 1])
            self.eta_G = self.beta * self.R_iG[-1]
            for i, d_Dp in enumerate(self.dD_iap):
                self.eta_D[i] = self.beta * d_Dp[-1]
            usize = len(self.u_G)
            for i in range(usize):
                axpy(-self.c_G[i], self.u_G[i], self.eta_G)
                for eta_D, u_D in zip(self.eta_D, self.u_D):
                    axpy(-self.c_G[i], u_D[i], eta_D)
            axpy(-1.0, self.R_iG[-1], nt_G)
            axpy(1.0, self.eta_G, nt_G)
            for D_p, d_Dp, eta_D in zip(D_ap, self.dD_iap, self.eta_D):
                axpy(-1.0, d_Dp[-1], D_p)
                axpy(1.0, eta_D, D_p)
            if self.step >= 2:
                del self.nt_iG[0]
                for D_ip in self.D_iap:
                    del D_ip[0]
        self.nt_iG.append(np.copy(nt_G))
        for D_ip, D_p in zip(self.D_iap, D_ap):
            D_ip.append(np.copy(D_p))
        self.step += 1
        return dNt


class DummyMixer:
    """Dummy mixer for TDDFT, i.e., it does not mix."""
    name = 'dummy'
    beta = 1.0
    nmaxold = 1
    weight = 1

    def __init__(self, *args, **kwargs):
        return

    def mix(self, basemixers, nt_sG, D_asp):
        return 0.0

    def get_basemixers(self, nspins):
        return []

    def todict(self):
        return {'name': 'dummy'}


class NotMixingMixer:
    name = 'no-mixing'

    def __init__(self, beta, nmaxold, weight):
        """Construct density-mixer object.
        Parameters: they are ignored for this mixer
        """

        # whatever parameters you give it doesn't do anything with them
        self.beta = 0
        self.nmaxold = 0
        self.weight = 0

    def initialize_metric(self, gd):
        self.gd = gd
        self.metric = None

    def reset(self):
        """Reset Density-history.

        Called at initialization and after each move of the atoms.

        my_nuclei:   All nuclei in local domain.
        """

        # Previous density:
        self.nt_iG = []  # Pseudo-electron densities

    def calculate_charge_sloshing(self, R_G):
        return self.gd.integrate(np.fabs(R_G))

    def mix_single_density(self, nt_G, D_ap):
        iold = len(self.nt_iG)

        dNt = np.inf
        if iold > 0:
            # Calculate new residual (difference between input and
            # output density):
            dNt = self.calculate_charge_sloshing(nt_G - self.nt_iG[-1])
        # Store new input density:
        self.nt_iG = [nt_G.copy()]

        return dNt

    # may presently be overridden by passing argument in constructor
    def dotprod(self, R1_G, R2_G, dD1_ap, dD2_ap):
        pass

    def estimate_memory(self, mem, gd):
        gridbytes = gd.bytecount()
        mem.subnode('nt_iG, R_iG', 2 * self.nmaxold * gridbytes)

    def __repr__(self):
        string = 'no mixing of density'
        return string


class SeparateSpinMixerDriver:
    name = 'separate'

    def __init__(self, basemixerclass, beta, nmaxold, weight):
        self.basemixerclass = basemixerclass

        self.beta = beta
        self.nmaxold = nmaxold
        self.weight = weight

    def get_basemixers(self, nspins):
        return [self.basemixerclass(self.beta, self.nmaxold, self.weight)
                for _ in range(nspins)]

    def mix(self, basemixers, nt_sG, D_asp):
        """Mix pseudo electron densities."""
        D_asp = D_asp.values()
        D_sap = []
        for s in range(len(nt_sG)):
            D_sap.append([D_sp[s] for D_sp in D_asp])
        dNt = 0.0
        for nt_G, D_ap, basemixer in zip(nt_sG, D_sap, basemixers):
            dNt += basemixer.mix_single_density(nt_G, D_ap)
        return dNt


class SpinSumMixerDriver:
    name = 'sum'
    mix_atomic_density_matrices = False

    def __init__(self, basemixerclass, beta, nmaxold, weight):
        self.basemixerclass = basemixerclass

        self.beta = beta
        self.nmaxold = nmaxold
        self.weight = weight

    def get_basemixers(self, nspins):
        if nspins == 1:
            raise ValueError('Spin sum mixer expects 2 or 4 components')
        return [self.basemixerclass(self.beta, self.nmaxold, self.weight)]

    def mix(self, basemixers, nt_sG, D_asp):
        assert len(basemixers) == 1
        basemixer = basemixers[0]
        D_asp = D_asp.values()

        collinear = len(nt_sG) == 2

        # Mix density
        if collinear:
            nt_G = nt_sG.sum(0)
        else:
            nt_G = nt_sG[0]

        if self.mix_atomic_density_matrices:
            if collinear:
                D_ap = [D_sp[0] + D_sp[1] for D_sp in D_asp]
            else:
                D_ap = [D_sp[0] for D_sp in D_asp]
            dNt = basemixer.mix_single_density(nt_G, D_ap)
            if collinear:
                dD_ap = [D_sp[0] - D_sp[1] for D_sp in D_asp]
                for D_sp, D_p, dD_p in zip(D_asp, D_ap, dD_ap):
                    D_sp[0] = 0.5 * (D_p + dD_p)
                    D_sp[1] = 0.5 * (D_p - dD_p)
        else:
            dNt = basemixer.mix_single_density(nt_G, D_asp)

        if collinear:
            dnt_G = nt_sG[0] - nt_sG[1]
            # Only new magnetization for spin density
            # dD_ap = [D_sp[0] - D_sp[1] for D_sp in D_asp]

            # Construct new spin up/down densities
            nt_sG[0] = 0.5 * (nt_G + dnt_G)
            nt_sG[1] = 0.5 * (nt_G - dnt_G)

        return dNt


class SpinSumMixerDriver2(SpinSumMixerDriver):
    name = 'sum2'
    mix_atomic_density_matrices = True


class SpinDifferenceMixerDriver:
    name = 'difference'

    def __init__(self, basemixerclass, beta, nmaxold, weight,
                 beta_m=0.7, nmaxold_m=2, weight_m=10.0):
        self.basemixerclass = basemixerclass
        self.beta = beta
        self.nmaxold = nmaxold
        self.weight = weight
        self.beta_m = beta_m
        self.nmaxold_m = nmaxold_m
        self.weight_m = weight_m

    def get_basemixers(self, nspins):
        if nspins == 1:
            raise ValueError('Spin difference mixer expects 2 or 4 components')
        basemixer = self.basemixerclass(self.beta, self.nmaxold, self.weight)
        if nspins == 2:
            basemixer_m = self.basemixerclass(self.beta_m, self.nmaxold_m,
                                              self.weight_m)
            return basemixer, basemixer_m
        else:
            basemixer_x = self.basemixerclass(self.beta_m, self.nmaxold_m,
                                              self.weight_m)
            basemixer_y = self.basemixerclass(self.beta_m, self.nmaxold_m,
                                              self.weight_m)
            basemixer_z = self.basemixerclass(self.beta_m, self.nmaxold_m,
                                              self.weight_m)
            return basemixer, basemixer_x, basemixer_y, basemixer_z

    def mix(self, basemixers, nt_sG, D_asp):
        D_asp = D_asp.values()

        if len(nt_sG) == 2:
            basemixer, basemixer_m = basemixers
        else:
            assert len(nt_sG) == 4
            basemixer, basemixer_x, basemixer_y, basemixer_z = basemixers

        if len(nt_sG) == 2:
            # Mix density
            nt_G = nt_sG.sum(0)
            D_ap = [D_sp[0] + D_sp[1] for D_sp in D_asp]
            dNt = basemixer.mix_single_density(nt_G, D_ap)

            # Mix magnetization
            dnt_G = nt_sG[0] - nt_sG[1]
            dD_ap = [D_sp[0] - D_sp[1] for D_sp in D_asp]
            basemixer_m.mix_single_density(dnt_G, dD_ap)
            # (The latter is not counted in dNt)

            # Construct new spin up/down densities
            nt_sG[0] = 0.5 * (nt_G + dnt_G)
            nt_sG[1] = 0.5 * (nt_G - dnt_G)
            for D_sp, D_p, dD_p in zip(D_asp, D_ap, dD_ap):
                D_sp[0] = 0.5 * (D_p + dD_p)
                D_sp[1] = 0.5 * (D_p - dD_p)
        else:
            # Mix density
            nt_G = nt_sG[0]
            D_ap = [D_sp[0] for D_sp in D_asp]
            dNt = basemixer.mix_single_density(nt_G, D_ap)

            # Mix magnetization
            Dx_ap = [D_sp[1] for D_sp in D_asp]
            Dy_ap = [D_sp[2] for D_sp in D_asp]
            Dz_ap = [D_sp[3] for D_sp in D_asp]

            basemixer_x.mix_single_density(nt_sG[1], Dx_ap)
            basemixer_y.mix_single_density(nt_sG[2], Dy_ap)
            basemixer_z.mix_single_density(nt_sG[3], Dz_ap)
        return dNt


class SpinDiagonalMixerDriver:
    name = 'diagonal'

    def __init__(self, basemixerclass, beta, nmaxold, weight,
                 beta_m=0.7, nmaxold_m=2, weight_m=10.0):
        self.basemixerclass = basemixerclass
        self.beta = beta
        self.nmaxold = nmaxold
        self.weight = weight
        self.beta_m = beta_m
        self.nmaxold_m = nmaxold_m
        self.weight_m = weight_m

    def get_basemixers(self, nspins):
        if nspins == 1 or nspins == 2:
            raise ValueError('Spin diagonal mixer expects 4 components')
        basemixer = self.basemixerclass(self.beta, self.nmaxold, self.weight)
        basemixer_u = self.basemixerclass(self.beta_m, self.nmaxold_m,
                                              self.weight_m)
        basemixer_d = self.basemixerclass(self.beta_m, self.nmaxold_m,
                                              self.weight_m)
        return basemixer, basemixer_u, basemixer_d

    def mix(self, basemixers, nt_sG, D_asp):
        assert len(nt_sG) == 4
        D_asp = D_asp.values()
        basemixer, basemixer_u, basemixer_d = basemixers

        # Mix density
        nt_G = nt_sG[0]
        D_ap = [D_sp[0] for D_sp in D_asp]
        dNt = basemixer.mix_single_density(nt_G, D_ap, blas=False)

        # Construct complex magnetization density
        rho_Gss = np.array([[nt_sG[3],  nt_sG[1] - 1j*nt_sG[2]],
                            [nt_sG[1] + 1j*nt_sG[2], -nt_sG[3]]]).transpose((2,3,4,0,1))
        D_apss =[np.array([[D_sp[3],  D_sp[1] - 1j*D_sp[2]],
                           [D_sp[1] + 1j*D_sp[2], -D_sp[3]]]).transpose((2, 0, 1)) for D_sp in D_asp]

        # Diagonalize magnetization density using singular value decomposition
        u_Gss, s_Gs, vh_Gss = np.linalg.svd(rho_Gss)
        D_avps = [np.linalg.svd(D_pss) for D_pss in D_apss]
        
        # Extract effective mz+ and mz- in diagonal space
        su_G, sd_G = (s_Gs[:, :, :, 0], s_Gs[:, :, :, 1])
        Du_ap = [D_vps[1][:, 0] for D_vps in D_avps]
        Dd_ap = [D_vps[1][:, 1] for D_vps in D_avps]
        
        # Mix, inplace update of sx_G and Dx_ap
        basemixer_u.mix_single_density(su_G, Du_ap, blas=False)
        basemixer_d.mix_single_density(sd_G, Dd_ap, blas=False)
        
        # Reconstruct the array structure of the spin-index and calculate new complex magnetization density
        sn_Gs = np.append(su_G[..., np.newaxis], sd_G[..., np.newaxis], axis=-1)
        Dsn_aps = [np.append(Du_p[..., np.newaxis], Dd_p[..., np.newaxis], axis=-1) for (Du_p, Dd_p) in zip(Du_ap, Dd_ap)]
        rhon_Gss = np.matmul((u_Gss * sn_Gs[..., None, :]), vh_Gss)
        Dn_apss =  [np.matmul((D_vpss[0] * Dsn_ps[..., None, :]), D_vpss[2]) for (D_vpss, Dsn_ps) in zip(D_avps, Dsn_aps)]
        
        # Construct new real magnetization density components from complex magnetization density
        nt_sG[1] = 0.5 * (rhon_Gss[:, :, :, 0, 1] + rhon_Gss[:, :, :, 1, 0])
        nt_sG[2] = 1j*0.5 * (rhon_Gss[:, :, :, 0, 1] - rhon_Gss[:, :, :, 1, 0])
        nt_sG[3] = 0.5 * (rhon_Gss[:, :, :, 1, 1] - rhon_Gss[:, :, :, 0, 0])
        for D_sp, Dn_pss in zip(D_asp, Dn_apss):
            D_sp[1] = 0.5 * (Dn_pss[:, 0, 1] + Dn_pss[:, 1, 0])
            D_sp[2] = 1j*0.5 * (Dn_pss[:, 0, 1] - Dn_pss[:, 1, 0])
            D_sp[3] = 0.5 * (Dn_pss[:, 0, 0] - Dn_pss[:, 1, 1])
        
        return dNt

class SpinDiagfulMixerDriver:
    name = 'diagonalspinful'

    def __init__(self, basemixerclass, beta, nmaxold, weight, g=None):
        self.basemixerclass = basemixerclass
        self.beta = beta
        self.nmaxold = nmaxold
        self.weight = weight
        if g == None:
            self.g = np.identity(2)
        else:
            self.g = g

    def get_basemixers(self, nspins):
        if nspins == 1 or nspins == 2:
            raise ValueError('Spin diagonal mixer expects 4 components')
        basemixer = self.basemixerclass(self.beta, self.nmaxold, self.weight)
        return [basemixer]

    def mix(self, basemixers, nt_sG, D_asp):
        assert len(nt_sG) == 4
        D_asp = D_asp.values()
        basemixer = basemixers[0]

        # Construct complex 4-density, transpose for SVD
        rho_Gss = np.array([[nt_sG[0] + nt_sG[3], nt_sG[1] - 1j*nt_sG[2]],
                            [nt_sG[1] + 1j*nt_sG[2], nt_sG[0] - nt_sG[3]]]).transpose((2,3,4,0,1))
        D_apss =[np.array([[D_sp[0] + D_sp[3],  D_sp[1] - 1j*D_sp[2]],
                           [D_sp[1] + 1j*D_sp[2], D_sp[0] - D_sp[3]]]).transpose((2, 0, 1))
                 for D_sp in D_asp]

        # Diagonalize magnetization density using singular value decomposition
        u_Gss, s_Gs, vh_Gss = np.linalg.svd(rho_Gss)
        D_avps = [np.linalg.svd(D_pss) for D_pss in D_apss]

        # Tranpose for mixer
        s_sG = s_Gs.transpose((3, 0, 1, 2))
        Ds_asp = [D_vps[1].transpose(1, 0) for D_vps in D_avps]

        # Mix
        dNut, dNdt = basemixer.mix_density(s_sG, Ds_asp, self.g, blas=False)

        # Transpose back to compatability with SVD eigenvectors
        s_Gs = s_sG.transpose((1, 2, 3, 0))
        Ds_aps =  [Ds_sp.transpose(1, 0) for Ds_sp in Ds_asp]

        # Reconstruct the complex 4-density
        rhon_Gss = np.matmul((u_Gss * s_Gs[..., None, :]), vh_Gss)
        Dn_apss = [np.matmul((D_vpss[0] * Ds_ps[..., None, :]), D_vpss[2]) for (D_vpss, Ds_ps) in zip(D_avps, Ds_aps)]
        
        # Construct new real magnetization density components from complex magnetization density
        nt_sG[0] = 0.5 * (rhon_Gss[:, :, :, 1, 1] + rhon_Gss[:, :, :, 0, 0])
        nt_sG[1] = 0.5 * (rhon_Gss[:, :, :, 0, 1] + rhon_Gss[:, :, :, 1, 0])
        nt_sG[2] = 1j*0.5 * (rhon_Gss[:, :, :, 0, 1] - rhon_Gss[:, :, :, 1, 0])
        nt_sG[3] = 0.5 * (rhon_Gss[:, :, :, 1, 1] - rhon_Gss[:, :, :, 0, 0])
        for D_sp,Dn_pss in zip(D_asp, Dn_apss):
            D_sp[0] = 0.5 * (Dn_pss[:, 0, 0] + Dn_pss[:, 1, 1])
            D_sp[1] = 0.5 * (Dn_pss[:, 0, 1] + Dn_pss[:, 1, 0])
            D_sp[2] = 1j*0.5 * (Dn_pss[:, 0, 1] - Dn_pss[:, 1, 0])
            D_sp[3] = 0.5 * (Dn_pss[:, 0, 0] - Dn_pss[:, 1, 1])
        
        return dNut+dNdt 


class SpinCylinderMixerDriver:
    name = 'cylinder'

    def __init__(self, basemixerclass, beta, nmaxold, weight,
                 beta_m=0.7, nmaxold_m=2, weight_m=10.0,
                 beta_t=1, nmaxold_t=1, weight_t=0):
        self.basemixerclass = basemixerclass
        self.beta = beta
        self.nmaxold = nmaxold
        self.weight = weight
        self.beta_m = beta_m
        self.nmaxold_m = nmaxold_m
        self.weight_m = weight_m
        self.beta_t = beta_t
        self.nmaxold_t = nmaxold_t
        self.weight_t = weight_t

    def get_basemixers(self, nspins):
        if nspins == 1:
            raise ValueError('Spin difference mixer expects 2 or 4 components')
        basemixer = self.basemixerclass(self.beta, self.nmaxold, self.weight)
        if nspins == 2:
            basemixer_m = self.basemixerclass(self.beta_m, self.nmaxold_m,
                                              self.weight_m)
            return basemixer, basemixer_m
        else:
            basemixer_r = self.basemixerclass(self.beta_m, self.nmaxold_m,
                                              self.weight_m)
            basemixer_t = self.basemixerclass(self.beta_t, self.nmaxold_t,
                                              self.weight_t)
            basemixer_z = self.basemixerclass(self.beta_m, self.nmaxold_m,
                                              self.weight_m)
            return basemixer, basemixer_r, basemixer_t, basemixer_z

    def mix(self, basemixers, nt_sG, D_asp):
        D_asp = D_asp.values()

        if len(nt_sG) == 2:
            basemixer, basemixer_m = basemixers
        else:
            assert len(nt_sG) == 4
            basemixer, basemixer_r, basemixer_t, basemixer_z = basemixers

        if len(nt_sG) == 2:
            # Mix density
            nt_G = nt_sG.sum(0)
            D_ap = [D_sp[0] + D_sp[1] for D_sp in D_asp]
            dNt = basemixer.mix_single_density(nt_G, D_ap)

            # Mix magnetization
            dnt_G = nt_sG[0] - nt_sG[1]
            dD_ap = [D_sp[0] - D_sp[1] for D_sp in D_asp]
            dMt = basemixer_m.mix_single_density(dnt_G, dD_ap)
            # (The latter is not counted in dNt)

            # Construct new spin up/down densities
            nt_sG[0] = 0.5 * (nt_G + dnt_G)
            nt_sG[1] = 0.5 * (nt_G - dnt_G)
            for D_sp, D_p, dD_p in zip(D_asp, D_ap, dD_ap):
                D_sp[0] = 0.5 * (D_p + dD_p)
                D_sp[1] = 0.5 * (D_p - dD_p)
        else:
            # Mix density
            nt_G = nt_sG[0]
            D_ap = [D_sp[0] for D_sp in D_asp]
            dNt = basemixer.mix_single_density(nt_G, D_ap)

            # Mix magnetization
            ntr_sG = np.sqrt(nt_sG[1]**2 + nt_sG[2]**2)
            ntt_sG = np.arctan2(nt_sG[2], nt_sG[1])

            Dr_ap = [np.sqrt(D_sp[1]**2 + D_sp[2]**2) for D_sp in D_asp]
            Dt_ap = [np.arctan2(D_sp[2], D_sp[1]) for D_sp in D_asp]
            Dz_ap = [D_sp[3] for D_sp in D_asp]

            basemixer_r.mix_single_density(ntr_sG, Dr_ap, blas=False)
            basemixer_t.mix_single_density(ntt_sG, Dt_ap, blas=False)
            basemixer_z.mix_single_density(nt_sG[3], Dz_ap, blas=False)
            
            nt_sG[1] = ntr_sG * np.cos(ntt_sG)
            nt_sG[2] = ntr_sG * np.sin(ntt_sG)
            for D_sp, Dr_p, Dt_p in zip(D_asp, Dr_ap, Dt_ap):
                D_sp[1] = Dr_p * np.cos(Dt_p)
                D_sp[2] = Dr_p * np.sin(Dt_p)

        return dNt


class SpinFulMixerDriver:
    name = 'spinful'

    def __init__(self, basemixerclass, beta, nmaxold, weight, g=None):
        self.basemixerclass = basemixerclass
        self.beta = beta
        self.nmaxold = nmaxold
        self.weight = weight
        self.g = g

    def get_basemixers(self, nspins):
        if nspins == 1:
            raise ValueError('Spinful mixer expects 2 or 4 spin channels')
            
        basemixer = self.basemixerclass(self.beta, self.nmaxold, self.weight)
        return [basemixer]

    def mix(self, basemixers, nt_sG, D_asp):
        D_asp = D_asp.values()
        basemixer = basemixers[0]
        if self.g is None:
            self.g = np.identity(len(nt_sG))

        dNt = basemixer.mix_density(nt_sG, D_asp, self.g, blas=False)

        return np.sum(dNt)


# Dictionaries to get mixers by name:
_backends = {}
_methods = {}
for cls in [FFTBaseMixer, BroydenBaseMixer, BaseMixer, NotMixingMixer]:
    _backends[cls.name] = cls  # type:ignore
for dcls in [SeparateSpinMixerDriver, SpinSumMixerDriver, SpinFulMixerDriver,
             SpinSumMixerDriver2, SpinDiagonalMixerDriver, SpinCylinderMixerDriver,
             SpinDiagfulMixerDriver, SpinDifferenceMixerDriver, DummyMixer]:
    _methods[dcls.name] = dcls  # type:ignore


# This function is used by Density to decide mixer parameters
# that the user did not explicitly provide, i.e., it fills out
# everything that is missing and returns a mixer "driver".
def get_mixer_from_keywords(pbc, nspins, **mixerkwargs):
    if mixerkwargs.get('name') == 'dummy':
        return DummyMixer()

    if mixerkwargs.get('backend') == 'no-mixing':
        mixerkwargs['beta'] = 0
        mixerkwargs['nmaxold'] = 0
        mixerkwargs['weight'] = 0

    if nspins == 1:
        mixerkwargs['method'] = SeparateSpinMixerDriver

    # The plan is to first establish a kwargs dictionary with all the
    # defaults, then we update it with values from the user.
    kwargs = {'backend': BaseMixer}

    if np.any(pbc):  # Works on array or boolean
        kwargs.update(beta=0.05, history=5, weight=50.0)
    else:
        kwargs.update(beta=0.25, history=3, weight=1.0)

    if nspins == 1:
        kwargs['method'] = SeparateSpinMixerDriver
    else:
        kwargs['method'] = SpinDifferenceMixerDriver

    # Clean up mixerkwargs (compatibility)
    if 'nmaxold' in mixerkwargs:
        assert 'history' not in mixerkwargs
        mixerkwargs['history'] = mixerkwargs.pop('nmaxold')

    # Now the user override:
    for key in kwargs:
        # Clean any 'None' values out as if they had never been passed:
        val = mixerkwargs.pop(key, None)
        if val is not None:
            kwargs[key] = val

    # Resolve keyword strings (like 'fft') into classes (like FFTBaseMixer):
    driver = _methods.get(kwargs['method'], kwargs['method'])
    baseclass = _backends.get(kwargs['backend'], kwargs['backend'])

    # We forward any remaining mixer kwargs to the actual mixer object.
    # Any user defined variables that do not really exist will cause an error.
    mixer = driver(baseclass, beta=kwargs['beta'],
                   nmaxold=kwargs['history'], weight=kwargs['weight'],
                   **mixerkwargs)
    return mixer


# This is the only object which will be used by Density, sod the others
class MixerWrapper:
    def __init__(self, driver, nspins, gd):
        self.driver = driver

        self.beta = driver.beta
        self.nmaxold = driver.nmaxold
        self.weight = driver.weight
        assert self.weight is not None, driver

        self.basemixers = self.driver.get_basemixers(nspins)
        for basemixer in self.basemixers:
            basemixer.initialize_metric(gd)

    def mix(self, nt_sR, D_asp=None):
        if D_asp is not None:
            return self.driver.mix(self.basemixers, nt_sR, D_asp)

        # new interface:
        density = nt_sR
        nspins = density.nt_sR.dims[0]
        D_asp = {a: D_sii.copy().reshape((nspins, -1))
                 for a, D_sii in density.D_asii.items()}
        error = self.driver.mix(self.basemixers,
                                density.nt_sR.data,
                                D_asp)
        for a, D_sii in density.D_asii.items():
            ni = D_sii.shape[1]
            D_sii[:] = D_asp[a].reshape((-1, ni, ni))
        return error

    def estimate_memory(self, mem, gd):
        for i, basemixer in enumerate(self.basemixers):
            basemixer.estimate_memory(mem.subnode('Mixer %d' % i), gd)

    def reset(self):
        for basemixer in self.basemixers:
            basemixer.reset()

    def __str__(self):
        lines = ['Density mixing:',
                 'Method: ' + self.driver.name,
                 'Backend: ' + self.driver.basemixerclass.name,
                 'Linear mixing parameter: %g' % self.beta,
                 f'old densities: {self.nmaxold}',
                 'Damping of long wavelength oscillations: %g' % self.weight]
        if self.weight == 1:
            lines[-1] += '  # (no daming)'
        return '\n  '.join(lines)


# Helper function to define old-style interfaces to mixers.
# Defines and returns a function which looks like a mixer class
def _definemixerfunc(method, backend):
    def getmixer(beta=None, nmaxold=None, weight=None, **kwargs):
        d = dict(method=method, backend=backend,
                 beta=beta, nmaxold=nmaxold, weight=weight)
        d.update(kwargs)
        return d
    return getmixer


Mixer = _definemixerfunc('separate', 'pulay')
MixerSum = _definemixerfunc('sum', 'pulay')
MixerSum2 = _definemixerfunc('sum2', 'pulay')
MixerDif = _definemixerfunc('difference', 'pulay')
MixerDiag = _definemixerfunc('diagonal', 'pulay')
MixerDiagFul = _definemixerfunc('diagonalspinful', 'pulay')
MixerCyl = _definemixerfunc('cylinder', 'pulay')
MixerFul = _definemixerfunc('spinful', 'pulay')
FFTMixer = _definemixerfunc('separate', 'fft')
FFTMixerSum = _definemixerfunc('sum', 'fft')
FFTMixerSum2 = _definemixerfunc('sum2', 'fft')
FFTMixerDif = _definemixerfunc('difference', 'fft')
BroydenMixer = _definemixerfunc('separate', 'broyden')
BroydenMixerSum = _definemixerfunc('sum', 'broyden')
BroydenMixerSum2 = _definemixerfunc('sum2', 'broyden')
BroydenMixerDif = _definemixerfunc('difference', 'broyden')

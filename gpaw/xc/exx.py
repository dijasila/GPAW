from __future__ import division

import sys
from math import pi

import numpy as np
from ase.units import Hartree
from ase.utils import prnt

import gpaw.mpi as mpi
from gpaw.response.pair import PairDensity
from gpaw.wavefunctions.pw import PWDescriptor
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.utilities import unpack, unpack2, packed_index


def pawexxvv(atomdata, D_ii):
    """PAW correction for valence-valence EXX energy."""
    ni = len(D_ii)
    V_ii = np.empty((ni, ni))
    for i1 in range(ni):
        for i2 in range(ni):
            V = 0.0
            for i3 in range(ni):
                p13 = packed_index(i1, i3, ni)
                for i4 in range(ni):
                    p24 = packed_index(i2, i4, ni)
                    V += atomdata.M_pp[p13, p24] * D_ii[i3, i4]
            V_ii[i1, i2] = V
    return V_ii

        
class EXX(PairDensity):
    def __init__(self, calc, kpts=None, bands=None, ecut=150.0,
                 world=mpi.world, txt=sys.stdout):
    
        PairDensity.__init__(self, calc, ecut, world=world, txt=txt)

        ecut /= Hartree
        
        if kpts is None:
            kpts = range(self.calc.wfs.kd.nibzkpts)
        
        if bands is None:
            bands = [0, self.nocc2]
            
        self.kpts = kpts
        self.bands = bands

        shape = (self.calc.wfs.nspins, len(kpts), bands[1] - bands[0])
        self.exxvv_sin = np.zeros(shape)   # valence-valence exchange energies
        self.exxvc_sin = np.zeros(shape)   # valence-core exchange energies
        self.f_sin = np.empty(shape)       # occupation numbers

        # The total EXX energy will not be calculated if we are only
        # interested in a few eigenvalues for a few k-points
        self.exx = np.nan    # total EXX energy
        self.exxvv = np.nan  # valence-valence
        self.exxvc = np.nan  # valence-core
        self.exxcc = 0.0     # core-core

        self.mysKn1n2 = None  # my (s, K, n1, n2) indices
        self.distribute_k_points_and_bands(self.nocc2)
        
        self.mykpts = [self.get_k_point(s, K, n1, n2)
                       for s, K, n1, n2 in self.mysKn1n2]

        dvq = (2 * pi)**3 / self.vol / self.calc.wfs.kd.nbzkpts
        qcut = (dvq / (4 * pi / 3))**(1 / 3)
        self.G0 = (4 * pi * qcut / dvq)**-0.5
        
        # PAW matrices:
        self.D_asii = []  # atomic density matrices
        self.V_asii = []  # valence-valence correction
        self.C_aii = []   # valence-core correction
        self.initialize_paw_exx_corrections()
        
    def calculate(self):
        kd = self.calc.wfs.kd
        for s in range(self.calc.wfs.nspins):
            for i, k1 in enumerate(self.kpts):
                K1 = kd.ibz2bz_k[k1]
                kpt1 = self.get_k_point(s, K1, *self.bands)
                self.f_sin[s, i] = kpt1.f_n
                for kpt2 in self.mykpts:
                    if kpt2.s == s:
                        self.calculate_q(i, kpt1, kpt2)
                
                self.calculate_paw_exx_corrections(i, kpt1)

        self.world.sum(self.exxvv_sin)
        
        # Calculate total energy if we have everything needed:
        if (len(self.kpts) == kd.nibzkpts and
            self.bands[0] == 0 and
            self.bands[1] >= self.nocc2):
            exxvv_i = (self.exxvv_sin * self.f_sin).sum(axis=2).sum(axis=0)
            exxvc_i = 2 * (self.exxvc_sin * self.f_sin).sum(axis=2).sum(axis=0)
            self.exxvv = np.dot(kd.weight_k[self.kpts], exxvv_i)
            self.exxvc = np.dot(kd.weight_k[self.kpts], exxvc_i)
            self.exx = self.exxvv + self.exxvc + self.exxcc
            prnt('Exact exchange energy:', file=self.fd)
            for kind, exx in [('valence-valence', self.exxvv),
                              ('valence-core', self.exxvc),
                              ('core-core', self.exxcc),
                              ('total', self.exx)]:
                prnt('%16s%11.3f eV' % (kind + ':', exx * Hartree),
                     file=self.fd)

        exx_sin = self.exxvv_sin + self.exxvc_sin
        prnt('EXX eigenvalue contributions in eV:', file=self.fd)
        prnt(exx_sin * Hartree, file=self.fd)
        
        return self.exx * Hartree, exx_sin * Hartree
        
    def calculate_q(self, i, kpt1, kpt2):
        wfs = self.calc.wfs
        q_c = wfs.kd.bzk_kc[kpt2.K] - wfs.kd.bzk_kc[kpt1.K]
        shift_c = 0
        qd = KPointDescriptor([q_c])
        pd = PWDescriptor(self.ecut, wfs.gd, complex, qd)
        Q_G = self.get_fft_indices(kpt1.K, kpt2.K, q_c, pd,
                                   kpt1.shift_c - kpt2.shift_c)

        Q_aGii = self.initialize_paw_corrections(pd, soft=True)
        
        for n in range(kpt1.n2 - kpt1.n1):
            ut1cc_R = kpt1.ut_nR[n].conj()
            C1_aGi = [np.dot(Q_Gii, P1_ni[n].conj())
                     for Q_Gii, P1_ni in zip(Q_aGii, kpt1.P_ani)]
            n_mG = self.calculate_pair_densities(ut1cc_R, C1_aGi, kpt2,
                                                 pd, Q_G)
            f_m = kpt2.f_n
            x = self.calculate_n(pd, n_mG, f_m)
            self.exxvv_sin[kpt1.s, i, n] += x

    def calculate_n(self, pd, n_mG, f_m):
        G_G = pd.G2_qG[0]**0.5
        if G_G[0] == 0.0:
            G_G[0] = self.G0
        
        x_mG = ((f_m**0.5)[:, np.newaxis] * n_mG / G_G).view(float)
        ex = -2 * pi / self.vol * np.vdot(x_mG, x_mG)
        return ex

    def initialize_paw_exx_corrections(self):
        for a, atomdata in enumerate(self.calc.wfs.setups):
            D_sii = []
            V_sii = []
            for D_p in self.calc.density.D_asp[a]:
                D_ii = unpack2(D_p)
                V_ii = pawexxvv(atomdata, D_ii)
                D_sii.append(D_ii)
                V_sii.append(V_ii)
            C_ii = unpack(atomdata.X_p)
            self.D_asii.append(D_sii)
            self.V_asii.append(V_sii)
            self.C_aii.append(C_ii)
            self.exxcc += atomdata.ExxC

    def calculate_paw_exx_corrections(self, i, kpt):
        dexxvv = 0.0
        exxvc = 0.0
        
        n1, n2 = self.bands
        s = kpt.s
        
        for D_sii, V_sii, C_ii, P_ni in zip(self.D_asii, self.V_asii,
                                            self.C_aii, kpt.P_ani):
            D_ii = D_sii[s]
            V_ii = V_sii[s]
            dexxvv -= np.vdot(D_ii, V_ii) / 2
            exxvc -= np.vdot(D_ii, C_ii)
            v_n = (np.dot(P_ni, V_ii) * P_ni.conj()).sum(axis=1).real
            c_n = (np.dot(P_ni, C_ii) * P_ni.conj()).sum(axis=1).real
            self.exxvv_sin[s, i] -= v_n / self.world.size
            self.exxvc_sin[s, i] -= c_n

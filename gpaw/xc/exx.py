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


class EXX(PairDensity):
    def __init__(self, calc, kpts=None, bands=None,
                 ecut=150.0,
                 world=mpi.world,
                 txt=sys.stdout):
    
        PairDensity.__init__(self, calc, ecut, world=world, txt=txt)

        ecut /= Hartree
        
        if kpts is None:
            kpts = range(self.calc.wfs.kd.nibzkpts)
        
        if bands is None:
            bands = [0, self.nocc2]
            
        self.kpts = kpts
        self.bands = bands

        shape = (self.calc.wfs.nspins, len(kpts), bands[1] - bands[0])
        self.ex_sin = np.zeros(shape)   # exchange energies
        self.eps_sin = np.zeros(shape)  # KS-eigenvalues
        self.f_sin = np.zeros(shape)    # occupation numbers
        
        self.mysKn1n2 = None  # my (s, K, n1, n2) indices
        self.distribute_k_points_and_bands(self.nocc2)
        
        self.mykpts = [self.get_k_point(s, K, n1, n2)
                       for s, K, n1, n2 in self.mysKn1n2]

        dvq = (2 * pi)**3 / self.vol / self.calc.wfs.kd.nbzkpts
        qcut = (dvq / (4 * pi / 3))**(1 / 3)
        self.G0 = (4 * pi * qcut / dvq)**-0.5
        
    def calculate(self):
        kd = self.calc.wfs.kd
        for s in range(self.calc.wfs.nspins):
            for i, k1 in enumerate(self.kpts):
                K1 = kd.ibz2bz_k[k1]
                kpt1 = self.get_k_point(s, K1, *self.bands)
                self.eps_sin[s, i] = kpt1.eps_n
                self.f_sin[s, i] = kpt1.f_n
                for kpt2 in self.mykpts:
                    if kpt2.s == s:
                        self.calculate_exx(i, kpt1, kpt2)
                        
        exx_i = (self.exx_sin * self.f_sin).sum(axis=2).sum(axis=0)
        self.energy = 0.5 * np.dot(kd.weight_k[self.kpts], exx_i)
        return self.energy * Hartree
        
    def calculate_exx(self, i, kpt1, kpt2):
        wfs = self.calc.wfs
        q_c = wfs.kd.bzk_kc[kpt1.K] - wfs.kd.bzk_kc[kpt2.K]
        shift_c = 0
        qd = KPointDescriptor([q_c])
        pd = PWDescriptor(self.ecut, wfs.gd, complex, qd)
        Q_G = self.get_fft_indices(kpt1.K, kpt2.K, q_c, pd,
                                   kpt1.shift_c - kpt2.shift_c)

        Q_aGii = self.calculate_paw_corrections(pd)
        
        for n in range(kpt1.n2 - kpt1.n1):
            ut1cc_R = kpt1.ut_nR[n].conj()
            C1_aGi = [np.dot(Q_Gii, P1_ni[n].conj())
                     for Q_Gii, P1_ni in zip(Q_aGii, kpt1.P_ani)]
            n_mG = self.calculate_pair_densities(ut1cc_R, C1_aGi, kpt2,
                                                 pd, Q_G)
            f_m = kpt2.f_n
            x = self.exx(pd, n_mG, f_m)
            self.ex_sin[kpt1.s, i, n] += x

    def exx(self, pd, n_mG, f_m):
        G_G = pd.G2_qG[0]**0.5
        if G_G[0] == 0.0:
            G_G[0] = self.G0
            
        x_G = np.dot(f_m, n_mG) / G_G
        ex = np.vdot(x_G, x_G).real
        return ex

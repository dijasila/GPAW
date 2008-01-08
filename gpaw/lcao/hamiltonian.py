from math import sqrt, pi

import numpy as npy

from gpaw.hamiltonian import Hamiltonian
from gpaw.utilities.blas import rk, r2k
from gpaw.utilities import unpack
from gpaw.lcao.overlap import TwoCenterIntegrals
from gpaw import debug
from _gpaw import overlap


class LCAOHamiltonian(Hamiltonian):
    """Hamiltonian class for LCAO-basis calculations"""

    def __init__(self, paw):
        Hamiltonian.__init__(self, paw)
        self.setups = paw.setups
        self.ibzk_kc = paw.ibzk_kc
        self.gamma = paw.gamma

    def initialize(self, cell_c):
        self.nao = 0
        for nucleus in self.nuclei:
            nucleus.initialize_atomic_orbitals(self.gd, 42, None)
            self.nao += nucleus.get_number_of_atomic_orbitals()

        tci = TwoCenterIntegrals(self.setups)

        R_dc = self.calculate_displacements(tci.rcmax, cell_c)
        
        nkpts = len(self.ibzk_kc)

        for nucleus1 in self.nuclei:
            pos1 = nucleus1.spos_c
            setup1 = nucleus1.setup
            ni1 = nucleus1.get_number_of_partial_waves()
            nucleus1.P_kmi = npy.zeros((nkpts, self.nao, ni1), complex)
            P_mi = npy.zeros((self.nao, ni1))
            for R in R_dc:
                i1 = 0 
                for j1, pt1 in enumerate(setup1.pt_j):
                    id1 = (setup1.symbol, j1)
                    l1 = pt1.get_angular_momentum_number()
                    for m1 in range(2 * l1 + 1):
                        self.p_overlap(R, i1, pos1, id1, l1, m1, P_mi, tci)
                        i1 += 1
                for k in range(nkpts):
                    nucleus1.P_kmi[k] += (P_mi *
                                          npy.exp(2j * pi *
                                                  npy.dot(self.ibzk_kc[k], R)))
                    
        self.T_kmm = npy.zeros((nkpts, self.nao, self.nao), complex)
        T_mm = npy.zeros((self.nao, self.nao))
        self.S_kmm = npy.zeros((nkpts, self.nao, self.nao), complex)
        S_mm = npy.zeros((self.nao, self.nao))

        for R in R_dc:
            i1 = 0
            for nucleus1 in self.nuclei:
                pos1 = nucleus1.spos_c
                setup1 = nucleus1.setup
                for j1, phit1 in enumerate(setup1.phit_j):
                    id1 = (setup1.symbol, j1)
                    l1 = phit1.get_angular_momentum_number()
                    for m1 in range(2 * l1 + 1):
                        self.st_overlap(R, i1, pos1, id1,
                                        l1, m1, S_mm, T_mm, tci)
                        i1 += 1
            for k in range(nkpts):            
                self.S_kmm[k] +=  S_mm * npy.exp(2j * pi *
                                                 npy.dot(self.ibzk_kc[k], R))
                self.T_kmm[k] +=  T_mm * npy.exp(2j * pi *
                                                 npy.dot(self.ibzk_kc[k], R))
                
        for nucleus in self.nuclei:
            for k in range(nkpts):
                self.S_kmm[k] += npy.dot(npy.dot(nucleus.P_kmi[k],
                                            nucleus.setup.O_ii),
                                    npy.transpose(nucleus.P_kmi[k]))

    def p_overlap(self, R, i1, pos1, id1, l1, m1, P_mi, tci):
        i2 = 0
        for nucleus2 in self.nuclei:
            pos2 = nucleus2.spos_c
            d = (R + pos1 - pos2) * self.gd.domain.cell_c
            setup2 = nucleus2.setup
            for j2, phit2 in enumerate(setup2.phit_j):
                id2 = (setup2.symbol, j2)
                l2 = phit2.get_angular_momentum_number()
                for m2 in range(2 * l2 + 1):
                    P = tci.p_overlap(id1, id2, l1, l2, m1, m2, d)
                    P_mi[i2, i1] = P
                    i2 += 1

    def st_overlap(self, R, i1, pos1, id1, l1, m1, S_mm, T_mm, tci):
       i2 = 0
       for nucleus2 in self.nuclei:
           pos2 = nucleus2.spos_c
           d = (pos1 - pos2 + R) * self.gd.domain.cell_c
           setup2 = nucleus2.setup
           for j2, phit2 in enumerate(setup2.phit_j):
               id2 = (setup2.symbol, j2)
               l2 = phit2.get_angular_momentum_number()
               for m2 in range(2 * l2 + 1):
                   S, T = tci.st_overlap(id1, id2, l1, l2,
                                         m1, m2, d)
                   S_mm[i1, i2] = S
                   T_mm[i1, i2] = T
                   i2 += 1

    def calculate_displacements(self, rmax, cell_c):
        n = [1 + int(2 * rmax / a) for a in cell_c ]
        n = [0, 0, 0]
        # XXXXX BC's !!!!!
        nd = (1 + 2 * n[0]) * (1 + 2 * n[1]) * (1 + 2 * n[2])
        R_dc = npy.empty((nd, 3))
        d = 0
        for d1 in range(-n[0], n[0] + 1):
            for d2 in range(-n[0], n[0] + 1):
                for d3 in range(-n[0], n[0] + 1):
                    R_dc[d, :] = d1, d2, d3
                    d += 1
        return R_dc

        
    def calculate_effective_potential_matrix(self, V_mm, s):
        box_b = []
        for nucleus in self.nuclei:
            if debug:	
                box_b.append(nucleus.phit_i.box_b[0].lfs)
            else:	
                box_b.extend(nucleus.phit_i.box_b)	
        assert len(box_b) == len(self.nuclei)
        from _gpaw import overlap
        from time import time as t
        t0 = t()
        overlap(box_b, self.vt_sG[s], V_mm)
        t1 = t()

    def calculate_effective_potential_matrix2(self, Vt_kmm, s):
        nb = 0
        for nucleus in self.nuclei:
            nb += len(nucleus.phit_i.box_b)

        m_b = npy.empty(nb, int)

        if self.gamma:
            phase_kb = npy.empty((0, 0), complex)
        else:
            phase_kb = npy.empty((nb, nkpts), complex) # XXX

        m = 0
        b1 = 0
        lfs_b = []
        for nucleus in self.nuclei:
            if debug:	
                box_b = [box.lfs for box in nucleus.phit_i.box_b]
            else:	
                box_bnucleus.phit_i.box_b
            b2 = b1 + len(box_b)
            m_b[b1:b2] = m
            lfs_b.extend(box_b)
            if not self.gamma:
                phase_bk[b1:b2] = nucleus.phit_i.phase_kb.T
            m += phit_i.ni
            b1 = b2

        overlap(lfs_b, m_b, phase_bk, self.vt_sG[s], Vt_kmm)

    def old_initialize(self):
        self.nao = 0
        for nucleus in self.nuclei:
            self.nao += nucleus.get_number_of_atomic_orbitals()
        self.phi_mG = self.gd.zeros(self.nao)

        m1 = 0
        for nucleus in self.nuclei:
            niao = nucleus.get_number_of_atomic_orbitals()
            m2 = m1 + niao
            nucleus.initialize_atomic_orbitals(self.gd, 42, None)
            nucleus.create_atomic_orbitals(self.phi_mG[m1:m2], 0)
            m1 = m2
        assert m2 == self.nao

        for nucleus in self.nuclei:
            ni = nucleus.get_number_of_partial_waves()
            nucleus.P_mi = npy.zeros((self.nao, ni))
            nucleus.pt_i.integrate(self.phi_mG, nucleus.P_mi)

        self.S_mm = npy.zeros((self.nao, self.nao))
        rk(self.gd.dv, self.phi_mG, 0.0, self.S_mm)
        
        # Filling up the upper triangle:
        for m in range(self.nao - 1):
            self.S_mm[m, m:] = self.S_mm[m:, m]

        for nucleus in self.nuclei:
            self.S_mm += npy.dot(npy.dot(nucleus.P_mi, nucleus.setup.O_ii),
                            npy.transpose(nucleus.P_mi))

        self.T_mm = npy.zeros((self.nao, self.nao))
        Tphi_mG = self.gd.zeros(self.nao)
        self.kin.apply(self.phi_mG, Tphi_mG)
        r2k(0.5 * self.gd.dv, self.phi_mG, Tphi_mG, 0.0, self.T_mm)

        # Filling up the upper triangle:
        for m in range(self.nao - 1):
            self.T_mm[m, m:] = self.T_mm[m:, m] 

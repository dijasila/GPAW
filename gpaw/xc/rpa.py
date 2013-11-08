import sys

import numpy as np
from ase.utils import devnull
from ase.dft.kpoints import monkhorst_pack
from scipy.special.orthogonal import p_roots

from gpaw.response.chi0 import Chi0
from gpaw.mpi import serial_comm, world
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.wavefunctions.pw import PWDescriptor


class RPACorrelation:
    def __init__(self, calc, omega_w, weight_w,
                 wcomm=serial_comm, kncomm=world, fd=sys.stdout):
        self.calc = calc
        self.omega_w = omega_w
        self.weight_w = weight_w

        assert len(omega_w) % wcomm.size == 0
        self.mynw = len(omega_w) // wcomm.size
        w1 = wcomm.rank * self.mynw
        w2 = w1 + self.mynw
        self.myomega_w = omega_w[w1:w2]
        self.kncomm = kncomm

        self.fd = fd
        
        self.qd = None  # q-points descriptor
        self.initialize_q_points()
    
    def initialize_q_points(self):
        kd = self.calc.wfs.kd
        shift_c = 0.5 * ((kd.N_c + 1) % 2) / kd.N_c
        bzq_qc = monkhorst_pack(kd.N_c) + shift_c
        self.qd = KPointDescriptor(bzq_qc)
        self.qd.set_symmetry(self.calc.atoms, self.calc.wfs.setups,
                             usesymm=True, N_c=self.calc.wfs.gd.N_c)
        
    def calculate(self, ecut_i, nbands=None):
        ecutmax = max(ecut_i)
        
        chi0 = Chi0(self.calc, 1j * self.omega_w, eta=0.0, fd=devnull,
                    comm=self.kncomm)
        
        energy_qiw = []
        for q_c in self.qd.ibzk_kc:
            thisqd = KPointDescriptor([q_c])
            pd = PWDescriptor(ecutmax, self.calc.wfs.gd, complex, thisqd)
            nG = pd.ngmax
            chi0_wGG = np.zeros((self.mynw, nG, nG), complex)
            chi0_wxvG = np.zeros((self.mynw, 2, 3, nG), complex)
            m1 = chi0.nocc1
            m2 = nbands or nG
            cut_G = None
            energy_iw = []
            for ecut in ecut_i:
                print >> self.fd, q_c, ecut,
                if ecut < ecutmax:
                    cut_G = np.arange(nG)[pd.G2_qG[0] <= 2 * ecut]
                    m2 = len(cut_G)
                e_w = self.calculate_q(chi0, pd, chi0_wGG, chi0_wxvG,
                                       m1, m2, cut_G)
                energy_iw.append(e_w)
                m1 = m2

            energy_qiw.append(energy_iw)
        
        energy_qiw = np.array(energy_qiw)
        print >> self.fd, np.dot(energy_qiw, self.weight_w).sum(1) / 4 / np.pi
        #E = np.dot(np.array(self.q_weights), np.array(E_q).real)

    def calculate_q(self, chi0, pd, chi0_wGG, chi0_wxvG, m1, m2, cut_G):
        chi0._calculate(pd, chi0_wGG, chi0_wxvG, m1, m2)
        if not pd.kd.gamma:
            e_w = self.calculate_energy(pd, chi0_wGG, cut_G)
        else:
            e_w = 0.0
            for v in range(3):
                chi0_wGG[:, 0] = chi0_wxvG[:, 0, v]
                chi0_wGG[:, :, 0] = chi0_wxvG[:, 1, v]
                e_w += self.calculate_energy(pd, chi0_wGG, cut_G) / 3
        return e_w
        
    def calculate_energy(self, pd, chi0_wGG, cut_G):
        G_G = pd.G2_qG[0]**0.5  # |G+q|
        if G_G[0] == 0.0:
            G_G[0] = 1.0
        nG = len(G_G)
        
        if cut_G is not None:
            G_G = G_G[cut_G]
            
        e_w = []
        for chi0_GG in chi0_wGG:
            if cut_G is not None:
                chi0_GG = chi0_GG[cut_G, cut_G]
            e_GG = np.eye(nG) - 4 * np.pi * chi0_GG / G_G / G_G[:, np.newaxis]
            e = np.log(np.linalg.det(e_GG)) + nG - np.trace(e_GG)
            e_w.append(e)
        e_w = np.array(e_w)
        print >> self.fd, np.dot(e_w, self.weight_w) / 4 / np.pi
        return e_w

        
def get_gauss_legendre_points(nw=16, frequency_cut=800.0, frequency_scale=2.0):
    y_w, weights_w = p_roots(nw)
    ys = 0.5 - 0.5 * y_w
    ys = ys[::-1]
    w = (-np.log(1 - ys))**frequency_scale
    w *= frequency_cut / w[-1]
    alpha = (-np.log(1 - ys[-1]))**frequency_scale / frequency_cut
    transform = (-np.log(1 - ys))**(frequency_scale - 1) \
        / (1 - ys) * frequency_scale / alpha
    omega_w = w * transform
    return omega_w, weights_w

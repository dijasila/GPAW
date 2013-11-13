import os
import sys

import numpy as np
from ase.units import Hartree
from ase.utils import devnull, prnt
from ase.dft.kpoints import monkhorst_pack
from scipy.special.orthogonal import p_roots

from gpaw import GPAW
import gpaw.mpi as mpi
from gpaw.response.chi0 import Chi0
from gpaw.wavefunctions.pw import PWDescriptor
from gpaw.kpt_descriptor import KPointDescriptor


class RPACorrelation:
    def __init__(self, calc, filename=None,
                 skip_gamma=False,
                 nfrequencies=16, frequency_cut=800.0, frequency_scale=2.0,
                 frequencies=None, weights=None,
                 wcomm=None, chicomm=None, world=mpi.world,
                 txt=sys.stdout):
    
        if isinstance(calc, str):
            calc = GPAW(calc, txt=None, communicator=mpi.serial_comm)
        self.calc = calc
        
        if world.rank != 0:
            txt = devnull
        self.fd = txt
        
        if frequencies is None:
            frequencies, weights = get_gauss_legendre_points(nfrequencies,
                                                             frequency_cut,
                                                             frequency_scale)
        
        self.omega_w = frequencies / Hartree
        self.weight_w = weights / Hartree
        prnt('Frequencies:', file=self.fd)
        prnt(', '.join('%.2f' % (o * Hartree) for o in self.omega_w), 'eV',
             file=self.fd)

        if wcomm is None:
            wcomm = 1
            
        if isinstance(wcomm, int):
            if wcomm == 1:
                wcomm = mpi.serial_comm
                chicomm = world
            else:
                r = world.rank
                s = world.size
                assert s % wcomm == 0
                n = s // wcomm  # size of skncomm
                wcomm = world.new_communicator(range(r % n, s, n))
                chicomm = world.new_communicator(range(r // n * n,
                                                       (r // n + 1) * n))
            
        assert len(self.omega_w) % wcomm.size == 0
        self.mynw = len(self.omega_w) // wcomm.size
        self.w1 = wcomm.rank * self.mynw
        self.w2 = self.w1 + self.mynw
        self.myomega_w = self.omega_w[self.w1:self.w2]
        self.wcomm = wcomm
        self.chicomm = chicomm
        self.world = world

        self.qd = None  # q-points descriptor
        self.initialize_q_points()
    
        # Energies for all q-vetors and cutoff energies:
        self.energy_qi = []
        
        self.filename = filename
    
    def initialize_q_points(self):
        kd = self.calc.wfs.kd
        assert -1 not in kd.bz2bz_ks
        shift_c = 0.5 * ((kd.N_c + 1) % 2) / kd.N_c
        bzq_qc = monkhorst_pack(kd.N_c) + shift_c
        self.qd = KPointDescriptor(bzq_qc)
        self.qd.set_symmetry(self.calc.atoms, self.calc.wfs.setups,
                             usesymm=True, N_c=self.calc.wfs.gd.N_c)
        
    def read(self):
        lines = open(self.filename).readlines()
        n = 0
        self.energy_qi = []
        nq = len(lines) // len(self.ecut_i)
        for q_c in self.qd.ibzk_kc[:nq]:
            self.energy_qi.append([])
            for ecut in self.ecut_i:
                q1, q2, q3, ec, energy = [float(x)
                                            for x in lines[n].split()]
                self.energy_qi[-1].append(energy)
                n += 1
            
                if (abs(q_c - (q1, q2, q3)).max() > 1e-5 or
                    abs(ecut - ec) > 1e-5):
                    self.energy_qi = []
                    return

        prnt('Read %d q-points from file: %s' % (nq, self.filename),
             file=self.fd)
            
    def write(self):
        if self.world.rank == 0 and self.filename:
            fd = open(self.filename, 'w')
            for energy_i, q_c in zip(self.energy_qi, self.qd.ibzk_kc):
                for energy, ecut in zip(energy_i, self.ecut_i):
                    prnt('%10.6f %10.6f %10.6f %10.6f %r' %
                         (tuple(q_c) + (ecut, energy)), file=fd)
        
    def calculate(self, ecut, nbands=None):
        """Calculate RPA correlation energy for one or several cutoffs.
        
        ecut: float or list of floats
            Plane-wave cutoff(s).
        nbands: int
            Number of bands (defaults to number of plane-waves).
        """
        
        if isinstance(ecut, (float, int)):
            ecut_i = [ecut]
        else:
            ecut_i = ecut
        self.ecut_i = np.asarray(ecut_i) / Hartree
        ecutmax = max(self.ecut_i)
        
        prnt('Cutoff energies:',
             ', '.join('%.3f' % e for e in self.ecut_i * Hartree), 'eV',
             file=self.fd)
        
        if self.filename and os.path.isfile(self.filename):
            self.read()
            self.world.barrier()

        chi0 = Chi0(self.calc, 1j * self.omega_w, eta=0.0, fd=devnull,
                    comm=self.chicomm)
        
        nq = len(self.energy_qi)
        for q_c, weight in zip(self.qd.ibzk_kc[nq:], self.qd.weight_k):
            thisqd = KPointDescriptor([q_c])
            pd = PWDescriptor(ecutmax, self.calc.wfs.gd, complex, thisqd)

            nG = pd.ngmax
            chi0_wGG = np.zeros((self.mynw, nG, nG), complex)
            if not q_c.any():
                # Wings (x=0,1) and head (G=0) for optical limit and three
                # directions (v=0,1,2):
                chi0_wxvG = np.zeros((self.mynw, 2, 3, nG), complex)
            else:
                chi0_wxvG = None
        
            # First not completely filled band:
            m1 = chi0.nocc1

            prnt('q:', q_c, file=self.fd)
            
            energy_i = []
            for ecut in self.ecut_i:
                if ecut == ecutmax:
                    # Nothing to cut away:
                    cut_G = None
                    m2 = nbands or nG
                else:
                    cut_G = np.arange(nG)[pd.G2_qG[0] <= 2 * ecut]
                    m2 = len(cut_G)
                    
                prnt('Cutoff energy%9.3f eV,' % (ecut * Hartree),
                     'bands %-10s' % ('%d-%d:' % (m1, m2 - 1)),
                     file=self.fd, end='', flush=True)
                
                energy = self.calculate_q(chi0, pd, weight,
                                          chi0_wGG, chi0_wxvG, m1, m2, cut_G)
                energy_i.append(energy)
                m1 = m2

            self.energy_qi.append(energy_i)
            self.write()
        
        e_i = np.array(self.energy_qi).sum(axis=0)
        prnt('Total correlation energy:', file=self.fd)
        prnt(', '.join('%.3f' % (e * Hartree) for e in e_i), 'eV',
             file=self.fd)
        
        self.extrapolate()
        
        return e_i * Hartree
        
    def extrapolate(self):
        e_i = np.array(self.energy_qi).sum(axis=0)
        ex_i = []
        for i in range(len(e_i) - 1):
            e1, e2 = e_i[i:i + 2]
            x1, x2 = self.ecut_i[i:i + 2]**-1.5
            ex = (e1 * x2 - e2 * x1) / (x2 - x1)
            ex_i.append(ex)
        
        prnt('Extrapolated:', file=self.fd)
        prnt(', '.join('%.3f' % (e * Hartree) for e in ex_i), 'eV',
             file=self.fd, flush=True)
        
        return e_i * Hartree
        
    def calculate_q(self, chi0, pd, weight,
                    chi0_wGG, chi0_wxvG, m1, m2, cut_G):
        chi0._calculate(pd, chi0_wGG, chi0_wxvG, m1, m2)
        if not pd.kd.gamma:
            e = self.calculate_energy(pd, chi0_wGG, cut_G) * weight
            prnt(' %.3f eV' % (e * Hartree), flush=True, file=self.fd)
        else:
            e = 0.0
            for v in range(3):
                chi0_wGG[:, 0] = chi0_wxvG[:, 0, v]
                chi0_wGG[:, :, 0] = chi0_wxvG[:, 1, v]
                ev = self.calculate_energy(pd, chi0_wGG, cut_G) * weight / 3
                e += ev
                prnt(' %.3f eV' % (ev * Hartree), end='', file=self.fd)
            prnt(flush=True, file=self.fd)

        return e
        
    def calculate_energy(self, pd, chi0_wGG, cut_G):
        """Evaluate RPA correlation energy from chi-0."""
        
        G_G = pd.G2_qG[0]**0.5  # |G+q|
        if G_G[0] == 0.0:
            G_G[0] = 1.0
        
        if cut_G is not None:
            G_G = G_G[cut_G]

        nG = len(G_G)
        
        e_w = []
        for chi0_GG in chi0_wGG:
            if cut_G is not None:
                chi0_GG = chi0_GG.take(cut_G, 0).take(cut_G, 1)

            e_GG = np.eye(nG) - 4 * np.pi * chi0_GG / G_G / G_G[:, np.newaxis]
            e = np.log(np.linalg.det(e_GG)) + nG - np.trace(e_GG)
            e_w.append(e.real)

        E_w = np.zeros_like(self.omega_w)
        self.wcomm.gather(np.array(e_w), 0, E_w)
        energy = np.dot(E_w, self.weight_w) / (4 * np.pi)
        return energy

        
def get_gauss_legendre_points(nw=16, frequency_cut=800.0, frequency_scale=2.0):
    y_w, weights_w = p_roots(nw)
    ys = 0.5 - 0.5 * y_w
    ys = ys[::-1]
    w = (-np.log(1 - ys))**frequency_scale
    w *= frequency_cut / w[-1]
    alpha = (-np.log(1 - ys[-1]))**frequency_scale / frequency_cut
    transform = (-np.log(1 - ys))**(frequency_scale - 1) \
        / (1 - ys) * frequency_scale / alpha
    return w, weights_w * transform

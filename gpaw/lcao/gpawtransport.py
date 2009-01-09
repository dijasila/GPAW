import pickle

from ase.transport.selfenergy import LeadSelfEnergy
from ase.transport.greenfunction import GreenFunction
from ase.transport.tools import *;print 'XXX??????'
from ase import Atoms, Atom, monkhorst_pack, Hartree
import ase
import numpy as np

from gpaw import GPAW, Mixer
from gpaw import restart as restart_gpaw
from gpaw.lcao.tools import get_realspace_hs, get_kspace_hs, \
     tri2full, remove_pbc
from gpaw.mpi import world
from gpaw.utilities.lapack import diagonalize
from gpaw.utilities import pack
from gpaw.lcao.IntCtrl import IntCtrl
from gpaw.lcao.CvgCtrl import *;print 'XXX??????'
from gpaw.utilities.timing import Timer

class PathInfo:
    def __init__(self, type):
    # equilibrium ------ eq
    # non-equilibrium ---ne
        self.type = type
        self.num = 0
        self.energy = []
        self.weight = []
        self.nres = 0
        self.sigma = [[], []]
        if type == 'eq':
            self.fermi_factor = []
        elif type == 'ne':
            self.fermi_factor = [[], []]
        else:
            raise TypeError('unkown PathInfo type')

    def add(self, elist, wlist, flist, siglist):
        self.num += len(elist)
        self.energy += elist
        self.weight += wlist
        if self.type == 'eq':
            self.fermi_factor += flist
        elif self.type == 'ne':
            for i in [0, 1]:
                self.fermi_factor[i] += flist[i]
        else:
            raise TypeError('unkown PathInfo type')
        for i in [0, 1]:
            self.sigma[i] += siglist[i]

    def set_nres(self, nres):
        self.nres = nres
    
class GPAWTransport:
    
    def __init__(self, atoms, pl_atoms, pl_cells, d=0):
        self.atoms = atoms
        print 'if not self.atoms.calc.initialized: self.atoms.calc.initialize(atoms)'
        self.pl_atoms = pl_atoms
        self.pl_cells = pl_cells
        self.d = d
        self.atoms_l = [None,None]
        self.h_skmm = None
        self.s_kmm = None
        self.h1_skmm = None
        self.s1_kmm = None
        self.h2_skmm = None
        self.s2_kmm = None

    def write_left_lead(self,filename):
        self.update_lead_hamiltonian(0)

    def write(self, filename):
        self.update_lead_hamiltonian(0)

        pl1 = self.h1_skmm.shape[-1]
        h1 = np.zeros((2*pl1, 2 * pl1), complex)
        s1 = np.zeros((2*pl1, 2 * pl1), complex)

        atoms1 = self.atoms_l[0]
        calc1 = atoms1.calc
        R_c = [0,0,0] 
        h1_sii, s1_ii = get_realspace_hs(self.h1_skmm,
                                         self.s1_kmm,
                                         calc1.ibzk_kc, 
                                         calc1.weight_k,
                                         R_c=R_c)
        R_c = [0,0,0]
        R_c[self.d] = 1.0
        h1_sij, s1_ij = get_realspace_hs(self.h1_skmm,
                                         self.s1_kmm,
                                         calc1.ibzk_kc, 
                                         calc1.weight_k,
                                         R_c=R_c)

        h1[:pl1, :pl1] = h1_sii[0]
        h1[pl1:2 * pl1, pl1:2 * pl1] = h1_sii[0]
        h1[:pl1, pl1:2 * pl1] = h1_sij[0]
        tri2full(h1, 'U')
        
        s1[:pl1,:pl1] = s1_ii
        s1[pl1:2*pl1,pl1:2*pl1] = s1_ii
        s1[:pl1,pl1:2*pl1] = s1_ij
        tri2full(s1, 'U')
        
        if calc1.wfs.world.rank == 0:
            print "Dumping lead 1 hamiltonian..."
            fd = file('lead1_' + filename, 'wb')
            pickle.dump((h1, s1), fd, 2)
            fd.close()

        world.barrier()
        
        self.update_lead_hamiltonian(1) 
        pl2 = self.h2_skmm.shape[-1]
        h2 = np.zeros((2 * pl2, 2 * pl2), complex)
        s2 = np.zeros((2 * pl2, 2 * pl2), complex)

        atoms2 = self.atoms_l[1]
        calc2 = atoms2.calc
        
        h2_sii, s2_ii = get_realspace_hs(self.h2_skmm,
                                         self.s2_kmm,
                                         calc2.ibzk_kc, 
                                         calc2.weight_k,
                                         R_c=(0,0,0))
        R_c = [0,0,0]
        R_c[self.d] = 1.0

        h2_sij, s2_ij = get_realspace_hs(self.h2_skmm,
                                         self.s2_kmm,
                                         calc1.ibzk_kc, 
                                         calc1.weight_k,
                                         R_c=R_c)


        h2[:pl2,:pl2] = h2_sii[0]
        h2[pl2:2*pl2,pl2:2*pl2] = h2_sii[0]
        h2[:pl2,pl2:2*pl2] = h2_sij[0]
        tri2full(h2,'U')
        
        s2[:pl2, :pl2] = s2_ii
        s2[pl2:2*pl2, pl2:2*pl2] = s2_ii
        s2[:pl2, pl2:2*pl2] = s2_ij
        tri2full(s2, 'U')

        if calc2.wfs.world.rank == 0:
            print "Dumping lead 2 hamiltonian..."
            fd = file('lead2_'+filename,'wb')
            pickle.dump((h2,s2),fd,2)
            fd.close()

        world.barrier()
        
        del self.atoms_l

        self.update_scat_hamiltonian()
        nbf_m = self.h_skmm.shape[-1]
        nbf = nbf_m + pl1 + pl2
        h = np.zeros((nbf, nbf), complex)
        s = np.zeros((nbf, nbf), complex)
        
        h_mm = self.h_skmm[0,0]
        s_mm = self.s_kmm[0]
        atoms = self.atoms
        remove_pbc(atoms, h_mm, s_mm, self.d)

        h[:2*pl1,:2*pl1] = h1
        h[-2*pl2:,-2*pl2:] = h2
        h[pl1:-pl2,pl1:-pl2] = h_mm

        s[:2*pl1,:2*pl1] = s1
        s[-2*pl2:,-2*pl2:] = s2
        s[pl1:-pl2,pl1:-pl2] = s_mm
  
        if atoms.calc.wfs.world.rank == 0:
            print "Dumping scat hamiltonian..."
            fd = file('scat_'+filename,'wb')
            pickle.dump((h,s),fd,2)
            fd.close()
        world.barrier()

    def update_lead_hamiltonian(self, l, flag=0):
        # flag: 0 for calculation, 1 for restart
        if flag == 0:
            self.atoms_l[l] = self.get_lead_atoms(l)
            atoms = self.atoms_l[l]
            atoms.get_potential_energy()
            atoms.calc.write('lead' + str(l) + '.gpw')
            if l == 0:
                self.h1_skmm, self.s1_kmm = self.get_hs(atoms)
                fd = file('leadhs0','wb')
                pickle.dump((self.h1_skmm, self.s1_kmm, self.nxklead), fd, 2)            
                fd.close()            
            elif l == 1:
                self.h2_skmm, self.s2_kmm = self.get_hs(atoms)
                fd = file('leadhs1','wb')
                pickle.dump((self.h2_skmm, self.s2_kmm, self.nxklead), fd, 2)
                fd.close()
        else:
            atoms, calc = restart_gpaw('lead' + str(l) + '.gpw')
            calc.set_positions()
            self.atoms_l[l] = atoms
            if l == 0:        
                fd = file('leadhs0','r') 
                self.h1_skmm, self.s1_kmm, self.nxklead = pickle.load(fd)
                fd.close()
            elif l == 1:
                fd = file('leadhs1','r') 
                self.h2_skmm, self.s2_kmm, self.nxklead = pickle.load(fd)
                fd.close()

    def update_scat_hamiltonian(self, restart=False):
        if not restart:
            atoms = self.atoms
            atoms.get_potential_energy()
            atoms.calc.write('scat.gpw')
            self.h_skmm, self.s_kmm = self.get_hs(atoms)
            fd = file('scaths', 'wb')
            pickle.dump((self.h_skmm, self.s_kmm), fd, 2)
            fd.close()
            calc = atoms.calc
            fd = file('nct_G.dat', 'wb')
            pickle.dump(calc.density.nct_G, fd, 2)
            fd.close()
            self.nct_G = np.copy(calc.density.nct_G)
                        
        else:
            atoms, calc = restart_gpaw('scat.gpw')
            calc.set_positions()
            self.atoms = atoms
            fd = file('scaths', 'r')
            self.h_skmm, self.s_kmm = pickle.load(fd)
            fd.close()
            fd = file('nct_G.dat', 'r')
            self.nct_G = pickle.load(fd)
            fd.close()
            
    def get_hs(self, atoms):
        calc = atoms.calc
        wfs = calc.wfs
        Ef = calc.get_fermi_level()
        eigensolver = wfs.eigensolver
        ham = calc.hamiltonian
        S_qMM = wfs.S_qMM.copy()
        for S_MM in S_qMM:
            tri2full(S_MM)
        H_sqMM = np.empty((wfs.nspins,) + S_qMM.shape, complex)
        for kpt in wfs.kpt_u:
            eigensolver.calculate_hamiltonian_matrix(ham, wfs, kpt)
            H_MM = eigensolver.H_MM
            tri2full(H_MM)
            H_MM *= Hartree
            H_MM -= Ef * S_qMM[kpt.q]
            H_sqMM[kpt.s, kpt.q] = H_MM

        return H_sqMM, S_qMM

    def get_lead_atoms(self, l):
        """l: 0, 1 correpsonding to left, right """
        atoms = self.atoms.copy()
        atomsl = atoms[self.pl_atoms[l]]
        atomsl.cell = self.pl_cells[l]
        atomsl.center() # ???
        atomsl.set_calculator(self.get_lead_calc(l))
        return atomsl

    def get_lead_calc(self, l, nkpts=21):
        p = self.atoms.calc.input_parameters.copy()
        p['nbands'] = None
        kpts = list(p['kpts'])
        if nkpts == 0:
            kpts[self.d] = 2 * int(17.0 / self.pl_cells[l][self.d]) + 1
        else:
            kpts[self.d] = nkpts
        self.nxklead = kpts[self.d]
        p['kpts'] = kpts
        if 'mixer' in p: # XXX Works only if spin-paired
            p['mixer'] = Mixer(0.1, 5, metric='new', weight=100.0)
        if 'txt' in p and p['txt'] != '-':
            p['txt'] = 'lead%i_' % (l + 1) + p['txt']
        return GPAW(**p)

    def read(self, filename):
        h, s = pickle.load(file('scat_' + filename))
        h1, s1 = pickle.load(file('lead1_' + filename))
        h2, s2 = pickle.load(file('lead2_' + filename))
        pl1 = len(h1) / 2 
        pl2 = len(h2) / 2
        self.h_skmm = h[:]
        self.s_kmm = s[:]
        self.h1_skmm = h1[:]
        self.s1_kmm = s1[:]
        self.h2_skmm = h2[:]
        self.s2_kmm = s2[:]
        self.atoms_l[0] = self.get_lead_atoms(0)
        self.atoms_l[1] = self.get_lead_atoms(1)
        
    def negf_prepare(self, filename, flag=0):
        # flag: 0 for calculation, 1 for restart
        self.update_lead_hamiltonian(0, flag)

        atoms1 = self.atoms_l[0]
        kpts = atoms1.calc.wfs.ibzk_kc

        self.nspins = self.h1_skmm.shape[0]
        self.nblead = self.h1_skmm.shape[-1]
        self.nyzk = kpts.shape[0] / self.nxklead        
        
        nxk = self.nxklead
        weight = np.array([1.0 / nxk] * nxk )
        xkpts = self.pick_out_xkpts(nxk, kpts)
        self.check_edge(self.s1_kmm, xkpts, weight)
        self.d1_skmm = self.generate_density_matrix(0, flag)        
        self.initial_lead(0)
       

        self.update_lead_hamiltonian(1, flag)
        self.d2_skmm = self.generate_density_matrix(1, flag)
        self.initial_lead(1)

        p = self.atoms.calc.input_parameters.copy()           
        self.nxkmol = p['kpts'][0]  
        self.update_scat_hamiltonian(flag)
        self.nbmol = self.h_skmm.shape[-1]
        self.d_skmm = self.generate_density_matrix(2, flag)
        self.initial_mol()        
 
        self.edge_density_mm = self.calc_edge_charge(self.d_syzkmm_ij,
                                                              self.s_yzkmm_ij)
        self.edge_charge = np.empty([self.nspins, self.nyzk])
        
        for i in range(self.nspins):
            for j in range(self.nyzk):
                self.edge_charge[i, j] = np.trace(self.edge_density_mm[i, j])
                print 'edge_charge[',i,']=', self.edge_charge[i, j]

    def initial_lead(self, lead):
        nxk = self.nxklead
        kpts = self.atoms_l[lead].calc.wfs.ibzk_kc
        if lead == 0:
            self.h1_syzkmm = self.substract_yzk(nxk, kpts, self.h1_skmm, 'h')
            self.s1_yzkmm = self.substract_yzk(nxk, kpts, self.s1_kmm)
            self.h1_syzkmm_ij = self.substract_yzk(nxk, kpts, self.h1_skmm, 'h',
                                               [1.0,0,0])
            self.s1_yzkmm_ij = self.substract_yzk(nxk, kpts, self.s1_kmm, 's',
                                               [1.0,0,0])
            self.d1_syzkmm = self.substract_yzk(nxk, kpts, self.d1_skmm, 'h')
            self.d1_syzkmm_ij = self.substract_yzk(nxk, kpts, self.d1_skmm, 'h',
                                               [1.0,0,0])
        elif lead == 1:
            self.h2_syzkmm = self.substract_yzk(nxk, kpts, self.h2_skmm, 'h')
            self.s2_yzkmm = self.substract_yzk(nxk, kpts, self.s2_kmm)
            self.h2_syzkmm_ij = self.substract_yzk(nxk, kpts, self.h2_skmm, 'h',
                                               [-1.0,0,0])
            self.s2_yzkmm_ij = self.substract_yzk(nxk, kpts, self.s2_kmm, 's',
                                              [-1.0,0,0])            
        else:
            print 'unkown lead index'

    def initial_mol(self):
        nxk = self.nxkmol
        kpts = self.atoms.calc.wfs.ibzk_kc
        self.h_syzkmm = self.substract_yzk(nxk, kpts, self.h_skmm, 'h')
        self.s_yzkmm = self.substract_yzk(nxk, kpts, self.s_kmm)
        self.s_yzkmm_ij = self.substract_yzk(nxk, kpts, self.s_kmm, 's',
                                            [1.0,0,0])
        self.d_syzkmm = self.substract_yzk(nxk, kpts, self.d_skmm, 'h')
        self.d_syzkmm_ij = self.fill_density_matrix()

    def substract_yzk(self, nxk, kpts, k_mm, hors='s', position=[0, 0, 0]):
        nyzk = self.nyzk
        weight = np.array([1.0 / nxk] * nxk )
        if hors not in 'hs':
            raise KeyError('hors should be h or s!')
        if hors == 'h':
            dim = k_mm.shape[:]
            dim = (dim[0],) + (dim[1] / nxk,) + dim[2:]
            yzk_mm = np.empty(dim, complex)
            dim = (dim[0],) + (nxk,) + dim[2:]
            xk_mm = np.empty(dim, complex)
        elif hors == 's':
            dim = k_mm.shape[:]
            dim = (dim[0] / nxk,) + dim[1:]
            yzk_mm = np.empty(dim, complex)
            dim = (nxk,) + dim[1:]
            xk_mm = np.empty(dim, complex)
        n = 0
        xkpts = self.pick_out_xkpts(nxk, kpts)
        for i in range(nyzk):
            n = i
            for j in range(nxk):
                if hors == 'h':
                    xk_mm[:, j] = np.copy(k_mm[:, n])
                elif hors == 's':
                    xk_mm[j] = np.copy(k_mm[n])
                n += nyzk
            if hors == 'h':
                yzk_mm[:, i] = get_realspace_hs(xk_mm, None,
                                               xkpts, weight, position)
            elif hors == 's':
                yzk_mm[i] = get_realspace_hs(None, xk_mm,
                                                   xkpts, weight, position)
        return yzk_mm   
            
    def check_edge(self, k_mm, xkpts, weight):
        tolx = 1e-6
        position = [2,0,0]
        nbf = k_mm.shape[-1]
        nxk = self.nxklead
        nyzk = self.nyzk
        xk_mm = np.empty([nxk, nbf, nbf], complex)
      
        num = 0       
        for i in range(nxk):
            xk_mm[i] = np.copy(k_mm[num])
            num += nyzk

        r_mm = np.empty([nbf, nbf], complex)
        r_mm = get_realspace_hs(None, xk_mm, xkpts, weight, position)
        matmax = np.max(abs(r_mm))

        if matmax > tolx:
            print 'Warning*: the principle layer should be lagger, \
                                                          matmax=%f' % matmax
    
    def calc_edge_charge(self, d_syzkmm_ij, s_yzkmm_ij):
        nspins = self.nspins
        nyzk = self.nyzk
        nbf = s_yzkmm_ij.shape[-1]
        edge_charge_mm = np.zeros([nspins, nyzk, nbf, nbf])
        for i in range(nspins):
            for j in range(nyzk):
                edge_charge_mm[i, j] += np.dot(d_syzkmm_ij[i, j],
                                               s_yzkmm_ij[i].T.conj())
                edge_charge_mm[i, j] += np.dot(d_syzkmm_ij[i, j].T.conj(),
                                               s_yzkmm_ij[i])
        return edge_charge_mm

    def pick_out_xkpts(self, nxk, kpts):
        nyzk = self.nyzk
        xkpts = np.zeros([nxk, 3])
        num = 0
        for i in range(nxk):
            xkpts[i, 0] = kpts[num, 0]
            num += nyzk
        return xkpts

    def generate_density_matrix(self, lead, flag=0):
        nxk = self.nxklead
        nyzk = self.nyzk
        if flag == 0:
            if lead == 0:
                calc = self.atoms_l[0].calc
                dim = self.h1_skmm.shape
                filename = 'd1_skmm.dat'
            elif lead == 1:
                calc = self.atoms_l[1].calc
                dim = self.h2_skmm.shape
                filename = 'd2_skmm.dat'
            elif lead == 2:
                calc = self.atoms.calc
                dim = self.h_skmm.shape
                nxk = self.nxkmol
                filename = 'd_skmm.dat'
            else:
                raise KeyError('invalid lead index in \
                                                     generate_density_matrix')
            d_skmm = np.empty(dim, complex)
            for kpt in calc.wfs.kpt_u:
                C_nm = kpt.C_nM
                f_nn = np.diag(kpt.f_n)
                d_skmm[kpt.s, kpt.k] = np.dot(C_nm.T.conj(),
                                              np.dot(f_nn, C_nm)) * nxk * nyzk
            fd = file(filename,'wb')
            pickle.dump(d_skmm, fd, 2)
            fd.close()
        else:
            if lead == 0:
                filename = 'd1_skmm.dat'
            elif lead == 1:
                filename = 'd2_skmm.dat'
            elif lead == 2:
                filename = 'd_skmm.dat'
            else:
                raise KeyError('invalid lead index in \
                                                     generate_density_matrix')
            fd = file(filename,'r')
            d_skmm = pickle.load(fd)
            fd.close()
        return d_skmm
    
    def fill_density_matrix(self):
        pl1 = self.h1_skmm.shape[-1]
        nbmol = self.h_skmm.shape[-1]
        nyzk = self.nyzk
        nspins = self.nspins
        d_syzkmm_ij = np.zeros([nspins, nyzk, nbmol, nbmol])
        d_syzkmm_ij[:, :, -pl1:, :pl1] = np.copy(self.d1_syzkmm_ij)
        return d_syzkmm_ij

    def boundary_check(self):
        tol = 1e-4
        pl1 = self.h1_skmm.shape[-1]
        matdiff = self.h_syzkmm[0, :, :pl1, :pl1] - self.h1_syzkmm[0]
        if self.nspins == 2:
             matdiff1 = self.h_syzkmm[1, :, :pl1, :pl1] - self.h1_syzkmm[1]
             float_diff = np.max(abs(matdiff - matdiff1)) 
             if float_diff > 1e-6:
                  print 'Warning!, float_diff between spins %f' % float_diff
        if self.bias == 0:
            ediff = np.sum(np.diag(matdiff[0])) / np.sum(
                                                   np.diag(self.s1_yzkmm[0]))
            print_diff = np.max(abs(matdiff - ediff * self.s1_yzkmm))
        else:
            ediff = matdiff[0,0,0]/ self.s1_yzkmm[0,0,0]
            print_diff = abs(ediff)
        if print_diff > tol:
            print 'Warning*: hamiltonian boundary difference %f' % print_diff
            
        nspins = len(self.h_syzkmm)
        for i in range(nspins):
            self.h_syzkmm[i] += (self.gate - ediff) * self.s_yzkmm
        
        matdiff = self.d_syzkmm[:, :, :pl1, :pl1] - self.d1_syzkmm
        if self.bias == 0:
            print_diff = np.max(abs(matdiff))
        else:
            print_diff= matdiff[0,0,0,0]
        if print_diff > tol:
            print 'Warning*: density boundary difference %f' % print_diff
            
    def get_selfconsistent_hamiltonian(self, bias=0, gate=0, verbose=0):
        #----- initialize---------
        self.verbose = verbose
        self.bias = bias
        self.gate = gate
        self.kt = self.atoms.calc.occupations.kT * Hartree
        fermi = 0
        if self.verbose:
            print 'prepare for selfconsistent calculation'
            print '[bias, gate] =', bias, gate, 'V'
            print 'lead_fermi_level =', fermi

        intctrl = IntCtrl(self.kt, fermi, bias)
        fcvg = CvgCtrl()
        dcvg = CvgCtrl()
        inputinfo = {'fasmethodname':'SCL_None', 'fmethodname':'CVG_None',
                     'falpha': 0.1, 'falphascaling':0.1, 'ftol':1e-3,
                     'fallowedmatmax':1e-4, 'fndiis':10, 'ftolx':1e-5,
                     'fsteadycheck': False,
                     'dasmethodname':'SCL_None', 'dmethodname':'CVG_broydn',
                     'dalpha': 0.01, 'dalphascaling':0.1, 'dtol':1e-3,
                     'dallowedmatmax':1e-4, 'dndiis':6, 'dtolx':1e-5,
                     'dsteadycheck': False}
        fcvg(inputinfo, 'f', dcvg)
        dcvg(inputinfo, 'd', fcvg)
        if self.verbose:
            print 'mix_factor =', inputinfo['dalpha']
        nspins = self.nspins
        nyzk = self.nyzk
        nbmol = self.nbmol
        den = np.empty([nspins, nyzk, nbmol, nbmol], complex)
        denocc = np.empty([nspins, nyzk, nbmol, nbmol], complex)

        self.selfenergies = [LeadSelfEnergy((self.h1_syzkmm[0,0],
                                                              self.s1_yzkmm[0]), 
                                            (self.h1_syzkmm_ij[0,0],
                                                           self.s1_yzkmm_ij[0]),
                                            (self.h1_syzkmm_ij[0,0],
                                                           self.s1_yzkmm_ij[0]),
                                             0),
                             LeadSelfEnergy((self.h2_syzkmm[0,0],
                                                              self.s2_yzkmm[0]), 
                                            (self.h2_syzkmm_ij[0,0],
                                                           self.s2_yzkmm_ij[0]),
                                            (self.h2_syzkmm_ij[0,0],
                                                           self.s2_yzkmm_ij[0]),
                                             0)]

        self.greenfunction = GreenFunction(selfenergies=self.selfenergies,
                                           H=self.h_syzkmm[0,0],
                                           S=self.s_yzkmm[0],
                                           eta=0)
        self.selfenergies[0].set_bias(bias/2.0)
        self.selfenergies[1].set_bias(-bias/2.0)
        
        self.eqpathinfo = []
        self.nepathinfo = []
       
        for s in range(nspins):
            self.eqpathinfo.append([])
            self.nepathinfo.append([])
            for k in range(nyzk):
                self.eqpathinfo[s].append(PathInfo('eq'))
                self.nepathinfo[s].append(PathInfo('ne'))

        self.boundary_check() 

        #-------get the path --------    
        for s in range(nspins):
            for k in range(nyzk):      
                den[s, k] = self.get_eqintegral_points(intctrl, s, k)
                denocc[s, k]= self.get_neintegral_points(intctrl, s, k)
        
        #-------begin the SCF ----------         
        self.step = 0
        self.cvgflag = 0
        calc = self.atoms.calc    
        timer = Timer()
        nxk = self.nxkmol
        kpts = self.atoms.calc.wfs.ibzk_kc
        while self.cvgflag == 0:
            f_syzkmm = fcvg.matcvg(self.h_syzkmm)
            timer.start('Fock2Den')
            for s in range(nspins):
                for k in range(nyzk):
                    self.d_syzkmm[s, k] = self.fock2den(intctrl, f_syzkmm, s, k)
            timer.stop('Fock2Den')
            if self.verbose:
                print'Fock2Den', timer.gettime('Fock2Den'), 'second'
            if nspins == 1:
                self.d_syzkmm = np.real(self.d_syzkmm) * 2
            else:
                self.d_syzkmm = np.real(self.d_syzkmm)

            d_syzkmm_out = dcvg.matcvg(self.d_syzkmm)
            self.cvgflag = fcvg.bcvg and dcvg.bcvg
            timer.start('Den2Fock')            
            self.h_skmm = self.den2fock(d_syzkmm_out)
            timer.stop('Den2Fock')
            if self.verbose:
                print'Den2Fock', timer.gettime('Den2Fock'),'second'
            self.h_syzkmm = self.substract_yzk(nxk, kpts , self.h_skmm, 'h')
            self.boundary_check()
            self.step +=  1
        return 1
    
    def get_eqintegral_points(self, intctrl, s, k):
        global zint, fint, tgtint, cntint
        maxintcnt = 500
        nblead = self.nblead
        nbmol = self.nbmol
        den = np.zeros([nbmol, nbmol])
        
        zint = [0] * maxintcnt
        fint = []
        tgtint = np.empty([2, maxintcnt, nblead, nblead], complex)
        cntint = -1

        self.selfenergies[0].h_ii = self.h1_syzkmm[s, k]
        self.selfenergies[0].s_ii = self.s1_yzkmm[k]
        self.selfenergies[0].h_ij = self.h1_syzkmm_ij[s, k]
        self.selfenergies[0].s_ij = self.s1_yzkmm_ij[k]
        self.selfenergies[0].h_im = self.h1_syzkmm_ij[s, k]
        self.selfenergies[0].s_im = self.s1_yzkmm_ij[k]

        self.selfenergies[1].h_ii = self.h2_syzkmm[s, k]
        self.selfenergies[1].s_ii = self.s2_yzkmm[k]
        self.selfenergies[1].h_ij = self.h2_syzkmm_ij[s, k]
        self.selfenergies[1].s_ij = self.s2_yzkmm_ij[k]
        self.selfenergies[1].h_im = self.h2_syzkmm_ij[s, k]
        self.selfenergies[1].s_im = self.s2_yzkmm_ij[k]

        self.greenfunction.H = self.h_syzkmm[s, k]
        self.greenfunction.S = self.s_yzkmm[k]

        #--eq-Integral-----
        [grsum, zgp, wgp, fcnt] = function_integral(self,
                                                        intctrl.eqintpath,
                                                        intctrl.eqinttol,
                                                       0, 'eqInt', intctrl)
        # res Calcu
        grsum += self.calgfunc(intctrl.eqresz, 'resInt', intctrl)    
        tmp = np.resize(grsum, [nbmol, nbmol])
        den += 1.j * (tmp - tmp.T.conj()) / np.pi / 2

        # --sort SGF --
        nres = len(intctrl.eqresz)
        self.eqpathinfo[s][k].set_nres(nres)
        elist = zgp + intctrl.eqresz
        wlist = wgp + [1.0] * nres

        fcnt = len(elist)
        sgforder = [0] * fcnt
        for i in range(fcnt):
            sgferr = np.min(abs(elist[i] - np.array(zint[:cntint + 1 ])))
                
            sgforder[i] = np.argmin(abs(elist[i]
                                              - np.array(zint[:cntint + 1])))
            if sgferr > 1e-15:
                print '--Warning--: SGF not Found. eqzgp[', i, ']=', elist[i]
                                             
        
        flist = fint[:]
        siglist = [[],[]]
        for i, num in zip(range(fcnt), sgforder):
            flist[i] = fint[num]
 
        sigma= np.empty([nblead, nblead], complex)
        for i in [0, 1]:
            for j, num in zip(range(fcnt), sgforder):
                sigma = tgtint[i, num]
                siglist[i].append(sigma)
        self.eqpathinfo[s][k].add(elist, wlist, flist, siglist)    
   
        if self.verbose: 
            if self.nspins == 1:
                print 'Eq_k[',k,']=', np.trace(np.dot(den,
                                                  self.s_yzkmm[k])) * 2
            else:
                print 'Eq_sk[', s , k , ']=', np.trace(np.dot(den,
                                                 self.s_yzkmm[k]))
        del fint, tgtint, zint
        return den 
    
    def get_neintegral_points(self, intctrl, s, k):
        # -----initialize-------
        intpathtol = 1e-8
        nblead = self.nblead  
        nbmol = self.nbmol
        nyzk = self.nyzk
        denocc = np.zeros([nbmol, nbmol], complex)
        denvir = np.zeros([nbmol, nbmol], complex)
        denloc = np.zeros([nbmol, nbmol], complex)

        # -------Init Global Value -----------
        global zint, fint, tgtint, cntint 
        maxintcnt = 500

        # -------ne--Integral---------------
        zint = [0] * maxintcnt
        tgtint = np.empty([2, maxintcnt, nblead, nblead], complex)

        self.selfenergies[0].h_ii = self.h1_syzkmm[s, k]
        self.selfenergies[0].s_ii = self.s1_yzkmm[k]
        self.selfenergies[0].h_ij = self.h1_syzkmm_ij[s, k]
        self.selfenergies[0].s_ij = self.s1_yzkmm_ij[k]
        self.selfenergies[0].h_im = self.h1_syzkmm_ij[s, k]
        self.selfenergies[0].s_im = self.s1_yzkmm_ij[k]

        self.selfenergies[1].h_ii = self.h2_syzkmm[s, k]
        self.selfenergies[1].s_ii = self.s2_yzkmm[k]
        self.selfenergies[1].h_ij = self.h2_syzkmm_ij[s, k]
        self.selfenergies[1].s_ij = self.s2_yzkmm_ij[k]
        self.selfenergies[1].h_im = self.h2_syzkmm_ij[s, k]
        self.selfenergies[1].s_im = self.s2_yzkmm_ij[k]

        self.greenfunction.H = self.h_syzkmm[s, k]
        self.greenfunction.S = self.s_yzkmm[k]

        for n in range(1, len(intctrl.neintpath)):
            cntint = -1
            fint = [[],[]]
            if intctrl.kt <= 0:
                neintpath = [intctrl.neintpath[n - 1] + intpathtol,
                             intctrl.neintpath[n] - intpathtol]
            else:
                neintpath = [intctrl.neintpath[n-1],intctrl.neintpath[n]]
            if intctrl.neintmethod== 1:
    
                # ----Auto Integral------
                sumga, zgp, wgp, nefcnt = function_integral(self,
                                                            neintpath,
                                                        intctrl.neinttol,
                                                            0, 'neInt',
                                                              intctrl)
    
                nefcnt = len(zgp)
                sgforder = [0] * nefcnt
                for i in range(nefcnt):
                    sgferr = np.min(np.abs(zgp[i] - np.array(
                                                     zint[:cntint + 1 ])))
                    sgforder[i] = np.argmin(np.abs(zgp[i] -
                                            np.array(zint[:cntint + 1])))
                    if sgferr > 1e-15:
                        print '--Warning: SGF not found, nezgp[', \
                                                           i, ']=', zgp[i]
            else:
                # ----Manual Integral------
                nefcnt = max(ceil(np.real(neintpath[1] -
                                                neintpath[0]) /
                                                intctrl.neintstep) + 1, 6)
                nefcnt = int(nefcnt)
                zgp = np.linspace(neintpath[0], neintpath[1], nefcnt)
                zgp = list(zgp)
                wgp = np.array([3.0 / 8, 7.0 / 6, 23.0 / 24] + [1] *
                         (nefcnt - 6) + [23.0 / 24, 7.0 / 6, 3.0 / 8]) * (
                                                          zgp[1] - zgp[0])
                wgp = list(wgp)
                sgforder = range(nefcnt)
                sumga = np.zeros([nbmol, nbmol], complex)
                for i in range(nefcnt):
                    if iscalcuocc:
                        sumga += self.calgfunc(zgp[i], 'neInt',
                                                      intctrl) * wgp[i]
                else:
                        self.calgfunc(zgp[i],'neInt', intctrl)
            denocc += sumga[0] / np.pi / 2
   
            flist = [[],[]]
            siglist = [[],[]]
            sigma= np.empty([nblead, nblead], complex)
            for i in [0, 1]:
                for j, num in zip(range(nefcnt), sgforder):
                    fermi_factor = fint[i][num]
                    sigma = tgtint[i, num]
                    flist[i].append(fermi_factor)    
                    siglist[i].append(sigma)
            self.nepathinfo[s][k].add(zgp, wgp, flist, siglist)
        # Loop neintpath
        neq = np.trace(np.dot(denocc, self.s_yzkmm[k]))
        if self.verbose:
            if self.nspins == 1:
                print 'NEQ_k[', k, ']=',np.real(neq) * 2, 'ImagPart=',\
                                                            np.imag(neq) * 2
            else:
                print 'NEQ_sk[', s,  k, ']=',np.real(neq), 'ImagPart=',\
                                                         np.imag(neq)
        del zint, tgtint
        if len(intctrl.neintpath) >= 2:
            del fint
        return denocc
        
    def calgfunc(self, zp, calcutype, intctrl):			 
        #calcutype = 
        #  - 'eqInt':  gfunc[Mx*Mx,nE] (default)
        #  - 'neInt':  gfunc[Mx*Mx,nE]
        #  - 'resInt': gfunc[Mx,Mx] = gr * fint
        #              fint = -2i*pi*kt
        global zint, fint, tgtint, cntint
        
        sgftol = 1e-10
        stepintcnt = 100
        nlead = 2
        nblead = self.nblead
        nbmol = self.nbmol
        gamma = np.zeros([nlead, nbmol, nbmol], complex)
        
        if type(zp) == list:
            pass
        elif type(zp) == np.ndarray:
            pass
        else:
            zp = [zp]
        nume = len(zp)
        if calcutype == 'resInt':
            gfunc = np.zeros([nbmol, nbmol], complex)
        else:
            gfunc = np.zeros([nume, nbmol, nbmol], complex)
        for i in range(nume):
            sigma = np.zeros([nbmol, nbmol], complex)
            if cntint + 1 >= len(zint):
                zint = zint + [0] * stepintcnt
                tmp = tgtint.shape[1]
                tmptgtint = np.copy(tgtint)
                tgtint = np.empty([2, tmp + stepintcnt, nblead, nblead],
                                                                      complex)
                tgtint[:, :tmp] = np.copy(tmptgtint)
                tgtint[:, tmp:tmp + stepintcnt] = np.zeros([2,
                                                 stepintcnt, nblead, nblead])
            cntint += 1
            zint[cntint] = zp[i]

            for j in [0, 1]:
                tgtint[j, cntint] = self.selfenergies[j](zp[i])
            
            sigma[:nblead, :nblead] += tgtint[0, cntint]
            sigma[-nblead:, -nblead:] += tgtint[1, cntint]
            gamma[0, :nblead, :nblead] = self.selfenergies[0].get_lambda(
                                                                        zp[i])
            gamma[1, -nblead:, -nblead:] = self.selfenergies[1].get_lambda(
                                                                        zp[i])
            gr = self.greenfunction.calculate(zp[i], sigma)       
        
            # --ne-Integral---
            if calcutype == 'neInt':
                gammaocc = np.zeros([nbmol, nbmol], complex)
                for n in [0, 1]:
                    fint[n].append(  fermidistribution(zp[i] -
                                         intctrl.leadfermi[n], intctrl.kt) - 
                                         fermidistribution(zp[i] -
                                             intctrl.minfermi, intctrl.kt) )
                    gammaocc += gamma[n] * fint[n][cntint]
                aocc = np.dot(gr, gammaocc)
                aocc = np.dot(aocc, gr.T.conj())
               
                gfunc[i] = aocc
            # --res-Integral --
            elif calcutype == 'resInt':
                fint.append(-2.j * np.pi * intctrl.kt)
                gfunc += gr * fint[cntint]
            #--eq-Integral--
            else:
                if intctrl.kt <= 0:
                    fint.append(1.0)
                else:
                    fint.append(fermidistribution(zp[i] -
                                                intctrl.minfermi, intctrl.kt))
                gfunc[i] = gr * fint[cntint]    
        return gfunc        

    def fock2den(self, intctrl, f_syzkmm, s, k):
        nyzk = self.nyzk
        nblead = self.nblead
        nbmol = self.nbmol
        den = np.zeros([nbmol, nbmol], complex)
        sigmatmp = np.zeros([nblead, nblead], complex)

        # missing loc states
        eqzp = self.eqpathinfo[s][k].energy
        nezp = self.nepathinfo[s][k].energy
        
        self.greenfunction.H = f_syzkmm[s, k]
        self.greenfunction.S = self.s_yzkmm[k]
        for i in range(len(eqzp)):
            sigma = np.zeros([nbmol, nbmol], complex)  
            sigma[:nblead, :nblead] += self.eqpathinfo[s][k].sigma[0][i]
            sigma[-nblead:, -nblead:] += self.eqpathinfo[s][k].sigma[1][i]
            gr = self.greenfunction.calculate(eqzp[i], sigma)
            fermifactor = self.eqpathinfo[s][k].fermi_factor[i]
            weight = self.eqpathinfo[s][k].weight[i]
            den += gr * fermifactor * weight
        den = 1.j * (den - den.T.conj()) / np.pi / 2
        for i in range(len(nezp)):
            sigma = np.zeros([nbmol, nbmol], complex)
            sigmalesser = np.zeros([nbmol, nbmol], complex)
            sigma[:nblead, :nblead] += self.nepathinfo[s][k].sigma[0][i]
            sigma[-nblead:, -nblead:] += self.nepathinfo[s][k].sigma[1][i]    
            gr = self.greenfunction.calculate(nezp[i], sigma)
            fermifactor = np.real(self.nepathinfo[s][k].fermi_factor[0][i])
           
            sigmatmp = self.nepathinfo[s][k].sigma[0][i]
            sigmalesser[:nblead, :nblead] += 1.0j * fermifactor * (
                                          sigmatmp - sigmatmp.T.conj())
            fermifactor = np.real(self.nepathinfo[s][k].fermi_factor[0][i])

            sigmatmp = self.nepathinfo[s][k].sigma[1][i] 
            sigmalesser[-nblead:, -nblead:] += 1.0j * fermifactor * (
                                          sigmatmp - sigmatmp.T.conj())       
            glesser = np.dot(sigmalesser, gr.T.conj())
            glesser = np.dot(gr, glesser)
            weight = self.nepathinfo[s][k].weight[i]            
            den += glesser * weight / np.pi / 2
        return den    

    def den2fock(self, d_yzkmm):
        nyzk = self.nyzk
        nbmol = self.nbmol
        atoms = self.atoms
        calc = atoms.calc
        kpt = calc.wfs.kpt_u
        density = calc.density
        
        timer = Timer()
        timer.start('getdensity')
        self.get_density(density, d_yzkmm , kpt)
        timer.stop('getdensity')
        print 'getdensity', timer.gettime('getdensity'), 'seconds'
        calc.update_kinetic()
        calc.hamiltonian.update(density)
        
        linear_potential = np.empty(calc.hamiltonian.vt_sG.shape)
        dimx = linear_potential.shape[0]
        dimyz = linear_potential.shape[1:]
        bias = self.bias / Hartree
        vx = np.linspace(bias/2, -bias/2, dimx)
        for i in range(dimx):
            linear_potential[i] = vx[i] * (np.zeros(dimyz) + 1)

        calc.hamiltonian.vt_sG += linear_potential
        #xcfunc = calc.hamiltonian.xc.xcfunc
        #calc.Enlxc = xcfunc.get_non_local_energy()
        #calc.Enlkin = xcfunc.get_non_local_kinetic_corrections()   
        h_skmm, s_kmm = self.get_hs(atoms)
        return h_skmm
    
    def get_density(self, density, d_syzkmm, kpt_u):
        #Calculate pseudo electron-density based on green function.
        wfs = self.atoms.calc.wfs
        nspins = self.nspins
        nxk = self.nxkmol
        nyzk = self.nyzk
        nbmol = self.nbmol
        relate_layer = 3
        dr_mm = np.zeros([nspins, relate_layer, nyzk,  nbmol, nbmol], complex)
        qr_mm = np.zeros([nspins, nyzk, self.nbmol, self.nbmol])
        
        for s in range(nspins):
            for i in range(relate_layer):
                for j in range(nyzk):
                    if i == 0:
                        dr_mm[s, i, j] = self.d_syzkmm_ij[s, j].T.conj()
                    elif i == 1:
                        dr_mm[s, i, j] += np.copy(d_syzkmm[s, j])
                        qr_mm[s, j] += np.dot(dr_mm[s, i, j], self.s_yzkmm[j])
                    elif i == 2:
                        dr_mm[s, i, j]= np.copy(self.d_syzkmm_ij[s, j])
                    else:
                        pass         
        qr_mm += self.edge_density_mm
        if self.verbose:
            for i in range(nspins):
                print 'spin', i, 'charge on atomic basis', np.diag(np.sum(qr_mm[s],axis=0))            
            print 'total charge'
            print np.trace(np.sum(np.sum(qr_mm, axis=0), axis=0))

        rvector = np.zeros([relate_layer, 3])
        xkpts = self.pick_out_xkpts(nxk, wfs.ibzk_kc)
        for i in range(relate_layer):
            rvector[i, 0] = i - (relate_layer - 1) / 2

        self.d_skmm.shape = (nspins, nxk, nyzk, nbmol, nbmol)
        for s in range(nspins):
            for i in range(nxk):
                for j in range(nyzk):
                    self.d_skmm[s, i, j] = get_kspace_hs(None, dr_mm[s, :, j],
                                                         rvector, xkpts[i])
                    self.d_skmm[s, i, j] = self.d_skmm[s, i, j] / (nxk *
                                                                   nyzk) 

        self.d_skmm.shape = (nspins, nxk * nyzk, nbmol, nbmol)

        for kpt in kpt_u:
            kpt.rho_MM = self.d_skmm[kpt.s, kpt.q]
        density.update(wfs)
        return density

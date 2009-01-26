import pickle

from ase.transport.selfenergy import LeadSelfEnergy
from ase.transport.greenfunction import GreenFunction
from ase.transport.tools import function_integral, fermidistribution
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
from gpaw.lcao.CvgCtrl import CvgCtrl
from gpaw.utilities.timing import Timer

class PathInfo:
    def __init__(self, type):
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

    def update_lead_hamiltonian(self, l, restart=False):
        if not restart:
            self.atoms_l[l] = self.get_lead_atoms(l)
            atoms = self.atoms_l[l]
            atoms.get_potential_energy()
            atoms.calc.write('lead' + str(l) + '.gpw')
            if l == 0:
                self.h1_skmm, self.s1_kmm = self.get_hs(atoms)
                fd = file('leadhs0','wb')
                pickle.dump((self.h1_skmm, self.s1_kmm,
                             self.atoms_l[0].calc.hamiltonian.vt_sG,
                             self.nxklead), fd, 2)            
                fd.close()            
            elif l == 1:
                self.h2_skmm, self.s2_kmm = self.get_hs(atoms)
                fd = file('leadhs1','wb')
                pickle.dump((self.h2_skmm, self.s2_kmm,
                             self.atoms_l[1].calc.hamiltonian.vt_sG,
                             self.nxklead), fd, 2)
                fd.close()
        else:
            atoms, calc = restart_gpaw('lead' + str(l) + '.gpw')
            calc.set_positions()
            self.atoms_l[l] = atoms
            self.atoms_l[l].calc = calc
            if l == 0:        
                fd = file('leadhs0','r') 
                self.h1_skmm, self.s1_kmm, \
                                   self.atoms_l[0].calc.hamiltonian.vt_sG, \
                                               self.nxklead = pickle.load(fd)
                fd.close()
            elif l == 1:
                fd = file('leadhs1','r') 
                self.h2_skmm, self.s2_kmm, \
                                   self.atoms_l[1].calc.hamiltonian.vt_sG, \
                                               self.nxklead = pickle.load(fd)
                fd.close()

    def update_scat_hamiltonian(self, restart=False):
        if not restart:
            atoms = self.atoms
            atoms.get_potential_energy()
            calc = atoms.calc
            calc.write('scat.gpw')
            self.h_skmm, self.s_kmm = self.get_hs(atoms)
            fd = file('scaths', 'wb')
            pickle.dump((self.h_skmm, self.s_kmm, calc.density.nct_G,
                                          calc.hamiltonian.vt_sG), fd, 2)
            fd.close()
                        
        else:
            atoms, calc = restart_gpaw('scat.gpw')
            calc.set_positions()
            self.atoms = atoms
            fd = file('scaths', 'r')
            self.h_skmm, self.s_kmm, calc.density.nct_G, \
                                    calc.hamiltonian.vt_sG = pickle.load(fd)
            fd.close()
            self.atoms.calc = calc
            
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
        atomsl.center()
        atomsl.set_calculator(self.get_lead_calc(l))
        return atomsl

    def get_lead_calc(self, l, nkpts=13):
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
        
    def negf_prepare(self, scat_restart=False, lead_restart=True):
        self.update_lead_hamiltonian(0, lead_restart)

        atoms1 = self.atoms_l[0]
        kpts = atoms1.calc.wfs.ibzk_kc

        self.nspins = self.h1_skmm.shape[0]
        self.nblead = self.h1_skmm.shape[-1]
        self.nyzk = kpts.shape[0] / self.nxklead        
        
        nxk = self.nxklead
        weight = np.array([1.0 / nxk] * nxk )
        xkpts = self.pick_out_xkpts(nxk, kpts)
        self.d1_skmm = self.generate_density_matrix('lead_l', lead_restart)        
        self.check_edge()
        self.initial_lead(0)

        self.update_lead_hamiltonian(1, lead_restart)
        self.d2_skmm = self.generate_density_matrix('lead_r', lead_restart)
        self.initial_lead(1)

        p = self.atoms.calc.input_parameters.copy()           
        self.nxkmol = p['kpts'][0]
        self.update_scat_hamiltonian(scat_restart)
        self.nbmol = self.h_skmm.shape[-1]
        self.d_skmm = self.generate_density_matrix('scat', scat_restart)
        self.initial_mol()        
 
        self.edge_density_mm = self.calc_edge_density(self.d_syzkmm_ij,
                                                              self.s_yzkmm_ij)
        self.edge_charge = np.zeros([self.nspins])
        for i in range(self.nspins):
            for j in range(self.nyzk):
                self.edge_charge[i] += np.trace(self.edge_density_mm[i, j])
            print 'edge_charge[%d]=%f' % (i, self.edge_charge[i])

    def initial_lead(self, lead):
        nspins = self.nspins
        nxk = self.nxklead
        nyzk = self.nyzk
        nblead = self.nblead
        kpts = self.atoms_l[lead].calc.wfs.ibzk_kc
        if lead == 0:
            self.h1_syzkmm = self.substract_yzk(nxk, kpts, self.h1_skmm, 'h')
            self.s1_yzkmm = self.substract_yzk(nxk, kpts, self.s1_kmm)
            self.h1_syzkmm_ij = self.substract_yzk(nxk, kpts, self.h1_skmm,
                                                   'h', [1.0,0,0])
            self.s1_yzkmm_ij = self.substract_yzk(nxk, kpts, self.s1_kmm,
                                                  's', [1.0,0,0])
            self.d1_syzkmm = self.substract_yzk(nxk, kpts, self.d1_skmm, 'h')
            self.d1_syzkmm_ij = self.substract_yzk(nxk, kpts, self.d1_skmm,
                                                   'h', [1.0,0,0])
        elif lead == 1:
            self.h2_syzkmm = self.substract_yzk(nxk, kpts, self.h2_skmm, 'h')
            self.s2_yzkmm = self.substract_yzk(nxk, kpts, self.s2_kmm)
            self.h2_syzkmm_ij = self.substract_yzk(nxk, kpts, self.h2_skmm,
                                                   'h', [-1.0,0,0])
            self.s2_yzkmm_ij = self.substract_yzk(nxk, kpts, self.s2_kmm,
                                                  's', [-1.0,0,0])            
        else:
            raise TypeError('unkown lead index')

    def initial_mol(self):
        nxk = self.nxkmol
        kpts = self.atoms.calc.wfs.ibzk_kc
        self.h_syzkmm = self.substract_yzk(nxk, kpts, self.h_skmm, 'h')
        self.s_yzkmm = self.substract_yzk(nxk, kpts, self.s_kmm)
        self.s_yzkmm_ij = self.substract_yzk(nxk, kpts, self.s_kmm, 's',
                                            [1.0,0,0])
        self.d_syzkmm = self.substract_yzk(nxk, kpts, self.d_skmm, 'h')
        self.d_syzkmm_ij = self.substract_yzk(nxk, kpts, self.d_skmm, 'h', [1.0,0,0])
        self.d_syzkmm_ij2 = self.fill_density_matrix()
        self.edge_nt_sG = self.atoms_l[0].calc.density.nt_sG 

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
            
    def check_edge(self):
        tolx = 1e-6
        position = [2,0,0]
        nxk = self.nxklead
        nyzk = self.nyzk
        kpts = self.atoms_l[0].calc.wfs.ibzk_kc
        s_yzkmm = self.substract_yzk(nxk, kpts,
                                                 self.s1_kmm, 's', position)
        matmax = np.max(abs(s_yzkmm))
        if matmax > tolx:
            print 'Warning*: the principle layer should be lagger, \
                                                          matmax=%f' % matmax        
    
    def calc_edge_density(self, d_syzkmm_ij, s_yzkmm_ij):
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
        return edge_charge_mm / nyzk

    def pick_out_xkpts(self, nxk, kpts):
        nyzk = self.nyzk
        xkpts = np.zeros([nxk, 3])
        num = 0
        for i in range(nxk):
            xkpts[i, 0] = kpts[num, 0]
            num += nyzk
        return xkpts

    def generate_density_matrix(self, region, restart=True):
        nxk = self.nxklead
        nyzk = self.nyzk
        if not restart:
            if region == 'lead_l':
                calc = self.atoms_l[0].calc
                dim = self.h1_skmm.shape
                filename = 'd1_skmm.dat'
            elif region == 'lead_r':
                calc = self.atoms_l[1].calc
                dim = self.h2_skmm.shape
                filename = 'd2_skmm.dat'
            elif region == 'scat':
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
            if region == 'lead_l':
                filename = 'd1_skmm.dat'
            elif region == 'lead_r':
                filename = 'd2_skmm.dat'
            elif region == 'scat':
                filename = 'd_skmm.dat'
            else:
                raise KeyError('invalid lead index in \
                                                     generate_density_matrix')
            fd = file(filename,'r')
            d_skmm = pickle.load(fd)
            fd.close()
        return d_skmm
    
    def fill_density_matrix(self):
        nblead = self.nblead
        nbmol = self.nbmol
        nyzk = self.nyzk
        nspins = self.nspins
        d_syzkmm_ij = np.zeros([nspins, nyzk, nbmol, nbmol], complex)
        d_syzkmm_ij[:, :, -nblead:, :nblead] = np.copy(self.d1_syzkmm_ij)                    
        return d_syzkmm_ij

    def boundary_check(self):
        tol = 1e-4
        pl1 = self.h1_skmm.shape[-1]
        self.do_shift = False
        matdiff = self.h_syzkmm[0, :, :pl1, :pl1] - self.h1_syzkmm[0]
        if self.nspins == 2:
             matdiff1 = self.h_syzkmm[1, :, :pl1, :pl1] - self.h1_syzkmm[1]
             float_diff = np.max(abs(matdiff - matdiff1)) 
             if float_diff > 1e-6:
                  print 'Warning!, float_diff between spins %f' % float_diff
        e_diff = matdiff[0,0,0] / self.s1_yzkmm[0,0,0]
        if abs(e_diff) > tol:
            print 'Warning*: hamiltonian boundary difference %f' %  e_diff
            self.do_shift = True
        self.zero_shift = e_diff
        matdiff = self.d_syzkmm[:, :, :pl1, :pl1] - self.d1_syzkmm
        print_diff = np.max(abs(matdiff))
        if print_diff > tol:
            print 'Warning*: density boundary difference %f' % print_diff
            
    def get_selfconsistent_hamiltonian(self, bias=0, gate=0, verbose=0):
        self.initialize_scf(bias, gate, verbose)  
        self.move_buffer()
        nbmol = self.nbmol_inner
        nspins = self.nspins
        nyzk = self.nyzk
        den = np.empty([nspins, nyzk, nbmol, nbmol], complex)
        denocc = np.empty([nspins, nyzk, nbmol, nbmol], complex)
        #denvir = np.empty([nspins, nyzk, nbmol, nbmol], complex)
       
        self.get_zero_float()
        self.boundary_check() 

        #-------get the path --------    
        for s in range(nspins):
            for k in range(nyzk):      
                den[s, k] = self.get_eqintegral_points(self.intctrl, s, k)
                denocc[s, k]= self.get_neintegral_points(self.intctrl, s, k)
        #        denvir[s, k]= self.get_neintegral_points(self.intctrl, s, k,
        #                                                          'neVirInt')
        
        #-------begin the SCF ----------         
        self.step = 0
        self.cvgflag = 0
        calc = self.atoms.calc    
        timer = Timer()
        nxk = self.nxkmol
        kpts = self.atoms.calc.wfs.ibzk_kc
        spin_coff = 3 - nspins
        max_steps = 100
        while self.cvgflag == 0 and self.step < max_steps:
            self.move_buffer()
            f_syzkmm_mol = self.fcvg.matcvg(self.h_syzkmm_mol)
            timer.start('Fock2Den')
            for s in range(nspins):
                for k in range(nyzk):
                    self.d_syzkmm_mol[s, k] = spin_coff * self.fock2den(
                                                            self.intctrl,
                                                            f_syzkmm_mol,
                                                            s, k)
            timer.stop('Fock2Den')
            if self.verbose:
                print'Fock2Den', timer.gettime('Fock2Den'), 'second'
            d_syzkmm_out = self.dcvg.matcvg(self.d_syzkmm)
            self.cvgflag = self.fcvg.bcvg and self.dcvg.bcvg
            timer.start('Den2Fock')            
            self.h_skmm = self.den2fock(d_syzkmm_out)
            timer.stop('Den2Fock')
            if self.verbose:
                print'Den2Fock', timer.gettime('Den2Fock'),'second'
         
            self.h_syzkmm = self.substract_yzk(nxk, kpts, self.h_skmm, 'h')
            if self.do_shift:
                for i in range(nspins):
                    self.h_syzkmm[i] -= self.zero_shift * self.s_yzkmm
            self.step +=  1
            
        return 1
 
    def initialize_scf(self, bias, gate, verbose, alpha=0.1):
        self.verbose = verbose
        self.bias = bias
        self.gate = gate
        self.kt = self.atoms.calc.occupations.kT * Hartree
        self.fermi = 0
        self.current = 0
        #self.buffer = self.nblead
        self.buffer = 0
        self.atoms.calc.density.transport = True       
        if self.verbose:
            print 'prepare for selfconsistent calculation'
            print '[bias, gate] =', bias, gate, 'V'
            print 'lead_fermi_level =', self.fermi

        self.intctrl = IntCtrl(self.kt, self.fermi, self.bias)
        self.fcvg = CvgCtrl()
        self.dcvg = CvgCtrl()
        inputinfo = {'fasmethodname':'SCL_None', 'fmethodname':'CVG_None',
                     'falpha': 0.1, 'falphascaling':0.1, 'ftol':1e-3,
                     'fallowedmatmax':1e-4, 'fndiis':10, 'ftolx':1e-5,
                     'fsteadycheck': False,
                     'dasmethodname':'SCL_None', 'dmethodname':'CVG_Broydn',
                     'dalpha': alpha, 'dalphascaling':0.1, 'dtol':1e-3,
                     'dallowedmatmax':1e-4, 'dndiis':6, 'dtolx':1e-5,
                     'dsteadycheck': False}
        self.fcvg(inputinfo, 'f', self.dcvg)
        self.dcvg(inputinfo, 'd', self.fcvg)
        if self.verbose:
            print 'mix_factor =', inputinfo['dalpha']
        
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
        Hmx = self.h_syzkmm[0,0]
        Smx = self.s_yzkmm[0]
        
        self.greenfunction = GreenFunction(selfenergies=self.selfenergies,
                                           H=Hmx, S=Smx, eta=0)

        self.selfenergies[0].set_bias(self.bias / 2.0)
        self.selfenergies[1].set_bias(-self.bias / 2.0)
       
        self.eqpathinfo = []
        self.nepathinfo = []
       
        for s in range(self.nspins):
            self.eqpathinfo.append([])
            self.nepathinfo.append([])
            for k in range(self.nyzk):
                self.eqpathinfo[s].append(PathInfo('eq'))
                self.nepathinfo[s].append(PathInfo('ne'))    
     
    def get_eqintegral_points(self, intctrl, s, k):
        global zint, fint, tgtint, cntint
        maxintcnt = 500
        nblead = self.nblead
        nbmol = self.nbmol_inner
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
        
        self.greenfunction.H = self.h_syzkmm_mol[s, k]
        self.greenfunction.S = self.s_yzkmm_mol[k]
       
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
                                                self.greenfunction.S)) * 2
            else:
                print 'Eq_sk[', s , k , ']=', np.trace(np.dot(den,
                                                 self.greenfunction.S))
        del fint, tgtint, zint
        return den 
    
    def get_neintegral_points(self, intctrl, s, k, calcutype='neInt'):
        intpathtol = 1e-8
        nblead = self.nblead
        nbmol = self.nbmol_inner
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
        
        self.greenfunction.H = self.h_syzkmm_mol[s, k]
        self.greenfunction.S = self.s_yzkmm_mol[k]

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
                                                            0, calcutype,
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
                nefcnt = max(np.ceil(np.real(neintpath[1] -
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
                sumga = np.zeros([1, nbmol, nbmol], complex)
                for i in range(nefcnt):
                    sumga += self.calgfunc(zgp[i], calcutype,
                                                      intctrl) * wgp[i]
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
        neq = np.trace(np.dot(denocc, self.greenfunction.S))
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
        nbmol = self.nbmol_inner
                
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
            gamma = np.zeros([nlead, nbmol, nbmol], complex)
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
            gamma[0, :nblead, :nblead] += self.selfenergies[0].get_lambda(
                                                                        zp[i])
            gamma[1, -nblead:, -nblead:] += self.selfenergies[1].get_lambda(
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
            elif calcutype == 'neVirInt':
                gammavir = np.zeros([nbmol, nbmol], complex)
                for n in [0, 1]:
                    fint[n].append(  fermidistribution(zp[i] -
                                         intctrl.maxfermi, intctrl.kt) - 
                                         fermidistribution(zp[i] -
                                             intctrl.leadfermi[n], intctrl.kt))
                    gammavir += gamma[n] * fint[n][cntint]
                avir = np.dot(gr, gammavir)
                avir = np.dot(avir, gr.T.conj())
                gfunc[i] = avir
                    
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
        nbmol = self.nbmol_inner
        den = np.zeros([nbmol, nbmol], complex)
        sigmatmp = np.zeros([nblead, nblead], complex)

        # missing loc states
        eqzp = self.eqpathinfo[s][k].energy
        nezp = self.nepathinfo[s][k].energy
        
        self.greenfunction.H = f_syzkmm[s, k]
        self.greenfunction.S = self.s_yzkmm_mol[k]
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
            fermifactor = np.real(self.nepathinfo[s][k].fermi_factor[1][i])

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
        linear_potential = self.get_linear_potential()
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
                        dr_mm[s, i, j] = np.copy(d_syzkmm[s, j])
                        qr_mm[s, j] += np.dot(dr_mm[s, i, j], self.s_yzkmm[j]) / nyzk
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

    def get_zero_float(self):
        calc = self.atoms.calc
        vmol = calc.hamiltonian.vt_sG
        vlead = self.atoms_l[0].calc.hamiltonian.vt_sG 
        dimy_lead = vlead.shape[2]
        dimz_lead = vlead.shape[3]
        self.zero_float = vmol[0,0, dimy_lead/2, dimz_lead /2
                                    ] - vlead[0,0, dimy_lead/2, dimz_lead /2]
   
    def neutualize(self, d_syzkmm):
        extend_layer = self.extend_layer
        nbmol = self.nbmol
        if extend_layer == 0 :
            dmx = self.d_syzkmm
            smx = self.s_yzkmm
        else:
            dmx = self.d_syzkmm[:,:,extend_layer:-extend_layer, extend_layer:-extend_layer]
            smx = self.s_yzkmm[:,extend_layer:-extend_layer, extend_layer:-extend_layer]
        qr_mm = np.empty([self.nspins, self.nyzk, nbmol, nbmol])
        for i in range(self.nspins):  
            for j in range(self.nyzk):
                qr_mm[i,j] = np.dot(dmx[i,j],smx[j])
        Qmol = np.trace(np.sum(np.sum(qr_mm, axis=0), axis=0)) + np.sum(self.edge_charge)/self.nyzk
        nlead_atoms = len(self.pl_atoms[0])
    
    
        Qmol = self.calc_total_charge(d_syzkmm)
        self.totalQ = self.nbmol * nlead_atoms / self.nblead
    
    
        print   'Warning! totalQmol,  Qmol = %f,%f'%(Qmol,self.totalQ)
        Qmol_plus = Qmol - self.totalQ
        spincoff= 3-self.nspins 
        nbatom = self.nblead / nlead_atoms 
        for i in range(self.nspins):
            for j in range(self.nyzk):
                for k in range(nbatom):                   
                    d_syzkmm[i, j, k, k] -= Qmol_plus / (nbatom * self.nyzk * 2* self.s_yzkmm[j,k,k])
                for k in range(-nbatom,0):
                    d_syzkmm[i, j, k, k] -= Qmol_plus / (nbatom * self.nyzk * 2* self.s_yzkmm[j,k,k])
        return d_syzkmm
    
    def calc_total_charge(self, d_syzkmm):
        nbmol = self.nbmol 
        qr_mm = np.empty([self.nspins, self.nyzk, nbmol, nbmol])
        for i in range(self.nspins):  
            for j in range(self.nyzk):
                qr_mm[i,j] = np.dot(d_syzkmm[i,j],self.s_yzkmm[j])
        Qmol = np.trace(np.sum(np.sum(qr_mm, axis=0), axis=0)) + np.sum(self.edge_charge)/self.nyzk
        return Qmol        

    def get_linear_potential(self):
        calc = self.atoms.calc
        linear_potential = np.zeros(calc.hamiltonian.vt_sG.shape)
        
        dimx = linear_potential.shape[1]
        dimyz = linear_potential.shape[2:]
        calc1 = self.atoms_l[0].calc
        dimx_lead = calc1.hamiltonian.vt_sG.shape[1]
           
        spacing_tol = 1e-5
        spacing_lead = calc1.domain.cell_c[0] / dimx_lead
        spacing_scat = calc.domain.cell_c[0] / dimx
        
        if abs(spacing_lead -spacing_scat) < spacing_tol:
            pass
        else:
            print 'Warning!, the spacing between the scat %f and lead %f is not small, \
                  the linear potential maybe not very precise' % (spacing_lead, spacing_scat)
        
        #buffer_dim = dimx_lead
        buffer_dim = 0
        scat_dim = dimx - buffer_dim * 2
        bias = self.bias / Hartree
        vx = np.empty([dimx])
        if buffer_dim !=0:
            vx[:buffer_dim] = bias / 2.0
            vx[-buffer_dim:] = -bias / 2.0        
            vx[buffer_dim : -buffer_dim] = np.linspace(bias/2.0, -bias/2.0, scat_dim)
        else:
            vx = np.linspace(bias/2.0, -bias/2.0, scat_dim)
        for s in range(self.nspins):
            for i in range(dimx):
                linear_potential[s, i] = vx[i] * (np.zeros(dimyz) + 1)
        return linear_potential
    
    
    def output(self, filename):
        fd = file(filename, 'wb')
        pickle.dump((self.h1_syzkmm, 
                     self.h1_syzkmm_ij, 
                     self.h_syzkmm,
                     self.s1_yzkmm, 
                     self.s1_yzkmm_ij, 
                     self.s_yzkmm,
                     self.d_syzkmm,
                     self.bias,
                     self.intctrl,
                     self.atoms.calc.hamiltonian.vt_sG,
                     self.atoms.calc.density.nt_sG,
                     self.eqpathinfo,
                     self.nepathinfo,
                     self.current,
                     self.step,
                     self.cvgflag
                     ), fd, 2)
        fd.close()
        
    def input(self, filename):
        fd = file(filename, 'r')
        (self.h1_syzkmm, 
                     self.h1_syzkmm_ij,
                     self.h_syzkmm,
                     self.s1_yzkmm, 
                     self.s1_yzkmm_ij,
                     self.s_yzkmm,
                     self.d_syzkmm,
                     self.bias,
                     self.intctrl,
                     #self.atoms.calc.hamiltonian.vt_sG,
                     self.vt_sG,
                     #self.atoms.calc.density.nt_sG,
                     self.nt_sG,
                     self.eqpathinfo,
                     self.nepathinfo,
                     self.current,
                     self.step,
                     self.cvgflag) = pickle.load(fd)
        fd.close()
       
    def set_calculator(self, e_points):
        from ase.transport.calculators import TransportCalculator
     
        h_scat = self.h_syzkmm[0,0]
        h_lead1 = self.double_size(self.h1_syzkmm[0,0],
                                   self.h1_syzkmm_ij[0,0])
        h_lead2 = self.double_size(self.h2_syzkmm[0,0],
                                   self.h2_syzkmm_ij[0,0])
       
        s_scat = self.s_yzkmm[0]
        s_lead1 = self.double_size(self.s1_yzkmm[0], self.s1_yzkmm_ij[0])
        s_lead2 = self.double_size(self.s2_yzkmm[0], self.s2_yzkmm_ij[0])
        
        tcalc = TransportCalculator(energies=e_points,
                                    h = h_scat,
                                    h1 = h_lead2,
                                    h2 = h_lead2,
                                    s = s_scat,
                                    s1 = s_lead2,
                                    s2 = s_lead2,
                                    dos = True
                                   )
        return tcalc
    
    def plot_dos(self, E_range, point_num = 30):
        e_points = np.linspace(E_range[0], E_range[1], point_num)
        tclac = self.set_calculator(e_points)
        tcalc.get_transmission()
        tcalc.get_dos()
       
        import pylab
        pylab.figure(1)
        pylab.subplot(211)
        pylab.plot(e_points, tcalc.T_e)
        pylab.ylabel('Transmission Coefficients')
        pylab.subplot(212)
        pylab.plot(e_points, tcalc.dos_e)
        pylab.ylabel('Density of States')
        pylab.xlabel('Energy (eV)')
        pylab.show()
    
    def double_size(self, m_ii, m_ij):
        dim = m_ii.shape[-1]
        mtx = np.empty([dim * 2, dim * 2])
        mtx[:dim, :dim] = m_ii
        mtx[-dim:, -dim:] = m_ii
        mtx[:dim, -dim:] = m_ij
        mtx[-dim:, :dim] = m_ij.T.conj()
        return mtx
    
    def get_current(self):
        E_Points, weight, fermi_factor = self.get_nepath_info()
        tcalc = self.set_calculator(E_Points)
        tcalc.initialize()
        tcalc.update()
        numE = len(E_Points) 
        current = [0, 0]
        for i in [0,1]:
            for j in range(numE):
                current[i] += tcalc.T_e[j] * weight[j] * fermi_factor[i][j]
        self.current = current[0] - current[1]
        return self.current
    
    def get_nepath_info(self):
        if hasattr(self, 'nepathinfo'):
            energy = self.nepathinfo[0][0].energy
            weight = self.nepathinfo[0][0].weight
            fermi_factor = self.nepathinfo[0][0].fermi_factor
      
        return energy, weight, fermi_factor
    
    
    def fill_extend_layer(self):
        pl1 = self.extend_layer
        if pl1 * 2 == self.nblead:
            for s in range(self.nspins):
                for k in range(self.nyzk):
                    self.d_syzkmm[s, k, :pl1, :pl1] = self.d1_syzkmm[s, k, :pl1, :pl1]
                    self.d_syzkmm[s, k, -pl1:, -pl1:] = self.d1_syzkmm[s, k, -pl1:, -pl1:]
                    self.d_syzkmm[s, k, :pl1, pl1:2:pl1] = self.d1_syzkmm[s, k, :pl1, pl1:2:pl1]
                    self.d_syzkmm[s, k, pl1:2:pl1, :pl1] = self.d1_syzkmm[s, k, pl1:2:pl1, :pl1]
                    self.d_syzkmm[s, k, -2*pl1:-pl1, -pl1:] = self.d1_syzkmm[s, k, -2*pl1:-pl1, -pl1:]
                    self.d_syzkmm[s, k, -pl1:, -2*pl1:-pl1] = self.d1_syzkmm[s, k, -pl1:, -2*pl1:-pl1]                    
        else:
            for s in range(self.nspins):
                for k in range(self.nyzk):
                    self.d_syzkmm[s, k, :pl1, :pl1] = self.d1_syzkmm[s, k, :pl1, :pl1]
                    self.d_syzkmm[s, k, -pl1:, -pl1:] = self.d1_syzkmm[s, k, -pl1:, -pl1:]
                    self.d_syzkmm[s, k, :pl1, pl1:2*pl1] = self.d1_syzkmm_ij[s, k]
                    self.d_syzkmm[s, k, pl1:2*pl1, :pl1] = self.d1_syzkmm_ij[s, k].T.conj()
                    self.d_syzkmm[s, k, -2*pl1:-pl1, -pl1:] = self.d1_syzkmm_ij[s, k]
                    self.d_syzkmm[s, k, -pl1:, -2*pl1:-pl1] = self.d1_syzkmm_ij[s, k].T.conj()
                    
    def fill_extend_layer2(self):
        shift = self.bias / 2.0
        pl1 = self.extend_layer
        if pl1 * 2 == self.nblead:
            for s in range(self.nspins):
                for k in range(self.nyzk):
                    self.h_syzkmm[s, k, :pl1, :pl1] = self.h1_syzkmm[s, k, :pl1, :pl1]
                    self.h_syzkmm[s, k, -pl1:, -pl1:] = self.h1_syzkmm[s, k, -pl1:, -pl1:]
                    self.h_syzkmm[s, k, :pl1, pl1:2:pl1] = self.h1_syzkmm[s, k, :pl1, pl1:2:pl1]
                    self.h_syzkmm[s, k, pl1:2:pl1, :pl1] = self.h1_syzkmm[s, k, pl1:2:pl1, :pl1]
                    self.h_syzkmm[s, k, -2*pl1:-pl1, -pl1:] = self.h1_syzkmm[s, k, -2*pl1:-pl1, -pl1:]
                    self.h_syzkmm[s, k, -pl1:, -2*pl1:-pl1] = self.h1_syzkmm[s, k, -pl1:, -2*pl1:-pl1]                    
        else:
            for s in range(self.nspins):
                for k in range(self.nyzk):
                    self.h_syzkmm[s, k, :pl1, :pl1] = self.h1_syzkmm[s, k] + shift * self.s1_yzkmm[s, k]
                    self.h_syzkmm[s, k, -pl1:, -pl1:] = self.h1_syzkmm[s, k] - shift * self.s1_yzkmm[s, k]
                    self.h_syzkmm[s, k, :pl1, pl1:2*pl1] = self.h1_syzkmm_ij[s, k] + shift * self.s1_yzkmm_ij[s,k]
                    self.h_syzkmm[s, k, pl1:2*pl1, :pl1] = self.h1_syzkmm_ij[s, k].T.conj() + shift * self.s1_yzkmm_ij[s,k].T.conj()
                    self.h_syzkmm[s, k, -2*pl1:-pl1, -pl1:] = self.h1_syzkmm_ij[s, k] - shift * self.s1_yzkmm_ij[s,k]
                    self.h_syzkmm[s, k, -pl1:, -2*pl1:-pl1] = self.h1_syzkmm_ij[s, k].T.conj() - shift * self.s1_yzkmm_ij[s,k].T.conj()
    
    def move_buffer(self):
        self.nbmol_inner = self.nbmol - 2 * self.buffer
        pl1 = self.buffer
        if pl1 == 0:
            self.h_syzkmm_mol = self.h_syzkmm
            self.d_syzkmm_mol = self.d_syzkmm
            self.s_yzkmm_mol = self.s_yzkmm
        else:
            self.h_syzkmm_mol = self.h_syzkmm[:, :, pl1: -pl1, pl1:-pl1]
            self.d_syzkmm_mol = self.d_syzkmm[:, :, pl1: -pl1, pl1:-pl1]
            self.s_yzkmm_mol = self.s_yzkmm[:, pl1: -pl1, pl1:-pl1]

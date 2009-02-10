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
    
    def __init__(self, atoms, pl_atoms, pl_cells, d=0, extend=False):
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
        self.extend = extend

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
            rank = world.rank
            atoms.calc.write('lead' + str(l) + '.gpw')
            if l == 0:
                self.h1_skmm, self.s1_kmm = self.get_hs(atoms)
                fd = file('leadhs0_' + str(rank),'wb')
                pickle.dump((self.h1_skmm, self.s1_kmm,
                             self.atoms_l[0].calc.hamiltonian.vt_sG,
                             self.ntklead), fd, 2)            
                fd.close()            
            elif l == 1:
                self.h2_skmm, self.s2_kmm = self.get_hs(atoms)
                fd = file('leadhs1_' + str(rank),'wb')
                pickle.dump((self.h2_skmm, self.s2_kmm,
                             self.atoms_l[1].calc.hamiltonian.vt_sG,
                             self.ntklead), fd, 2)
                fd.close()
        else:
            atoms, calc = restart_gpaw('lead' + str(l) + '.gpw')
            calc.set_positions()
            self.atoms_l[l] = atoms
            self.atoms_l[l].calc = calc
            rank = world.rank
            if l == 0:        
                fd = file('leadhs0_' + str(rank),'r') 
                self.h1_skmm, self.s1_kmm, \
                                   self.atoms_l[0].calc.hamiltonian.vt_sG, \
                                               self.ntklead = pickle.load(fd)
                fd.close()
            elif l == 1:
                fd = file('leadhs1_' + str(rank),'r') 
                self.h2_skmm, self.s2_kmm, \
                                   self.atoms_l[1].calc.hamiltonian.vt_sG, \
                                               self.ntklead = pickle.load(fd)
                fd.close()
        self.dimt_lead = self.atoms_l[0].calc.hamiltonian.vt_sG.shape[-1]
        
    def update_scat_hamiltonian(self, restart=False):
        if not restart:
            atoms = self.atoms
            atoms.get_potential_energy()
            calc = atoms.calc
            rank = world.rank
            calc.write('scat.gpw')
            self.h_skmm, self.s_kmm = self.get_hs(atoms)
            fd = file('scaths_' + str(rank), 'wb')
            pickle.dump((self.h_skmm, self.s_kmm, calc.density.nct_G,
                                          calc.hamiltonian.vt_sG), fd, 2)
            fd.close()
                        
        else:
            atoms, calc = restart_gpaw('scat.gpw')
            calc.set_positions()
            rank = world.rank
            self.atoms = atoms
            fd = file('scaths_' + str(rank), 'r')
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

    def get_lead_calc(self, l, nkpts=15):
        p = self.atoms.calc.input_parameters.copy()
        p['nbands'] = None
        kpts = list(p['kpts'])
        if nkpts == 0:
            kpts[self.d] = 2 * int(17.0 / self.pl_cells[l][self.d]) + 1
        else:
            kpts[self.d] = nkpts
        self.ntklead = kpts[self.d]
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
        world.barrier()
        self.update_lead_hamiltonian(1, lead_restart)
        world.barrier()
        if self.extend:
            self.extend_scat()
        
        p = self.atoms.calc.input_parameters.copy()           
        self.ntkmol = p['kpts'][self.d]
        self.update_scat_hamiltonian(scat_restart)
        world.barrier()
        self.nspins = self.h1_skmm.shape[0]
        self.nblead = self.h1_skmm.shape[-1]
        self.nbmol = self.h_skmm.shape[-1]
        kpts = self.atoms_l[0].calc.wfs.ibzk_kc  
        self.npk = kpts.shape[0] / self.ntklead  
        self.allocate_cpus()

        self.d1_skmm = self.generate_density_matrix('lead_l', lead_restart)        
        self.check_edge()
        self.initial_lead(0)

        self.d2_skmm = self.generate_density_matrix('lead_r', lead_restart)
        self.initial_lead(1)

        self.d_skmm = self.generate_density_matrix('scat', scat_restart)
        self.initial_mol()        

        self.edge_density_mm = self.calc_edge_density(self.d_spkmm_ij,
                                                              self.s_pkmm_ij)
        self.edge_charge = np.zeros([self.nspins])
        for i in range(self.nspins):
            for j in range(self.my_npk):
                self.edge_charge[i] += np.trace(self.edge_density_mm[i, j])
            
        self.kpt_comm.sum(self.edge_charge)
        if world.rank == 0:
            for i in range(self.nspins):  
                total_edge_charge  = self.edge_charge[i] / self.npk
                print 'edge_charge[%d]=%f' % (i, total_edge_charge)
        self.boundary_check()
        
        del self.atoms_l

    def initial_lead(self, lead):
        nspins = self.nspins
        ntk = self.ntklead
        npk = self.my_npk
        nblead = self.nblead
        kpts = self.my_lead_kpts
        position = [0, 0, 0]
        if lead == 0:
            position[self.d] = 1.0
            self.h1_spkmm = self.substract_pk(ntk, kpts, self.h1_skmm, 'h')
            self.s1_pkmm = self.substract_pk(ntk, kpts, self.s1_kmm)
            self.h1_spkmm_ij = self.substract_pk(ntk, kpts, self.h1_skmm,
                                                   'h', position)
            self.s1_pkmm_ij = self.substract_pk(ntk, kpts, self.s1_kmm,
                                                  's', position)
            self.d1_spkmm = self.substract_pk(ntk, kpts, self.d1_skmm, 'h')
            self.d1_spkmm_ij = self.substract_pk(ntk, kpts, self.d1_skmm,
                                                   'h', position)
        elif lead == 1:
            position[self.d] = -1.0
            self.h2_spkmm = self.substract_pk(ntk, kpts, self.h2_skmm, 'h')
            self.s2_pkmm = self.substract_pk(ntk, kpts, self.s2_kmm)
            self.h2_spkmm_ij = self.substract_pk(ntk, kpts, self.h2_skmm,
                                                   'h', position)
            self.s2_pkmm_ij = self.substract_pk(ntk, kpts, self.s2_kmm,
                                                  's', position)            
        else:
            raise TypeError('unkown lead index')

    def initial_mol(self):
        ntk = self.ntkmol
        kpts = self.my_kpts
        position = [0,0,0]
        position[self.d] = 1.0
        self.h_spkmm = self.substract_pk(ntk, kpts, self.h_skmm, 'h')
        self.s_pkmm = self.substract_pk(ntk, kpts, self.s_kmm)
        self.s_pkmm_ij = self.substract_pk(ntk, kpts, self.s_kmm, 's',
                                            position)
        self.d_spkmm = self.substract_pk(ntk, kpts, self.d_skmm, 'h')
        self.d_spkmm_ij = self.substract_pk(ntk, kpts, self.d_skmm,
                                                    'h', position)
        #self.d_spkmm_ij2 = self.fill_density_matrix()

    def substract_pk(self, ntk, kpts, k_mm, hors='s', position=[0, 0, 0]):
        npk = self.my_npk
        weight = np.array([1.0 / ntk] * ntk )
        if hors not in 'hs':
            raise KeyError('hors should be h or s!')
        if hors == 'h':
            dim = k_mm.shape[:]
            dim = (dim[0],) + (dim[1] / ntk,) + dim[2:]
            pk_mm = np.empty(dim, complex)
            dim = (dim[0],) + (ntk,) + dim[2:]
            tk_mm = np.empty(dim, complex)
        elif hors == 's':
            dim = k_mm.shape[:]
            dim = (dim[0] / ntk,) + dim[1:]
            pk_mm = np.empty(dim, complex)
            dim = (ntk,) + dim[1:]
            tk_mm = np.empty(dim, complex)
        n = 0
        tkpts = self.pick_out_tkpts(ntk, kpts)
        for i in range(npk):
            n = i * ntk
            for j in range(ntk):
                if hors == 'h':
                    tk_mm[:, j] = np.copy(k_mm[:, n + j])
                elif hors == 's':
                    tk_mm[j] = np.copy(k_mm[n + j])
            if hors == 'h':
                pk_mm[:, i] = get_realspace_hs(tk_mm, None,
                                               tkpts, weight, position)
            elif hors == 's':
                pk_mm[i] = get_realspace_hs(None, tk_mm,
                                                   tkpts, weight, position)
        return pk_mm   
            
    def check_edge(self):
        tolx = 1e-6
        position = [0,0,0]
        position[self.d] = 2.0
        ntk = self.ntklead
        npk = self.npk
        kpts = self.my_lead_kpts
        s_pkmm = self.substract_pk(ntk, kpts, self.s1_kmm, 's', position)
        matmax = np.max(abs(s_pkmm))
        if matmax > tolx:
            print 'Warning*: the principle layer should be lagger, \
                                                          matmax=%f' % matmax        
    
    def calc_edge_density(self, d_spkmm_ij, s_pkmm_ij):
        nspins = self.nspins
        npk = self.my_npk
        nbf = s_pkmm_ij.shape[-1]
        edge_charge_mm = np.zeros([nspins, npk, nbf, nbf])
        for i in range(nspins):
            for j in range(npk):
                edge_charge_mm[i, j] += np.dot(d_spkmm_ij[i, j],
                                               s_pkmm_ij[j].T.conj())
                edge_charge_mm[i, j] += np.dot(d_spkmm_ij[i, j].T.conj(),
                                               s_pkmm_ij[j])
        return edge_charge_mm

    def pick_out_tkpts(self, ntk, kpts):
        npk = self.npk
        tkpts = np.zeros([ntk, 3])
        for i in range(ntk):
            tkpts[i, 2] = kpts[i, 2]
        return tkpts

    def generate_density_matrix(self, region, restart=True):
        ntk = self.ntklead
        npk = self.npk
        rank = world.rank
        if not restart:
            if region == 'lead_l':
                calc = self.atoms_l[0].calc
                dim = self.h1_skmm.shape
                filename = 'd1_skmm_' + str(rank)
            elif region == 'lead_r':
                calc = self.atoms_l[1].calc
                dim = self.h2_skmm.shape
                filename = 'd2_skmm_' + str(rank)
            elif region == 'scat':
                calc = self.atoms.calc
                dim = self.h_skmm.shape
                ntk = self.ntkmol
                filename = 'd_skmm_' + str(rank)
            else:
                raise KeyError('invalid lead index in \
                                                     generate_density_matrix')
            d_skmm = np.empty(dim, complex)
            for kpt in calc.wfs.kpt_u:
                C_nm = kpt.C_nM
                f_nn = np.diag(kpt.f_n)
                d_skmm[kpt.s, kpt.q] = np.dot(C_nm.T.conj(),
                                              np.dot(f_nn, C_nm)) * ntk * npk
            fd = file(filename,'wb')
            pickle.dump(d_skmm, fd, 2)
            fd.close()
        else:
            
            if region == 'lead_l':
                filename = 'd1_skmm_' + str(rank)
            elif region == 'lead_r':
                filename = 'd2_skmm_' + str(rank)
            elif region == 'scat':
                filename = 'd_skmm_'  + str(rank)
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
        npk = self.my_npk
        nspins = self.nspins
        d_spkmm_ij = np.zeros([nspins, npk, nbmol, nbmol], complex)
        d_spkmm_ij[:, :, -nblead:, :nblead] = np.copy(self.d1_spkmm_ij)                    
        return d_spkmm_ij

    def boundary_check(self):
        tol = 1e-4
        pl1 = self.h1_skmm.shape[-1]
        self.do_shift = False
        matdiff = self.h_spkmm[0, :, :pl1, :pl1] - self.h1_spkmm[0]
        if self.nspins == 2:
             matdiff1 = self.h_spkmm[1, :, :pl1, :pl1] - self.h1_spkmm[1]
             float_diff = np.max(abs(matdiff - matdiff1)) 
             if float_diff > 1e-6:
                  print 'Warning!, float_diff between spins %f' % float_diff
        e_diff = matdiff[0,0,0] / self.s1_pkmm[0,0,0]
        if abs(e_diff) > tol:
            print 'Warning*: hamiltonian boundary difference(rank=%d) %f' % (
                                                           world.rank, e_diff)
            self.do_shift = True
        self.zero_shift = e_diff
        matdiff = self.d_spkmm[:, :, :pl1, :pl1] - self.d1_spkmm
        print_diff = np.max(abs(matdiff))
        if print_diff > tol:
            print 'Warning*: density boundary difference(rank=%d) %f' % (
                                                       world.rank, print_diff)
            
    def extend_scat(self):
        lead_atoms_num = len(self.pl_atoms[0])
        atoms_inner = self.atoms.copy()
        atoms_inner.center()
        atoms = self.atoms_l[0] + atoms_inner + self.atoms_l[1]
        atoms.set_pbc(atoms_inner._pbc)
        cell = self.atoms._cell.copy()
        cell[2, 2] += self.pl_cells[0][2] * 2
        atoms.set_cell(cell)
        for i in range(lead_atoms_num):
            atoms.positions[i, 2] -= self.pl_cells[0][2]
        for i in range(-lead_atoms_num, 0):
            atoms.positions[i, 2] += self.atoms._cell[2, 2]
        atoms.calc = self.atoms.calc
        self.atoms = atoms
        self.atoms.center()

    def get_selfconsistent_hamiltonian(self, bias=0, gate=0,
                                                    cal_loc=False, verbose=0):
        self.initialize_scf(bias, gate, cal_loc, verbose)  
        self.move_buffer()
        nbmol = self.nbmol_inner
        nspins = self.nspins
        npk = self.my_npk
        den = np.empty([nspins, npk, nbmol, nbmol], complex)
        denocc = np.empty([nspins, npk, nbmol, nbmol], complex)
        if self.cal_loc:
            #denvir = np.empty([nspins, npk, nbmol, nbmol], complex)
            denloc = np.empty([nspins, npk, nbmol, nbmol], complex)            

 

        world.barrier()
        #-------get the path --------    
        for s in range(nspins):
            for k in range(npk):      
                den[s, k] = self.get_eqintegral_points(self.intctrl, s, k)
                denocc[s, k] = self.get_neintegral_points(self.intctrl, s, k)
                if self.cal_loc:
                    #denvir[s, k]= self.get_neintegral_points(self.intctrl,
                    #                                         s, k, 'neVirInt')
                    denloc[s, k] = self.get_neintegral_points(self.intctrl,
                                                              s, k, 'locInt')                    
       
        #-------begin the SCF ----------         
        self.step = 0
        self.cvgflag = 0
        calc = self.atoms.calc    
        timer = Timer()
        ntk = self.ntkmol
        kpts = self.my_kpts
        spin_coff = 3 - nspins
        max_steps = 200
        while self.cvgflag == 0 and self.step < max_steps:
            if self.master:
                print '----------------step %d -------------------' % self.step
            self.move_buffer()
            f_spkmm_mol = self.fcvg.matcvg(self.h_spkmm_mol)
            timer.start('Fock2Den')
            for s in range(nspins):
                for k in range(npk):
                    self.d_spkmm_mol[s, k] = spin_coff * self.fock2den(
                                                            self.intctrl,
                                                            f_spkmm_mol,
                                                            s, k)
            timer.stop('Fock2Den')
            if self.verbose and self.master:
                print'Fock2Den', timer.gettime('Fock2Den'), 'second'
            self.collect_density_matrix()
            d_stkmm_out = self.dcvg.matcvg(self.d_stkmm)
            d_spkmm_out = self.distribute_density_matrix(d_stkmm_out)
            self.cvgflag = self.fcvg.bcvg and self.dcvg.bcvg
            timer.start('Den2Fock')            
            self.h_skmm = self.den2fock(d_spkmm_out)
            timer.stop('Den2Fock')
            if self.verbose and self.master:
                print'Den2Fock', timer.gettime('Den2Fock'),'second'
         
            self.h_spkmm = self.substract_pk(ntk, kpts, self.h_skmm, 'h')
            if self.do_shift:
                for i in range(nspins):
                    self.h_spkmm[i] -= self.zero_shift * self.s_pkmm
            self.step +=  1
            
        return 1
 
    def initialize_scf(self, bias, gate, cal_loc, verbose, alpha=0.1):
        self.verbose = verbose
        self.master = (world.rank==0)
        self.bias = bias
        self.gate = gate
        self.cal_loc = cal_loc and self.bias != 0
        self.kt = self.atoms.calc.occupations.kT * Hartree
        self.fermi = 0
        self.current = 0
        if self.nblead == self.nbmol:
            self.buffer = 0
        else:
            self.buffer = self.nblead
        self.atoms.calc.density.transport = True       

        self.intctrl = IntCtrl(self.kt, self.fermi, self.bias)
        self.fcvg = CvgCtrl(self.master)
        self.dcvg = CvgCtrl(self.master)
        inputinfo = {'fasmethodname':'SCL_None', 'fmethodname':'CVG_None',
                     'falpha': 0.1, 'falphascaling':0.1, 'ftol':1e-3,
                     'fallowedmatmax':1e-4, 'fndiis':10, 'ftolx':1e-5,
                     'fsteadycheck': False,
                     'dasmethodname':'SCL_None', 'dmethodname':'CVG_Broydn',
                     'dalpha': alpha, 'dalphascaling':0.1, 'dtol':1e-4,
                     'dallowedmatmax':1e-4, 'dndiis':6, 'dtolx':1e-5,
                     'dsteadycheck': False}
        self.fcvg(inputinfo, 'f', self.dcvg)
        self.dcvg(inputinfo, 'd', self.fcvg)
        
        self.selfenergies = [LeadSelfEnergy((self.h1_spkmm[0,0],
                                                            self.s1_pkmm[0]), 
                                            (self.h1_spkmm_ij[0,0],
                                                         self.s1_pkmm_ij[0]),
                                            (self.h1_spkmm_ij[0,0],
                                                         self.s1_pkmm_ij[0]),
                                             0),
                             LeadSelfEnergy((self.h2_spkmm[0,0],
                                                            self.s2_pkmm[0]), 
                                            (self.h2_spkmm_ij[0,0],
                                                         self.s2_pkmm_ij[0]),
                                            (self.h2_spkmm_ij[0,0],
                                                         self.s2_pkmm_ij[0]),
                                             0)]

        self.greenfunction = GreenFunction(selfenergies=self.selfenergies,
                                           H=self.h_spkmm[0,0],
                                           S=self.s_pkmm[0], eta=0)

        self.selfenergies[0].set_bias(self.bias / 2.0)
        self.selfenergies[1].set_bias(-self.bias / 2.0)
       
        self.eqpathinfo = []
        self.nepathinfo = []
        if self.cal_loc:
            #self.virpathinfo = [] 
            self.locpathinfo = []
             
       
        for s in range(self.nspins):
            self.eqpathinfo.append([])
            self.nepathinfo.append([])
            if self.cal_loc:
                #self.virpathinfo.append([])
                self.locpathinfo.append([])                
            if self.cal_loc:
                self.locpathinfo.append([])
            for k in self.my_pk:
                self.eqpathinfo[s].append(PathInfo('eq'))
                self.nepathinfo[s].append(PathInfo('ne'))    
                if self.cal_loc:
                    #self.virpathinfo[s].append(PathInfo('ne'))
                    self.locpathinfo[s].append(PathInfo('eq'))
        if self.master:
            print '------------------Transport SCF-----------------------'
            print 'Mixer: %s,  Mixing factor: %s,  tol_Ham=%f, tol_Den=%f ' % (
                                      inputinfo['dmethodname'],
                                      inputinfo['dalpha'],
                                      inputinfo['ftol'],
                                      inputinfo['dtol']) 
            print 'bias = %f (V), gate = %f (V)' % (bias, gate)

     
    def get_eqintegral_points(self, intctrl, s, k):
        #global zint, fint, tgtint, cntint
        maxintcnt = 500
        nblead = self.nblead
        nbmol = self.nbmol_inner
        den = np.zeros([nbmol, nbmol])
        
        self.zint = [0] * maxintcnt
        self.fint = []
        self.tgtint = np.empty([2, maxintcnt, nblead, nblead], complex)
        self.cntint = -1

        self.selfenergies[0].h_ii = self.h1_spkmm[s, k]
        self.selfenergies[0].s_ii = self.s1_pkmm[k]
        self.selfenergies[0].h_ij = self.h1_spkmm_ij[s, k]
        self.selfenergies[0].s_ij = self.s1_pkmm_ij[k]
        self.selfenergies[0].h_im = self.h1_spkmm_ij[s, k]
        self.selfenergies[0].s_im = self.s1_pkmm_ij[k]

        self.selfenergies[1].h_ii = self.h2_spkmm[s, k]
        self.selfenergies[1].s_ii = self.s2_pkmm[k]
        self.selfenergies[1].h_ij = self.h2_spkmm_ij[s, k]
        self.selfenergies[1].s_ij = self.s2_pkmm_ij[k]
        self.selfenergies[1].h_im = self.h2_spkmm_ij[s, k]
        self.selfenergies[1].s_im = self.s2_pkmm_ij[k]
        
        self.greenfunction.H = self.h_spkmm_mol[s, k]
        self.greenfunction.S = self.s_pkmm_mol[k]
       
        #--eq-Integral-----
        [grsum, zgp, wgp, fcnt] = function_integral(self, 'eqInt')
        # res Calcu
        grsum += self.calgfunc(intctrl.eqresz, 'resInt')    
        grsum.shape = (nbmol, nbmol)
        den += 1.j * (grsum - grsum.T.conj()) / np.pi / 2

        # --sort SGF --
        nres = len(intctrl.eqresz)
        self.eqpathinfo[s][k].set_nres(nres)
        elist = zgp + intctrl.eqresz
        wlist = wgp + [1.0] * nres

        fcnt = len(elist)
        sgforder = [0] * fcnt
        for i in range(fcnt):
            sgferr = np.min(abs(elist[i] -
                                      np.array(self.zint[:self.cntint + 1 ])))
                
            sgforder[i] = np.argmin(abs(elist[i]
                                     - np.array(self.zint[:self.cntint + 1])))
            if sgferr > 1e-12:
                print 'Warning: SGF not Found. eqzgp[%d]= %f' %(i, elist[i])
                                             
        
        flist = self.fint[:]
        siglist = [[],[]]
        for i, num in zip(range(fcnt), sgforder):
            flist[i] = self.fint[num]
 
        sigma= np.empty([nblead, nblead], complex)
        for i in [0, 1]:
            for j, num in zip(range(fcnt), sgforder):
                sigma = self.tgtint[i, num]
                siglist[i].append(sigma)
        self.eqpathinfo[s][k].add(elist, wlist, flist, siglist)    
   
        if self.verbose and self.master: 
            if self.nspins == 1:
                print 'Eq_k[%d]= %f '% (k, np.trace(np.dot(den,
                                                self.greenfunction.S)) * 2 )
            else:
                print 'Eq_sk[%d, %d]= %f' %( s, k, np.trace(np.dot(den,
                                                 self.greenfunction.S)))
        del self.fint, self.tgtint, self.zint
        return den 
    
    def get_neintegral_points(self, intctrl, s, k, calcutype='neInt'):
        intpathtol = 1e-8
        nblead = self.nblead
        nbmol = self.nbmol_inner
        den = np.zeros([nbmol, nbmol], complex)
        maxintcnt = 500

        self.zint = [0] * maxintcnt
        self.tgtint = np.empty([2, maxintcnt, nblead, nblead], complex)

        self.selfenergies[0].h_ii = self.h1_spkmm[s, k]
        self.selfenergies[0].s_ii = self.s1_pkmm[k]
        self.selfenergies[0].h_ij = self.h1_spkmm_ij[s, k]
        self.selfenergies[0].s_ij = self.s1_pkmm_ij[k]
        self.selfenergies[0].h_im = self.h1_spkmm_ij[s, k]
        self.selfenergies[0].s_im = self.s1_pkmm_ij[k]

        self.selfenergies[1].h_ii = self.h2_spkmm[s, k]
        self.selfenergies[1].s_ii = self.s2_pkmm[k]
        self.selfenergies[1].h_ij = self.h2_spkmm_ij[s, k]
        self.selfenergies[1].s_ij = self.s2_pkmm_ij[k]
        self.selfenergies[1].h_im = self.h2_spkmm_ij[s, k]
        self.selfenergies[1].s_im = self.s2_pkmm_ij[k]
        
        self.greenfunction.H = self.h_spkmm_mol[s, k]
        self.greenfunction.S = self.s_pkmm_mol[k]

        if calcutype == 'neInt' or calcutype == 'neVirInt':
            for n in range(1, len(intctrl.neintpath)):
                self.cntint = -1
                self.fint = [[],[]]
                if intctrl.kt <= 0:
                    neintpath = [intctrl.neintpath[n - 1] + intpathtol,
                                 intctrl.neintpath[n] - intpathtol]
                else:
                    neintpath = [intctrl.neintpath[n-1],intctrl.neintpath[n]]
                if intctrl.neintmethod== 1:
    
                    # ----Auto Integral------
                    sumga, zgp, wgp, nefcnt = function_integral(self,
                                                                    calcutype)
    
                    nefcnt = len(zgp)
                    sgforder = [0] * nefcnt
                    for i in range(nefcnt):
                        sgferr = np.min(np.abs(zgp[i] - np.array(
                                            self.zint[:self.cntint + 1 ])))
                        sgforder[i] = np.argmin(np.abs(zgp[i] -
                                       np.array(self.zint[:self.cntint + 1])))
                        if sgferr > 1e-15:
                            print '--Warning: SGF not found, nezgp[%d]=%f' % (
                                                                    i, zgp[i])
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
                        sumga += self.calgfunc(zgp[i], calcutype) * wgp[i]
                den += sumga[0] / np.pi / 2
                flist = [[],[]]
                siglist = [[],[]]
                sigma= np.empty([nblead, nblead], complex)
                for i in [0, 1]:
                    for j, num in zip(range(nefcnt), sgforder):
                        fermi_factor = self.fint[i][num]
                        sigma = self.tgtint[i, num]
                        flist[i].append(fermi_factor)    
                        siglist[i].append(sigma)
                if calcutype == 'neInt':
                    self.nepathinfo[s][k].add(zgp, wgp, flist, siglist)
                elif calcutype == 'neVirInt':
                    self.virpathinfo[s][k].add(zgp, wgp, flist, siglist)
        # Loop neintpath
        elif calcutype == 'locInt':
            self.cntint = -1
            self.fint =[]
            sumgr, zgp, wgp, locfcnt = function_integral(self, 'locInt')
            # res Calcu :minfermi
            sumgr -= self.calgfunc(intctrl.locresz[0, :], 'resInt')
               
            # res Calcu :maxfermi
            sumgr += self.calgfunc(intctrl.locresz[1, :], 'resInt')
            
            sumgr.shape = (nbmol, nbmol)
            den = 1.j * (sumgr - sumgr.T.conj()) / np.pi / 2
         
            # --sort SGF --
            nres = intctrl.locresz.shape[-1]
            self.locpathinfo[s][k].set_nres(2 * nres)
            loc_e = intctrl.locresz.copy()
            loc_e.shape = (2 * nres, )
            elist = zgp + loc_e.tolist()
            wlist = wgp + [-1.0] * nres + [1.0] * nres
            fcnt = len(elist)
            sgforder = [0] * fcnt
            for i in range(fcnt):
                sgferr = np.min(abs(elist[i] -
                                      np.array(self.zint[:self.cntint + 1 ])))
                
                sgforder[i] = np.argmin(abs(elist[i]
                                     - np.array(self.zint[:self.cntint + 1])))
                if sgferr > 1e-12:
                    print 'Warning: SGF not Found. eqzgp[%d]= %f' %(i,
                                                                     elist[i])
            flist = self.fint[:]
            siglist = [[],[]]
            for i, num in zip(range(fcnt), sgforder):
                flist[i] = self.fint[num]
            sigma= np.empty([nblead, nblead], complex)
            for i in [0, 1]:
                for j, num in zip(range(fcnt), sgforder):
                    sigma = self.tgtint[i, num]
                    siglist[i].append(sigma)
            self.locpathinfo[s][k].add(elist, wlist, flist, siglist)           
        neq = np.trace(np.dot(den, self.greenfunction.S))
        if self.verbose and self.master:
            if self.nspins == 1:
                print '%s NEQ_k[%d]= %f + %f j' % (calcutype, k,
                                           np.real(neq) * 2, np.imag(neq) * 2)
            else:
                print '%s NEQ_sk[%d, %d]= %f, %f j' % (calcutype, s, k,
                                                np.real(neq), np.imag(neq))
        del self.zint, self.tgtint
        if len(intctrl.neintpath) >= 2:
            del self.fint
        return den
         
    def calgfunc(self, zp, calcutype):			 
        #calcutype = 
        #  - 'eqInt':  gfunc[Mx*Mx,nE] (default)
        #  - 'neInt':  gfunc[Mx*Mx,nE]
        #  - 'resInt': gfunc[Mx,Mx] = gr * fint
        #              fint = -2i*pi*kt
      
        intctrl = self.intctrl
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
            if self.cntint + 1 >= len(self.zint):
                self.zint += [0] * stepintcnt
                tmp = self.tgtint.shape[1]
                tmptgtint = np.copy(self.tgtint)
                self.tgtint = np.empty([2, tmp + stepintcnt, nblead, nblead],
                                                                      complex)
                self.tgtint[:, :tmp] = tmptgtint
                self.tgtint[:, tmp:tmp + stepintcnt] = np.zeros([2,
                                                 stepintcnt, nblead, nblead])
            self.cntint += 1
            self.zint[self.cntint] = zp[i]

            for j in [0, 1]:
                self.tgtint[j, self.cntint] = self.selfenergies[j](zp[i])
            
            sigma[:nblead, :nblead] += self.tgtint[0, self.cntint]
            sigma[-nblead:, -nblead:] += self.tgtint[1, self.cntint]
            gamma[0, :nblead, :nblead] += self.selfenergies[0].get_lambda(
                                                                        zp[i])
            gamma[1, -nblead:, -nblead:] += self.selfenergies[1].get_lambda(
                                                                        zp[i])
            gr = self.greenfunction.calculate(zp[i], sigma)       
        
            # --ne-Integral---
            if calcutype == 'neInt':
                gammaocc = np.zeros([nbmol, nbmol], complex)
                for n in [0, 1]:
                    self.fint[n].append(  fermidistribution(zp[i] -
                                         intctrl.leadfermi[n], intctrl.kt) - 
                                         fermidistribution(zp[i] -
                                             intctrl.minfermi, intctrl.kt) )
                    gammaocc += gamma[n] * self.fint[n][self.cntint]
                aocc = np.dot(gr, gammaocc)
                aocc = np.dot(aocc, gr.T.conj())
               
                gfunc[i] = aocc
            elif calcutype == 'neVirInt':
                gammavir = np.zeros([nbmol, nbmol], complex)
                for n in [0, 1]:
                    self.fint[n].append(fermidistribution(zp[i] -
                                         intctrl.maxfermi, intctrl.kt) - 
                                         fermidistribution(zp[i] -
                                             intctrl.leadfermi[n], intctrl.kt))
                    gammavir += gamma[n] * self.fint[n][self.cntint]
                avir = np.dot(gr, gammavir)
                avir = np.dot(avir, gr.T.conj())
                gfunc[i] = avir
            # --local-Integral--
            elif calcutype == 'locInt':
                # fmax-fmin
                self.fint.append( fermidistribution(zp[i] -
                                    intctrl.maxfermi, intctrl.kt) - \
                                    fermidistribution(zp[i] -
                                    intctrl.minfermi, intctrl.kt) )
                gfunc[i] = gr * self.fint[self.cntint]
 
            # --res-Integral --
            elif calcutype == 'resInt':
                self.fint.append(-2.j * np.pi * intctrl.kt)
                gfunc += gr * self.fint[self.cntint]
            #--eq-Integral--
            else:
                if intctrl.kt <= 0:
                    self.fint.append(1.0)
                else:
                    self.fint.append(fermidistribution(zp[i] -
                                                intctrl.minfermi, intctrl.kt))
                gfunc[i] = gr * self.fint[self.cntint]    
        return gfunc        
    
    '''
    def calgfunc(self,  zp, calcutype):
        if type(zp)==list or type(zp) ==np.ndarray:
            pass
        else:
            zp = [zp]
        gfunc = np.empty([len(zp)])
        for i in range(len(zp)):
            gfunc[i] = 1
        return gfunc
    '''
    def fock2den(self, intctrl, f_spkmm, s, k):
        nblead = self.nblead
        nbmol = self.nbmol_inner
        den = np.zeros([nbmol, nbmol], complex)
        denocc = np.zeros([nbmol, nbmol], complex)
        if self.cal_loc:
            denvir = np.zeros([nbmol, nbmol], complex)
            denloc = np.zeros([nbmol, nbmol], complex)
        sigmatmp = np.zeros([nblead, nblead], complex)

        # missing loc states
        eqzp = self.eqpathinfo[s][k].energy
        
        self.greenfunction.H = f_spkmm[s, k]
        self.greenfunction.S = self.s_pkmm_mol[k]
        
        for i in range(len(eqzp)):
            sigma = np.zeros([nbmol, nbmol], complex)  
            sigma[:nblead, :nblead] += self.eqpathinfo[s][k].sigma[0][i]
            sigma[-nblead:, -nblead:] += self.eqpathinfo[s][k].sigma[1][i]
            gr = self.greenfunction.calculate(eqzp[i], sigma)
            fermifactor = self.eqpathinfo[s][k].fermi_factor[i]
            weight = self.eqpathinfo[s][k].weight[i]
            den += gr * fermifactor * weight
        den = 1.j * (den - den.T.conj()) / np.pi / 2

        if self.cal_loc:
            eqzp = self.locpathinfo[s][k].energy
            for i in range(len(eqzp)):
                sigma = np.zeros([nbmol, nbmol], complex)  
                sigma[:nblead, :nblead] += self.locpathinfo[s][k].sigma[0][i]
                sigma[-nblead:, -nblead:] += self.locpathinfo[s][k].sigma[1][i]
                gr = self.greenfunction.calculate(eqzp[i], sigma)
                fermifactor = self.locpathinfo[s][k].fermi_factor[i]
                weight = self.locpathinfo[s][k].weight[i]
                denloc += gr * fermifactor * weight
            denloc = 1.j * (denloc - denloc.T.conj()) / np.pi / 2

        nezp = self.nepathinfo[s][k].energy
        
        intctrl = self.intctrl
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
            denocc += glesser * weight / np.pi / 2
            if self.cal_loc:
                sigmalesser = np.zeros([nbmol, nbmol], complex)
                fermifactor = fermidistribution(nezp[i] -
                                              intctrl.maxfermi, intctrl.kt)-\
                              fermidistribution(nezp[i] -
                                             intctrl.leadfermi[0], intctrl.kt)
                fermifactor = np.real(fermifactor)
                sigmatmp = self.nepathinfo[s][k].sigma[0][i]
                sigmalesser[:nblead, :nblead] += 1.0j * fermifactor * (
                                                   sigmatmp - sigmatmp.T.conj())
                fermifactor = fermidistribution(nezp[i] -
                                         intctrl.maxfermi, intctrl.kt) -  \
                              fermidistribution(nezp[i] -
                                         intctrl.leadfermi[1], intctrl.kt)
                fermifactor = np.real(fermifactor)
                sigmatmp = self.nepathinfo[s][k].sigma[1][i] 
                sigmalesser[-nblead:, -nblead:] += 1.0j * fermifactor * (
                                                   sigmatmp - sigmatmp.T.conj())       
                glesser = np.dot(sigmalesser, gr.T.conj())
                glesser = np.dot(gr, glesser)
                weight = self.nepathinfo[s][k].weight[i]            
                denvir += glesser * weight / np.pi / 2
        if self.cal_loc:
            weight_mm = self.integral_diff_weight(denocc, denvir,
                                                                 'transiesta')
            diff = (denloc - (denocc + denvir)) * weight_mm
            den += denocc + diff
            percents = np.sum( diff * diff ) / np.sum( denocc * denocc )
            print 'local percents %f' % percents
        else:
            den += denocc
        den = (den + den.T.conj()) / 2
        return den    

    def den2fock(self, d_pkmm):
        self.get_density(d_pkmm)
        calc = self.atoms.calc
        calc.update_kinetic()
        density = calc.density
        calc.hamiltonian.update(density)
        linear_potential = self.get_linear_potential()
        calc.hamiltonian.vt_sG += linear_potential
        #xcfunc = calc.hamiltonian.xc.xcfunc
        #calc.Enlxc = xcfunc.get_non_local_energy()
        #calc.Enlkin = xcfunc.get_non_local_kinetic_corrections()   
        h_skmm, s_kmm = self.get_hs(self.atoms)
 
        return h_skmm
    
    def get_density(self,d_spkmm):
        #Calculate pseudo electron-density based on green function.
        calc = self.atoms.calc
        density = calc.density
        wfs = calc.wfs

        nspins = self.nspins
        ntk = self.ntkmol
        npk = self.my_npk
        nbmol = self.nbmol
        relate_layer_num = 3
        dr_mm = np.zeros([nspins, npk, relate_layer_num,
                                                 nbmol, nbmol], complex)
        qr_mm = np.zeros([nspins, npk, nbmol, nbmol])
        
        for s in range(nspins):
            for i in range(relate_layer_num):
                for j in range(self.my_npk):
                    if i == 0:
                        dr_mm[s, j, i] = self.d_spkmm_ij[s, j].T.conj()
                    elif i == 1:
                        dr_mm[s, j, i] = np.copy(d_spkmm[s, j])
                        qr_mm[s, j] += np.dot(dr_mm[s, j, i],
                                                    self.s_pkmm[j]) 
                    elif i == 2:
                        dr_mm[s, j, i]= np.copy(self.d_spkmm_ij[s, j])
                    else:
                        pass
        qr_mm += self.edge_density_mm
        world.barrier()
        self.kpt_comm.sum(qr_mm)
        qr_mm /= self.npk
     
        if self.master:
            if self.verbose:
                for i in range(nspins):
                   print 'spin[%d] charge on atomic basis =' % i
                   print np.diag(np.sum(qr_mm[i],axis=0))

            qr_mm = np.sum(np.sum(qr_mm, axis=0), axis=0)
            natom_inlead = len(self.pl_atoms[0])
            nb_atom = self.nblead / natom_inlead
            pl1 = self.buffer + self.nblead
            natom_print = pl1 / nb_atom 
            edge_charge0 = np.diag(qr_mm[:pl1,:pl1])
            edge_charge1 = np.diag(qr_mm[-pl1:, -pl1:])
            edge_charge0.shape = (natom_print, nb_atom)
            edge_charge1.shape = (natom_print, nb_atom)
            edge_charge0 = np.sum(edge_charge0,axis=1)
            edge_charge1 = np.sum(edge_charge1,axis=1)
            print '***charge distribution at edges***'
            if self.verbose:
                info = []
                for i in range(natom_print):
                    info.append('--' +  str(edge_charge0[i])+'--')
                print info
                info = []
                for i in range(natom_print):
                    info.append('--' +  str(edge_charge1[i])+'--')
                print info
            else:
                edge_charge0.shape = (natom_print / natom_inlead, natom_inlead)
                edge_charge1.shape = (natom_print / natom_inlead, natom_inlead)                
                edge_charge0 = np.sum(edge_charge0,axis=1)
                edge_charge1 = np.sum(edge_charge1,axis=1)
                info = ''
                for i in range(natom_print / natom_inlead):
                    info += '--' +  str(edge_charge0[i]) + '--'
                info += '---******---'
                for i in range(natom_print / natom_inlead):
                    info += '--' +  str(edge_charge1[i]) + '--'
                print info
            print '***total charge***'
            print np.trace(qr_mm)            

        rvector = np.zeros([relate_layer_num, 3])
        tkpts = self.pick_out_tkpts(ntk, self.my_kpts)
        for i in range(relate_layer_num):
            rvector[i, self.d] = i - (relate_layer_num - 1) / 2
        
        self.d_skmm.shape = (nspins, npk, ntk, nbmol, nbmol)
        test = 0
        test1 = 0
        test2 = 0
        for s in range(nspins):
            for i in range(ntk):
                for j in range(npk):
                    self.d_skmm[s, j, i] = get_kspace_hs(None, dr_mm[s, j, :],
                                                         rvector, tkpts[i])
                    self.d_skmm[s, j, i] /=  ntk * self.npk
                    test += np.max(abs(self.d_skmm[s, j,i] - self.d_skmm[s, j , i].T.conj()))
                    if i==0:
                        test1 += np.max(abs(d_spkmm[s, j] - d_spkmm[s, j].T.conj()))
                        test2 += np.max(abs(self.d_spkmm[s, j] - self.d_spkmm[s, j].T.conj()))
        self.d_skmm.shape = (nspins, ntk * npk, nbmol, nbmol)
        print 'sym_test', test, test1, test2
        for kpt in calc.wfs.kpt_u:
            kpt.rho_MM = self.d_skmm[kpt.s, kpt.q]
        density.update(wfs)
        return density

    def calc_total_charge(self, d_spkmm):
        nbmol = self.nbmol 
        qr_mm = np.empty([self.nspins, self.my_npk, nbmol, nbmol])
        for i in range(self.nspins):  
            for j in range(self.my_npk):
                qr_mm[i,j] = np.dot(d_spkmm[i, j],self.s_pkmm[j])
        Qmol = np.trace(np.sum(np.sum(qr_mm, axis=0), axis=0))
        Qmol += np.sum(self.edge_charge)
        Qmol = self.kpt_comm.sum(Qmol) / self.npk
        return Qmol        

    def get_linear_potential(self):
        calc = self.atoms.calc
        linear_potential = np.zeros(calc.hamiltonian.vt_sG.shape)
        
        dimt = linear_potential.shape[-1]
        dimp = linear_potential.shape[1:3]
        dimt_lead = self.dimt_lead
        if self.nblead == self.nbmol:
            buffer_dim = 0
        else:
            buffer_dim = dimt_lead
        scat_dim = dimt - buffer_dim * 2
        bias = self.bias / Hartree
        vt = np.empty([dimt])
        if buffer_dim !=0:
            vt[:buffer_dim] = bias / 2.0
            vt[-buffer_dim:] = -bias / 2.0        
            vt[buffer_dim : -buffer_dim] = np.linspace(bias/2.0,
                                                         -bias/2.0, scat_dim)
        else:
            vt = np.linspace(bias/2.0, -bias/2.0, scat_dim)
        for s in range(self.nspins):
            for i in range(dimt):
                linear_potential[s,:,:,i] = vt[i] * (np.zeros(dimp) + 1)
        return linear_potential
    
    
    def output(self, filename):
        filename1 = filename + str(world.rank)
        fd = file(filename1, 'wb')
        pickle.dump((self.h1_spkmm, 
                     self.h1_spkmm_ij, 
                     self.h_spkmm,
                     self.s1_pkmm, 
                     self.s1_pkmm_ij, 
                     self.s_pkmm,
                     self.d_spkmm,
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
        filename1 = filename + str(world.rank)
        fd = file(filename1, 'r')
        (self.h1_spkmm, 
                     self.h1_spkmm_ij,
                     self.h_spkmm,
                     self.s1_pkmm, 
                     self.s1_pkmm_ij,
                     self.s_pkmm,
                     self.d_spkmm,
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
     
        h_scat = self.h_spkmm[0,0]
        h_lead1 = self.double_size(self.h1_spkmm[0,0],
                                   self.h1_spkmm_ij[0,0])
        h_lead2 = self.double_size(self.h2_spkmm[0,0],
                                   self.h2_spkmm_ij[0,0])
       
        s_scat = self.s_pkmm[0]
        s_lead1 = self.double_size(self.s1_pkmm[0], self.s1_pkmm_ij[0])
        s_lead2 = self.double_size(self.s2_pkmm[0], self.s2_pkmm_ij[0])
        
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
        tcalc = self.set_calculator(e_points)
        tcalc.get_transmission()
        tcalc.get_dos()
        f1 = self.intctrl.leadfermi[0] * (np.zeros([10, 1]) + 1)
        f2 = self.intctrl.leadfermi[1] * (np.zeros([10, 1]) + 1)
        a1 = np.max(tcalc.T_e)
        a2 = np.max(tcalc.dos_e)
        l1 = np.linspace(0, a1, 10)
        l2 = np.linspace(0, a2, 10)
       
        import pylab
        pylab.figure(1)
        pylab.subplot(211)
        pylab.plot(e_points, tcalc.T_e, 'b-o', f1, l1, 'r--', f2, l1, 'r--')
        pylab.ylabel('Transmission Coefficients')
        pylab.subplot(212)
        pylab.plot(e_points, tcalc.dos_e, 'b-o', f1, l2, 'r--', f2, l2, 'r--')
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
    
    def move_buffer(self):
        self.nbmol_inner = self.nbmol - 2 * self.buffer
        pl1 = self.buffer
        if pl1 == 0:
            self.h_spkmm_mol = self.h_spkmm
            self.d_spkmm_mol = self.d_spkmm
            self.s_pkmm_mol = self.s_pkmm
        else:
            self.h_spkmm_mol = self.h_spkmm[:, :, pl1: -pl1, pl1:-pl1]
            self.d_spkmm_mol = self.d_spkmm[:, :, pl1: -pl1, pl1:-pl1]
            self.s_pkmm_mol = self.s_pkmm[:, pl1: -pl1, pl1:-pl1]
            
    def allocate_cpus(self):
        rank = world.rank
        size = world.size
        npk = self.npk
        npk_each = npk / size
        r0 = rank * npk_each
        self.my_pk = np.arange(r0, r0 + npk_each)
        self.my_npk = npk_each
        self.kpt_comm = world.new_communicator(np.arange(size))

        self.my_kpts = np.empty((npk_each * self.ntkmol, 3), complex)
        kpts = self.atoms.calc.wfs.ibzk_kc
        for i in range(self.ntkmol):
            for j, k in zip(range(npk_each), self.my_pk):
                self.my_kpts[j * self.ntkmol + i] = kpts[k * self.ntkmol + i]        

        self.my_lead_kpts = np.empty((npk_each * self.ntklead, 3), complex)
        kpts = self.atoms_l[0].calc.wfs.ibzk_kc
        for i in range(self.ntklead):
            for j, k in zip(range(npk_each), self.my_pk):
                self.my_lead_kpts[j * self.ntklead + i] = kpts[
                                                        k * self.ntklead + i]         

    def real_projecting(self):
        world.barrier()
        self.h_spmm = np.sum(self.h_spkmm, axis=1)
        self.s_pmm = np.sum(self.s_pkmm, axis=0)
        self.d_spmm = np.sum(self.d_spkmm, axis=1)
        self.kpt_comm.sum(self.h_spmm)
        self.kpt_comm.sum(self.s_pmm)
        self.kpt_comm.sum(self.d_spmm)
        self.h_spmm /= self.npk
        self.s_pmm /= self.npk
        self.d_spmm /= self.npk

        
    def collect_density_matrix(self):
        world.barrier()
        npk = self.npk
        self.d_stkmm = np.zeros([self.nspins, npk, self.nbmol, self.nbmol], complex)
        for pk, kk in zip(self.my_pk, range(self.my_npk)):
            self.d_stkmm[:, pk] = self.d_spkmm[:, kk]
        self.kpt_comm.sum(self.d_stkmm)
        for i in range(npk):
            self.d_stkmm[0,i] = (self.d_stkmm[0,i] + self.d_stkmm[0,i].T.conj()) / 2

    def distribute_density_matrix(self, d_stkmm):
        world.barrier()
        dagger_tol = 1e-9
        d_spkmm = np.empty(self.d_spkmm.shape, complex)
        for pk, kk in zip(self.my_pk, range(self.my_npk)):
            if np.max(abs(d_stkmm[0, pk] -
                          d_stkmm[0, pk].T.conj())) > dagger_tol:
                print 'Warning, density matrix is not dagger symmetric'
            d_spkmm[0, kk] = (d_stkmm[0, pk] + d_stkmm[0, pk].T.conj()) / 2.
        return d_spkmm
     
    def integral_diff_weight(self, denocc, denvir, method='transiesta'):
        if method=='transiesta':
            weight = denocc * denocc.conj() / (denocc * denocc.conj() +
                                               denvir * denvir.conj())
        return weight

    def revoke_right_lead(self):
        self.h2_spkmm = self.h1_spkmm
        self.s2_pkmm = self.s1_pkmm

        self.h2_spkmm_ij = np.empty(self.h1_spkmm_ij.shape, complex)
        self.s2_pkmm_ij = np.empty(self.s1_pkmm_ij.shape, complex)

        nspins = self.h2_spkmm.shape[0]
        npk = self.h2_spkmm.shape[1]

        for s in range(nspins):
            for k in range(npk):
                 self.h2_spkmm_ij[s,k] = self.h1_spkmm_ij[s,k].T.conj()
        for k in range(npk):
            self.s2_pkmm_ij[k] = self.s1_pkmm_ij[k].T.conj()

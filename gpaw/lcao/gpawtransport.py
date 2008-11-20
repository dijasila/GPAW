from ase import Atoms, Atom, monkhorst_pack, Hartree
import ase
from gpaw import GPAW, Mixer, restart
from gpaw.lcao.tools import get_realspace_hs, get_realspace_h, tri2full, remove_pbc
import pickle
import numpy as npy
from gpaw.mpi import world
from gpaw.utilities.lapack import diagonalize
from ase.transport.tools import mytextwrite1, dagger

class GPAWTransport:
    
    def __init__(self, atoms, pl_atoms, pl_cells, d=0):
        self.atoms = atoms
        if not self.atoms.calc.initialized:
            self.atoms.calc.initialize(atoms)
        self.pl_atoms = pl_atoms
        self.pl_cells = pl_cells
        self.d = d
        self.atoms_l = [None,None]
        #
        #self.leadnxk = [None,None]
        #----
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
        h1 = npy.zeros((2*pl1, 2 * pl1), complex)
        s1 = npy.zeros((2*pl1, 2 * pl1), complex)

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
        
        if calc1.master:
            print "Dumping lead 1 hamiltonian..."
            fd = file('lead1_' + filename, 'wb')
            pickle.dump((h1, s1), fd, 2)
            fd.close()

        world.barrier()
        
        self.update_lead_hamiltonian(1) 
        pl2 = self.h2_skmm.shape[-1]
        h2 = npy.zeros((2 * pl2, 2 * pl2), complex)
        s2 = npy.zeros((2 * pl2, 2 * pl2), complex)

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
        
        s2[:pl2,:pl2] = s2_ii
        s2[pl2:2*pl2,pl2:2*pl2] = s2_ii
        s2[:pl2,pl2:2*pl2] = s2_ij
        tri2full(s2,'U')

        if calc2.master:
            print "Dumping lead 2 hamiltonian..."
            fd = file('lead2_'+filename,'wb')
            pickle.dump((h2,s2),fd,2)
            fd.close()

        world.barrier()
        
        del self.atoms_l

        self.update_scat_hamiltonian()
        nbf_m = self.h_skmm.shape[-1]
        nbf = nbf_m + pl1 + pl2
        h = npy.zeros((nbf, nbf), complex)
        s = npy.zeros((nbf, nbf), complex)
        
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
  
        if atoms.calc.master:
            print "Dumping scat hamiltonian..."
            fd = file('scat_'+filename,'wb')
            pickle.dump((h,s),fd,2)
            fd.close()
        world.barrier()

    def update_lead_hamiltonian(self, l, flag=0):
        # flag: 0 for calculation, 1 for read from file
        if flag == 0:
            self.atoms_l[l] = self.get_lead_atoms(l)
            atoms = self.atoms_l[l]
            atoms.get_potential_energy()
            atoms.calc.write('lead' + str(l) + '.gpw')
            if l == 0:
                self.h1_skmm, self.s1_kmm = self.get_hs(atoms)
                fd = file('leadhs0','wb')
                pickle.dump((self.h1_skmm, self.s1_kmm), fd, 2)            
                fd.close()            
            elif l == 1:
                self.h2_skmm, self.s2_kmm = self.get_hs(atoms)
                fd = file('leadhs1','wb')
                pickle.dump((self.h2_skmm, self.s2_kmm), fd, 2)
                fd.close()
        else:
            atoms, calc = restart('lead' + str(l) + '.gpw')
            self.atoms_l[l] = atoms
            if l == 0:        
                fd = file('leadhs0','r') 
                self.h1_skmm, self.s1_kmm = pickle.load(fd)
                fd.close()
            elif l == 1:
                fd = file('leadhs1','r') 
                self.h2_skmm, self.s2_kmm = pickle.load(fd)
                fd.close()

    def update_scat_hamiltonian(self, flag=0):
        # flag: 0 for calculation, 1 for read from file
        if flag == 0:
            atoms = self.atoms
            atoms.get_potential_energy()
            atoms.calc.write('scat.gpw')
            self.h_skmm, self.s_kmm = self.get_hs(atoms)
            fd = file('scaths', 'wb')
            pickle.dump((self.h_skmm, self.s_kmm), fd, 2)
            fd.close()
            calc0 = atoms.calc
            nm = sum([nucleus.get_number_of_atomic_orbitals()
                                                for nucleus in calc0.nuclei])
            phit_mG = calc0.gd.zeros(nm)
            for nucleus in calc0.nuclei:
                m1 = nucleus.m
                m2 = m1 + nucleus.get_number_of_atomic_orbitals()
                nucleus.phit_i.add(phit_mG[m1:m2], npy.identity(m2 - m1))
            fd = file('locfun_g.dat', 'wb')
            pickle.dump((phit_mG, calc0.density.nct_G), fd, 2)
            fd.close()
            self.phit_mG = npy.copy(phit_mG)
            self.nct_G = npy.copy(calc0.density.nct_G)
                        
        else :
            atoms, calc = restart('scat.gpw')
            self.atoms = atoms
            fd = file('scaths', 'r')
            self.h_skmm, self.s_kmm = pickle.load(fd)
            fd.close()
            fd = file('locfun_g.dat', 'r')
            self.phit_mG, self.nct_G = pickle.load(fd)
            fd.close()
            

    def get_hs(self, atoms):
        calc = atoms.calc
        Ef = calc.get_fermi_level()
        print'fermi_level'
        print Ef
        eigensolver = calc.eigensolver
        ham = calc.hamiltonian
        Vt_skmm = eigensolver.Vt_skmm
        ham.calculate_effective_potential_matrix(Vt_skmm)
        ibzk_kc = calc.ibzk_kc
        nkpts = len(ibzk_kc)
        nspins = calc.nspins
        weight_k = calc.weight_k
        nao = calc.nao
        h_skmm = npy.zeros((nspins, nkpts, nao, nao), complex)
        s_kmm = npy.zeros((nkpts, nao, nao), complex)
        for k in range(nkpts):
            s_kmm[k] = ham.S_kmm[k]
            tri2full(s_kmm[k])
            for s in range(nspins):
                h_skmm[s,k] = calc.eigensolver.get_hamiltonian_matrix(ham,
                                                                      k=k,
                                                                      s=s)
                tri2full(h_skmm[s, k])
                h_skmm[s,k] *= Hartree
                #h_skmm[s,k] -= Ef * s_kmm[k]

        return h_skmm, s_kmm

    def get_lead_atoms(self, l):
        """l: 0, 1 correpsonding to left, right """
        atoms = self.atoms.copy()
        atomsl = Atoms(pbc=atoms.pbc, cell=self.pl_cells[l])
    
        for a in self.pl_atoms[l]:
            atomsl.append(atoms[a])
       
        atomsl.center()
        atomsl.set_calculator(self.get_lead_calc(l))
        return atomsl

    def get_lead_calc(self, l):
        p = self.atoms.calc.input_parameters.copy()
        p['nbands'] = None
        ##
        #kpts = [1, 1, 1]
        #kpts = list(p['kpts'])
        #kpts[self.d] = 2 * int(17.0 / self.pl_cells[l][self.d]) + 1
        #kpts[self.d] = 13
        #self.leadnxk[l] = kpts[self.d]
        #p['kpts'] = kpts
   
        if 'mixer' in p: # XXX Works only if spin-paired
            p['mixer'] = Mixer(0.1, 5, metric='new', weight=100.0)

        if 'txt' in p and p['txt'] != '-':
            p['txt'] = 'lead%i_' % (l + 1) + p['txt']
        return GPAW(**p)
        
    ##
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
        #self.atoms_l[0] = self.get_lead_atoms(0)
        #self.atoms_l[1] = self.get_lead_atoms(1)
        
    def prepare(self, filename, flag=0):
        # flag: 0 for calculation, 1 for read from file
        self.update_lead_hamiltonian(0, flag)
        atoms1 = self.atoms_l[0]
        calc1 = atoms1.calc
        kpts = npy.copy(calc1.ibzk_kc)
        p = self.atoms.calc.input_parameters.copy()
        nxk = p['kpts'][0]
        #nxk = 13
        nyzk = kpts.shape[0] / nxk
        pl1 = self.h1_skmm.shape[-1]
        nspin = self.h1_skmm.shape[0]

        h = npy.zeros((nspin, nyzk, 2*pl1, 2 * pl1), complex)
        d = npy.zeros((nspin, nyzk, 2*pl1, 2 * pl1), complex)
        s = npy.zeros((nyzk, 2*pl1, 2 * pl1), complex)

        h_sii, s_ii, h_sij, s_ij = self.substract_yzk(nxk, kpts,
                                                    self.h1_skmm, self.s1_kmm)
        for i in range(nyzk):
            h[0, i, :pl1, :pl1] = h_sii[0][i]
            h[0, i, pl1:2 * pl1, pl1:2 * pl1] = h_sii[0][i]
            h[0, i, :pl1, pl1:2 * pl1] = h_sij[0][i]
            tri2full(h[0][i], 'U')
         
            s[i, :pl1, :pl1] = s_ii[i]
            s[i, pl1:2*pl1, pl1:2*pl1] = s_ii[i]
            s[i, :pl1, pl1:2*pl1] = s_ij
            tri2full(s[i], 'U')        
        self.h1_syzkmm = npy.copy(h)
        self.s1_yzkmm = npy.copy(s)
        
        self.d1_skmm = npy.empty(self.h1_skmm.shape, complex)
        
        for kpt in calc1.kpt_u:
            C_nm = kpt.C_nm
            f_nn = npy.diag(kpt.f_n)
            self.d1_skmm[kpt.s, kpt.k] = npy.dot(dagger(C_nm), npy.dot(f_nn, C_nm))
        d_sii, s_ii,  d_sij , s_ij = self.substract_yzk(nxk, kpts, self.d1_skmm, self.s1_kmm)
        
        for i in range(nyzk):
            d[0, i, :pl1, :pl1] = d_sii[0][i]
            d[0, i, pl1:2 * pl1, pl1:2 * pl1] = d_sii[0][i]
            d[0, i, :pl1, pl1:2 * pl1] = d_sij[0][i]
            tri2full(d[0][i], 'U')
        self.d1_syzkmm = npy.copy(d)       
        
        
                
        self.update_lead_hamiltonian(1, flag)
        
        pl2 = self.h2_skmm.shape[-1]
        h = npy.zeros((nspin, nyzk, 2*pl2, 2 * pl2), complex)
        s = npy.zeros((nyzk, 2*pl2, 2 * pl2), complex)
        
        h_sii, s_ii, h_sij, s_ij = self.substract_yzk(nxk, kpts,
                                                    self.h2_skmm, self.s2_kmm)
        for i in range(nyzk):
            h[0, i, :pl1, :pl1] = h_sii[0][i]
            h[0, i, pl1:2 * pl1, pl1:2 * pl1] = h_sii[0][i]
            h[0, i, :pl1, pl1:2 * pl1] = h_sij[0][i]
            tri2full(h[0][i], 'U')
         
            s[i, :pl1, :pl1] = s_ii[i]
            s[i, pl1:2*pl1, pl1:2*pl1] = s_ii[i]
            s[i, :pl1, pl1:2*pl1] = s_ij
            tri2full(s[i], 'U')        
        self.h2_syzkmm = npy.copy(h)
        self.s2_yzkmm = npy.copy(s)

        
        self.update_scat_hamiltonian(flag)
        #p = self.atoms.calc.input_parameters.copy()
        #nxk= p['kpts']
        #nxk = 13
        kpts = self.atoms.calc.ibzk_kc
        nyzk = kpts.shape[0] / nxk
        self.h_syzkmm, self.s_yzkmm, self.h_syzkmm_ij, self.s_yzkmm_ij = \
                        self.substract_yzk(nxk, kpts, self.h_skmm, self.s_kmm)
         
        
        
        
        
        
       
    def substract_yzk(self, nxk, kpts, h_skmm, s_kmm):
        dimh = h_skmm.shape[:]
        nyzk = dimh[1] / nxk
        dimh = (dimh[0],) + (dimh[1] / nxk,) + dimh[2:]
      
        h_syzkmm = npy.empty(dimh, complex)
        s_yzkmm = npy.empty(dimh[1:], complex)
        
        h_syzkmm_ij = npy.empty(dimh, complex)
        s_yzkmm_ij = npy.empty(dimh[1:], complex)

        dimh = (dimh[0],) + (nxk,) + dimh[2:]
        h_sxkmm = npy.empty(dimh, complex)
        s_xkmm = npy.empty(dimh[1:], complex)
        
        xkpts = npy.empty([nxk, 3])
        weight = npy.array([1.0 / nxk] * nxk )
        for i in range(nyzk):
            n = i
            for j in range(nxk):
                xkpts[j] = kpts[n]
                h_sxkmm[:, j] = npy.copy(h_skmm[:, n])
                s_xkmm[j] = npy.copy(s_kmm[n])
                n += nyzk
            h_syzkmm[:, i], s_yzkmm[i] = get_realspace_hs(h_sxkmm, s_xkmm,
                                                   xkpts, weight, R_c=(0,0,0)) 
            h_syzkmm_ij[:, i], s_yzkmm_ij[i] = get_realspace_hs(h_sxkmm, s_xkmm,
                                                   xkpts, weight, R_c=(1.0,0,0))
                
        return h_syzkmm, s_yzkmm, h_syzkmm_ij, s_yzkmm_ij    
            
    def substract_yzk1(self, nxk, kpts, h_skmm):
        dimh = h_skmm.shape[:]
        nyzk = dimh[1] / nxk
        dimh = (dimh[0],) + (dimh[1] / nxk,) + dimh[2:]
      
        h_syzkmm = npy.empty(dimh, complex)
        
        h_syzkmm_ij = npy.empty(dimh, complex)

        dimh = (dimh[0],) + (nxk,) + dimh[2:]
        h_sxkmm = npy.empty(dimh, complex)
        
        xkpts = npy.empty([nxk, 3])
        weight = npy.array([1.0 / nxk] * nxk )
        for i in range(nyzk):
            n = i
            for j in range(nxk):
                xkpts[j] = kpts[n]
                h_sxkmm[:, j] = npy.copy(h_skmm[:, n])
                n += nyzk
            h_syzkmm[:, i] = get_realspace_h(h_sxkmm,
                                                   xkpts, weight, R_c=(0,0,0)) 
            h_syzkmm_ij[:, i]= get_realspace_h(h_sxkmm,
                                                   xkpts, weight, R_c=(1.0,0,0))
                
        return h_syzkmm, h_syzkmm_ij,  

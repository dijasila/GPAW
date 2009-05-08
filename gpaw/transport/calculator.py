import pickle

from gpaw.transport.selfenergy import LeadSelfEnergy, CellSelfEnergy
from gpaw.transport.greenfunction import GreenFunction
from ase.transport.tools import function_integral, fermidistribution
from ase import Atoms, Atom, monkhorst_pack, Hartree, Bohr
import ase
import numpy as np

import gpaw
from gpaw import GPAW
from gpaw import Mixer, MixerDif
from gpaw import restart as restart_gpaw
from gpaw.transport.tools import k2r_hs, r2k_hs, tri2full, dot
from gpaw.transport.surrounding import Surrounding
from gpaw.grid_descriptor import GridDescriptor
from gpaw.lcao.tools import get_realspace_hs
from gpaw.mpi import world
from gpaw.utilities.lapack import diagonalize
from gpaw.utilities import pack
from gpaw.transport.intctrl import IntCtrl
from gpaw.utilities.timing import Timer

class PathInfo:
    def __init__(self, type, nlead):
        self.type = type
        self.num = 0
        self.lead_num = nlead
        self.energy = []
        self.weight = []
        self.nres = 0
        self.sigma = []
        for i in range(nlead):
            self.sigma.append([])
        if type == 'eq':
            self.fermi_factor = []
        elif type == 'ne':
            self.fermi_factor = []
            for i in range(nlead):
                self.fermi_factor.append([[], []])
        else:
            raise TypeError('unkown PathInfo type')

    def add(self, elist, wlist, flist, siglist):
        self.num += len(elist)
        self.energy += elist
        self.weight += wlist
        if self.type == 'eq':
            self.fermi_factor += flist
        elif self.type == 'ne':
            for i in range(self.lead_num):
                for j in [0, 1]:
                    self.fermi_factor[i][j] += flist[i][j]
        else:
            raise TypeError('unkown PathInfo type')
        for i in range(self.lead_num):
            self.sigma[i] += siglist[i]

    def set_nres(self, nres):
        self.nres = nres
    
class Transport(GPAW):
    
    def __init__(self, **transport_kwargs):
        self.set_transport_kwargs(**transport_kwargs)
        if self.scat_restart:
            GPAW.__init__(self, self.restart_file + '.gpw')
            self.set_positions()
            self.verbose = self.transport_parameters['verbose']
        else:
            GPAW.__init__(self, **self.gpw_kwargs)            
            
    def set_transport_kwargs(self, **transport_kwargs):
        kw = transport_kwargs  
        p =  self.set_default_transport_parameters()
        self.gpw_kwargs = kw.copy()
        for key in kw:
            if key in ['use_lead', 'identical_leads',
                       'pl_atoms', 'pl_cells', 'pl_kpts',
                       'use_buffer', 'buffer_atoms', 'edge_atoms', 'bias',
                       'lead_restart',
                       
                       'use_env', 'env_atoms', 'env_cells', 'env_kpts',
                       'env_use_buffer', 'env_buffer_atoms', 'env_edge_atoms',
                       'env_bias', 'env_pbc', 'env_restart',                     
                       
                       'LR_leads', 'gate',  'cal_loc',
                       'recal_path', 'use_qzk_boundary', 'use_linear_vt_mm',
                       'scat_restart', 'save_file', 'restart_file', 'fixed_boundary']:
                
                del self.gpw_kwargs[key]
            #----descript the lead-----    
            if key in ['use_lead']:
                p['use_lead'] = kw['use_lead']
            if key in ['identical_leads']:
                p['identical_leads'] = kw['identical_leads']
            if key in ['pl_atoms']:
                p['pl_atoms'] = kw['pl_atoms']
            if key in ['pl_cells']:
                p['pl_cells'] = kw['pl_cells']
            if key in ['pl_kpts']:
                p['pl_kpts'] = kw['pl_kpts']
            if key in ['use_buffer']:
                p['use_buffer'] = kw['use_buffer']
            if key in ['buffer_atoms']:
                p['buffer_atoms'] = kw['buffer_atoms']
            if key in ['edge_atoms']:
                p['edge_atoms'] = kw['edge_atoms']
            if key in ['bias']:
                p['bias'] = kw['bias']                
            if key in ['lead_restart']:
                p['lead_restart'] = kw['lead_restart']
            #----descript the environment----   
            if key in ['use_env']:
                p['use_env'] = kw['use_env']
            if key in ['env_atoms']:
                p['env_atoms'] = kw['env_atoms']
            if key in ['env_cells']:
                p['env_cells'] = kw['env_cells']
            if key in ['env_kpts']:
                p['env_kpts'] = kw['env_kpts']
            if key in ['env_buffer_atoms']:
                p['env_buffer_atoms'] = kw ['env_buffer_atoms']
            if key in ['env_edge_atoms']:
                p['env_edge_atoms'] = kw['env_edge_atoms']
            if key in ['env_pbc']:
                p['env_pbc'] = kw['env_pbc']
            if key in ['env_bias']:
                p['env_bias'] = kw['env_bias']
            if key in ['env_restart']:
                p['env_restart'] = kw['env_restart']
            #----descript the scattering region----     
            if key in ['LR_leads']:         
                p['LR_leads'] = kw['LR_leads']
            if key in ['gate']:
                p['gate'] = kw['gate']
            if key in ['cal_loc']:
                p['cal_loc'] = kw['cal_loc']
            if key in ['recal_path']:
                p['recal_path'] = kw['recal_path']
            if key in ['use_qzk_boundary']:
                p['use_qzk_boundary'] = kw['use_qzk_boundary']
            if key in ['use_linear_vt_mm']:
                p['use_linear_vt_mm'] = kw['use_linear_vt_mm']
            if key in ['scat_restart']:
                p['scat_restart'] = kw['scat_restart']
            if key in ['save_file']:
                p['save_file'] = kw['save_file']
            if key in ['restart_file']:
                p['restart_file'] = kw['restart_file']
            if key in ['fixed_boundary']:
                p['fixed_boundary'] = kw['fixed_boundary']
            if key in ['spinpol']:
                p['spinpol'] = kw['spinpol']
            if key in ['verbose']:
                p['verbose'] = kw['verbose']

        self.transport_parameters = p

        self.use_lead = p['use_lead']
        self.identical_leads = p['identical_leads']
        self.pl_atoms = p['pl_atoms']
        self.lead_num = len(self.pl_atoms)
        self.bias = p['bias']

        if self.use_lead:
            self.pl_cells = p['pl_cells']
            self.pl_kpts = p['pl_kpts']
            self.lead_restart = p['lead_restart']
            self.use_buffer = p['use_buffer']
            self.buffer_atoms = p['buffer_atoms']
            self.edge_atoms = p['edge_atoms']
            assert self.lead_num == len(self.pl_cells)
            #assert self.lead_num == len(self.buffer_atoms)
            #assert self.lead_num == len(self.edge_atoms[0])
            assert self.lead_num == len(self.bias)
            
        self.use_env = p['use_env']
        self.env_atoms = p['env_atoms']
        self.env_num = len(self.env_atoms)
        self.env_bias = p['env_bias']

        if self.use_env:
            self.env_cells = p['env_cells']
            self.env_kpts = p['env_kpts']
            self.env_buffer_atoms = p['env_buffer_atoms']
            self.env_edge_atoms = p['env_edge_atoms']
            self.env_pbc = p['env_pbc']
            self.env_restart = p['env_restart']
            assert self.env_num == len(self.env_cells)
            assert self.env_num == len(self.env_buffer_atoms)
            assert self.env_num == len(self.env_edge_atoms[0])
            assert self.env_num == len(self.env_bias)

        self.LR_leads = p['LR_leads']            
        self.gate = p['gate']
        self.cal_loc = p['cal_loc']
        self.recal_path = p['recal_path']
        self.use_qzk_boundary = p['use_qzk_boundary']
        self.use_linear_vt_mm = p['use_linear_vt_mm']
        self.scat_restart = p['scat_restart']
        self.save_file = p['save_file']
        self.restart_file = p['restart_file']
        self.fixed = p['fixed_boundary']
        self.spinpol = p['spinpol']
        self.verbose = p['verbose']
        self.d = p['d']
       
        if self.scat_restart and self.restart_file == None:
            self.restart_file = 'scat'
        
        self.master = (world.rank==0)
    
        bias = self.bias + self.env_bias
        self.cal_loc = self.cal_loc and max(abs(bias)) != 0
 
        if self.use_linear_vt_mm:
            self.use_buffer = False
        
        if self.LR_leads and self.lead_num != 2:
            raise RuntimeErrir('wrong way to use keyword LR_leads')
       
        self.initialized_transport = False
       
        self.atoms_l = [None] * self.lead_num
        self.atoms_e = [None] * self.env_num
        
        kpts = kw['kpts']
        if np.product(kpts) == kpts[self.d]:
            self.gpw_kwargs['usesymm'] = None
        else:
            self.gpw_kwargs['usesymm'] = False
   
    def set_default_transport_parameters(self):
        p = {}
        p['use_lead'] = True
        p['identical_leads'] = True
        p['pl_atoms'] = []
        p['pl_cells'] = []
        p['pl_kpts'] = []
        p['use_buffer'] = True
        p['buffer_atoms'] = None
        p['edge_atoms'] = None
        p['bias'] = []
        p['d'] = 2
        p['lead_restart'] = False

        p['use_env'] = False
        p['env_atoms'] = []
        p['env_cells']  = []
        p['env_kpts'] = []
        p['env_use_buffer'] = False
        p['env_buffer_atoms'] = None
        p['env_edge_atoms'] = None
        p['env_pbc'] = True
        p['env_bias'] = []
        p['env_restart'] = False
        
        p['LR_leads'] = True
        p['gate'] = 0
        p['cal_loc'] = False
        p['recal_path'] = False
        p['use_qzk_boundary'] = False 
        p['use_linear_vt_mm'] = False
        p['scat_restart'] = False
        p['save_file'] = True
        p['restart_file'] = None
        p['fixed_boundary'] = False
        p['spinpol'] = False
        p['verbose'] = False
        return p     

    def set_atoms(self, atoms):
        self.atoms = atoms.copy()
        
    def initialize_transport(self, dry=False):
        if not self.initialized:
            self.initialize()
            self.set_positions()
        self.nspins = self.wfs.nspins

        if self.LR_leads:
            self.ntkmol = self.gpw_kwargs['kpts'][self.d]
            self.ntklead = self.pl_kpts[self.d]
            if self.ntkmol == len(self.wfs.bzk_kc):
                self.npk = 1
                self.kpts = self.wfs.bzk_kc
            else:
                self.npk = len(self.wfs.ibzk_kc) / self.ntkmol
                self.kpts = self.wfs.ibzk_kc
        else:
            self.npk = 1
            self.kpts = self.wfs.ibzk_kc
            self.ntkmol = len(self.kpts)
            if self.use_lead:
                self.ntklead = np.product(self.pl_kpts)

        self.gamma = len(self.kpts) == 1
        self.nbmol = self.wfs.setups.nao

        if self.use_lead:
            if self.LR_leads:
                self.dimt_lead = []
                self.dimt_buffer = []
            self.nblead = []
            self.edge_index = [[None] * self.lead_num, [None] * self.lead_num]

        if self.use_env:
            self.nbenv = []
            self.env_edge_index = [[None] * self.env_num, [None] * self.env_num]

        for i in range(self.lead_num):
            self.atoms_l[i] = self.get_lead_atoms(i)
            calc = self.atoms_l[i].calc
            atoms = self.atoms_l[i]
            if not calc.initialized:
                calc.initialize(atoms)
                calc.set_positions(atoms)
            self.nblead.append(calc.wfs.setups.nao)
            if self.LR_leads:
                self.dimt_lead.append(calc.gd.N_c[self.d])

        for i in range(self.env_num):
            self.atoms_e[i] = self.get_env_atoms(i)
            calc = self.atoms_e[i].calc
            atoms = self.atoms_e[i]
            if not calc.initialized:
                calc.initialize(atoms)
                calc.set_positions(atoms)
            self.nbenv.append(calc.wfs.setups.nao)

        if self.use_lead:
            if self.LR_leads and self.npk == 1:    
                self.lead_kpts = self.atoms_l[0].calc.wfs.bzk_kc
            else:
                self.lead_kpts = self.atoms_l[0].calc.wfs.ibzk_kc
        if self.use_env:
            self.env_kpts = self.atoms_e[0].calc.wfs.ibzk_kc               
        self.allocate_cpus()
        self.initialize_matrix()
       
        if self.use_lead:
            self.get_lead_index()
            self.get_buffer_index()
        if self.use_env:
            self.get_env_index()
            self.get_env_buffer_index()

        if self.use_lead:
            if self.nbmol <= np.sum(self.nblead):
                self.use_buffer = False
                if self.master:
                    self.text('Moleucle is too small, force not to use buffer')
            
        if self.use_lead:
            if self.use_buffer: 
                self.buffer = [len(self.buffer_index[i])
                                                   for i in range(self.lead_num)]
                self.print_index = self.buffer_index
            else:
                self.buffer = [0] * self.lead_num
                self.print_index = self.lead_index
            
        if self.use_env:
            self.env_buffer = [len(self.env_buffer_index[i])
                                               for i in range(self.env_num)]
        self.set_buffer()
            
        self.current = 0
        self.linear_mm = None

        if not dry:        
            for i in range(self.lead_num):
                if self.identical_leads and i > 0:
                    self.update_lead_hamiltonian(i, 'lead0')    
                else:
                    self.update_lead_hamiltonian(i)
                self.initialize_lead(i)

            for i in range(self.env_num):
                self.update_env_hamiltonian(i)
                self.initialize_env(i)

        self.fermi = self.lead_fermi[0]
        self.text('*****should change here*******')
        world.barrier()
        if self.use_lead:
            self.check_edge()
            self.get_edge_density()
        
        del self.atoms_l
        del self.atoms_e

        self.initialized_transport = True
        
        if self.fixed:
            self.atoms.calc = self
            self.surround = Surrounding(h=0.3,
                                        xc='PBE',
                                        basis='szp',
                                        kpts=(1,1,13),
                                        width=0.01,
                                        mode='lcao',
                                        usesymm=False,
                                        mixer=Mixer(0.1, 5, metric='new', weight=100.0),
                                        name='test',
                                        type='LR',
                                        atoms=self.atoms,
                                        h_c=self.gd.h_c,
                                        directions=['z-','z+'],
                                        N_c=self.gd.N_c,
                                        pl_atoms=self.pl_atoms,
                                        pl_cells=self.pl_cells,
                                        pl_kpts=self.pl_kpts,
                                        pl_pbcs=[(0,0,1),(0,0,1)],
                                        bias=self.bias)
            self.surround.initialize()
            self.surround.calculate_sides()
            self.surround.combine_vHt_g()
            self.surround.combine_vt_sG()

    def get_lead_index(self):
        basis_list = [setup.niAO for setup in self.wfs.setups]
        lead_basis_list = []
        for i in range(self.lead_num):
            calc = self.atoms_l[i].calc
            lead_basis_list.append([setup.niAO for setup in calc.wfs.setups])
            for j, lj  in zip(self.pl_atoms[i], range(len(self.pl_atoms[i]))):
                begin = np.sum(np.array(basis_list[:j], int))
                l_begin = np.sum(np.array(lead_basis_list[i][:lj], int))
                if self.edge_atoms == None:
                    self.edge_index[0][i] = -i * lead_basis_list[i][-1]
                    self.edge_index[1][i] = -i * lead_basis_list[i][-1]
                    assert self.lead_num == 2
                else:
                    if j == self.edge_atoms[1][i]:
                        self.edge_index[1][i] = begin
                    if lj == self.edge_atoms[0][i]:
                        self.edge_index[0][i] = l_begin
                for n in range(basis_list[j]):
                    self.lead_index[i].append(begin + n) 
            self.lead_index[i] = np.array(self.lead_index[i], int)
            
    def get_env_index(self):
        basis_list = [setup.niAO for setup in self.wfs.setups]
        env_basis_list = []
        for i in range(self.env_num):
            calc = self.atoms_e[i].calc
            env_basis_list.append([setup.niAO for setup in calc.wfs.setups])
            for j, lj in zip(self.env_atoms[i], range(len(self.env_atoms[i]))):
                begin = np.sum(np.array(basis_list[:j], int))
                l_begin = np.sum(np.array(env_basis_list[i][:lj], int))
                if j == self.env_edge_atoms[1][i]:
                    self.env_edge_index[1][i] = begin
                if lj == self.env_edge_atoms[0][i]:
                    self.env_edge_index[0][i] = l_begin
                for n in range(basis_list[j]):
                    self.env_index[i].append(begin + n)
            self.env_index[i] = np.array(self.env_index[i], int)
            
    def get_buffer_index(self):
        if not self.use_buffer:
            self.dimt_buffer = [0] * self.lead_num
        elif self.LR_leads:
            for i in range(self.lead_num):
                if i == 0:
                    self.buffer_index[i] = self.lead_index[i] - self.nblead[i]
                if i == 1:
                    self.buffer_index[i] = self.lead_index[i] + self.nblead[i]
            self.dimt_buffer = self.dimt_lead
        else:
            basis_list = [setup.niAO for setup in self.wfs.setups]
            for i in range(self.lead_num):
                for j in self.buffer_atoms[i]:
                    begin = np.sum(np.array(basis_list[:j], int))
                    for n in range(basis_list[j]):
                        self.buffer_index[i].append(begin + n) 
                self.buffer_index[i] = np.array(self.buffer_index[i], int)
                self.dimt_buffer.append(self.dimt_lead[i] *
                                        len(self.buffer_atoms[i]) /
                                           len(self.pl_atoms[i]))
    
    def get_env_buffer_index(self):
        basis_list = [setup.niAO for setup in self.wfs.setups]
        for i in range(self.env_num):
            for j in self.env_buffer_atoms[i]:
                begin = np.sum(np.array(basis_list[:j], int))
                for n in range(basis_list[j]):
                    self.env_buffer_index[i].append(begin + n)
            self.env_buffer_index[i] = np.array(self.env_buffer_index[i], int)
           
    def initialize_matrix(self):
        if self.use_lead:
            self.hl_skmm = []
            self.dl_skmm = []
            self.sl_kmm = []
            self.hl_spkmm = []
            self.dl_spkmm = []
            self.sl_pkmm = []
            self.hl_spkcmm = []
            self.dl_spkcmm = []
            self.sl_pkcmm = []
            self.ed_pkmm = []
            self.lead_index = []
            self.inner_lead_index = []
            self.buffer_index = []
            self.lead_fermi = np.empty([self.lead_num])

        if self.use_env:
            self.he_skmm = []
            self.de_skmm = []
            self.se_kmm = []
            self.he_smm = []
            self.de_smm = []
            self.se_mm = []
            self.env_index = []
            self.inner_env_index = []
            self.env_buffer_index = []
            self.env_ibzk_kc = []
            self.env_weight = []
            self.env_fermi = np.empty([self.lead_num])

        npk = self.my_npk
        if self.npk == 1:
            dtype = float
        else:
            dtype = complex
            
        for i in range(self.lead_num):
            ns = self.atoms_l[i].calc.wfs.nspins        
            nk = len(self.my_lead_kpts)
            nb = self.nblead[i]
            self.hl_skmm.append(np.empty((ns, nk, nb, nb), complex))
            self.dl_skmm.append(np.empty((ns, nk, nb, nb), complex))
            self.sl_kmm.append(np.empty((nk, nb, nb), complex))

            self.hl_spkmm.append(np.empty((ns, npk, nb, nb), dtype))
            self.dl_spkmm.append(np.empty((ns, npk, nb, nb), dtype))
            self.sl_pkmm.append(np.empty((npk, nb, nb), dtype))

            if self.LR_leads:
                self.hl_spkcmm.append(np.empty((ns, npk, nb, nb), dtype))
                self.dl_spkcmm.append(np.empty((ns, npk, nb, nb), dtype))
                self.sl_pkcmm.append(np.empty((npk, nb, nb), dtype))

            self.ed_pkmm.append(np.empty((ns, npk, nb, nb)))

            self.lead_index.append([])
            self.inner_lead_index.append([])
            self.buffer_index.append([])
       
        for i in range(self.env_num):
            calc = self.atoms_e[i].calc
            ns = calc.wfs.nspins
            nk = len(calc.wfs.ibzk_kc)
            nb = self.nbenv[i]
            self.he_skmm.append(np.empty((ns, nk, nb, nb), complex))
            self.de_skmm.append(np.empty((ns, nk, nb, nb), complex))
            self.se_kmm.append(np.empty((nk, nb, nb), complex))
            self.he_smm.append(np.empty((ns, nb, nb)))
            self.de_smm.append(np.empty((ns, nb, nb)))
            self.se_mm.append(np.empty((nb, nb)))
            self.env_index.append([])
            self.inner_env_index.append([])
            self.env_buffer_index.append([])
            self.env_ibzk_kc.append([])
            self.env_weight.append([])
                
        if self.use_lead:
            self.ec = np.empty([self.lead_num, ns])        

        if self.gamma:
            dtype = float
        else:
            dtype = complex
 
        ns = self.nspins
        nk = len(self.my_kpts)
        nb = self.nbmol
        self.h_skmm = np.empty((ns, nk, nb, nb), dtype)
        self.d_skmm = np.empty((ns, nk, nb, nb), dtype)
        self.s_kmm = np.empty((nk, nb, nb), dtype)
        
        if self.npk == 1:
            dtype = float
        else:
            dtype = complex        
        
        self.h_spkmm = np.empty((ns, npk, nb, nb), dtype)
        self.d_spkmm = np.empty((ns, npk, nb, nb), dtype)
        self.s_pkmm = np.empty((npk, nb, nb), dtype)

        self.h_spkcmm = np.empty((ns, npk, nb, nb), dtype)
        self.d_spkcmm = np.empty((ns, npk, nb, nb), dtype)
        self.s_pkcmm = np.empty((npk, nb, nb), dtype)
        
    def allocate_cpus(self):
        rank = world.rank
        size = world.size
        npk = self.npk
        parsize_energy = size // npk
        if size > npk:
            assert size % npk == 0
        else:
            assert npk % size == 0
 
        r0 = (rank // parsize_energy) * parsize_energy
        ranks = np.arange(r0, r0 + parsize_energy)
        self.energy_comm = world.new_communicator(ranks)
        
        r0 =  rank % parsize_energy
        ranks = np.arange(r0, r0 + size, parsize_energy)
        self.pkpt_comm = world.new_communicator(ranks)

        npk_each = npk / self.pkpt_comm.size           
        pk0 = self.pkpt_comm.rank * npk_each
        self.my_pk = np.arange(pk0, pk0 + npk_each)
        self.my_npk = npk_each
    
        self.my_kpts = np.empty((npk_each * self.ntkmol, 3))
        kpts = self.kpts
        for i in range(self.ntkmol):
            for j, k in zip(range(npk_each), self.my_pk):
                self.my_kpts[j * self.ntkmol + i] = kpts[k * self.ntkmol + i]        

        self.my_lead_kpts = np.empty([npk_each * self.ntklead, 3])
        kpts = self.lead_kpts
        for i in range(self.ntklead):
            for j, k in zip(range(npk_each), self.my_pk):
                self.my_lead_kpts[j * self.ntklead + i] = kpts[
                                                        k * self.ntklead + i] 
        self.my_enum = 20
        e0 = self.energy_comm.rank * self.my_enum
        self.my_ep = np.arange(e0, e0 + self.my_enum)
        self.total_ep = self.my_enum * self.energy_comm.size
        
    def distribute_energy_points(self):
        rank = world.rank
        self.par_energy_index = np.empty([self.nspins, self.my_npk, 2, 2], int)
        for s in range(self.nspins):
            for k in range(self.my_npk):
                neeq = self.eqpathinfo[s][k].num
                neeq_each = neeq // self.energy_comm.size
                
                if neeq % self.energy_comm.size != 0:
                    neeq_each += 1
                begin = rank % self.energy_comm.size * neeq_each
                if (rank + 1) % self.energy_comm.size == 0:
                    end = neeq 
                else:
                    end = begin + neeq_each
                    
                self.par_energy_index[s, k, 0] = [begin, end]

                nene = self.nepathinfo[s][k].num
                nene_each = nene // self.pkpt_comm.size
                if nene % self.energy_comm.size != 0:
                    nene_each += 1
                begin = rank % self.energy_comm.size * nene_each
                if (rank + 1) % self.energy_comm.size == 0:
                    end = nene
                else:
                    end = begin + nene_each
      
                self.par_energy_index[s, k, 1] = [begin, end] 

    def update_lead_hamiltonian(self, l, restart_file=None):
        if not self.lead_restart:
            atoms = self.atoms_l[l]
            atoms.get_potential_energy()
            kpts = self.lead_kpts 
            self.hl_skmm[l], self.sl_kmm[l] = self.get_hs(atoms.calc)
            self.lead_fermi[l] = atoms.calc.get_fermi_level()
            self.dl_skmm[l] = self.initialize_density_matrix('lead', l)
            if self.save_file:
                atoms.calc.write('lead' + str(l) + '.gpw')                    
                self.pl_write('lead' + str(l) + '.mat',
                                                  (self.lead_fermi[l],
                                                   self.hl_skmm[l],
                                                   self.dl_skmm[l],
                                                   self.sl_kmm[l]))            
        else:
            if restart_file == None:
                restart_file = 'lead' + str(l)
            atoms, calc = restart_gpaw(restart_file +'.gpw')
            calc.set_positions()
            self.atoms_l[l] = atoms
            (self.lead_fermi[l],
             self.hl_skmm[l],
             self.dl_skmm[l],
             self.sl_kmm[l]) = self.pl_read(restart_file +'.mat')

    def update_env_hamiltonian(self, l):
        if not self.env_restart:
            atoms = self.atoms_e[l]
            atoms.get_potential_energy()
            self.he_skmm[l], self.se_kmm[l] = self.get_hs(atoms.calc)
            self.env_fermi[l] = atoms.calc.get_fermi_level()
            self.de_skmm[l] = self.initialize_density_matrix('env', l)
            if self.save_file:
                atoms.calc.write('env' + str(l) + '.gpw')                    
                self.pl_write('env' + str(l) + '.mat',
                                                  (self.he_skmm[l],
                                                   self.de_skmm[l],
                                                   self.se_kmm[l]))            
        else:
            atoms, calc = restart_gpaw('env' + str(l) + '.gpw')
            calc.set_positions()
            self.atoms_e[l] = atoms
            (self.he_skmm[l],
             self.de_skmm[l],
             self.se_kmm[l]) = self.pl_read('env' + str(l) + '.mat')
            
    def update_scat_hamiltonian(self, atoms):
        if not self.scat_restart:
            if atoms is None:
                atoms = self.atoms
            if not self.fixed:
                GPAW.get_potential_energy(self, atoms)
            self.atoms = atoms.copy()
            rank = world.rank
            
            if self.fixed:
                from gpaw import restart
                self.atomsf, self.calcf=restart('scat.gpw')
                self.calcf.set_positions(self.atomsf)
                self.h_skmmf, self.s_kmmf = self.get_hs(self.calcf)
            
            self.h_skmm, self.s_kmm = self.get_hs(self)
            #self.h_skmm[0,0] += self.surround.get_potential_matrix_projection()
            #self.s_kmm[0] += self.surround.get_overlap_matrix_projection()
            if self.fixed:
                self.h_spkmm=self.h_skmmf.copy()
                self.s_pkmm=self.s_kmmf.copy()
                
            if self.gamma:
                self.h_skmm = np.real(self.h_skmm).copy()
            
            if not self.fixed:
                self.d_skmm = self.initialize_density_matrix('scat')
            else:
                self.d_skmm = np.zeros(self.h_skmm.shape)
            
            if self.save_file and not self.fixed:
                self.write('scat.gpw')
                self.pl_write('scat.mat', (self.h_skmm,
                                           self.d_skmm,
                                           self.s_kmm))
            self.save_file = False
        else:
            self.h_skmm, self.d_skmm, self.s_kmm = self.pl_read(
                                                     self.restart_file + '.mat')
            self.set_text('restart.txt', self.verbose)
            self.scat_restart = False
            
    def get_hs(self, calc):
        wfs = calc.wfs
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
            H_sqMM[kpt.s, kpt.q] = H_MM
        return H_sqMM, S_qMM

    def get_lead_atoms(self, l):
        """Here is a multi-terminal version """
        atoms = self.atoms.copy()
        atomsl = atoms[self.pl_atoms[l]]
        atomsl.cell = self.pl_cells[l]
        atomsl.center()
        atomsl._pbc[self.d] = True
        atomsl.set_calculator(self.get_lead_calc(l))
        return atomsl

    def get_env_atoms(self, l):
        atoms = self.atoms.copy()
        atomsl = atoms[self.env_atoms[l]]
        atomsl.cell = self.env_cells[l]
        atomsl.center()
        atomsl._pbc = np.array(self.env_pbc, dtype=bool)
        atomsl.set_calculator(self.get_env_calc(l))
        return atomsl
    
    def get_lead_calc(self, l):
        p = self.gpw_kwargs.copy()
        p['nbands'] = None
        if not hasattr(self, 'pl_kpts') or self.pl_kpts==None:
            kpts = self.kpts
            kpts[self.d] = 2 * int(25.0 / self.pl_cells[l][self.d]) + 1
        else:
            kpts = self.pl_kpts
        p['kpts'] = kpts
        if 'mixer' in p:
            if not self.spinpol:
                p['mixer'] = Mixer(0.1, 5, metric='new', weight=100.0)
            else:
                p['mixer'] = MixerDif(0.1, 5, metric='new', weight=100.0)
        if 'txt' in p and p['txt'] != '-':
            p['txt'] = 'lead%i_' % (l + 1) + p['txt']
        return gpaw.GPAW(**p)
    
    def get_env_calc(self, l):
        p = self.gpw_kwargs.copy()
        p['nbands'] = None
        #p['usesymm'] = True
        p['kpts'] = self.env_kpts
        if 'mixer' in p:
            p['mixer'] = Mixer(0.1, 5, metric='new', weight=100.0)
        if 'txt' in p and p['txt'] != '-':
            p['txt'] = 'env%i_' % (l + 1) + p['txt']
        return gpaw.GPAW(**p)

    def negf_prepare(self, atoms=None):
        if not self.initialized_transport:
            self.initialize_transport()
        self.update_scat_hamiltonian(atoms)
        world.barrier()
        if not self.fixed:
            self.initialize_mol()
        self.boundary_check()
    
    def initialize_lead(self, l):
        nspins = self.nspins
        ntk = self.ntklead
        nblead = self.nblead[l]
        kpts = self.my_lead_kpts
        position = [0, 0, 0]
        spk = self.substract_pk
        if l == 0:
            position[self.d] = 1.0
        elif l == 1:
            position[self.d] = -1.0
        else:
            raise NotImplementError('can no deal with multi-terminal now')
        self.hl_spkmm[l] = spk(ntk, kpts, self.hl_skmm[l], 'h')
        self.sl_pkmm[l] = spk(ntk, kpts, self.sl_kmm[l])
        self.hl_spkcmm[l] = spk(ntk, kpts, self.hl_skmm[l], 'h', position)
        self.sl_pkcmm[l] = spk(ntk, kpts, self.sl_kmm[l], 's', position)
        self.dl_spkmm[l] = spk(ntk, kpts, self.dl_skmm[l], 'h')
        self.dl_spkcmm[l] = spk(ntk, kpts, self.dl_skmm[l], 'h', position)

    def initialize_env(self, l):
        wfs = self.atoms_e[l].calc.wfs
        kpts = wfs.ibzk_kc
        weight = wfs.weight_k
        self.he_smm[l], self.se_mm[l] = get_realspace_hs(self.he_skmm[l],
                                                         self.se_kmm[l],
                                                         kpts,
                                                         weight,
                                                         R_c=(0,0,0),
                                                         usesymm=False)
        self.de_smm[l] = get_realspace_hs(self.de_skmm[l],
                                          None,
                                          kpts,
                                          weight,
                                          R_c=(0,0,0),
                                          usesymm=False)
        self.env_ibzk_kc[l] = kpts
        self.env_weight[l] = weight
        
    def initialize_mol(self):
        ntk = self.ntkmol
        kpts = self.my_kpts
        self.h_spkmm = self.substract_pk(ntk, kpts, self.h_skmm, 'h')
        self.s_pkmm = self.substract_pk(ntk, kpts, self.s_kmm)
        self.d_spkmm = self.substract_pk(ntk, kpts, self.d_skmm, 'h')
        #This line only for two_terminal
        if self.LR_leads:
            self.s_pkcmm , self.d_spkcmm = self.fill_density_matrix()
        else:
            self.s_pkcmm = self.substract_pk(ntk, kpts, self.s_kmm, 's', (0,0,1.))
            self.d_spkcmm = self.substract_pk(ntk, kpts, self.d_skmm, 'h', (0,0,1.))
            print 'need something more for 2-D surrounding'

    def substract_pk(self, ntk, kpts, k_mm, hors='s', position=[0, 0, 0]):
        npk = self.my_npk
        weight = np.array([1.0 / ntk] * ntk )
        if hors not in 'hs':
            raise KeyError('hors should be h or s!')
        if hors == 'h':
            dim = k_mm.shape[:]
            dim = (dim[0],) + (dim[1] / ntk,) + dim[2:]
            pk_mm = np.empty(dim, k_mm.dtype)
            dim = (dim[0],) + (ntk,) + dim[2:]
            tk_mm = np.empty(dim, k_mm.dtype)
        elif hors == 's':
            dim = k_mm.shape[:]
            dim = (dim[0] / ntk,) + dim[1:]
            pk_mm = np.empty(dim, k_mm.dtype)
            dim = (ntk,) + dim[1:]
            tk_mm = np.empty(dim, k_mm.dtype)
        if self.LR_leads:
            tkpts = self.pick_out_tkpts(ntk, kpts)
        else:
            tkpts = kpts
        for i in range(npk):
            n = i * ntk
            for j in range(ntk):
                if hors == 'h':
                    tk_mm[:, j] = np.copy(k_mm[:, n + j])
                elif hors == 's':
                    tk_mm[j] = np.copy(k_mm[n + j])
            if hors == 'h':
                pk_mm[:, i] = k2r_hs(tk_mm, None, tkpts, weight, position)
            elif hors == 's':
                pk_mm[i] = k2r_hs(None, tk_mm, tkpts, weight, position)
        return pk_mm   
            
    def check_edge(self):
        tolx = 1e-6
        position = [0,0,0]
        position[self.d] = 2.0
        ntk = self.ntklead
        kpts = self.my_lead_kpts
        s_pkmm = self.substract_pk(ntk, kpts, self.sl_kmm[0], 's', position)
        matmax = np.max(abs(s_pkmm))
        if matmax > tolx:
            self.text('Warning*: the principle layer should be lagger, \
                                                          matmax=%f' % matmax)
    
    def get_edge_density(self):
        for n in range(self.lead_num):
            for i in range(self.nspins):
                for j in range(self.my_npk):
                    self.ed_pkmm[n][i, j] = dot(self.dl_spkcmm[n][i, j],
                                                 self.sl_pkcmm[n][j].T.conj())
                    self.ec[n, i] += np.trace(self.ed_pkmm[n][i, j])   
        self.pkpt_comm.sum(self.ec)
        self.ed_pkmm *= 3 - self.nspins
        self.ec *= 3 - self.nspins
        if self.master:
            for i in range(self.nspins):
                for n in range(self.lead_num):
                    total_edge_charge  = self.ec[n, i] / self.npk
                self.text('edge_charge[%d]=%f' % (i, total_edge_charge))

    def pick_out_tkpts(self, ntk, kpts):
        npk = self.npk
        tkpts = np.zeros([ntk, 3])
        for i in range(ntk):
            tkpts[i, self.d] = kpts[i, self.d]
        return tkpts

    def initialize_density_matrix(self, region, l=0):
        npk = self.npk
        if region == 'lead':
            ntk = self.ntklead
            calc = self.atoms_l[l].calc
            d_skmm = np.empty(self.hl_skmm[l].shape, self.hl_skmm[l].dtype)
            nk = ntk * npk
            weight = [1. / nk] * nk
   
        if region == 'env':
            calc = self.atoms_e[l].calc
            weight = calc.wfs.weight_k
            d_skmm = np.empty(self.he_skmm[l].shape, self.he_skmm[l].dtype)
        
        if region == 'scat':
            ntk = self.ntkmol
            calc = self
            d_skmm = np.empty(self.h_skmm.shape, self.h_skmm.dtype)
            nk = ntk * npk
            weight = [1. / nk] * nk
            
        for kpt, i in zip(calc.wfs.kpt_u, range(len(calc.wfs.kpt_u))):
            C_nm = kpt.C_nM
            f_nn = np.diag(kpt.f_n)
            d_skmm[kpt.s, kpt.q] = np.dot(C_nm.T.conj(),
                                          np.dot(f_nn, C_nm)) / weight[i]
        return d_skmm
    
    def fill_density_matrix(self):
        nb = self.nblead[0]
        dtype = self.s_pkmm.dtype
        s_pkcmm = np.zeros(self.s_pkmm.shape, dtype)
        s_pkcmm[:, -nb:, :nb] = self.sl_pkcmm[0]
        d_spkcmm = np.zeros(self.d_spkmm.shape, dtype)
        d_spkcmm[:, :, -nb:, :nb] = self.dl_spkcmm[0]                    
        return s_pkcmm, d_spkcmm

    def boundary_check(self):
        tol = 5.e-4
        if self.use_lead:
            ham_diff = np.empty([self.lead_num])
            den_diff = np.empty([self.lead_num])
            self.e_float = np.empty([self.lead_num])
            for i in range(self.lead_num):
                ind = self.lead_index[i]
                dim = len(ind)
                ind = np.resize(ind, [dim, dim])
                h_mat_diff = self.h_spkmm[:, :, ind.T, ind] - self.hl_spkmm[i]
                d_mat_diff = self.d_spkmm[:, :, ind.T, ind] - self.dl_spkmm[i]
                ham_diff[i] = np.max(abs(h_mat_diff))
                den_diff[i] = np.max(abs(d_mat_diff))
                im = self.edge_index[1][i]
                il = self.edge_index[0][i]
                self.e_float[i] = (self.h_spkmm[0, 0, im, im] -
                                          self.hl_spkmm[i][0, 0, il, il]) / \
                                          self.sl_pkmm[i][0, il, il]
            
            self.edge_ham_diff = np.max(ham_diff)
            self.edge_den_diff = np.max(den_diff)
            #for i in range(self.lead_num):
            #    self.hl_spkmm[i][:] += self.sl_pkmm[i] * self.e_float[i]
            #    self.hl_spkcmm[i][:] += self.sl_pkcmm[i] * self.e_float[i]
            #    self.hl_skmm[i][:] += self.sl_kmm[i] * self.e_float[i]
            #self.h_spkmm[:] -= self.e_float[0] * self.s_pkmm
            #self.h_spkcmm[:] -= self.e_float[0] * self.s_pkcmm
            self.text('********cancel boundary shift***********')

            for i in range(self.lead_num):
                if ham_diff[i] > tol and self.master:
                    self.text('Warning*: hamiltonian boundary difference lead %d  %f' %
                                                                 (i, ham_diff[i]))
                if den_diff[i] > tol and self.master:
                    self.text('Warning*: density boundary difference lead %d %f' % 
                                                             (i, den_diff[i]))

        if self.use_env:
            env_ham_diff = np.empty([self.env_num])
            env_den_diff = np.empty([self.env_num])            
            self.env_e_float = np.empty([self.env_num])
            for i in range(self.env_num):
                ind = self.env_index[i]
                dim = len(ind)
                ind = np.resize(ind, [dim, dim])
                env_h_mat_diff = self.h_skmm[:, :, ind.T, ind] - \
                                                               self.he_skmm[i]
                env_d_mat_diff = self.d_skmm[:, :, ind.T, ind] - \
                                                               self.de_skmm[i]            
                env_ham_diff[i] = np.max(abs(env_h_mat_diff))
                env_den_diff[i] = np.max(abs(env_d_mat_diff))
                im = self.env_edge_index[1][i]
                ie = self.env_edge_index[0][i]
                self.env_e_float[i] = (self.h_skmm[0, 0, im, im] -
                                       self.he_skmm[i][0, 0, ie, ie]) / \
                                       self.se_kmm[i][0, ie, ie]
            self.env_edge_ham_diff = np.max(env_ham_diff)
            self.env_dege_den_diff = np.max(env_den_diff)
        
            for i in range(self.env_num):
                self.he_skmm[i][:] += self.se_kmm[i] * self.env_e_float[i]
                self.he_smm[i][:] += self.se_mm[i] * self.env_e_float[i]

            
    def get_selfconsistent_hamiltonian(self):
        self.initialize_scf()
        while not self.cvgflag and self.step < self.max_steps:
            self.iterate()
            self.cvgflag = self.d_cvg and self.h_cvg
            self.step +=  1
        
        self.scf.converged = self.cvgflag
        for kpt in self.wfs.kpt_u:
            kpt.rho_MM = None
        if not self.scf.converged:
            raise RuntimeError('Transport do not converge in %d steps' %
                                                              self.max_steps)
    
    def get_hamiltonian_matrix(self):
        self.timer.start('HamMM')            
        self.den2fock()
        self.timer.stop('HamMM')
        self.h_spkmm = self.substract_pk(self.ntkmol, self.kpts,
                                         self.h_skmm, 'h')
        if self.master:
            self.text('HamMM', self.timer.gettime('HamMM'), 'second')        
  
    def get_density_matrix(self):
        self.timer.start('DenMM')
        ind = self.inner_mol_index
        dim = len(ind)
        ind = np.resize(ind, [dim, dim])
        if self.use_qzk_boundary:
            self.fill_lead_with_scat()
            for i in range(self.lead_num):
                self.selfenergies[i].set_bias(0)
        if self.recal_path:
            nb = self.nbmol_inner
            ns = self.nspins
            npk = self.my_npk
            den = np.empty([ns, npk, nb, nb], complex)
            denocc = np.empty([ns, npk, nb, nb], complex)
            if self.cal_loc:
                denloc = np.empty([ns, npk, nb, nb], complex) 
            for s in range(self.nspins):
                for k in range(self.my_npk):
                    den[s, k] = self.get_eqintegral_points(s, k)
                    denocc[s, k] = self.get_neintegral_points(s, k)
                    if self.cal_loc:
                        denloc[s, k] = self.get_neintegral_points(s, k,
                                                                  'locInt')
                    self.d_spkmm[s, k, ind.T, ind] = self.spin_coff * (
                                                              den[s, k] +
                                                              denocc[s, k])
        else:
            for s in range(self.nspins):
                for k in range(self.my_npk):
                    self.d_spkmm[s, k, ind.T, ind] = self.spin_coff *   \
                                                     self.fock2den(s, k)
        self.timer.stop('DenMM')
        if self.master:
            n_epoint = len(self.eqpathinfo[0][0].energy) + len(
                                         self.nepathinfo[0][0].energy)
            self.text('Energy Points on integral path %d' % n_epoint)
            self.text('DenMM', self.timer.gettime('DenMM'), 'second')

    def iterate(self):
        if self.master:
            self.text('----------------step %d -------------------'
                                                                % self.step)
        self.h_cvg = self.check_convergence('h')
        self.get_density_matrix()
        self.get_hamiltonian_matrix()
        self.d_cvg = self.check_convergence('d')
        self.txt.flush()
        
    def check_convergence(self, var):
        cvg = False
        if var == 'h':
            if self.step > 0:
                self.diff_h = np.max(abs(self.hamiltonian.vt_sG -
                                    self.ham_vt_old))
                if self.master:
                    self.text('hamiltonian: diff = %f  tol=%f' % (self.diff_h,
                                                  self.ham_vt_tol))
                if self.diff_h < self.ham_vt_tol:
                    cvg = True
            self.ham_vt_old = np.copy(self.hamiltonian.vt_sG)
        if var == 'd':
            if self.step > 0:
                self.diff_d = self.density.mixer.get_charge_sloshing()
                if self.step == 1:
                    self.min_diff_d = self.diff_d
                elif self.diff_d < self.min_diff_d:
                    self.min_diff_d = self.diff_d
                    #self.output('step')
                if self.master:
                    self.text('density: diff = %f  tol=%f' % (self.diff_d,
                                                  self.scf.max_density_error))
                if self.diff_d < self.scf.max_density_error:
                    cvg = True
        return cvg
 
    def initialize_scf(self):
        bias = self.bias + self.env_bias
        self.intctrl = IntCtrl(self.occupations.kT * Hartree,
                                                        self.fermi, bias)

        
        self.selfenergies = []
        
        if self.use_lead:
            for i in range(self.lead_num):
                self.selfenergies.append(LeadSelfEnergy((self.hl_spkmm[i][0,0],
                                                             self.sl_pkmm[i][0]), 
                                                (self.hl_spkcmm[i][0,0],
                                                             self.sl_pkcmm[i][0]),
                                                (self.hl_spkcmm[i][0,0],
                                                             self.sl_pkcmm[i][0]),
                                                 1e-8))
    
                self.selfenergies[i].set_bias(self.bias[i])
            
        if self.use_env:
            self.env_selfenergies = []
            for i in range(self.env_num):
                self.env_selfenergies.append(CellSelfEnergy((self.he_skmm[i],
                                                             self.se_kmm[i]),
                                                            (self.he_smm[i],
                                                             self.se_mm[i]),
                                                             self.env_ibzk_kc[i],
                                                             self.env_weight[i],
                                                            1e-8))

        self.greenfunction = GreenFunction(selfenergies=self.selfenergies,
                                           H=self.h_spkmm[0,0],
                                           S=self.s_pkmm[0], eta=0)

        self.calculate_integral_path()
        self.distribute_energy_points()
    
        if self.master:
            self.text('------------------Transport SCF-----------------------') 
            bias_info = 'Bias:'
            for i in range(self.lead_num):
                bias_info += 'lead' + str(i) + ': ' + str(self.bias[i]) + 'V'
            self.text(bias_info)
            self.text('Gate: %f V' % self.gate)

        #------for check convergence------
        self.ham_vt_old = np.empty(self.hamiltonian.vt_sG.shape)
        self.ham_vt_diff = None
        self.ham_vt_tol = 1e-4
        
        self.step = 0
        self.cvgflag = False
        self.spin_coff = 3. - self.nspins
        self.max_steps = 200
        self.h_cvg = False
        self.d_cvg = False
        
    def initialize_path(self):
        self.eqpathinfo = []
        self.nepathinfo = []
        self.locpathinfo = []
        for s in range(self.nspins):
            self.eqpathinfo.append([])
            self.nepathinfo.append([])
            if self.cal_loc:
                self.locpathinfo.append([])                
            if self.cal_loc:
                self.locpathinfo.append([])
            for k in self.my_pk:
                self.eqpathinfo[s].append(PathInfo('eq', self.lead_num + self.env_num))
                self.nepathinfo[s].append(PathInfo('ne', self.lead_num + self.env_num))    
                if self.cal_loc:
                    self.locpathinfo[s].append(PathInfo('eq',
                                                         self.lead_num + self.env_num))
                    
    def calculate_integral_path(self):
        self.initialize_path()
        nb = self.nbmol_inner
        ns = self.nspins
        npk = self.my_npk
        den = np.empty([ns, npk, nb, nb], complex)
        denocc = np.empty([ns, npk, nb, nb], complex)
        if self.cal_loc:
            denloc = np.empty([ns, npk, nb, nb], complex)            
        for s in range(ns):
            for k in range(npk):      
                den[s, k] = self.get_eqintegral_points(s, k)
                denocc[s, k] = self.get_neintegral_points(s, k)
                if self.cal_loc:
                    denloc[s, k] = self.get_neintegral_points(s, k, 'locInt')        
        
    def get_eqintegral_points(self, s, k):
        maxintcnt = 50
        nbmol = self.nbmol_inner
        den = np.zeros([nbmol, nbmol], self.d_spkmm.dtype)
        intctrl = self.intctrl
        
        self.zint = [0] * maxintcnt
        self.fint = []

        self.tgtint = []
        for i in range(self.lead_num):
            nblead = self.nblead[i]
            self.tgtint.append(np.empty([maxintcnt, nblead, nblead],
                                                       complex))
        for i in range(self.env_num):
            nbenv = self.nbenv[i]
            self.tgtint.append(np.empty([maxintcnt, nbenv, nbenv],
                                                           complex))
        self.cntint = -1

        if self.use_lead:
            sg = self.selfenergies
            for i in range(self.lead_num):
                sg[i].h_ii = self.hl_spkmm[i][s, k]
                sg[i].s_ii = self.sl_pkmm[i][k]
                sg[i].h_ij = self.hl_spkcmm[i][s, k]
                sg[i].s_ij = self.sl_pkcmm[i][k]
                sg[i].h_im = self.hl_spkcmm[i][s, k]
                sg[i].s_im = self.sl_pkcmm[i][k]
        
        if self.use_env:
            print 'Attention here, maybe confusing npk and nk'
            env_sg = self.env_selfenergies
            for i in range(self.env_num):
                env_sg[i].h_skmm = self.he_skmm[i]
                env_sg[i].s_kmm = self.se_kmm[i]
        
        ind = self.inner_mol_index
        dim = len(ind)
        ind = np.resize(ind, [dim, dim])
        self.greenfunction.H = self.h_spkmm[s, k, ind.T, ind]
        self.greenfunction.S = self.s_pkmm[k, ind.T, ind]
       
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
                self.text('Warning: SGF not Found. eqzgp[%d]= %f %f'
                                                        %(i, elist[i],sgferr))
        flist = []
        siglist = []
        for i in range(self.lead_num + self.env_num):
            siglist.append([])
        for i in sgforder:
            flist.append(self.fint[i])

        for i in range(self.lead_num + self.env_num):
            for j in sgforder:
                sigma = self.tgtint[i][j]
                siglist[i].append(sigma)
        self.eqpathinfo[s][k].add(elist, wlist, flist, siglist)    
   
        del self.fint, self.tgtint, self.zint
        return den 
    
    def get_neintegral_points(self, s, k, calcutype='neInt'):
        intpathtol = 1e-8
        nbmol = self.nbmol_inner
        den = np.zeros([nbmol, nbmol], complex)
        maxintcnt = 50
        intctrl = self.intctrl

        self.zint = [0] * maxintcnt
        self.tgtint = []
        for i in range(self.lead_num):
            nblead = self.nblead[i]
            self.tgtint.append(np.empty([maxintcnt, nblead, nblead], complex))
        
        for i in range(self.env_num):
            nbenv = self.nbenv[i]
            self.tgtint.append(np.empty([maxintcnt, nbenv, nbenv],
                                                                     complex))
        if self.use_lead:    
            sg = self.selfenergies
            for i in range(self.lead_num):
                sg[i].h_ii = self.hl_spkmm[i][s, k]
                sg[i].s_ii = self.sl_pkmm[i][k]
                sg[i].h_ij = self.hl_spkcmm[i][s, k]
                sg[i].s_ij = self.sl_pkcmm[i][k]
                sg[i].h_im = self.hl_spkcmm[i][s, k]
                sg[i].s_im = self.sl_pkcmm[i][k]

        if self.use_env:
            print 'Attention here, maybe confusing npk and nk'
            env_sg = self.env_selfenergies
            for i in range(self.env_num):
                env_sg[i].h_skmm = self.he_skmm[i]
                env_sg[i].s_kmm = self.se_kmm[i]
                
        ind = self.inner_mol_index
        dim = len(ind)
        ind = np.resize(ind, [dim, dim])
        self.greenfunction.H = self.h_spkmm[s, k, ind.T, ind]
        self.greenfunction.S = self.s_pkmm[k, ind.T, ind]

        if calcutype == 'neInt' or calcutype == 'neVirInt':
            for n in range(1, len(intctrl.neintpath)):
                self.cntint = -1
                self.fint = []
                for i in range(self.lead_num + self.env_num):
                    self.fint.append([[],[]])
                if intctrl.kt <= 0:
                    neintpath = [intctrl.neintpath[n - 1] + intpathtol,
                                 intctrl.neintpath[n] - intpathtol]
                else:
                    neintpath = [intctrl.neintpath[n-1], intctrl.neintpath[n]]
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
                        if sgferr > 1e-12:
                            self.text('--Warning: SGF not found, \
                                    nezgp[%d]=%f %f' % (i, zgp[i], sgferr))
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
                flist = [] 
                siglist = []
                for i in range(self.lead_num + self.env_num):
                    flist.append([[],[]])
                    siglist.append([])
                for l in range(self.lead_num):
                    #nblead = self.nblead[l]
                    #sigma= np.empty([nblead, nblead], complex)
                    for j in sgforder:
                        for i in [0, 1]:
                            fermi_factor = self.fint[l][i][j]
                            flist[l][i].append(fermi_factor)   
                        sigma = self.tgtint[l][j]
                        siglist[l].append(sigma)
                if self.use_env:
                    nl = self.lead_num
                    for l in range(self.env_num):
                        for j in sgforder:
                            for i in [0, 1]:
                                fermi_factor = self.fint[l + nl][i][j]
                                flist[l + nl][i].append(fermi_factor)
                            sigma = self.tgtint[l + nl][j]
                            siglist[l + nl].append(sigma)
                self.nepathinfo[s][k].add(zgp, wgp, flist, siglist)
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
                    self.text('Warning: SGF not Found. eqzgp[%d]= %f %f'
                                                        %(i, elist[i],sgferr))
            flist = []
            siglist = []
            for i in range(self.lead_num + self.env_num):
                siglist.append([])
            for i in sgforder:
                flist.append(self.fint[i])
            #sigma= np.empty([nblead, nblead], complex)
            for i in range(self.lead_num):
                for j in sgforder:
                    sigma = self.tgtint[i][j]
                    siglist[i].append(sigma)
            if self.use_env:
                nl = self.lead_num
                for i in range(self.env_num):
                    for j in sgforder:
                        sigma = self.tgtint[i + nl][j]
                        siglist[i + nl].append(sigma)
            self.locpathinfo[s][k].add(elist, wlist, flist, siglist)           
        
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
        stepintcnt = 50
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
            gamma = np.zeros([self.lead_num, nbmol, nbmol], complex)
            if self.use_env:
                env_gamma = np.zeros([self.env_num, nbmol, nbmol], complex)
            if self.cntint + 1 >= len(self.zint):
                self.zint += [0] * stepintcnt
                for n in range(self.lead_num):
                    nblead = self.nblead[n]
                    tmp = self.tgtint[n].shape[0]
                    tmptgt = np.copy(self.tgtint[n])
                    self.tgtint[n] = np.empty([tmp + stepintcnt,
                                                nblead, nblead], complex)
                    self.tgtint[n][:tmp] = tmptgt
                if self.use_env:
                    nl = self.lead_num
                    for n in range(self.env_num):
                        nbenv = self.nbenv[n]
                        tmp = self.tgtint[n + nl].shape[0]
                        tmptgt = np.copy(self.tgtint[n + nl])
                        self.tgtint[n + nl] = np.empty([tmp + stepintcnt,
                                                       nbenv, nbenv], complex)
                        self.tgtint[n + nl][:tmp] = tmptgt
            self.cntint += 1
            self.zint[self.cntint] = zp[i]

            for j in range(self.lead_num):
                self.tgtint[j][self.cntint] = self.selfenergies[j](zp[i])
            
            if self.use_env:
                nl = self.lead_num
                for j in range(self.env_num):
                    self.tgtint[j + nl][self.cntint] = \
                                               self.env_selfenergies[j](zp[i])
                    
            for j in range(self.lead_num):
                ind = self.inner_lead_index[j]
                dim = len(ind)
                ind = np.resize(ind, [dim, dim])
                sigma[ind.T, ind] += self.tgtint[j][self.cntint]             
                gamma[j, ind.T, ind] += self.selfenergies[j].get_lambda(zp[i])
            
            if self.use_env:
                nl = self.lead_num
                for j in range(self.env_num):
                    ind = self.inner_env_index[j]
                    dim = len(ind)
                    ind = np.resize(ind, [dim, dim])
                    sigma[ind.T, ind] += self.tgtint[j + nl][self.cntint]
                    env_gamma[j, ind.T, ind] += \
                                    self.env_selfenergies[j].get_lambda(zp[i])

            gr = self.greenfunction.calculate(zp[i], sigma)       
        
            # --ne-Integral---
            kt = intctrl.kt
            if calcutype == 'neInt':
                gammaocc = np.zeros([nbmol, nbmol], complex)
                for n in range(self.lead_num):
                    lead_ef = intctrl.leadfermi[n]
                    min_ef = intctrl.minfermi
                    max_ef = intctrl.maxfermi
                    self.fint[n][0].append(fermidistribution(zp[i] - lead_ef,
                                           kt) - fermidistribution(zp[i] -
                                          min_ef, kt))
                    self.fint[n][1].append(fermidistribution(zp[i] - max_ef,
                                           kt) - fermidistribution(zp[i] -
                                            lead_ef, kt))                    
                    gammaocc += gamma[n] * self.fint[n][0][self.cntint]
                if self.use_env:
                    nl = self.lead_num
                    for n in range(self.env_num):
                        env_ef = intctrl.envfermi[n]
                        min_ef = intctrl.minfermi
                        max_ef = intctrl.maxfermi
                        self.fint[n + nl][0].append(
                                         fermidistribution(zp[i] - env_ef, kt)
                                      - fermidistribution(zp[i] - min_ef, kt))
                        self.fint[n + nl][1].append(
                                         fermidistribution(zp[i] - max_ef, kt)
                                      - fermidistribution(zp[i] - env_ef, kt))
                        gammaocc += env_gamma[n] * \
                                             self.fint[n + nl][0][self.cntint]
                        
                aocc = dot(gr, gammaocc)
                aocc = dot(aocc, gr.T.conj())
                gfunc[i] = aocc

            elif calcutype == 'neVirInt':
                gammavir = np.zeros([nbmol, nbmol], complex)
                for n in range(self.lead_num):
                    lead_ef = intctrl.leadfermi[n]
                    min_ef = intctrl.minfermi
                    max_ef = intctrl.maxfermi
                    self.fint[n][0].append(fermidistribution(zp[i] - lead_ef,
                                           kt) - fermidistribution(zp[i] -
                                          min_ef, kt))
                    self.fint[n][1].append(fermidistribution(zp[i] - max_ef,
                                           kt) - fermidistribution(zp[i] -
                                            lead_ef, kt))
                    gammavir += gamma[n] * self.fint[n][1][self.cntint]
                if self.use_env:
                    nl = self.lead_num
                    for n in range(self.env_num):
                        env_ef = intctrl.envfermi[n]
                        min_ef = intctrl.minfermi
                        max_ef = intctrl.maxfermi
                        self.fint[n + nl][0].append(
                                         fermidistribution(zp[i] - env_ef, kt)
                                      - fermidistribution(zp[i] - min_ef, kt))
                        self.fint[n + nl][1].append(
                                         fermidistribution(zp[i] - max_ef, kt)
                                      - fermidistribution(zp[i] - env_ef, kt))
                        gammavir += env_gamma[n] * \
                                             self.fint[n + nl][1][self.cntint]                
                avir = dot(gr, gammavir)
                avir = dot(avir, gr.T.conj())
                gfunc[i] = avir
            # --local-Integral--
            elif calcutype == 'locInt':
                # fmax-fmin
                max_ef = intctrl.maxfermi
                min_ef = intctrl.minfermi
                self.fint.append(fermidistribution(zp[i] - max_ef, kt) - 
                                 fermidistribution(zp[i] - min_ef, kt) )
                gfunc[i] = gr * self.fint[self.cntint]
 
            # --res-Integral --
            elif calcutype == 'resInt':
                self.fint.append(-2.j * np.pi * kt)
                gfunc += gr * self.fint[self.cntint]
            #--eq-Integral--
            else:
                if kt <= 0:
                    self.fint.append(1.0)
                else:
                    min_ef = intctrl.minfermi
                    self.fint.append(fermidistribution(zp[i] - min_ef, kt))
                gfunc[i] = gr * self.fint[self.cntint]    
        return gfunc        
    
    def fock2den(self, s, k):
        intctrl = self.intctrl

  
        ind = self.inner_mol_index
        dim = len(ind)
        ind = np.resize(ind, [dim, dim])
        self.greenfunction.H = self.h_spkmm[s, k, ind.T, ind]
        self.greenfunction.S = self.s_pkmm[k, ind.T, ind]

        den = self.eq_fock2den(s, k)
        denocc = self.ne_fock2den(s, k, ov='occ')    
        den += denocc

        if self.cal_loc:
            denloc = self.eq_fock2den(s, k, el='loc')
            denvir = self.ne_fock2den(s, k, ov='vir')
            weight_mm = self.integral_diff_weight(denocc, denvir,
                                                                 'transiesta')
            diff = (denloc - (denocc + denvir)) * weight_mm
            den += diff
            percents = np.sum( diff * diff ) / np.sum( denocc * denocc )
            self.text('local percents %f' % percents)
        den = (den + den.T.conj()) / 2
        return den    

    def ne_fock2den(self, s, k, ov='occ'):
        pathinfo = self.nepathinfo[s][k]
        nbmol = self.nbmol_inner
        den = np.zeros([nbmol, nbmol], complex)
        begin = self.par_energy_index[s, k, 1, 0]
        end = self.par_energy_index[s, k, 1, 1]        
        zp = pathinfo.energy

        for i in range(begin, end):
            sigma = np.zeros(den.shape, complex)
            sigmalesser = np.zeros(den.shape, complex)
            for n in range(self.lead_num):
                ind = self.inner_lead_index[n]
                dim = len(ind)
                ind = np.resize(ind, [dim, dim])
                sigma[ind.T, ind] += pathinfo.sigma[n][i]
            if self.use_env:
                nl = self.lead_num
                for n in range(self.env):
                    ind = self.inner_env_index[n]
                    dim = len(ind)
                    ind = np.resize(ind, [dim, dim])
                    sigma[ind.T, ind] += pathinfo.sigma[n + nl][i]
            gr = self.greenfunction.calculate(zp[i], sigma)

            for n in range(self.lead_num):
                ind = self.inner_lead_index[n]
                dim = len(ind)
                ind = np.resize(ind, [dim, dim])                
                sigmatmp = pathinfo.sigma[n][i]
                if ov == 'occ':
                    fermifactor = np.real(pathinfo.fermi_factor[n][0][i])
                elif ov == 'vir':
                    fermifactor = np.real(pathinfo.fermi_factor[n][1][i])                    
                sigmalesser[ind.T, ind] += 1.0j * fermifactor * (
                                          sigmatmp - sigmatmp.T.conj())
            
            if self.use_env:
                nl = self.lead_num
                for n in range(self.env_num):
                    ind = self.inner_env_index[n]
                    dim = len(ind)
                    ind = np.resize(ind, [dim, dim])
                    sigmatmp = pathinfo[n + nl][i]
                    if ov == 'occ':
                        fermifactor = np.real(
                                         pathinfo.fermi_factor[n + nl][0][i])
                    elif ov == 'vir':
                        fermifactor = np.real(
                                         pathinfo.fermi_factor[n + nl][1][i])
                    sigmalesser[ind.T, ind] += 1.0j * fermifactor * (
                                             sigmatmp - sigmatmp.T.conj())
   
            glesser = dot(sigmalesser, gr.T.conj())
            glesser = dot(gr, glesser)
            weight = pathinfo.weight[i]            
            den += glesser * weight / np.pi / 2
        self.energy_comm.sum(den)
        return den  

    def eq_fock2den(self, s, k, el='eq'):
        if el =='loc':
            pathinfo = self.locpathinfo[s][k]
        else:
            pathinfo = self.eqpathinfo[s][k]
        
        nbmol = self.nbmol_inner
        den = np.zeros([nbmol, nbmol], complex)
        begin = self.par_energy_index[s, k, 0, 0]
        end = self.par_energy_index[s, k, 0, 1]
        zp = pathinfo.energy
        for i in range(begin, end):
            sigma = np.zeros(den.shape, complex)
            for n in range(self.lead_num):
                ind = self.inner_lead_index[n]
                dim = len(ind)
                ind = np.resize(ind, [dim, dim])
                sigma[ind.T, ind] += pathinfo.sigma[n][i]
            if self.use_env:
                nl = self.lead_num
                for n in range(self.env_num):
                    ind = self.inner_env_index[n]
                    dim = len(ind)
                    ind = np.resize(ind, [dim, dim])
                    sigma[ind.T, ind] += pathinfo.sigma[n + nl][i]
            gr = self.greenfunction.calculate(zp[i], sigma)
            fermifactor = pathinfo.fermi_factor[i]
            weight = pathinfo.weight[i]
            den += gr * fermifactor * weight
        self.energy_comm.sum(den)
        den = 1.j * (den - den.T.conj()) / np.pi / 2            
        return den

    def den2fock(self):
        self.timer.start('get_density')
        self.get_density()
        self.timer.stop('get_density')
        self.text('get_density', self.timer.gettime('get_density'))
        if self.fixed:
            self.density.rhot_g += self.surround.get_extra_density()
           
        #self.update_kinetic()
        self.timer.start('update_ham')
        self.hamiltonian.update(self.density)
        self.timer.stop('update_ham')
        self.text('update_ham', self.timer.gettime('update_ham'))
        
        if self.LR_leads and not self.use_linear_vt_mm:
            self.hamiltonian.vt_sG += self.get_linear_potential()
        self.timer.start('get_hs')
        self.h_skmm, self.s_kmm = self.get_hs(self)
        self.timer.stop('get_hs')
        self.text('get_hs', self.timer.gettime('get_hs'))
        if self.fixed:
            self.h_skmm[0,0] += self.surround.get_potential_matrix_projection()
            self.s_kmm[0] += self.surround.get_overlap_matrix_projection()

        if self.use_linear_vt_mm:
            if self.linear_mm == None:
                self.linear_mm = self.get_linear_potential_matrix()            

            self.h_skmm += self.linear_mm
   
    def get_forces(self, atoms):
        if (atoms.positions != self.atoms.positions).any():
            self.scf.converged = False
        if hasattr(self.scf, 'converged') and self.scf.converged:
            pass
        else:
            self.negf_prepare(atoms)
            self.get_selfconsistent_hamiltonian()
        self.forces.F_av = None
        f = GPAW.get_forces(self, atoms)
        return f
    
    def get_potential_energy(self, atoms=None, force_consistent=False):
        if hasattr(self.scf, 'converged') and self.scf.converged:
            pass
        else:
            self.negf_prepare()
            self.get_selfconsistent_hamiltonian()
        if force_consistent:
            # Free energy:
            return Hartree * self.hamiltonian.Etot
        else:
            # Energy extrapolated to zero Kelvin:
            return Hartree * (self.hamiltonian.Etot + 0.5 * self.hamiltonian.S)
       
    def get_density(self):
        #Calculate pseudo electron-density based on green function.
        ns = self.nspins
        ntk = self.ntkmol
        npk = self.my_npk
        nb = self.nbmol
        dr_mm = np.zeros([ns, npk, 3, nb, nb], self.d_spkmm.dtype)
        qr_mm = np.zeros([ns, npk, nb, nb])
        
        for s in range(ns):
            for i in range(npk):
                dr_mm[s, i, 0] = self.d_spkcmm[s, i].T.conj()
                dr_mm[s, i, 1] = self.d_spkmm[s, i]
                dr_mm[s, i, 2]= self.d_spkcmm[s, i]
                qr_mm[s, i] += dot(dr_mm[s, i, 1], self.s_pkmm[i]) 
        if ntk != 1:
            for i in range(self.lead_num):
                ind = self.print_index[i]
                dim = len(ind)
                ind = np.resize(ind, [dim, dim])
                qr_mm[:, :, ind.T, ind] += self.ed_pkmm[i]
        self.pkpt_comm.sum(qr_mm)
        qr_mm /= self.npk
        world.barrier()
        
        if self.master:
            self.print_boundary_charge(qr_mm)
           
        rvector = np.zeros([3, 3])
        rvector[:, self.d] = [-1, 0, 1]
        tkpts = self.pick_out_tkpts(ntk, self.my_kpts)

        self.d_skmm.shape = (ns, npk, ntk, nb, nb)
        for s in range(ns):
            if ntk != 1:
                for i in range(ntk):
                    for j in range(npk):
                        self.d_skmm[s, j, i] = r2k_hs(None,
                                                      dr_mm[s, j, :],
                                                      rvector,
                                                      tkpts[i])
                        self.d_skmm[s, j, i] /=  ntk * self.npk 
            else:
                for j in range(npk):
                    self.d_skmm[s, j, 0] =  dr_mm[s, j, 1]
                    self.d_skmm[s, j, 0] /= self.npk 
        self.d_skmm.shape = (ns, ntk * npk, nb, nb)

        if self.fixed:
            fd=file('d_skmm', 'r')
            self.d_skmm = pickle.load(fd)
            fd.close()
            
        for kpt in self.wfs.kpt_u:
            if self.fixed:
                kpt.P_aMi = self.calcf.wfs.kpt_u[0].P_aMi.copy()
                print 'change here'
            kpt.rho_MM = self.d_skmm[kpt.s, kpt.q]
        if self.fixed:
            self.wfs.fixed = True
            self.wfs.boundary_nt_sG = self.surround.combine_nt_sG()
        self.density.update(self.wfs)

    
    def print_boundary_charge(self, qr_mm):
        qr_mm = np.sum(np.sum(qr_mm, axis=0), axis=0)
        edge_charge = []
        natom_inlead = np.empty([self.lead_num], int)
        natom_print = np.empty([self.lead_num], int)
        
        for i in range(self.lead_num):
            natom_inlead[i] = len(self.pl_atoms[i])
            nb_atom = self.nblead[i] / natom_inlead[i]
            if self.use_buffer:
                pl1 = self.buffer[i]
            else:
                pl1 = self.nblead[i]
            natom_print[i] = pl1 / nb_atom
            ind = self.print_index[i]
            dim = len(ind)
            ind = np.resize(ind, [dim, dim])
            edge_charge.append(np.diag(qr_mm[ind.T, ind]))
            edge_charge[i].shape = (natom_print[i], nb_atom)
            edge_charge[i] = np.sum(edge_charge[i], axis=1)
        
        self.text('***charge distribution at edges***')
        if self.verbose:
            for n in range(self.lead_num):
                info = []
                for i in range(natom_print[n]):
                    info.append('--' +  str(edge_charge[n][i])+'--')
                self.text(info)

        else:
            info = ''
            for n in range(self.lead_num):
                edge_charge[n].shape = (natom_print[n] / natom_inlead[n],
                                                             natom_inlead[n])
                edge_charge[n] = np.sum(edge_charge[n],axis=1)
                nl = int(natom_print[n] / natom_inlead[n])
                for i in range(nl):
                    info += '--' +  str(edge_charge[n][i]) + '--'
                if n != 1:
                    info += '---******---'
            self.text(info)
        self.text('***total charge***')
        self.text(np.trace(qr_mm)) 

    def calc_total_charge(self, d_spkmm):
        nbmol = self.nbmol 
        qr_mm = np.empty([self.nspins, self.my_npk, nbmol, nbmol])
        for i in range(self.nspins):  
            for j in range(self.my_npk):
                qr_mm[i,j] = dot(d_spkmm[i, j], self.s_pkmm[j])
        Qmol = np.trace(np.sum(np.sum(qr_mm, axis=0), axis=0))
        Qmol += np.sum(self.ec)
        Qmol = self.pkpt_comm.sum(Qmol) / self.npk
        return Qmol        

    def get_linear_potential(self):
        linear_potential = np.zeros(self.hamiltonian.vt_sG.shape)
        dimt = linear_potential.shape[-1]
        dimp = linear_potential.shape[1:3]
        buffer_dim = self.dimt_buffer
        scat_dim = dimt - np.sum(buffer_dim)
        bias= np.array(self.bias)
        bias /= Hartree
        vt = np.empty([dimt])
        if (np.array(buffer_dim) != 0).any():
            vt[:buffer_dim[0]] = bias[0]
            vt[-buffer_dim[1]:] = bias[1]         
            vt[buffer_dim[0]: -buffer_dim[1]] = np.linspace(bias[0],
                                                         bias[1], scat_dim)
        else:
            vt = np.linspace(bias[0], bias[1], scat_dim)
        for s in range(self.nspins):
            for i in range(dimt):
                linear_potential[s,:,:,i] = vt[i] * (np.zeros(dimp) + 1)
        return linear_potential
    
    def output(self, filename):
        self.pl_write(filename + '.mat', (self.h_skmm,
                                          self.d_skmm,
                                          self.s_kmm))
        world.barrier()
        self.write(filename + '.gpw')
        if self.master:
            fd = file(filename, 'wb')
            pickle.dump((
                        self.bias,
                        self.gate,
                        self.intctrl,
                        self.eqpathinfo,
                        self.nepathinfo,
                        self.forces,
                        self.current,
                        self.step,
                        self.cvgflag
                        ), fd, 2)
            fd.close()
        world.barrier()

    def input(self, filename):
        GPAW.__init__(self, filename + '.gpw')
        self.set_positions()
        fd = file(filename, 'rb')
        (self.bias,
         self.gate,
         self.intctrl,
         self.eqpathinfo,
         self.nepathinfo,
         self.forces,
         self.current,
         self.step,
         self.cvgflag
         ) = pickle.load(fd)
        fd.close()
        self.h_skmm, self.d_skmm, self.s_kmm = self.pl_read(filename + '.mat')
        (self.h1_skmm,
                 self.d1_skmm,
                 self.s1_kmm) = self.pl_read('lead0.mat', collect=True)
        (self.h2_skmm,
                 self.d2_skmm,
                 self.s2_kmm) = self.pl_read('lead1.mat', collect=True)
        self.initialize_transport()
        self.initialize_mol()
     
    def set_calculator(self, e_points, leads=[0,1]):
        from ase.transport.calculators import TransportCalculator
        ind = self.inner_mol_index
        dim = len(ind)
        ind = np.resize(ind, [dim, dim])
     
        h_scat = self.h_spkmm[:, :, ind.T, ind]
        h_scat = np.sum(h_scat[0, :], axis=0) / self.npk
        h_scat = np.real(h_scat)
        
        l1 = leads[0]
        l2 = leads[1]
        
        h_lead1 = self.double_size(np.sum(self.hl_spkmm[l1][0], axis=0),
                                   np.sum(self.hl_spkcmm[l1][0], axis=0))
        h_lead2 = self.double_size(np.sum(self.hl_spkmm[l2][0], axis=0),
                                   np.sum(self.hl_spkcmm[l2][0], axis=0))
        h_lead1 /= self.npk
        h_lead2 /= self.npk
        
        h_lead1 = np.real(h_lead1)
        h_lead2 = np.real(h_lead2)
        
        s_scat = np.sum(self.s_pkmm[:, ind.T, ind], axis=0) / self.npk
        s_scat = np.real(s_scat)
        
        s_lead1 = self.double_size(np.sum(self.sl_pkmm[l1], axis=0),
                                   np.sum(self.sl_pkcmm[l1], axis=0))
        s_lead2 = self.double_size(np.sum(self.sl_pkmm[l2], axis=0),
                                   np.sum(self.sl_pkcmm[l2], axis=0))
        
        s_lead1 /= self.npk
        s_lead2 /= self.npk
        
        s_lead1 = np.real(s_lead1)
        s_lead2 = np.real(s_lead2)
        
        tcalc = TransportCalculator(energies=e_points,
                                    h = h_scat,
                                    h1 = h_lead1,
                                    h2 = h_lead1,
                                    s = s_scat,
                                    s1 = s_lead1,
                                    s2 = s_lead1,
                                    dos = True
                                   )
        return tcalc
    
    def plot_dos(self, E_range, point_num = 30, leads=[0,1]):
        e_points = np.linspace(E_range[0], E_range[1], point_num)
        
        tcalc = self.set_calculator(e_points, leads)
        tcalc.get_transmission()
        tcalc.get_dos()
        
        f1 = self.intctrl.leadfermi[leads[0]] * (np.zeros([10, 1]) + 1)
        f2 = self.intctrl.leadfermi[leads[1]] * (np.zeros([10, 1]) + 1)
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
        
    def plot_v(self, vt=None, tit=None, ylab=None,
                                             l_MM=False, plot_buffer=True):
        import pylab
        self.use_linear_vt_mm = l_MM
        if vt == None:
            vt = self.hamiltonian.vt_sG + self.get_linear_potential()
        dim = vt.shape
        for i in range(3):
            vt = np.sum(vt, axis=0) / dim[i]
        db = self.dimt_buffer
        if plot_buffer:
            td = len(vt)
            pylab.plot(range(db[0]), vt[:db[0]] * Hartree, 'g--o')
            pylab.plot(range(db[0], td - db[1]),
                               vt[db[0]: -db[1]] * Hartree, 'b--o')
            pylab.plot(range(td - db[1], td), vt[-db[1]:] * Hartree, 'g--o')
        elif db[1]==0:
            pylab.plot(vt[db[0]:] * Hartree, 'b--o')
        else:
            pylab.plot(vt[db[0]: db[1]] * Hartree, 'b--o')
        if ylab == None:
            ylab = 'energy(eV)'
        pylab.ylabel(ylab)
        if tit == None:
            tit = 'bias=' + str(self.bias)
        pylab.title(tit)
        pylab.show()

    def plot_d(self, nt=None, tit=None, ylab=None, plot_buffer=True):
        import pylab
        if nt == None:
            nt = self.density.nt_sG
        dim = nt.shape
        for i in range(3):
            nt = np.sum(nt, axis=0) / dim[i]
        db = self.dimt_buffer
        if plot_buffer:
            td = len(nt)
            pylab.plot(range(db[0]), nt[:db[0]], 'g--o')
            pylab.plot(range(db[0], td - db[1]), nt[db[0]: -db[1]], 'b--o')
            pylab.plot(range(td - db[1], td), nt[-db[1]:], 'g--o')
        elif db[1] == 0:
            pylab.plot(nt[db[0]:], 'b--o')            
        else:
            pylab.plot(nt[db[0]: db[1]], 'b--o')            
        if ylab == None:
            ylab = 'density'
        pylab.ylabel(ylab)
        if tit == None:
            tit = 'bias=' + str(self.bias)
        pylab.title(tit)
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
                current[i] += tcalc.T_e[j] * weight[j] * fermi_factor[i][0][j]
        self.current = current[0] - current[1]
        return self.current
    
    def plot_eigen_channel(self, energy=[0]):
        tcalc = self.set_calculator(energy)
        tcalc.initialize()
        tcalc.update()
        T_MM = tcalc.T_MM[0]
        from gpaw.utilities.lapack import diagonalize
        nmo = T_MM.shape[-1]
        T = np.zeros([nmo])
        info = diagonalize(T_MM, T, self.s_pkmm[0])
        dmo = np.empty([nmo, nmo, nmo])
        for i in range(nmo):
            dmo[i] = np.dot(T_MM[i].T.conj(),T_MM[i])
        basis_functions = self.wfs.basis_functions
        for i in range(nmo):
            wt = self.gd.zeros(1)
            basis_functions.construct_density(dmo[i], wt[0], 0)
            import pylab
            wt=np.sum(wt, axis=2) / wt.shape[2] 
            if abs(T[i]) > 0.001:
                pylab.matshow(wt[0])
                pylab.title('T=' + str(T[i]))
                pylab.show()     
    
    def get_nepath_info(self):
        if hasattr(self, 'nepathinfo'):
            energy = self.nepathinfo[0][0].energy
            weight = self.nepathinfo[0][0].weight
            fermi_factor = self.nepathinfo[0][0].fermi_factor
      
        return energy, weight, fermi_factor
    
    def set_buffer(self):
        self.nbmol_inner = self.nbmol 
        if self.use_lead:
            self.nbmol_inner -= np.sum(self.buffer)
        if self.use_env:
            self.nbmol_inner -= np.sum(self.env_buffer)
        ind = np.arange(self.nbmol)
        buffer_ind = []

        for i in range(self.lead_num):
            buffer_ind += list(self.buffer_index[i])
        for i in range(self.env_num):
            buffer_ind += list(self.env_buffer_index[i])

        ind = np.delete(ind, buffer_ind)
        self.inner_mol_index = ind

        for i in range(self.lead_num):
             self.inner_lead_index[i] = np.searchsorted(ind,
                                                           self.lead_index[i])
        for i in range(self.env_num):
             self.inner_env_index[i] = np.searchsorted(ind,
                                                            self.env_index[i])

    def integral_diff_weight(self, denocc, denvir, method='transiesta'):
        if method=='transiesta':
            eta = 1e-16
            weight = denocc * denocc.conj() / (denocc * denocc.conj() +
                                               denvir * denvir.conj() + eta)
        return weight

    def pl_write(self, filename, matlist):
        if type(matlist)!= tuple:
            matlist = (matlist,)
            nmat = 1
        else:
            nmat = len(matlist)
        total_matlist = []

        for i in range(nmat):
            if type(matlist[i]) == np.ndarray:
                dim = matlist[i].shape
                if len(dim) == 4:
                    dim = (dim[0],) + (dim[1] * self.pkpt_comm.size,) + dim[2:]
                elif len(dim) == 3:
                    dim = (dim[0] * self.pkpt_comm.size,) + dim[1:]
                else:
                    raise RuntimeError('wrong matrix dimension for pl_write')
                if self.pkpt_comm.rank == 0:
                    totalmat = np.empty(dim, dtype=matlist[i].dtype)
                    self.pkpt_comm.gather(matlist[i], 0, totalmat)
                    total_matlist.append(totalmat)
                else:
                    self.pkpt_comm.gather(matlist[i], 0)                    
            else:
                total_matlist.append(matlist[i])
        if world.rank == 0:
            fd = file(filename, 'wb')
            pickle.dump(total_matlist, fd, 2)
            fd.close()
        world.barrier()

    def pl_read(self, filename, collect=False):
        fd = file(filename, 'rb')
        total_matlist = pickle.load(fd)
        fd.close()
        nmat= len(total_matlist)
        matlist = []
        for i in range(nmat):
            if type(total_matlist[i]) == np.ndarray and not collect:
                dim = total_matlist[i].shape
                if len(dim) == 4:
                    dim = (dim[0],) + (dim[1] / self.pkpt_comm.size,) + dim[2:]
                elif len(dim) == 3:
                    dim = (dim[0] / self.pkpt_comm.size,) + dim[1:]
                else:
                    raise RuntimeError('wrong matrix dimension for pl_read')
                local_mat = np.empty(dim, dtype=total_matlist[i].dtype)
                self.pkpt_comm.scatter(total_matlist[i], local_mat, 0)
            elif type(total_matlist[i]) == np.ndarray:
                local_mat = np.empty(total_matlist[i].shape,
                                             dtype= total_matlist[i].dtype)
                local_mat = total_matlist[i]
                self.pkpt_comm.broadcast(local_mat, 0)
            else:
                local_mat = np.zeros([1], dtype=int)
                local_mat[0] = total_matlist[i]
                self.pkpt_comm.broadcast(local_mat, 0)
                local_mat = local_mat[0]
            matlist.append(local_mat)
        return matlist

    def fill_lead_with_scat(self):
        for  i in range(self.lead_num):
            ind = self.inner_lead_index[i]
            dim = len(dim)
            ind = np.resize(ind, [dim, dim])
            self.hl_spkmm[i] = self.h_spkmm[:, :, ind.T, ind]
            self.sl_pkmm[i] = self.s_pkmm[:, ind.T, ind]
            
        self.h1_spkmm = self.h_spkmm_mol[:, :, :nblead, :nblead]
        self.s1_pkmm = self.s_pkmm_mol[:, :nblead, :nblead]
        self.h1_spkmm_ij = self.h_spkmm_mol[:, :, :nblead, nblead:2 * nblead]
        self.s1_spkmm_ij = self.s_pkmm_mol[:, :nblead, nblead:2 * nblead]
        
        self.h2_spkmm = self.h_spkmm_mol[:, :, -nblead:, -nblead:]
        self.s2_pkmm = self.s_pkmm_mol[:, -nblead:, -nblead:]
        self.h2_spkmm_ij = self.h_spkmm_mol[:, :, -nblead:, -nblead*2 : -nblead]
        self.s2_spkmm_ij = self.s_pkmm_mol[:, -nblead:, -nblead*2 : -nblead]

    def get_lead_layer_num(self):
        tol = 1e-4
        temp = []
        for lead_atom in self.atoms_l[0]:
            for i in range(len(temp)):
                if abs(atom.position[self.d] - temp[i]) < tol:
                    break
                temp.append(atom.position[self.d])

    def get_linear_potential_matrix(self):
        # only ok for self.d = 2 now
        N_c = self.gd.N_c.copy()
        h_c = self.gd.h_c
        nn = 64
        N_c[self.d] += nn
        pbc = self.atoms._pbc
        cell = N_c * h_c
        from gpaw.grid_descriptor import GridDescriptor
        GD = GridDescriptor(N_c, cell, pbc)
        from gpaw.lfc import BasisFunctions
        basis_functions = BasisFunctions(GD,     
                                        [setup.phit_j
                                        for setup in self.wfs.setups],
                                        self.wfs.kpt_comm,
                                        cut=True)
        pos = self.atoms.positions.copy()
        for i in range(len(pos)):
            pos[i, self.d] += nn * h_c[self.d] / 2.
        spos_ac = np.linalg.solve(np.diag(cell) * Bohr, pos.T).T
        basis_functions.set_positions(spos_ac) 
        linear_potential = GD.zeros(self.nspins)
        dim_s = self.gd.N_c[self.d] #scat
        dim_t = linear_potential.shape[3]#transport direction
        dim_p = linear_potential.shape[1:3] #transverse 
        bias = np.array(self.bias) /Hartree
        vt = np.empty([dim_t])
        vt[:nn / 2] = bias[0] / 2.0
        vt[-nn / 2:] = bias[1] / 2.0
        vt[nn / 2: -nn / 2] = np.linspace(bias[0]/2.0, bias[1]/2.0, dim_s)
        for s in range(self.nspins):
             for i in range(dim_t):
                  linear_potential[s,:,:,i] = vt[i] * (np.zeros(dim_p) + 1)
        wfs = self.wfs
        nq = len(wfs.ibzk_qc)
        nao = wfs.setups.nao
        H_sqMM = np.empty([wfs.nspins, nq, nao, nao])
        H_MM = np.empty([nao, nao]) 
        for kpt in wfs.kpt_u:
            basis_functions.calculate_potential_matrix(linear_potential[kpt.s],
                                                       H_MM, kpt.q)
            tri2full(H_MM)
            H_MM *= Hartree
            H_sqMM[kpt.s, kpt.q] = H_MM      
        return H_sqMM            

    def estimate_transport_matrix_memory(self):
        self.initialize_transport(dry=True)
        sum = 0
        ns = self.nspins
        if self.use_lead:
            nk = len(self.my_lead_kpts)
            nb = max(self.nblead)
            npk = self.my_npk
            unit_real = np.array(1,float).itemsize
            unit_complex = np.array(1, complex).itemsize
            if self.npk == 1:
                unit = unit_real
            else:
                unit = unit_complex
            sum += self.lead_num * (2 * ns + 1)* nk * nb**2 * unit_complex
            sum += self.lead_num * (2 * ns + 1)* npk * nb**2 * unit
            
            if self.LR_leads:
                sum += ( 2 * ns + 1) * npk * nb ** 2 * unit
            sum += ns * npk * nb**2 * unit
            
            ntgt = 300
            sum += self.lead_num * ns * ntgt * nb**2 * unit_complex
        if self.use_env:
            nk = len(self.env_kpts)
            nb = self.nbenv
            sum += self.env_num * (2 * ns + 1) * nk * nb ** 2 * unit_complex
            sum += self.env_num * (2 * ns + 1) * nb ** 2 * unit_real
            
            sum += self.env_num * ns * ntgt * nb**2 * unit_complex
            
        if self.gamma:
            unit = unit_real
        else:
            unit = unit_complex
        nk = len(self.my_kpts)
        nb = self.nbmol
        sum += (2*ns + 1) * nk * nb**2 * unit
        
        if self.npk == 1:
            unit = unit_real
        else:
            unit = unit_complex
        sum += 2 * (2* ns + 1) * npk * nb**2 * unit
        return sum
    
        
        

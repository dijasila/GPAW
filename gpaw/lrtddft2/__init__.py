"""Linear response TDDFT-class with indexed matrix storage.

"""
import os
import sys
import datetime
import time
import pickle
import math
from glob import glob

import numpy as np
from ase.units import Hartree, alpha
import gpaw.mpi
from gpaw.xc import XC
from gpaw.poisson import PoissonSolver
from gpaw.fd_operators import Gradient
from gpaw.utilities import pack
from gpaw.utilities.lapack import diagonalize
from gpaw.utilities.tools import coordinates
from gpaw.utilities.tools import pick
from gpaw.parameters import InputParameters
from gpaw.blacs import BlacsGrid, Redistributor


#from gpaw.output import initialize_text_stream



#####################################################
class LrTDDFTindexed:
    def __init__(self, 
                 basefilename,
                 calc,
                 xc = None,
                 min_occ=None, max_occ=None, 
                 min_unocc=None, max_unocc=None,
                 max_energy_diff=None,
                 eh_communicator=None,
                 recalculate=None,
                 txt=None):
        self.txt = None
        if gpaw.mpi.world.rank == 0:
            if txt is None: txt = sys.stdout
            self.txt = txt

        
        self.ready_indices = []
        self.kss_list = None
        self.evectors = None
        self.recalculate = recalculate

        self.basefilename = basefilename
        self.xc_name = xc
        self.xc = XC(xc)        
        self.min_occ = min_occ
        self.max_occ = max_occ
        self.min_unocc = min_unocc
        self.max_unocc = max_unocc
        self.max_energy_diff = max_energy_diff # / units.Hartree


        self.calc = calc
        self.eh_comm = eh_communicator
        self.stride = self.eh_comm.size
        self.offset = self.eh_comm.rank
        #print 'Offset and stride = ', self.offset, ' / ', self.stride

        if calc is None:
            self.domain_comm = gpaw.mpi.serial_comm
        else:
            self.domain_comm = calc.density.gd.comm

        self.deriv_scale = 1e-5
        self.min_pop_diff = 1e-3
        self.K_parts = None

        self.timer = calc.timer
        self.timer.start('LrTDDFT')

        # read
        self.read(basefilename)

        # initialize wfs, paw corrections and xc
        #if calc is not None:
        #    self.calc.converge_wave_functions()
        #    if self.calc.density.nct_G is None:   self.calc.set_positions()
        #    self.xc.initialize(self.calc.density, self.calc.hamiltonian, self.calc.wfs, self.calc.occupations)


        # > FIXME
        assert len(self.calc.wfs.kpt_u) == 1, "LrTDDFTindexed does not support more than one k-point/spin."
        self.kpt_ind = 0
        # <
         
        nbands = len(self.calc.wfs.kpt_u[self.kpt_ind].f_n)
        if self.min_occ is None:
            self.min_occ = 0
        if self.min_unocc is None:
            self.min_unocc = self.min_occ
        if self.max_occ is None:
            self.max_occ = nbands - 1
        else:
            self.max_occ = min(self.max_occ, nbands - 1)
        if self.max_unocc is None:
            self.max_unocc = self.max_occ
        if self.max_energy_diff is None:
            self.max_energy_diff = 1e9

        # write info file
        if self.eh_comm.parent is not None:  self.eh_comm.parent.barrier()
        else:                                gpaw.mpi.world.barrier()
        if gpaw.mpi.world.rank == 0:
            f = open(basefilename+'.lr_info','a+')
            f.write('# LrTDDFTindexed\n')
            f.write('%20s = %s\n' % ('xc_name', self.xc_name))
            f.write('%20s = %s\n' % ('',''))
            f.write('%20s = %d\n' % ('min_occ', self.min_occ))
            f.write('%20s = %d\n' % ('min_unocc', self.min_unocc))
            f.write('%20s = %d\n' % ('max_occ', self.max_occ))
            f.write('%20s = %d\n' % ('max_unocc', self.max_unocc))
            f.write('%20s = %18.12lf\n' % ('max_energy_diff',self.max_energy_diff))
            f.write('%20s = %s\n' % ('',''))
            f.write('%20s = %18.12lf\n' % ('deriv_scale', self.deriv_scale))
            f.write('%20s = %18.12lf\n' % ('min_pop_diff', self.min_pop_diff))
            if self.K_parts is not None:
                f.write('%20s = %d\n' % ('K_parts', self.K_parts))
            f.close()


        self.calc_ready = False
        self.kss_list_ready = False
        self.ks_prop_ready = False
        self.K_matrix_ready = False


    def read(self, basename):
        info_file = basename+'.lr_info'
        if os.path.exists(info_file) and os.path.isfile(info_file):
            info_file = open(info_file,'r')
            for line in info_file:
                if line[0] == '#': continue
                if len(line.split()) <= 2: continue
                key = line.split()[0]
                value = line.split()[2]
                if key == 'K_parts': self.K_parts = int(value)
                # .....
                # FIXME: do something, like warn if changed
                # ... 
            info_file.close()            
                
        ready_files = glob(basename + '.ready_rows.*')
        for ready_file in ready_files:
            if os.path.isfile(ready_file):
                for line in open(ready_file,'r'):
                    line = line.split()
                    self.ready_indices.append([int(line[0]),int(line[1])])

        kss_file = basename+'.KS_singles'
        if os.path.exists(kss_file) and os.path.isfile(kss_file):
            self.kss_list = []
            kss_file = open(kss_file)
            for line in kss_file:
                line = line.split()
                i, p = int(line[0]), int(line[1])
                ediff, fdiff = float(line[2]), float(line[3])
                dm = np.array([float(line[4]),float(line[5]),float(line[6])])
                mm = np.array([float(line[7]),float(line[8]),float(line[9])])
                kss = KSSingle(i,p) # see below
                kss.energy_diff = ediff
                kss.pop_diff = fdiff
                kss.dip_mom_r = dm
                kss.magn_mom = mm
                if not kss in self.kss_list:
                    self.kss_list.append(kss)
            if len(self.kss_list) <= 0: self.kss_list = None
            kss_file.close()


    # omega_k = sqrt(lambda_k)
    def get_excitation_energy(self, k):
        self.calculate_excitations()
        return np.sqrt(self.evalues[k])

    # populations F**2
    #def get_excitation_weights(self, k, threshold=0.01):
    #    self.calculate_excitations()
    #    x = np.power(self.get_local_evector(k), 2)
    #    x = x[x > threshold]
    #    self.eh_comm.sum(x)
    #    return x

    def get_oscillator_strength(self, k):
        self.calculate_excitations()
        dm = np.zeros(3)
        for kss_ip in self.kss_list:
            i = kss_ip.occ_ind
            p = kss_ip.unocc_ind
            # c = sqrt(ediff_ip / omega_n) sqrt(population_ip) * F^(n)_ip
            c = np.sqrt(kss_ip.energy_diff / self.get_excitation_energy(k))
            c *= np.sqrt(kss_ip.pop_diff) * self.get_local_eig_coeff(k,self.ind_map[(i,p)])
            # dm_n = c * dm_ip
            dm[0] += c * kss_ip.dip_mom_r[0]
            dm[1] += c * kss_ip.dip_mom_r[1]
            dm[2] += c * kss_ip.dip_mom_r[2]

        self.eh_comm.sum(dm)

        # osc = 2 * omega |dm|**2 / 3
        osc = 2. * self.get_excitation_energy(k) * (dm[0]*dm[0]+dm[1]*dm[1]+dm[2]*dm[2]) / 3.
        return osc

    def get_rotatory_strength(self, k):

        self.calculate_excitations()
        dm = np.zeros(3)
        magn = np.zeros(3)

        # J. Chem. Phys., Vol. 116, No. 16, 22 April 2002
        for kss_ip in self.kss_list:
            i = kss_ip.occ_ind
            p = kss_ip.unocc_ind
            # c = sqrt(ediff_ip / omega_n) sqrt(population_ip) * F^(n)_ip
            c1 = np.sqrt(kss_ip.energy_diff / self.get_excitation_energy(k))
            c2 = np.sqrt(kss_ip.pop_diff) * self.get_local_eig_coeff(k,self.ind_map[(i,p)])
            # dm_n = c * dm_ip
            dm[0] += c1*c2 * kss_ip.dip_mom_r[0]
            dm[1] += c1*c2 * kss_ip.dip_mom_r[1]
            dm[2] += c1*c2 * kss_ip.dip_mom_r[2]

            magn[0] += c2/c1 * kss_ip.magn_mom[0]
            magn[1] += c2/c1 * kss_ip.magn_mom[1]
            magn[2] += c2/c1 * kss_ip.magn_mom[2]

        self.eh_comm.sum(dm)
        self.eh_comm.sum(magn)
        return - ( dm[0] * magn[0] + dm[1] * magn[1] + dm[2] * magn[2] )


    def get_spectrum(self, min_energy=0.0, max_energy=30.0, energy_step=0.01, width=0.1, units='eVcgs'):
        self.calculate_excitations()
        max_energy = max_energy / Hartree
        min_energy = min_energy / Hartree
        energy_step = energy_step / Hartree
        width = width / Hartree

        w = np.arange(min_energy, max_energy, energy_step)
        n = len(w)
        data = np.zeros((n, 3))
        data[:,0] = w * Hartree
        S = data[:,1]
        R = data[:,2]
            
        for (k, omega2) in enumerate(self.evalues):
            #if gpaw.mpi.world.rank == 0:
            #    print k, str(datetime.datetime.now())
            c = self.get_oscillator_strength(k) / width / math.sqrt(6.28318530717959)
            S += c * np.exp( (-5./width/width) * np.power(w-self.get_excitation_energy(k),2) ) 

            c = self.get_rotatory_strength(k) / width / np.sqrt(6.28318530717959)
            R += c * np.exp( (-5./width/width) * np.power(w-self.get_excitation_energy(k),2) ) 

        if units == 'eVcgs':
            w *= Hartree
            S /= Hartree
            R *= 64604.8164 # from turbomole

        return data
        

###
# Utility
###

    def index_of_kss(self,i,p):
        for (ind,kss) in enumerate(self.kss_list):
            if kss.occ_ind == i and kss.unocc_ind == p:
                return ind
        return None


    def get_evector(self, k):
        # First get the eigenvector for the rank 0 and
        # then broadcast it  
        evec = np.zeros_like(self.evectors[0])
        # Rank owning this evector
        off = k % self.stride
        k_local = k // self.stride
        if off == self.offset:
            if self.offset == 0:
                evec[:] = self.evectors[k_local]
            else:
                self.eh_comm.send(self.evectors[k_local], 0, 123)
        else:
            if self.offset == 0:
                self.eh_comm.receive(evec, off, 123)
        # Broadcast
        self.eh_comm.broadcast(evec, 0)
        return evec


    def get_local_index(self,k):
        if k % self.stride != self.offset: return None
        return k // self.stride

    def get_local_eig_coeff(self, k, ip):
        kloc = self.get_local_index(k)
        if kloc is None: return 0.0
        return self.evectors[kloc][ip]
    


####
# Somewhat ugly things
####

    """
    Parameters:
    ----------------------------------------------------------------------
    recalculate | None     = don't recalculate anything if not needed
                | 'all'    = recalculate everything (matrix and eigenvectors)
                | 'matrix' = (re)calculate only matrix (no diagonalization)
                | 'eigen'  = (re)calculate only eigenvectors from the current
                |            matrix (on-the-fly)
    """
    def calculate_excitations(self):
        if self.recalculate == 'all' or self.recalculate == 'matrix':
            self.kss_list = None
            self.evalues = None
            self.evectors = None
            self.ready_indices = []
            rrfn = self.basefilename + '.ready_rows.' + '%04dof%04d' % (self.offset, self.stride)
            Kfn = self.basefilename + '.K_matrix.' + '%04dof%04d' % (self.offset, self.stride)
            open(rrfn,'w').close()
            open(Kfn,'w').close()
            
        if self.recalculate == 'eigen':
            self.calculate_KS_singles()
            self.evalues = None
            self.evectors = None

        if self.evectors is None:
            if ( self.recalculate is None 
                 or self.recalculate == 'all'
                 or self.recalculate == 'matrix' ):
                self.calculate_KS_singles()
                self.calculate_KS_properties()
                self.calculate_K_matrix()

            if self.eh_comm.parent is not None:  self.eh_comm.parent.barrier()
            else:                                gpaw.mpi.world.barrier()
    
            if self.recalculate == 'matrix':
                self.recalculate = None
                return None
            # recalculate only first time
            self.recalculate = None

            # diagonalize
            self.diagonalize2()




    def diagonalize2(self):
        par = self.calc.input_parameters
        sl_lrtddft = par.parallel['sl_lrtddft']

        self.ind_map = {}          # (i,p) to matrix index map
        nrow = len(self.kss_list)  # total rows
        nloc = 0                   # local rows

        # create indexing
        for (ip,kss) in enumerate(self.kss_list):
            self.ind_map[(kss.occ_ind,kss.unocc_ind)] = ip

        
        # read all files
        # (K_parts from calculate_K_matrix or from lr_info-file.)
        rrfns = glob(self.basefilename + '.ready_rows.*')
        for rrfn in rrfns:
            # read all ready rows files
            # to decide what indices are already calculated
            for line in open(rrfn,'r'):
                i = int(line.split()[0])
                p = int(line.split()[1])
                key = (i,p)
                
                # if key not in self.kss_list, drop it
                # i.e. we are calculating just part of the whole matrix
                ip = self.index_of_kss(i,p)
                if ip is not None:
                    assert ip == self.ind_map[key], 'List index %d is not equal to ind_map index %d for key (%d,%d)\n' % (ip,self.ind_map[key],i,p)
                    if self.get_local_index(ip) is not None: nloc += 1


        # Matrix build
        omega_matrix = np.zeros((nloc,nrow))
        omega_matrix[:,:] = np.NAN
        Kfns = glob(self.basefilename + '.K_matrix.*')
        for Kfn in Kfns:
            for line in open(Kfn,'r'):
                line = line.split()
                ipkey = (int(line[0]), int(line[1]))
                jqkey = (int(line[2]), int(line[3]))
                Kvalue = float(line[4])
                # if not in index map, ignore
                if ( not ipkey in self.ind_map
                     or not jqkey in self.ind_map ):
                    continue
                # if ip on this this proc
                if self.get_local_index(self.ind_map[ipkey]) is not None:
                    # add value to matrix
                    lip = self.get_local_index(self.ind_map[ipkey])
                    jq = self.ind_map[jqkey]
                    omega_matrix[lip,jq] = Kvalue
                # if jq on this this proc
                if self.get_local_index(self.ind_map[jqkey]) is not None:
                    # add value to matrix
                    ljq = self.get_local_index(self.ind_map[jqkey])
                    ip = self.ind_map[ipkey]
                    omega_matrix[ljq,ip] = Kvalue

        if np.isnan(np.sum(np.sum(omega_matrix))):
            raise RuntimeError('Not all required LrTDDFT matrix elements could be found.')

        # Add diagonal values
        for kss in self.kss_list:
            key = (kss.occ_ind, kss.unocc_ind)
            if key in self.ind_map:
                ip = self.ind_map[key]
                lip = self.get_local_index(ip)
                if lip is None: continue
                omega_matrix[lip,ip] += kss.energy_diff * kss.energy_diff



        # calculate eigenvalues
        self.evalues = np.zeros(nrow)
        # ScaLapack
        if sl_lrtddft is not None:
            # print 'eh_comm', self.eh_comm.size, self.eh_comm.parent
            ksl = LrTDDFTLayouts(sl_lrtddft, nrow, self.domain_comm,
                                 self.eh_comm)
            self.evectors = omega_matrix
            ksl.diagonalize(self.evectors, self.evalues)

        # Serial Lapack
        else:
            # local to global
            self.evectors = omega_matrix
            omega_matrix = np.zeros((nrow,nrow))
            for ip in range(nrow):
                lip = self.get_local_index(ip)
                if lip is None: continue
                omega_matrix[ip,:] = self.evectors[lip,:]

            #if gpaw.mpi.world.rank == 0:
            #    print omega_matrix
                
            # broadcast to all
            self.eh_comm.sum(omega_matrix)
            #self.eh_comm.broadcast(omega_matrix,0)

            #if gpaw.mpi.world.rank == 0:
            #    print omega_matrix

            omega_matrix2 = omega_matrix.copy()

            # diagonalize
            diagonalize(omega_matrix, self.evalues)

            #if gpaw.mpi.world.rank == 0:
            #    for i in range(nrow):
            #        for j in range(nrow):
            #            print "%9.3le" % omega_matrix[i,j],
            #        print

            #if gpaw.mpi.world.rank == 0:
            #    print omega_matrix

            # global to local
            for ip in range(nrow):
                lip = self.get_local_index(ip)
                if lip is None: continue
                #print '#####', self.offset, ip, lip
                self.evectors[lip,:] = omega_matrix[ip,:]

#            if gpaw.mpi.world.rank == 0:
#                n = 0; m = 0
#                print np.vdot(omega_matrix[:,n], np.dot(omega_matrix2,omega_matrix[:,m]))
#                print np.vdot(omega_matrix[n,:], np.dot(omega_matrix2,omega_matrix[m,:]))
#
#            x = np.zeros(nrow)
#            k = 0
#            for (ip,kss) in enumerate(self.kss_list):
#                i = kss.occ_ind
#                p = kss.unocc_ind
#                x[ip] = self.get_local_eig_coeff(k,self.ind_map[(i,p)])
#            print np.sqrt(np.vdot(x,np.dot(omega_matrix2,x)))

            

    def diagonalize(self):
        """Diagonalizes the Omega matrix. Use ScaLAPACK if available."""

        par = InputParameters()  # We need ScaLAPACK parameters
        sl_omega = par.parallel['sl_lrtddft']

        # Determine index map
        # Read the calculated rows from all read_rows files. K_matrix elements
        # are calculated with a stride comm.size, the global index of eh-pair
        # is thus given by g = l * stride + (stride + offset) % stride

        self.ind_map = {}
        nind = 0
        for off in range(self.stride):
            local_ind = 0
            rrfn = self.basefilename + '.ready_rows.' + '%04dof%04d' % (off, self.stride)
            for line in open(rrfn,'r'):
                i = int(line.split()[0])
                p = int(line.split()[1])
                key = (i,p)
                # if key not in self.kss_list, drop it
                found = False
                for kss in self.kss_list:
                    if kss.occ_ind == i and kss.unocc_ind == p:
                        found = True
                        break
                if found and not key in self.ind_map:
                    global_ind = local_ind * self.stride + (self.stride + off) % self.stride 
                    self.ind_map[key] = global_ind
                    local_ind += 1
                    nind += 1

        # Read omega matrix and diagonalize

        if sl_omega is not None:
            # Use ScaLAPACK
            mynind = len(range(self.offset, nind, self.stride))
            omega_matrix = np.zeros((mynind, nind))
            Kfn = self.basefilename + '.K_matrix.' + '%04dof%04d' % (self.offset, self.stride)
            for line in open(Kfn,'r'):
                line = line.split()
                ipkey = (int(line[0]), int(line[1]))
                jqkey = (int(line[2]), int(line[3]))
                Kvalue = float(line[4])
                if not ipkey in self.ind_map or not jqkey in self.ind_map:
                    continue
                ip = self.ind_map[ipkey]
                # Debugging stuff
                off = ip % self.stride
                assert off == self.offset
                myip = ip // self.stride
                jq = self.ind_map[jqkey]
                omega_matrix[myip,jq] = Kvalue

            # Add diagonal values
            for kss in self.kss_list:
                key = (kss.occ_ind, kss.unocc_ind)
                if key in self.ind_map:
                    global_ind = self.ind_map[key]
                    if (global_ind % self.stride) == self.offset:
                        local_ind = global_ind // self.stride
                        omega_matrix[local_ind, global_ind] += kss.energy_diff * kss.energy_diff

            ksl = LrTDDFTLayouts(sl_omega, nind, self.domain_comm, self.eh_comm)
            self.evectors = omega_matrix
            self.evalues = np.zeros(nind)
            ksl.diagonalize(self.evectors, self.evalues)
            
        else:
            # Serial LAPACK
            omega_matrix = np.zeros((nind,nind))
            for off in range(self.stride):
                Kfn = self.basefilename + '.K_matrix.' + '%04dof%04d' % (off, self.stride)
                for line in open(Kfn,'r'):
                    line = line.split()
                    ipkey = (int(line[0]), int(line[1]))
                    jqkey = (int(line[2]), int(line[3]))
                    Kvalue = float(line[4])
                    if not ipkey in self.ind_map or not jqkey in self.ind_map:
                        continue
                    ip = self.ind_map[ipkey]
                    jq = self.ind_map[jqkey]
                    omega_matrix[ip,jq] = Kvalue


            diag = np.zeros(nind)
            for kss in self.kss_list:
                key = (kss.occ_ind,kss.unocc_ind)
                if key in self.ind_map:
                    diag[self.ind_map[key]] = kss.energy_diff * kss.energy_diff
            for (k,value) in enumerate(diag):
                omega_matrix[k,k] += value


#            if gpaw.mpi.world.rank == 0:
#                for i in range(nind):
#                    for j in range(nind):
#                        print '%18.12lf ' % omega_matrix[i,j],
#                    print

            # diagonalize
            self.evalues = np.zeros(nind)
            diagonalize(omega_matrix, self.evalues)
            self.evectors = omega_matrix[self.offset::self.stride].copy()

        return self.evectors



    def calculate_KS_singles(self):
        if self.kss_list_ready: return 

        eps_n = self.calc.wfs.kpt_u[self.kpt_ind].eps_n      # eigen energies
        f_n = self.calc.wfs.kpt_u[self.kpt_ind].f_n          # occupations

        # create Kohn-Sham single excitation list with energy filter
        old_kss_list = self.kss_list
        self.kss_list = []
        for i in range(self.min_occ, self.max_occ+1):
            for p in range(self.min_unocc, self.max_unocc+1):
                deps_pi = eps_n[p] - eps_n[i]
                df_ip = f_n[i] - f_n[p]
                if np.abs(deps_pi) <= self.max_energy_diff and df_ip > self.min_pop_diff:
                    # i, p, deps, df, mur, muv, magn
                    kss = KSSingle(i,p)
                    kss.energy_diff = deps_pi
                    kss.pop_diff = df_ip
                    self.kss_list.append(kss)

        # sort by energy diff
        def energy_diff_cmp(kss_ip,kss_jq):
            ediff = kss_ip.energy_diff - kss_jq.energy_diff
            if ediff < 0: return -1
            elif ediff > 0: return 1
            return 0
        self.kss_list = sorted(self.kss_list, cmp=energy_diff_cmp)            


        # remove old and add new, but only add to the end of the list
        # (otherwise lower triangle matrix is not filled completely)
        if old_kss_list is not None:
            #if gpaw.mpi.world.rank == 0:
            #    for kss in self.kss_list:
            #        print kss
            #    for kss in old_kss_list:
            #        print kss
                
            new_kss_list = self.kss_list
            self.kss_list = []
            # if in both lists
            for kss_o in old_kss_list:
                found = False
                for kss_n in new_kss_list:
                    if ( kss_o.occ_ind == kss_n.occ_ind
                         and kss_o.unocc_ind == kss_n.unocc_ind ):
                        found = True
                        break
                if found:
                    self.kss_list.append(kss_o)
                else:
                    #if gpaw.mpi.world.rank == 0:
                    #    print 'Dropped transition ', kss_o.occ_ind, kss_o.unocc_ind
                    pass

            # if only in new list
            app_list = []
            for kss_n in new_kss_list:
                found = False
                for kss in self.kss_list:
                    if kss.occ_ind == kss_n.occ_ind and kss.unocc_ind == kss_n.unocc_ind:
                        found = True
                        break
                if not found:
                    app_list.append(kss_n)
                    #if gpaw.mpi.world.rank == 0:
                    #    print 'Added transition ', kss_o.occ_ind, kss_o.unocc_ind

            self.kss_list += app_list


        self.kss_list_ready = True


    def calculate_pair_density(self, kpt, kss_ip, dnt_Gip, dnt_gip, drhot_gip):
        self.pair_density.initialize(kpt, kss_ip.occ_ind, kss_ip.unocc_ind)
        self.pair_density.get(dnt_Gip, dnt_gip, drhot_gip)


    def calculate_KS_properties(self):
        self.calculate_KS_singles()

        if self.ks_prop_ready: return
        self.ks_prop_ready = True
        for kss_ip in self.kss_list:
            if kss_ip.dip_mom_r is None or kss_ip.magn_mom is None:
                self.ks_prop_ready = False
                break
        if self.ks_prop_ready: return
        

        # initialize wfs, paw corrections and xc
        if not self.calc_ready:
            self.calc.converge_wave_functions()
            spos_ac = self.calc.initialize_positions()
            self.calc.wfs.initialize(self.calc.density, 
                                     self.calc.hamiltonian, spos_ac)

            self.xc.initialize(self.calc.density, self.calc.hamiltonian, self.calc.wfs, self.calc.occupations)
            self.calc_ready = True


        dnt_Gip = self.calc.wfs.gd.empty()
        dnt_gip = self.calc.density.finegd.empty()
        drhot_gip = self.calc.density.finegd.empty()
        
        grad_psit2_G = [self.calc.wfs.gd.empty(),self.calc.wfs.gd.empty(),self.calc.wfs.gd.empty()]
        
        dtype = pick(self.calc.wfs.kpt_u[self.kpt_ind].psit_nG,0).dtype
        

        grad = []
        for c in range(3):
            grad.append(Gradient(self.calc.wfs.gd, c, dtype=dtype,n=2))


        self.pair_density = LRiPairDensity(self.calc.density) # see below


        # loop over all KS single excitations
        for kss_ip in self.kss_list:
            if kss_ip.dip_mom_r is not None and kss_ip.magn_mom is not None: continue
            

#            print 'KS single properties for ', kss_ip.occ_ind, ' => ', kss_ip.unocc_ind
            # Dipole moment
            self.calculate_pair_density( self.calc.wfs.kpt_u[self.kpt_ind],
                                         kss_ip, dnt_Gip, dnt_gip, drhot_gip )
            kss_ip.dip_mom_r = self.calc.density.finegd.calculate_dipole_moment(drhot_gip)


            # Magnetic transition dipole
            # J. Chem. Phys., Vol. 116, No. 16, 22 April 2002

            # coordinate vector r
            r_cG, r2_G = coordinates(self.calc.wfs.gd)
            # gradients
            for c in range(3):
                grad[c].apply(self.pair_density.psit2_G, grad_psit2_G[c], self.calc.wfs.kpt_u[self.kpt_ind].phase_cd)
                    
            # <psi1|r x grad|psi2>
            #    i  j  k
            #    x  y  z   = (y pz - z py)i + (z px - x pz)j + (x py - y px)
            #    px py pz
            magn_g = np.zeros(3)
            magn_g[0] = self.calc.wfs.gd.integrate(self.pair_density.psit1_G *
                                                   (r_cG[1] * grad_psit2_G[2] -
                                                    r_cG[2] * grad_psit2_G[1]))
            magn_g[1] = self.calc.wfs.gd.integrate(self.pair_density.psit1_G * 
                                                   (r_cG[2] * grad_psit2_G[0] -
                                                    r_cG[0] * grad_psit2_G[2]))
            magn_g[2] = self.calc.wfs.gd.integrate(self.pair_density.psit1_G * 
                                                   (r_cG[0] * grad_psit2_G[1] -
                                                    r_cG[1] * grad_psit2_G[0]))
            
            # augmentation contributions
            magn_a = np.zeros(3)
            for a, P_ni in self.calc.wfs.kpt_u[self.kpt_ind].P_ani.items():
                Pi_i = P_ni[kss_ip.occ_ind]
                Pp_i = P_ni[kss_ip.unocc_ind]
                rnabla_iiv = self.calc.wfs.setups[a].rnabla_iiv
                for c in range(3):
                    for i1, Pi in enumerate(Pi_i):
                        for i2, Pp in enumerate(Pp_i):
                            magn_a[c] += Pi * Pp * rnabla_iiv[i1, i2, c]
            self.calc.wfs.gd.comm.sum(magn_a)
                
            kss_ip.magn_mom = alpha / 2. * (magn_g + magn_a)






        if gpaw.mpi.world.rank == 0:
            self.kss_file = open(self.basefilename+'.KS_singles','w')
            for kss_ip in self.kss_list:
                format = '%08d %08d  %18.12lf %18.12lf  '
                format += '%18.12lf %18.12lf %18.12lf '
                format += '%18.12lf %18.12lf %18.12lf\n'
                self.kss_file.write(format % (kss_ip.occ_ind, kss_ip.unocc_ind,
                                              kss_ip.energy_diff,
                                              kss_ip.pop_diff,
                                              kss_ip.dip_mom_r[0],
                                              kss_ip.dip_mom_r[1],
                                              kss_ip.dip_mom_r[2],
                                              kss_ip.magn_mom[0],
                                              kss_ip.magn_mom[1],
                                              kss_ip.magn_mom[2]))
            self.kss_file.close()
            
        self.ks_prop_ready = True
        


##############################
# The big ugly thing
##############################

    def calculate_K_matrix(self, fH=True, fxc=True):
        self.calculate_KS_singles()

        # check if done before allocating
        if self.K_matrix_ready: return
        self.K_matrix_ready = True
        nrows = 0
        for (ip,kss_ip) in enumerate(self.kss_list):
            if ip % self.stride != self.offset:  continue
            if [kss_ip.occ_ind,kss_ip.unocc_ind] in self.ready_indices:continue
            self.K_matrix_ready = False
            nrows += 1
        if self.K_matrix_ready: return


        # initialize wfs, paw corrections and xc
        if not self.calc_ready:
            self.calc.converge_wave_functions()
            if self.calc.density.nct_G is None:   self.calc.set_positions()
            self.xc.initialize(self.calc.density, self.calc.hamiltonian, self.calc.wfs, self.calc.occupations)
            self.calc_ready = True


        self.K_parts = self.stride
        if gpaw.mpi.world.rank == 0:
            info_file = open(self.basefilename + '.lr_info','a+')
            info_file.write('%20s = %d\n' % ('K_parts',self.stride))
            info_file.close()


        Kfn = self.basefilename + '.K_matrix.' + '%04dof%04d' % (self.offset, self.stride)
        rrfn = self.basefilename + '.ready_rows.' + '%04dof%04d' % (self.offset, self.stride)
        logfn = self.basefilename + '.log.' + '%04dof%04d' % (self.offset, self.stride)

        if self.calc.density.finegd.comm.rank == 0:
            self.Kfile = open(Kfn, 'a+')
            self.ready_file = open(rrfn,'a+')
            self.log_file = open(logfn,'a+')


        self.poisson = PoissonSolver(nn=self.calc.hamiltonian.poisson.nn)
        self.poisson.set_grid_descriptor(self.calc.density.finegd)
        self.poisson.initialize()

        self.pair_density = LRiPairDensity(self.calc.density)

        dnt_Gip = self.calc.wfs.gd.empty()
        dnt_gip = self.calc.density.finegd.empty()
        drhot_gip = self.calc.density.finegd.empty()
        dnt_Gjq = self.calc.wfs.gd.empty()
        dnt_gjq = self.calc.density.finegd.empty()
        drhot_gjq = self.calc.density.finegd.empty()

        nt_g = self.calc.density.finegd.zeros(self.calc.density.nt_sg.shape[0])

        dVht_gip = self.calc.density.finegd.empty()
        dVxct_gip = self.calc.density.finegd.zeros(self.calc.density.nt_sg.shape[0])
        dVxct_gip_2 = self.calc.density.finegd.zeros(self.calc.density.nt_sg.shape[0])


        # for timings
        irow = 0
        tp = None
        t0 = None
        tm = None
        # outer loop over KS singles
        for (ip,kss_ip) in enumerate(self.kss_list):
            if ip % self.stride != self.offset:  continue

            if [kss_ip.occ_ind,kss_ip.unocc_ind] in self.ready_indices:  continue

            # timings
            if self.calc.density.finegd.comm.rank == 0:
                if irow % 10 == 0: tm = t0;  t0 = tp;  tp = time.time()
                if tm is None:
                    eta = -1.0
                    self.log_file.write('Calculating pair %5d => %5d  ( %s, ETA %9.1lfs )\n' % (kss_ip.occ_ind, kss_ip.unocc_ind, str(datetime.datetime.now()), eta))
                else:
                    a = (.5*tp - t0 + .5*tm)/10./10.
                    b = .5*(tp - tm)/10. - 2.*a*(irow-10)
                    c = t0 - a*(irow-10)*(irow-10) - b*(irow-10)
                    eta = a*nrows*nrows + b * nrows + c - tp
                    self.log_file.write('Calculating pair %5d => %5d  ( %s, ETA %9.1lfs )\n' % (kss_ip.occ_ind, kss_ip.unocc_ind, str(datetime.datetime.now()), eta))
                self.log_file.flush()
                irow += 1


            dnt_Gip[:] = 0.0
            dnt_gip[:] = 0.0
            drhot_gip[:] = 0.0

            # pair density
            self.timer.start('Pair density')
            self.calculate_pair_density(self.calc.wfs.kpt_u[self.kpt_ind], kss_ip, 
                                        dnt_Gip, dnt_gip, drhot_gip)

            self.timer.stop('Pair density')
            # smooth hartree potential
            dVht_gip[:] = 0.0
            self.timer.start('Poisson')
            self.poisson.solve(dVht_gip, drhot_gip, charge=None)
            self.timer.stop('Poisson')

            self.timer.start('Smooth XC')
            # smooth xc potential
            # finite difference plus
            nt_g[self.kpt_ind][:] = self.deriv_scale * dnt_gip
            nt_g[self.kpt_ind][:] += self.calc.density.nt_sg[self.kpt_ind]
            dVxct_gip[:] = 0.0
            self.xc.calculate(self.calc.density.finegd, nt_g, dVxct_gip)
            # finite difference minus
            nt_g[self.kpt_ind][:] = -self.deriv_scale * dnt_gip
            nt_g[self.kpt_ind][:] += self.calc.density.nt_sg[self.kpt_ind]
            dVxct_gip_2[:] = 0.0
            self.xc.calculate(self.calc.density.finegd, nt_g, dVxct_gip_2)
            dVxct_gip -= dVxct_gip_2
            # finite difference approx for fxc
            # vxc = (vxc+ - vxc-) / 2h
            dVxct_gip *= 1./(2.*self.deriv_scale)
            self.timer.stop('Smooth XC')


            # XC corrections
            I_asp = {}
            i = kss_ip.occ_ind
            p = kss_ip.unocc_ind
            # FIXME
            kss_ip.spin = 0

            self.timer.start('Atomic XC')
            for a, P_ni in self.calc.wfs.kpt_u[kss_ip.spin].P_ani.items():
                I_sp = np.zeros_like(self.calc.density.D_asp[a])
                I_sp_2 = np.zeros_like(self.calc.density.D_asp[a])

                Pip_ni = self.calc.wfs.kpt_u[kss_ip.spin].P_ani[a]
                Dip_ii = np.outer(Pip_ni[i], Pip_ni[p])
                Dip_p  = pack(Dip_ii)

                # finite difference plus
                D_sp = self.calc.density.D_asp[a].copy()
                D_sp[kss_ip.spin] += self.deriv_scale * Dip_p
                self.xc.calculate_paw_correction(self.calc.wfs.setups[a], D_sp, I_sp)

                # finite difference minus
                D_sp_2 = self.calc.density.D_asp[a].copy()
                D_sp_2[kss_ip.spin] -= self.deriv_scale * Dip_p
                self.xc.calculate_paw_correction(self.calc.wfs.setups[a], D_sp_2, I_sp_2)

                # finite difference
                I_asp[a] = (I_sp - I_sp_2) / (2.*self.deriv_scale)
            self.timer.stop('Atomic XC')
                

            # calculate whole row before write to file (store row to K)
            K = []

            # inner loop over KS singles
            for (jq,kss_jq) in enumerate(self.kss_list):
                i = kss_ip.occ_ind
                p = kss_ip.unocc_ind
                j = kss_jq.occ_ind
                q = kss_jq.unocc_ind


                # only lower triangle
                if ip < jq: continue

                dnt_Gjq[:] = 0.0
                dnt_gjq[:] = 0.0
                drhot_gjq[:] = 0.0

                self.timer.start('Pair density')
                self.calculate_pair_density(self.calc.wfs.kpt_u[self.kpt_ind], kss_jq, 
                                            dnt_Gjq, dnt_gjq, drhot_gjq)
                self.timer.stop('Pair density')


                self.timer.start('Integrate')
                Ig = 0.0
                # Hartree smooth part, RHOT_JQ HERE???
                if fH:
                    Ig += self.calc.density.finegd.integrate(dVht_gip, drhot_gjq)
                # XC smooth part
                if fxc:
                    Ig += self.calc.density.finegd.integrate(dVxct_gip[self.kpt_ind], dnt_gjq)
                self.timer.stop('Integrate')
                
                # FIXME
                kss_ip.spin = kss_jq.spin = 0
                # atomic corrections
                self.timer.start('Atomic corrections')
                Ia = 0.
                for a, P_ni in self.calc.wfs.kpt_u[kss_ip.spin].P_ani.items():
                    Pip_ni = self.calc.wfs.kpt_u[kss_ip.spin].P_ani[a]
                    Dip_ii = np.outer(Pip_ni[i], Pip_ni[p])
                    Dip_p = pack(Dip_ii)

                    Pjq_ni = self.calc.wfs.kpt_u[kss_jq.spin].P_ani[a]
                    Djq_ii = np.outer(Pjq_ni[j], Pjq_ni[q])
                    Djq_p = pack(Djq_ii)

                    # Hartree part
                    C_pp = self.calc.wfs.setups[a].M_pp
                    #   ----
                    # 2 >      P   P  C    P  P
                    #   ----    ip  jr prst ks qt
                    #   prst
                    if fH:
                        Ia += 2.0 * np.dot(Djq_p, np.dot(C_pp, Dip_p))

                    # XC part, CHECK THIS JQ EVERWHERE!!!
                    if fxc:
                        Ia += np.dot(I_asp[a][kss_jq.spin], Djq_p)
                    
                self.timer.stop('Atomic corrections')

                Ia = self.calc.density.finegd.comm.sum(Ia)
                
                Itot = Ig + Ia
                
                # K_ip,jq += 2*sqrt(f_ip*f_jq*eps_pi*eps_qj)<ip|dH|jq>

                K.append( [i,p,j,q,
                           2.*np.sqrt(kss_ip.pop_diff * kss_jq.pop_diff 
                                   * kss_ip.energy_diff * kss_jq.energy_diff)
                           * Itot] )

            # write  i p j q Omega_Hxc -2345.789012345678
            self.timer.start('Write Omega')
            if self.calc.density.finegd.comm.rank == 0:
                for k in K:
                    [i,p,j,q,Kipjq] = k
                    self.Kfile.write("%5d %5d %5d %5d %18.12lf\n" % (i,p,j,q,Kipjq))
                    #if i != j or p != q: 
                    #    self.Kfile.write("%5d %5d %5d %5d %18.12lf\n" % (j,q,i,p,Kipjq))
                self.Kfile.flush()
                self.ready_file.write("%d %d\n" % (kss_ip.occ_ind, kss_ip.unocc_ind))
                self.ready_file.flush()
            self.timer.stop('Write Omega')

            self.ready_indices.append([kss_ip.occ_ind,kss_ip.unocc_ind])
        
        if self.calc.density.finegd.comm.rank == 0:
            self.Kfile.close()
            self.ready_file.close()
            self.log_file.close()


    def __del__(self):
        self.timer.stop('LrTDDFT')



    


###############################################################################
# Small utility classes and functions
###############################################################################

class LRiPairDensity:
    def  __init__(self, density):
        """Initialization needs density instance"""
        self.density = density

    def initialize(self, kpt, n1, n2):
        """Set wave function indices."""
        self.n1 = n1
        self.n2 = n2
        self.spin = kpt.s
        self.P_ani = kpt.P_ani
        self.psit1_G = pick(kpt.psit_nG, n1)
        self.psit2_G = pick(kpt.psit_nG, n2)

    def get(self, nt_G, nt_g, rhot_g):
        np.multiply(self.psit1_G.conj(), self.psit2_G, nt_G)
        self.density.interpolator.apply(nt_G, nt_g)
        rhot_g[:] = nt_g[:]
        
        # Determine the compensation charges for each nucleus
        Q_aL = {}
        for a, P_ni in self.P_ani.items():
            assert P_ni.dtype == float
            # Generate density matrix
            P1_i = P_ni[self.n1]
            P2_i = P_ni[self.n2]
            D_ii = np.outer(P1_i.conj(), P2_i)
            # allowed to pack as used in the scalar product with
            # the symmetric array Delta_pL
            D_p  = pack(D_ii)
            #FIXME: CHECK THIS D_p  = pack(D_ii, tolerance=1e30)
            
            # Determine compensation charge coefficients:
            Q_aL[a] = np.dot(D_p, self.density.setups[a].Delta_pL)

        # Add compensation charges
        self.density.ghat.add(rhot_g, Q_aL)



###############################################################################
class KSSingle:
    def __init__(self, occ_ind, unocc_ind):
        self.occ_ind = occ_ind
        self.unocc_ind = unocc_ind
        self.energy_diff = None
        self.pop_diff = None
        self.dip_mom_r = None
        self.dip_mom_v = None
        self.magn_mom = None

    def __str__(self):
        if self.dip_mom_r is not None and self.dip_mom_v is not None and self.magn_mom is not None:
            str = "# KS single excitation from state %05d to state %05d: dE_pi = %18.12lf, f_pi = %18.12lf,  dmr_ip = (%18.12lf, %18.12lf, %18.12lf), dmv_ip = (%18.12lf, %18.12lf, %18.12lf), dmm_ip = %18.12lf" \
                % ( self.occ_ind, \
                        self.unocc_ind, \
                        self.energy_diff, \
                        self.pop_diff, \
                        self.dip_mom_r[0], self.dip_mom_r[1], self.dip_mom_r[2], \
                        self.dip_mom_v[0], self.dip_mom_v[1], self.dip_mom_v[2], \
                        self.magn_mom )
        elif self.energy_diff is not None and self.pop_diff is not None:
            str = "# KS single excitation from state %05d to state %05d: dE_pi = %18.12lf, f_pi = %18.12lf" \
                % ( self.occ_ind, \
                        self.unocc_ind,  \
                        self.energy_diff, \
                        self.pop_diff )
        elif self.occ_ind is not None and self.unocc_ind is not None:
            str = "# KS single excitation from state %05d to state %05d" \
                % ( self.occ_ind, self.unocc_ind )
        else:
            raise RuntimeError("Uninitialized KSSingle")
        return str
    

class LrTDDFTLayouts:
    """BLACS layout for distributed Omega matrix in linear response
       time-dependet DFT calculations"""

    def __init__(self, sl_omega, nkq, domain_comm, eh_comm):
        mcpus, ncpus, blocksize = tuple(sl_omega)
        self.world = eh_comm.parent
        self.domain_comm = domain_comm
        # All the ranks within domain communicator contain the omega matrix
        # construct new communicator only on domain masters
        eh_ranks = np.arange(eh_comm.size) * domain_comm.size
        self.eh_comm2 = self.world.new_communicator(eh_ranks)

        self.eh_grid = BlacsGrid(self.eh_comm2, eh_comm.size, 1)
        self.eh_descr = self.eh_grid.new_descriptor(nkq, nkq, 1, nkq)
        self.diag_grid = BlacsGrid(self.world, mcpus, ncpus)
        self.diag_descr = self.diag_grid.new_descriptor(nkq, nkq,
                                                        blocksize,
                                                        blocksize)
        self.redistributor_in = Redistributor(self.world,
                                              self.eh_descr,
                                              self.diag_descr)
        self.redistributor_out = Redistributor(self.world,
                                               self.diag_descr,
                                               self.eh_descr)

    def diagonalize(self, Om, eps_n):

        O_nn = self.diag_descr.empty(dtype=float)
        if self.eh_descr.blacsgrid.is_active():
            O_nN = Om
        else:
            O_nN = np.empty((0,0), dtype=float)

        self.redistributor_in.redistribute(O_nN, O_nn)
        self.diag_descr.diagonalize_dc(O_nn.copy(), O_nn, eps_n, 'L')
        self.redistributor_out.redistribute(O_nn, O_nN)
        self.world.broadcast(eps_n, 0)
        # Broadcast eigenvectors within domains
        if not self.eh_descr.blacsgrid.is_active():
            O_nN = Om
        self.domain_comm.broadcast(O_nN, 0)

###############################################################################
def lr_communicators(world, dd_size, eh_size):
    """Create communicators for LrTDDFT calculation.

    Parameters:
    ------------------------------------------------------------------------------
    world        | MPI world communicator (gpaw.mpi.world)
    dd_size      | Over how many processes is domain distributed.
    eh_size      | Over how many processes are electron-hole pairs distributed.
    ------------------------------------------------------------------------------
    Note: Sizes must match, i.e., world.size must be equal to dd_size x eh_size,
          e.g., 1024 = 64*16
    Tip:  Use enough processes for domain decomposition (dd_size) to fit everything
          into memory, and use the remaining processes for electron-hole pairs
          as it is trivially parallel.

    Returns: dd_comm, eh_comm
    ------------------------------------------------------------------------------
    dd_comm      | MPI communicator for domain decomposition
                 | Pass this to ground state calculator object (GPAW object).
    eh_comm      | MPI communicator for electron hole pairs
                 | Pass this to linear response calculator (LrTDDFT object).
    ------------------------------------------------------------------------------

    Example:

    dd_comm, eh_comm = lr_communicators(gpaw.mpi.world, 4, 2)
    txt = 'lr_%04d_%04d.txt' % (dd_comm.rank, eh_comm.rank)
    lr = LrTDDFT(GPAW('unocc.gpw', communicator=dd_comm), eh_comm=eh_comm, txt=txt)
    """

    if world.size != dd_size * eh_size:
        raise RuntimeError('Domain decomposition processes (dd_size) times electron-hole (eh_size) processes does not match with total processes (world size != dd_size * eh_size)')

    dd_ranks = []
    eh_ranks = []
    for k in range(world.size):
        if k / dd_size == world.rank / dd_size:
            dd_ranks.append(k)
        if k % dd_size == world.rank % dd_size:
            eh_ranks.append(k)
    #print 'Proc #%05d DD : ' % world.rank, dd_ranks, '\n', 'Proc #%05d EH : ' % world.rank, eh_ranks
    return world.new_communicator(dd_ranks), world.new_communicator(eh_ranks)


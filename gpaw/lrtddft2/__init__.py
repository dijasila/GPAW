"""Module for linear response TDDFT class with indexed K-matrix storage."""

import os
import sys
import datetime
import time
import pickle
import math
import glob

import numpy as np
import ase.units
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
"""Linear response TDDFT (Casida) class with indexed K-matrix storage."""
class LrTDDFTindexed:
    def __init__(self, 
                 basefilename,
                 calc,
                 xc = None,
                 min_occ=None, max_occ=None, 
                 min_unocc=None, max_unocc=None,
                 max_energy_diff=None,
                 recalculate=None,
                 eh_communicator=None,
                 txt='-'):
        """Initialize linear response TDDFT without calculating anything.

        Input parameters:

        basefilename
          All files associated with this calculation are stored as
          *<basefilename>.<extension>*

        calc
          Ground state calculator (if you are using eh_communicator,
          you need to take care that calc has suitable communicator.)

        xc
          Name of the exchange-correlation kernel (fxc) used in calculation.
          (optional)

        min_occ
          Index of the first occupied state to be included in the calculation.
          (optional)
          
        max_occ
          Index of the last occupied state (inclusive) to be included in the
          calculation. (optional)
 
        min_unocc
          Index of the first unoccupied state to be included in the
          calculation. (optional)

        max_unocc
          Index of the last unoccupied state (inclusive) to be included in the
          calculation. (optional)

        max_energy_diff
          Noninteracting Kohn-Sham excitations above this value are not
          included in the calculation. Atomic units = Hartree! (optional)

        recalculate
          | Force recalculation.
          | 'eigen'  : recalculate only eigensystem (useful for on-the-fly
          |            spectrum calcualtions and convergence checking)
          | 'matrix' : recalculate matrix without solving the eigensystem
          | 'all'    : recalculate everything
          | None     : do not recalculate anything if not needed (default)

        eh_communicator
          Communicator for parallelizing over electron-hole pairs (i.e.,
          rows of K-matrix). Note that calculator must have suitable
          communicator, which can be assured by using lr_communicators
          to create both communicators.

        txt
          Filename for text output
        """
        
        # Init internal stuff
        self.ready_indices = []   # K-matrix indices already calculated
        self.kss_list = None      # List of noninteracting Kohn-Sham single
                                  # excitations
        self.evectors = None      # eigenvectors of the Casida equation
        self.pair_density = None  # pair density class

        # Save input params
        self.basefilename = basefilename 
        self.xc_name = xc
        self.xc = XC(xc)        
        self.min_occ = min_occ
        self.max_occ = max_occ
        self.min_unocc = min_unocc
        self.max_unocc = max_unocc
        self.max_energy_diff = max_energy_diff # / units.Hartree
        self.recalculate = recalculate
        # Don't init calculator yet if it's not needed (to save memory)
        self.calc = calc
        # Input paramers?
        self.deriv_scale = 1e-5   # fxc finite difference step
        self.min_pop_diff = 1e-3  # ignore transition if population
                                  # difference is below this value


        # Parent communicator
        self.parent_comm = None
        
        # Decomposition domain communicator
        if calc is None:
            self.dd_comm = gpaw.mpi.serial_comm
        else:
            self.dd_comm = calc.density.gd.comm
            self.parent_comm = self.dd_comm.parent.parent # extra wrapper in calc so we need double parent

        # Parameters for parallelization over rows of K-matrix
        if eh_communicator is None:
            self.eh_comm = gpaw.mpi.serial_comm
        else:
            self.eh_comm = eh_communicator
            if self.parent_comm is None:
                self.parent_comm = self.eh_comm.parent
            # Check that parent_comm is valid
            elif ( self.parent_comm != self.eh_comm.parent ):
                raise RuntimeError('Invalid communicators in LrTDDFTindexed (Do not have same parent MPI communicator).')
        self.stride = self.eh_comm.size
        self.offset = self.eh_comm.rank

        # Parent communicator
        if self.parent_comm is None:
            self.parent_comm = self.gpaw.mpi.serial_comm

        # Init text output
        if self.parent_comm.rank == 0 and txt is not None:
            if txt == '-':
                self.txt = sys.stdout
            elif isintance(txt,str):
                self.txt = open(txt,'w')
            else:
                self.txt = txt
        else:
            self.txt = open(os.devnull,'w')


        # Timer
        self.timer = calc.timer
        self.timer.start('LrTDDFT')


        # If a previous calculation exists
        # read LR_info, KS_singles, and ready_rows
        self.read(self.basefilename)


        # Only spin unpolarized calculations are supported atm
        # > FIXME
        assert len(self.calc.wfs.kpt_u) == 1, "LrTDDFTindexed does not support more than one k-point/spin."
        self.kpt_ind = 0
        # <

        # If min/max_occ/unocc were not given, initialized them to include
        # everything: min_occ/unocc => 0, max_occ/unocc to nubmer of wfs,
        # energy diff to numerical infinity
        nbands = len(self.calc.wfs.kpt_u[self.kpt_ind].f_n)
        if self.min_occ is None:
            self.min_occ = 0
        if self.min_unocc is None:
            self.min_unocc = self.min_occ
        if self.max_occ is None:
            self.max_occ = nbands - 1
        if self.max_unocc is None:
            self.max_unocc = self.max_occ
        if self.max_energy_diff is None:
            self.max_energy_diff = 1e9


        # Write info file
        self.parent_comm.barrier()
        if self.parent_comm.rank == 0:
            self.write_info(self.basefilename+'.LR_info')

        # Flags to prevent repeated calculation
        self.calc_ready = False
        self.kss_list_ready = False
        self.ks_prop_ready = False
        self.K_matrix_ready = False



    def read(self, basename):
        info_file = basename+'.LR_info'
        if os.path.exists(info_file) and os.path.isfile(info_file):
            self.read_info(info_file)
        else:
            return

        # Read ALL ready_rows files
        ready_files = glob.glob(basename+'.ready_rows.*')
        for ready_file in ready_files:
            if os.path.isfile(ready_file):
                for line in open(ready_file,'r'):
                    line = line.split()
                    self.ready_indices.append([int(line[0]),int(line[1])])

        # Read KS_singles file if exists
        # occ_index | unocc index | energy diff | population diff |
        #   dmx | dmy | dmz | magnx | magny | magnz
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
                assert self.index_of_kss(i,p) is None, 'KS transition %d->%d found twice in KS_singles files.' % (i,p)
                self.kss_list.append(kss)
            if len(self.kss_list) <= 0: self.kss_list = None
            kss_file.close()



    def write_info(self, fname):
        f = open(fname,'a+')
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
        f.write('%20s = %s\n' % ('',''))
        f.close()


    def read_info(self, fname):
        info_file = open(fname,'r')
        for line in info_file:
            if line[0] == '#': continue
            if len(line.split()) <= 2: continue
            key = line.split()[0]
            value = line.split()[2]
            # .....
            # FIXME: do something, like warn if changed
            # ... 
        info_file.close()


    # omega_k = sqrt(lambda_k)
    def get_excitation_energy(self, k, units='a.u.'):
        """Get excitation energy for kth interacting transition

        Input parameters:

        k
          Transition index (indexing starts from zero).

        units
          Units for excitation energy: 'a.u.' (Hartree) or 'eV'.
        """
        self.calculate_excitations() # Calculates only on the first time
        if units == 'a.u.':
            return np.sqrt(self.evalues[k])
        elif units == 'eV':
            return np.sqrt(self.evalues[k]) * ase.units.Hartree
        else:
            raise RuntimeError('Unknown units.')

    # populations F**2
    #def get_excitation_weights(self, k, threshold=0.01):
    #    self.calculate_excitations()
    #    x = np.power(self.get_local_evector(k), 2)
    #    x = x[x > threshold]
    #    self.eh_comm.sum(x)
    #    return x

    def get_oscillator_strength(self, k):
        """Get oscillator strength for an interacting transition

        Returns oscillator strength of kth interacting transition.

        Input parameters:

        k
          Transition index (indexing starts from zero).
        """
        self.calculate_excitations()
        dm = np.zeros(3)
        for kss_ip in self.kss_list:
            i = kss_ip.occ_ind
            p = kss_ip.unocc_ind
            # c = sqrt(ediff_ip / omega_n) sqrt(population_ip) * F^(n)_ip
            c = np.sqrt(kss_ip.energy_diff / self.get_excitation_energy(k))
            c *= np.sqrt(kss_ip.pop_diff) * self.get_local_eig_coeff(k,self.index_map[(i,p)])
            # dm_n = c * dm_ip
            dm[0] += c * kss_ip.dip_mom_r[0]
            dm[1] += c * kss_ip.dip_mom_r[1]
            dm[2] += c * kss_ip.dip_mom_r[2]

        self.eh_comm.sum(dm)

        # osc = 2 * omega |dm|**2 / 3
        osc = 2. * self.get_excitation_energy(k) * (dm[0]*dm[0]+dm[1]*dm[1]+dm[2]*dm[2]) / 3.
        return osc

    def get_rotatory_strength(self, k, units='a.u.'):
        """Get rotatory strength for an interacting transition

        Returns rotatory strength of kth interacting transition.

        Input parameters:

        k
          Transition index (indexing starts from zero).

        units
          Units for rotatory strength: 'a.u.' or 'cgs'.
        """
        self.calculate_excitations()
        dm = np.zeros(3)
        magn = np.zeros(3)

        # J. Chem. Phys., Vol. 116, No. 16, 22 April 2002
        for kss_ip in self.kss_list:
            i = kss_ip.occ_ind
            p = kss_ip.unocc_ind
            # c = sqrt(ediff_ip / omega_n) sqrt(population_ip) * F^(n)_ip
            c1 = np.sqrt(kss_ip.energy_diff / self.get_excitation_energy(k))
            c2 = np.sqrt(kss_ip.pop_diff) * self.get_local_eig_coeff(k,self.index_map[(i,p)])
            # dm_n = c * dm_ip
            dm[0] += c1*c2 * kss_ip.dip_mom_r[0]
            dm[1] += c1*c2 * kss_ip.dip_mom_r[1]
            dm[2] += c1*c2 * kss_ip.dip_mom_r[2]

            magn[0] += c2/c1 * kss_ip.magn_mom[0]
            magn[1] += c2/c1 * kss_ip.magn_mom[1]
            magn[2] += c2/c1 * kss_ip.magn_mom[2]

        self.eh_comm.sum(dm)
        self.eh_comm.sum(magn)

        if units == 'a.u.':
            return - ( dm[0] * magn[0] + dm[1] * magn[1] + dm[2] * magn[2] )
        elif units == 'cgs':
            return - 64604.8164 * ( dm[0] * magn[0] + dm[1] * magn[1] + dm[2] * magn[2] )
        else:
            raise RuntimeError('Unknown units.')



    def get_transitions(self, filename=None, min_energy=0.0, max_energy=30.0, units='eVcgs'):
        """Get transitions: energy, dipole strength and rotatory strength.

        Returns transitions as (w,S,R) where w is an array of frequencies,
        S is an array of corresponding dipole strengths, and R is an array of
        corresponding rotatory strengths.

        Input parameters:

        min_energy
          Minimum energy 

        min_energy
          Maximum energy

        units
          Units for spectrum: 'a.u.' or 'eVcgs'
        """

        self.calculate_excitations()

        if units == 'eVcgs':
            convf = 1/ase.units.Hartree
        else:
            convf = 1.
            
        max_energy = max_energy * convf
        min_energy = min_energy * convf

        w = []
        S = []
        R = []
        
        print >> self.txt, 'Calculating transitions (', str(datetime.datetime.now()), ').',
        for (k, omega2) in enumerate(self.evalues):
            if k % 10 == 0:
                print >> self.txt, '.',
                self.txt.flush()
            
            w.append(self.get_excitation_energy(k))
            S.append(self.get_oscillator_strength(k))
            R.append(self.get_rotatory_strength(k))

        w = np.array(w)
        S = np.array(S)
        R = np.array(R)

        print >> self.txt, ''
                
        if units == 'a.u.':
            pass
        elif units == 'eVcgs':
            w *= ase.units.Hartree
            S /= ase.units.Hartree
            R *= 64604.8164 # from turbomole
        else:
            raise RuntimeError('Unknown units.')


        if filename is not None and gpaw.mpi.world.rank == 0:
            sfile = open(filename,'w')
            for (ww,SS,RR) in zip(w,S,R):
                sfile.write("%12.8lf  %12.8lf  %12.8lf\n" % (ww,SS,RR))
            sfile.close()

        return (w,S,R)
        


    
    def get_spectrum(self, filename=None, min_energy=0.0, max_energy=30.0, energy_step=0.01, width=0.1, units='eVcgs'):
        """Get spectrum for dipole and rotatory strength.

        Returns folded spectrum as (w,S,R) where w is an array of frequencies,
        S is an array of corresponding dipole strengths, and R is an array of
        corresponding rotatory strengths.

        Input parameters:

        min_energy
          Minimum energy 

        min_energy
          Maximum energy

        energy_step
          Spacing between calculated energies

        width
          Width of the Gaussian


        units
          Units for spectrum: 'a.u.' or 'eVcgs'
        """

        self.calculate_excitations()

        if units == 'eVcgs':
            convf = 1/ase.units.Hartree
        else:
            convf = 1.
            
        max_energy = max_energy * convf
        min_energy = min_energy * convf
        energy_step = energy_step * convf
        width = width * convf

        w = np.arange(min_energy, max_energy, energy_step)
        S = w*0.
        R = w*0.

        (ww,SS,RR) = self.get_transitions(min_energy=min_energy, max_energy=max_energy, units='a.u.')
                
        print >> self.txt, 'Calculating spectrum (', str(datetime.datetime.now()), ').',
        for (k, omega2) in enumerate(self.evalues):
            if k % 10 == 0:
                print >> self.txt, '.',
                self.txt.flush()
            
            c = SS[k] / width / math.sqrt(2*np.pi)
            S += c * np.exp( (-.5/width/width) * np.power(w-ww[k],2) ) 

            c = RR[k] / width / np.sqrt(2*np.pi)
            R += c * np.exp( (-.5/width/width) * np.power(w-ww[k],2) ) 

        print >> self.txt, ''
                
        if units == 'a.u.':
            pass
        elif units == 'eVcgs':
            w *= ase.units.Hartree
            S /= ase.units.Hartree
            R *= 64604.8164 # from turbomole
        else:
            raise RuntimeError('Unknown units.')

        if filename is not None and gpaw.mpi.world.rank == 0:
            sfile = open(filename,'w')
            for (ww,SS,RR) in zip(w,S,R):
                sfile.write("%12.8lf  %12.8lf  %12.8lf\n" % (ww,SS,RR))
            sfile.close()

        return (w,S,R)
        


###
# Utility
###

    def index_of_kss(self,i,p):
        for (ind,kss) in enumerate(self.kss_list):
            if kss.occ_ind == i and kss.unocc_ind == p:
                return ind
        return None


# Probably does not work
#    def get_evector(self, k):
#        # First get the eigenvector for the rank 0 and
#        # then broadcast it  
#        evec = np.zeros_like(self.evectors[0])
#        # Rank owning this evector
#        off = k % self.stride
#        k_local = k // self.stride
#        if off == self.offset:
#            if self.offset == 0:
#                evec[:] = self.evectors[k_local]
#            else:
#                self.eh_comm.send(self.evectors[k_local], 0, 123)
#        else:
#            if self.offset == 0:
#                self.eh_comm.receive(evec, off, 123)
#        # Broadcast
#        self.eh_comm.broadcast(evec, 0)
#        return evec


    def get_local_index(self,k):
        if k % self.stride != self.offset:
            return None
        return k // self.stride

    def get_local_eig_coeff(self, k, ip):
        kloc = self.get_local_index(k)
        if kloc is None:
            return 0.0
        return self.evectors[kloc][ip]
    


####
# Somewhat ugly things
####

    def calculate_excitations(self):
        """Calculates linear response excitations.

        This function is called implicitely by get_excitation_energy, etc.
        but it can be called directly to force calculation.
        """
        # If recalculating matrix, clear everything also delete files
        if self.recalculate == 'all' or self.recalculate == 'matrix':
            self.kss_list = None
            self.evalues = None
            self.evectors = None
            self.ready_indices = []
            # delete files
            if self.parent_comm.rank == 0:
                for ready_file in glob.glob(self.basefilename+'.ready_rows.*'):
                    os.remove(ready_file)
                for K_file in glob.glob(self.basefilename + '.K_matrix.*'):
                    os.remove(K_file)

        # if only recalculating eigen system, reconstruct kss_list and clear
        # eigenvalues and eigenvectors
        if self.recalculate == 'eigen':
            self.calculate_KS_singles()
            self.evalues = None
            self.evectors = None

        # If eigenvectors are not there, we may have to calculate
        # KS properties and K-matrix, and we have to diagonalize
        if self.evectors is None:

            # Calculate KS properties and K-matrix if needed
            if ( self.recalculate is None 
                 or self.recalculate == 'all'
                 or self.recalculate == 'matrix' ):
                self.calculate_KS_singles()
                self.calculate_KS_properties()
                self.calculate_K_matrix()

            # Wait... we don't want to read incomplete files
            self.parent_comm.barrier()

            # If only matrix was to be recalculate, exit
            if self.recalculate == 'matrix':
                self.recalculate = None # Recalculate only first time 
                return

            # Diagonalize
            self.diagonalize()

            # Recalculate only first time
            self.recalculate = None
            
            # FIXME: write interacting transitions properties to a file
            # for fast recalculation of spectrum


    # Diagonalize Casida matrix
    def diagonalize(self):
        # Are we using ScaLapack
        par = self.calc.input_parameters
        sl_lrtddft = par.parallel['sl_lrtddft']

        # Init
        self.index_map = {}        # (i,p) to matrix index map
        nrow = len(self.kss_list)  # total rows
        nloc = 0                   # local rows

        # Create indexing
        for (ip,kss) in enumerate(self.kss_list):
            self.index_map[(kss.occ_ind,kss.unocc_ind)] = ip

        print >> self.txt, 'Reading data for diagonalize (', str(datetime.datetime.now()), '). ',
            
        # Read ALL ready_rows files
        for rrfn in glob.glob(self.basefilename + '.ready_rows.*'):
            for line in open(rrfn,'r'):
                i = int(line.split()[0])
                p = int(line.split()[1])
                key = (i,p)
                
                # if key not in self.kss_list, drop it
                # i.e. we are calculating just part of the whole matrix
                ip = self.index_of_kss(i,p)
                if ip is not None:
                    assert ip == self.index_map[key], 'List index %d is not equal to ind_map index %d for key (%d,%d)\n' % (ip,self.index_map[key],i,p)
                    if self.get_local_index(ip) is not None: nloc += 1
        
        # Matrix build
        omega_matrix = np.zeros((nloc,nrow))
        omega_matrix[:,:] = np.NAN # fill with NaNs to detect problems
        # Read ALL K_matrix files
        for Kfn in glob.glob(self.basefilename + '.K_matrix.*'): 
            print >> self.txt, '.',
            self.txt.flush()
            for line in open(Kfn,'r'):
                line = line.split()
                ipkey = (int(line[0]), int(line[1]))
                jqkey = (int(line[2]), int(line[3]))
                Kvalue = float(line[4])
                # if not in index map, ignore
                if ( not ipkey in self.index_map
                     or not jqkey in self.index_map ):
                    continue
                # if ip on this this proc
                if self.get_local_index(self.index_map[ipkey]) is not None:
                    # add value to matrix
                    lip = self.get_local_index(self.index_map[ipkey])
                    jq = self.index_map[jqkey]
                    omega_matrix[lip,jq] = Kvalue
                # if jq on this this proc
                if self.get_local_index(self.index_map[jqkey]) is not None:
                    # add value to matrix
                    ljq = self.get_local_index(self.index_map[jqkey])
                    ip = self.index_map[ipkey]
                    omega_matrix[ljq,ip] = Kvalue
        print >> self.txt, ''

        # If any NaNs found, we did not read all matrix elements... BAD
        if np.isnan(np.sum(np.sum(omega_matrix))):
            raise RuntimeError('Not all required LrTDDFT matrix elements could be found.')

        # Add diagonal values
        for kss in self.kss_list:
            key = (kss.occ_ind, kss.unocc_ind)
            if key in self.index_map:
                ip = self.index_map[key]
                lip = self.get_local_index(ip)
                if lip is None: continue
                omega_matrix[lip,ip] += kss.energy_diff * kss.energy_diff


        # Calculate eigenvalues
        self.evalues = np.zeros(nrow)


        print >> self.txt, 'Diagonalizing (', str(datetime.datetime.now()), ')'
        
        # ScaLapack
        if sl_lrtddft is not None:
            ksl = LrTDDFTLayouts(sl_lrtddft, nrow, self.dd_comm,
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

            # broadcast to all
            self.eh_comm.sum(omega_matrix)

            # diagonalize
            diagonalize(omega_matrix, self.evalues)

            # global to local
            for ip in range(nrow):
                lip = self.get_local_index(ip)
                if lip is None: continue
                self.evectors[lip,:] = omega_matrix[ip,:]

            
# OLD VERSION... does not work anymore
#    def diagonalize2(self):
#        """Diagonalizes the Omega matrix. Use ScaLAPACK if available."""
#
#        par = InputParameters()  # We need ScaLAPACK parameters
#        sl_omega = par.parallel['sl_lrtddft']
#
#        # Determine index map
#        # Read the calculated rows from all read_rows files. K_matrix elements
#        # are calculated with a stride comm.size, the global index of eh-pair
#        # is thus given by g = l * stride + (stride + offset) % stride
#
#        self.index_map = {}
#        nind = 0
#        for off in range(self.stride):
#            local_ind = 0
#            rrfn = self.basefilename + '.ready_rows.' + '%04dof%04d' % (off, self.stride)
#            for line in open(rrfn,'r'):
#                i = int(line.split()[0])
#                p = int(line.split()[1])
#                key = (i,p)
#                # if key not in self.kss_list, drop it
#                found = False
#                for kss in self.kss_list:
#                    if kss.occ_ind == i and kss.unocc_ind == p:
#                        found = True
#                        break
#                if found and not key in self.index_map:
#                    global_ind = local_ind * self.stride + (self.stride + off) % self.stride 
#                    self.index_map[key] = global_ind
#                    local_ind += 1
#                    nind += 1
#
#        # Read omega matrix and diagonalize
#
#        if sl_omega is not None:
#            # Use ScaLAPACK
#            mynind = len(range(self.offset, nind, self.stride))
#            omega_matrix = np.zeros((mynind, nind))
#            Kfn = self.basefilename + '.K_matrix.' + '%04dof%04d' % (self.offset, self.stride)
#            for line in open(Kfn,'r'):
#                line = line.split()
#                ipkey = (int(line[0]), int(line[1]))
#                jqkey = (int(line[2]), int(line[3]))
#                Kvalue = float(line[4])
#                if not ipkey in self.index_map or not jqkey in self.index_map:
#                    continue
#                ip = self.index_map[ipkey]
#                # Debugging stuff
#                off = ip % self.stride
#                assert off == self.offset
#                myip = ip // self.stride
#                jq = self.index_map[jqkey]
#                omega_matrix[myip,jq] = Kvalue
#
#            # Add diagonal values
#            for kss in self.kss_list:
#                key = (kss.occ_ind, kss.unocc_ind)
#                if key in self.index_map:
#                    global_ind = self.index_map[key]
#                    if (global_ind % self.stride) == self.offset:
#                        local_ind = global_ind // self.stride
#                        omega_matrix[local_ind, global_ind] += kss.energy_diff * kss.energy_diff
#
#            ksl = LrTDDFTLayouts(sl_omega, nind, self.dd_comm, self.eh_comm)
#            self.evectors = omega_matrix
#            self.evalues = np.zeros(nind)
#            ksl.diagonalize(self.evectors, self.evalues)
#            
#        else:
#            # Serial LAPACK
#            omega_matrix = np.zeros((nind,nind))
#            for off in range(self.stride):
#                Kfn = self.basefilename + '.K_matrix.' + '%04dof%04d' % (off, self.stride)
#                for line in open(Kfn,'r'):
#                    line = line.split()
#                    ipkey = (int(line[0]), int(line[1]))
#                    jqkey = (int(line[2]), int(line[3]))
#                    Kvalue = float(line[4])
#                    if not ipkey in self.index_map or not jqkey in self.index_map:
#                        continue
#                    ip = self.index_map[ipkey]
#                    jq = self.index_map[jqkey]
#                    omega_matrix[ip,jq] = Kvalue
#
#
#            diag = np.zeros(nind)
#            for kss in self.kss_list:
#                key = (kss.occ_ind,kss.unocc_ind)
#                if key in self.index_map:
#                    diag[self.index_map[key]] = kss.energy_diff * kss.energy_diff
#            for (k,value) in enumerate(diag):
#                omega_matrix[k,k] += value
#
#
##            if gpaw.mpi.world.rank == 0:
##                for i in range(nind):
##                    for j in range(nind):
##                        print '%18.12lf ' % omega_matrix[i,j],
##                    print
#
#            # diagonalize
#            self.evalues = np.zeros(nind)
#            diagonalize(omega_matrix, self.evalues)
#            self.evectors = omega_matrix[self.offset::self.stride].copy()
#
#        return self.evectors



    # Create kss_list
    def calculate_KS_singles(self):
        # If ready, then done
        if self.kss_list_ready: return 

        # shorthands
        eps_n = self.calc.wfs.kpt_u[self.kpt_ind].eps_n      # eigen energies
        f_n = self.calc.wfs.kpt_u[self.kpt_ind].f_n          # occupations

        # Create Kohn-Sham single excitation list with energy filter
        old_kss_list = self.kss_list   # save old list for later
        self.kss_list = []             # create a completely new list
        # Occupied loop
        for i in range(self.min_occ, self.max_occ+1):
            # Unoccupied loop
            for p in range(self.min_unocc, self.max_unocc+1):
                deps_pi = eps_n[p] - eps_n[i] # energy diff
                df_ip = f_n[i] - f_n[p]       # population diff
                # Filter it
                if np.abs(deps_pi) <= self.max_energy_diff and df_ip > self.min_pop_diff:
                    # i, p, deps, df, mur, muv, magn
                    kss = KSSingle(i,p)
                    kss.energy_diff = deps_pi
                    kss.pop_diff = df_ip
                    self.kss_list.append(kss)


        # Sort by energy diff
        def energy_diff_cmp(kss_ip,kss_jq):
            ediff = kss_ip.energy_diff - kss_jq.energy_diff
            if ediff < 0: return -1
            elif ediff > 0: return 1
            return 0
        self.kss_list = sorted(self.kss_list, cmp=energy_diff_cmp)            


        # Remove old transitions and add new, but only add to the end of
        # the list (otherwise lower triangle matrix is not filled completely)
        if old_kss_list is not None:
            new_kss_list = self.kss_list   # required list
            self.kss_list = []             # final list with correct order
            # Old list first
            # If in new and old lists
            for kss_o in old_kss_list:
                found = False
                for kss_n in new_kss_list:
                    if ( kss_o.occ_ind == kss_n.occ_ind
                         and kss_o.unocc_ind == kss_n.unocc_ind ):
                        found = True
                        break
                if found: 
                    self.kss_list.append(kss_o) # Found, add to final list
                else:
                    pass                        # else drop
                
            # Now old transitions which are not in new list where dropped

            # If only in new list
            app_list = []
            for kss_n in new_kss_list:
                found = False
                for kss in self.kss_list:
                    if kss.occ_ind == kss_n.occ_ind and kss.unocc_ind == kss_n.unocc_ind:
                        found = True
                        break
                if not found:
                    app_list.append(kss_n) # Not found, add to final list
                else:
                    pass                   # else skip to avoid duplicates

            # Create the final list
            self.kss_list += app_list

        # Prevent repeated work
        self.kss_list_ready = True


    # Calculates pair density of (i,p) transition (and given kpt)
    # dnt_Gp: pair density without compensation charges on coarse grid
    # dnt_gp: pair density without compensation charges on fine grid      (XC)
    # drhot_gp: pair density with compensation charges on fine grid  (poisson)
    def calculate_pair_density(self, kpt, kss_ip, dnt_Gip, dnt_gip, drhot_gip):
        if self.pair_density is None:
            self.pair_density = LRiPairDensity(self.calc.density)
        self.pair_density.initialize(kpt, kss_ip.occ_ind, kss_ip.unocc_ind)
        self.pair_density.get(dnt_Gip, dnt_gip, drhot_gip)
        

    # Calculate dipole moment and magnetic moment for noninteracting
    # Kohn-Sham transitions
    def calculate_KS_properties(self):
        # Check that kss_list is up-to-date
        self.calculate_KS_singles()

        # Check if we already have them, and if yes, we are done
        if self.ks_prop_ready: return
        self.ks_prop_ready = True
        for kss_ip in self.kss_list:
            if kss_ip.dip_mom_r is None or kss_ip.magn_mom is None:
                self.ks_prop_ready = False
                break
        if self.ks_prop_ready: return
        

        # Initialize wfs, paw corrections and xc, if not done yet
        if not self.calc_ready:
            self.calc.converge_wave_functions()
            spos_ac = self.calc.initialize_positions()
            self.calc.wfs.initialize(self.calc.density, 
                                     self.calc.hamiltonian, spos_ac)
            self.xc.initialize(self.calc.density, self.calc.hamiltonian, self.calc.wfs, self.calc.occupations)
            self.calc_ready = True


        # Init pair densities
        dnt_Gip = self.calc.wfs.gd.empty()
        dnt_gip = self.calc.density.finegd.empty()
        drhot_gip = self.calc.density.finegd.empty()

        # Init gradients of wfs
        grad_psit2_G = [self.calc.wfs.gd.empty(),self.calc.wfs.gd.empty(),self.calc.wfs.gd.empty()]


        # Init gradient operators
        grad = []
        dtype = pick(self.calc.wfs.kpt_u[self.kpt_ind].psit_nG,0).dtype
        for c in range(3):
            grad.append(Gradient(self.calc.wfs.gd, c, dtype=dtype,n=2))


        # Loop over all KS single excitations
        for kss_ip in self.kss_list:
            # If have dipole moment and magnetic moment, already done and skip
            if kss_ip.dip_mom_r is not None and kss_ip.magn_mom is not None: continue
            
            # Dipole moment
            self.calculate_pair_density( self.calc.wfs.kpt_u[self.kpt_ind],
                                         kss_ip, dnt_Gip, dnt_gip, drhot_gip )
            kss_ip.dip_mom_r = self.calc.density.finegd.calculate_dipole_moment(drhot_gip)


            # Magnetic transition dipole
            # J. Chem. Phys., Vol. 116, No. 16, 22 April 2002

            # Coordinate vector r
            r_cG, r2_G = coordinates(self.calc.wfs.gd)
            # Gradients
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
            
            # augmentation contributions to magnetic moment
            magn_a = np.zeros(3)
            for a, P_ni in self.calc.wfs.kpt_u[self.kpt_ind].P_ani.items():
                Pi_i = P_ni[kss_ip.occ_ind]
                Pp_i = P_ni[kss_ip.unocc_ind]
                rnabla_iiv = self.calc.wfs.setups[a].rnabla_iiv
                for c in range(3):
                    for i1, Pi in enumerate(Pi_i):
                        for i2, Pp in enumerate(Pp_i):
                            magn_a[c] += Pi * Pp * rnabla_iiv[i1, i2, c]
            self.calc.wfs.gd.comm.sum(magn_a) # sum up from different procs

            # FIXME: Why we have alpha (fine structure constant?) here=
            kss_ip.magn_mom = ase.units.alpha / 2. * (magn_g + magn_a)



        # Wait... to avoid io problems, and write KS_singles file
        self.parent_comm.barrier()
        if self.parent_comm.rank == 0:
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
        self.parent_comm.barrier()
            
        self.ks_prop_ready = True      # avoid repeated work
        


##############################
# The big ugly thing
##############################

    # Calculate K-matrix
    def calculate_K_matrix(self, fH_pre=1.0, fxc_pre=1.0):
        """Calculates K-matrix.

        Only for pros.

        fH_pre
          Prefactor for Hartree term

        fxc_pre
          Prefactor for exchange-correlation term
        """
        
        # Check that kss_list is up-to-date
        self.calculate_KS_singles()

        # Check if already done before allocating
        if self.K_matrix_ready:
            return

        # Loop over all transitions
        self.K_matrix_ready = True  # mark done... if not, it's changed
        nrows = 0                   # number of rows for timings
        for (ip,kss_ip) in enumerate(self.kss_list):
            i = kss_ip.occ_ind
            p = kss_ip.unocc_ind

            # if not mine, skip it
            if self.get_local_index(ip) is None:
                continue
            # if already calculated, skip it
            if [i,p] in self.ready_indices:
                continue                          

            self.K_matrix_ready = False  # something not calculated, must do it
            nrows += 1

        # If matrix was ready, done
        if self.K_matrix_ready: return    


        # Initialize wfs, paw corrections and xc, if not done yet
        if not self.calc_ready:
            self.calc.converge_wave_functions()
            spos_ac = self.calc.initialize_positions()
            self.calc.wfs.initialize(self.calc.density, 
                                     self.calc.hamiltonian, spos_ac)
            self.xc.initialize(self.calc.density, self.calc.hamiltonian, self.calc.wfs, self.calc.occupations)
            self.calc_ready = True


        # Filenames
        Kfn = self.basefilename + '.K_matrix.' + '%04dof%04d' % (self.offset, self.stride)
        rrfn = self.basefilename + '.ready_rows.' + '%04dof%04d' % (self.offset, self.stride)
        logfn = self.basefilename + '.log.' + '%04dof%04d' % (self.offset, self.stride)

        # Open only on dd_comm root
        if self.dd_comm.rank == 0:
            self.Kfile = open(Kfn, 'a+')
            self.ready_file = open(rrfn,'a+')
            self.log_file = open(logfn,'a+')

        # Init Poisson solver
        self.poisson = PoissonSolver(nn=self.calc.hamiltonian.poisson.nn)
        self.poisson.set_grid_descriptor(self.calc.density.finegd)
        self.poisson.initialize()

        # Allocate grids for densities and potentials
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


        # Init timings
        [irow, tp, t0, tm, ap, bp, cp] = [0, None, None, None, None, None, None]
        
        #################################################################
        # Outer loop over KS singles
        for (ip,kss_ip) in enumerate(self.kss_list):
            # if not mine, skip it
            if self.get_local_index(ip) is None:
                continue                          
            # if already calculated, skip it
            if [kss_ip.occ_ind,kss_ip.unocc_ind] in self.ready_indices:
                continue                          

            # timings
            if self.dd_comm.rank == 0:
                # on every 10th transition, update ETA
                neta = 10
                if irow % neta == 0:
                    tm = t0;  t0 = tp;  tp = time.time()
                    # 2nd order fit:
                    #   t(irow) = a irow0**2 + b irow0 + c
                    #   t0'' = 2 a                        => a = t0''/2
                    #   t0'  = 2 a irow0 + b              => b = t0' - 2 a irow0
                    #   t0   = a irow0**2 + b irow0 + c   => c = t0 - a irow**2 - b irow0
                    # eta = t(nrows)
                    if tm is not None:
                        a = .5*(tp - 2*t0 + tm)/float(neta*neta)
                        b = (tp - tm)/(2.*neta) - 2.*a*float(irow-neta)
                        c = t0 - a*float((irow-neta)*(irow-neta)) - b*float(irow-neta)
                        #self.log_file.write('Calculated parameters for ETA: %12.6lf %12.6lf %12.6lf\n' % (a,b,c))
                        # Do some mixing to avoid large oscillations...
                        if ap is not None: a = .25*a + .75*ap
                        if bp is not None: b = .25*b + .75*bp
                        if cp is not None: c = .25*c + .75*cp
                        ap = a; bp = b; cp = c
                        #self.log_file.write('Averaged parameters for ETA: %12.6lf %12.6lf %12.6lf\n' % (a,b,c))
                    # No ETA available yet
                    else:
                        a = 0.0
                        b = 0.0
                        c = tp
                eta = a*nrows*nrows + b * nrows + c - time.time()
                self.log_file.write('Calculating pair %5d => %5d  ( %s, ETA %9.1lfs )\n' % (kss_ip.occ_ind, kss_ip.unocc_ind, str(datetime.datetime.now()), eta))
                self.log_file.flush()
                irow += 1



            # Pair density
            self.timer.start('Pair density')
            dnt_Gip[:] = 0.0
            dnt_gip[:] = 0.0
            drhot_gip[:] = 0.0
            self.calculate_pair_density(self.calc.wfs.kpt_u[self.kpt_ind],
                                        kss_ip, dnt_Gip, dnt_gip, drhot_gip)
            self.timer.stop('Pair density')
            
            # Smooth Hartree potential
            self.timer.start('Poisson')
            dVht_gip[:] = 0.0
            self.poisson.solve(dVht_gip, drhot_gip, charge=None)
            self.timer.stop('Poisson')

            # 
            # FIXME: All XC stuff should be in its own function
            # xc_energy_density()
            #

            # Smooth xc potential
            # (finite difference approximation from xc-potential)
            self.timer.start('Smooth XC')
            # finite difference plus,  vxc+ = vxc(n + deriv_scale * dn) 
            nt_g[self.kpt_ind][:] = self.deriv_scale * dnt_gip
            nt_g[self.kpt_ind][:] += self.calc.density.nt_sg[self.kpt_ind]
            dVxct_gip[:] = 0.0
            self.xc.calculate(self.calc.density.finegd, nt_g, dVxct_gip)
            # finite difference minus, vxc+ = vxc(n - deriv_scale * dn)
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
            # FIXME, only spin unpolarized works
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


            #################################################################
            # Inner loop over KS singles            
            K = [] # storage for row before writing to file
            for (jq,kss_jq) in enumerate(self.kss_list):
                i = kss_ip.occ_ind
                p = kss_ip.unocc_ind
                j = kss_jq.occ_ind
                q = kss_jq.unocc_ind


                # Only lower triangle
                if ip < jq: continue

                # Pair density dn_jq
                self.timer.start('Pair density')
                dnt_Gjq[:] = 0.0
                dnt_gjq[:] = 0.0
                drhot_gjq[:] = 0.0
                self.calculate_pair_density(self.calc.wfs.kpt_u[self.kpt_ind], kss_jq, 
                                            dnt_Gjq, dnt_gjq, drhot_gjq)
                self.timer.stop('Pair density')


                self.timer.start('Integrate')
                Ig = 0.0
                # Hartree smooth part, RHOT_JQ HERE???
                Ig += fH_pre * self.calc.density.finegd.integrate(dVht_gip, drhot_gjq)
                # XC smooth part
                Ig += fxc_pre * self.calc.density.finegd.integrate(dVxct_gip[self.kpt_ind], dnt_gjq)
                self.timer.stop('Integrate')


                # FIXME: make function of the following loop
                # FIXME, only spin unpolarized works atm
                kss_ip.spin = kss_jq.spin = 0
                # Atomic corrections
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
                    Ia += fH_pre * 2.0 * np.dot(Djq_p, np.dot(C_pp, Dip_p))

                    # XC part, CHECK THIS JQ EVERWHERE!!!
                    Ia += fxc_pre * np.dot(I_asp[a][kss_jq.spin], Djq_p)
                    
                self.timer.stop('Atomic corrections')
                Ia = self.dd_comm.sum(Ia)

                # Total integral
                Itot = Ig + Ia
                
                # K_ip,jq += 2*sqrt(f_ip*f_jq*eps_pi*eps_qj)<ip|dH|jq>
                K.append( [i,p,j,q,
                           2.*np.sqrt(kss_ip.pop_diff * kss_jq.pop_diff 
                                   * kss_ip.energy_diff * kss_jq.energy_diff)
                           * Itot] )

            # Write  i p j q Omega_Hxc       (format: -2345.789012345678)
            self.timer.start('Write Omega')
            # Write only on dd_comm root
            if self.dd_comm.rank == 0:
                
                # Write only lower triangle of K-matrix
                for [i,p,j,q,Kipjq] in K:
                    self.Kfile.write("%5d %5d %5d %5d %18.12lf\n" % (i,p,j,q,Kipjq))
                self.Kfile.flush() # flush K-matrix before ready_rows

                # Write and flush ready rows
                self.ready_file.write("%d %d\n" % (kss_ip.occ_ind,
                                                   kss_ip.unocc_ind))
                self.ready_file.flush()
                
            self.timer.stop('Write Omega')

            # Update ready rows before continuing
            self.ready_indices.append([kss_ip.occ_ind,kss_ip.unocc_ind])

        # Close files on dd_comm root
        if self.dd_comm.rank == 0:
            self.Kfile.close()
            self.ready_file.close()
            self.log_file.close()


    def __del__(self):
        self.timer.stop('LrTDDFT')



    


###############################################################################
# Small utility classes and functions
###############################################################################

class LRiPairDensity:
    """Pair density calculator class."""
    
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
        """Get pair densities.

        nt_G
          Pair density without compensation charges on the coarse grid

        nt_g
          Pair density without compensation charges on the fine grid

        rhot_g
          Pair density with compensation charges on the fine grid
        """
        # Coarse grid product
        np.multiply(self.psit1_G.conj(), self.psit2_G, nt_G)
        # Interpolate to fine grid
        self.density.interpolator.apply(nt_G, nt_g)
        
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
        rhot_g[:] = nt_g[:]
        self.density.ghat.add(rhot_g, Q_aL)



###############################################################################
# Container class for noninteracting Kohn-Sham single transition
class KSSingle:
    """Container class for noninteracting Kohn-Sham single transition."""
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
    

###############################
# BLACS layout for ScaLAPACK
class LrTDDFTLayouts:
    """BLACS layout for distributed Omega matrix in linear response
       time-dependet DFT calculations"""

    def __init__(self, sl_omega, nkq, dd_comm, eh_comm):
        mcpus, ncpus, blocksize = tuple(sl_omega)
        self.world = eh_comm.parent
        self.dd_comm = dd_comm
        # All the ranks within domain communicator contain the omega matrix
        # construct new communicator only on domain masters
        eh_ranks = np.arange(eh_comm.size) * dd_comm.size
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
        self.dd_comm.broadcast(O_nN, 0)


###############################################################################
def lr_communicators(world, dd_size, eh_size):
    """Create communicators for LrTDDFT calculation.

    Input parameters:
    
    world
      MPI parent communicator (usually gpaw.mpi.world)
      
    dd_size
      Over how many processes is domain distributed.
      
    eh_size
      Over how many processes are electron-hole pairs distributed.
      
    ---------------------------------------------------------------------------
    Note
      Sizes must match, i.e., world.size must be equal to
      dd_size x eh_size, e.g., 1024 = 64*16

    Tip
      Use enough processes for domain decomposition (dd_size) to fit everything
      (easily) into memory, and use the remaining processes for electron-hole
      pairs as K-matrix build is trivially parallel over them.

    ---------------------------------------------------------------------------
    
    Returns:

    dd_comm
      MPI communicator for domain decomposition.
      Pass this to the ground state calculator object (GPAW object).
      
    eh_comm
      MPI communicator for K-matrix.
      Pass this to the linear response calculator (LrTDDFTindexed object).

    ----------------------------------------------------------------------

    Example (for 8 MPI processes):

    dd_comm, eh_comm = lr_communicators(gpaw.mpi.world, 4, 2)
    txt = 'lr_%04d_%04d.txt' % (dd_comm.rank, eh_comm.rank)
    lr = LrTDDFTindexed(GPAW('unocc.gpw', communicator=dd_comm), eh_comm=eh_comm, txt=txt)
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


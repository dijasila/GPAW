import sys
from time import ctime
import numpy as np
from ase.parallel import paropen
from ase.units import Hartree
from gpaw import GPAW
from gpaw.response.df import DF
from gpaw.utilities import devnull
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.mpi import rank, size, world
from gpaw.response.parallel import parallel_partition, \
     parallel_partition_list, set_communicator
from scipy.special.orthogonal import p_roots
from gpaw.response.cell import set_Gvectors
from gpaw.response.cuda import BaseCuda
import _gpaw

class RPACorrelation:

    def __init__(self,
                 calc,
                 vcut=None,
                 cell=None,
                 txt=None,
                 tag=None,
                 cuda=False,
                 nmultix=1,
                 sync=False,
                 qsym=True):
        
        self.calc = calc
        self.tag = tag
        self.cell = cell
        self.cuda = cuda
        self.nmultix=nmultix
        self.sync = sync
        
        if txt is None:
            if rank == 0:
                #self.txt = devnull
                self.txt = sys.stdout
            else:
                sys.stdout = devnull
                self.txt = devnull
        else:
            assert type(txt) is str
            from ase.parallel import paropen
            self.txt = paropen(txt, 'w')

        self.qsym = qsym
        self.vcut = vcut
        self.nspins = calc.wfs.nspins
        self.bz_k_points = calc.wfs.bzk_kc
        self.atoms = calc.get_atoms()
        self.setups = calc.wfs.setups
        self.bz_q_points = calc.wfs.kd.get_bz_q_points(first=True)
        if qsym == False:
            self.ibz_q_points = self.bz_q_points
            self.q_weights = (np.ones(len(self.bz_q_points))
                              / len(self.bz_q_points))
        else:
            op_scc = calc.wfs.kd.symmetry.op_scc
            self.ibz_q_points = calc.wfs.kd.get_ibz_q_points(self.bz_q_points,
                                                             op_scc)[0]
            self.q_weights = calc.wfs.kd.q_weights
        
        self.print_initialization()
        self.initialized = 0
   
    def get_rpa_correlation_energy(self,
                                   kcommsize=1,
                                   dfcommsize=world.size,
                                   directions=None,
                                   skip_gamma=False,
                                   ecutlist=[10., 20., 30.],
                                   smooth_cut=None,
                                   nbands=None,
                                   gauss_legendre=None,
                                   frequency_cut=None,
                                   frequency_scale=None,
                                   w=None,
                                   extrapolate=False,
                                   restart=None):

        self.ecutlist_e = np.sort(ecutlist)
        self.necut = len(self.ecutlist_e)
        ecut = max(self.ecutlist_e)
            
        self.initialize_calculation(w,
                                    ecut,
                                    smooth_cut,
                                    nbands,
                                    kcommsize,
                                    extrapolate,
                                    gauss_legendre,
                                    frequency_cut,
                                    frequency_scale)

        assert dfcommsize == world.size
        self.dfcomm = world
        E_qe = np.zeros((len(self.ibz_q_points), self.necut))
        if restart is not None:
            import os.path
            if os.path.isfile(restart):
                f = paropen(restart, 'r')
                lines = f.readlines()
                for iline, line in enumerate(lines):
                    tmp = line[1:-2].split()
                    for iecut in range(self.necut):
                        E_qe[iline, iecut] = eval(tmp[iecut])
                print >> self.txt, 'Correlation energy obtained ' \
                          +'from %s q-points obtained from restart file: ' \
                          % (iline+1), restart
                        
                f.close()
                qstart = iline + 1
            else:
                qstart = 0
        else:
            qstart = 0
            
        for index, q in enumerate(self.ibz_q_points):
            if index < qstart:
                continue
            if abs(np.dot(q, q))**0.5 < 1.e-5:
                E_q0_e = np.zeros(self.necut)
                if skip_gamma:
                    print >> self.txt, \
                          'Not calculating q at the Gamma point'
                    print >> self.txt
                else:
                    if directions is None:
                        directions = [[0, 1/3.], [1, 1/3.], [2, 1/3.]]
                    for d in directions:                                   
                        E_q0_e += self.E_q(q,
                                         index=index,
                                         direction=d[0]) * d[1]
                E_qe[index,:] = E_q0_e
            else:
                E_qe[index,:] = self.E_q(q, index=index)
                
            if restart is not None:
                f = paropen(restart, 'a')
                print >> f, E_qe[index,:]
                f.close()

        E_e = np.dot(np.array(self.q_weights), E_qe)
        self.E_qe = E_qe

        print >> self.txt, 'RPA correlation energy:'
        for iecut, ecut in enumerate(self.ecutlist_e):
            print >> self.txt, 'ecut %s : E_c = %s eV' % (ecut, E_e[iecut])
        print >> self.txt
        if self.cuda:
            print >> self.txt, 'Destroy Cuda !'
            self.cu.destroy()

        print >> self.txt, 'Calculation completed at:  ', ctime()
        print >> self.txt
        print >> self.txt, \
              '------------------------------------------------------'
        print >> self.txt
        return E_e

    def get_E_q(self,
                kcommsize=1,
                index=None,
                q=[0., 0., 0.],
                direction=0,
                integrated=True,
                ecut=10,
                smooth_cut=None,
                nbands=None,
                gauss_legendre=None,
                frequency_cut=None,
                frequency_scale=None,
                w=None,
                extrapolate=False):

        self.initialize_calculation(w, ecut, smooth_cut,
                                    nbands, kcommsize, extrapolate,
                                    gauss_legendre, frequency_cut,
                                    frequency_scale)
        self.dfcomm = world
        E_q = self.E_q(q,
                       direction=direction,
                       integrated=integrated)
        
        print >> self.txt, 'Calculation completed at:  ', ctime()
        print >> self.txt
        print >> self.txt, \
              '------------------------------------------------------'

        return E_q


    def E_q(self,
            q,
            index=None,
            direction=0,
            integrated=True):

        if abs(np.dot(q, q))**0.5 < 1.e-5:
            q = [0.,0.,0.]
            q[direction] = 1.e-5
            optical_limit = True
        else:
            optical_limit = False

        dummy = DF(calc=self.calc,
                   eta=0.0,
                   w=self.w * 1j,
                   q=q,
                   ecut=self.ecut,
                   G_plus_q=True,
                   optical_limit=optical_limit,
                   cell=self.cell,
                   hilbert_trans=False)
        dummy.txt = devnull
        dummy.initialize(simple_version=True)
        npw = dummy.npw
        del dummy

        if self.nbands is None:
            nbands = npw
        else:
            nbands = self.nbands

        if self.txt is sys.stdout:
            txt = 'response.txt'
        else:
            txt='response_'+self.txt.name
        df = DF(calc=self.calc,
                xc=None,
                nbands=nbands,
                eta=0.0,
                q=q,
                txt=txt,
                vcut=self.vcut,
                w=self.w * 1j,
                ecut=self.ecut,
                smooth_cut=self.smooth_cut,
                G_plus_q=True,
                cell=self.cell,
                kcommsize=self.kcommsize,
                comm=self.dfcomm,
                cuda=self.cuda,
                cu=self.cu,
                nmultix=self.nmultix,
                sync=self.sync,
                optical_limit=optical_limit,
                hilbert_trans=False)
        
        if index is None:
            print >> self.txt, 'Calculating KS response function at:'
        else:
            print >> self.txt, '#', index, \
                  '- Calculating KS response function at:'
        if optical_limit:
            print >> self.txt, 'q = [0 0 0] -', 'Polarization: ', direction
        else:
            print >> self.txt, 'q = [%1.6f %1.6f %1.6f] -' \
                  % (q[0],q[1],q[2]), '%s planewaves' % npw

        df.initialize()
        Nw_local = df.Nw_local
        local_E_q_we = np.zeros((Nw_local, self.necut), dtype=complex)
        E_q_we = np.empty((len(self.w), self.necut), complex)

        mband_e = np.zeros(self.necut, int)
        for iecut, ecut in enumerate(self.ecutlist_e):
            if ecut < max(self.ecutlist_e):
                npw, Gvec_Gc, Gindex_G = set_Gvectors(df.acell_cv, df.bcell_cv,
                                                  df.nG, np.ones(3)*ecut/Hartree, q=df.q_c)
                mband_e[iecut] = npw
                for ipw in range(npw):
                    Gindex_G[ipw] = np.where(np.abs(df.Gvec_Gc - Gvec_Gc[ipw]).sum(1) < 1e-10)[0]
                    assert np.abs(df.Gvec_Gc[Gindex_G[ipw]] - Gvec_Gc[ipw]).sum() < 1e-10

            if iecut == 0:
                mstart = 0
            else:
                mstart = mband_e[iecut-1]
            if ecut == max(self.ecutlist_e):
                mend = df.nbands
            else:
                mend = mband_e[iecut]
            if self.cuda:
                overwritechi0 = True
            else:
                overwritechi0 = False
            e_wGG = df.get_dielectric_matrix(xc='RPA',overwritechi0=overwritechi0,
                                             initialized=True, mstart=mstart, mend=mend)
            for i in range(Nw_local):
                if ecut == max(self.ecutlist_e):
                    local_E_q_we[i, iecut] = (np.log(np.linalg.det(e_wGG[i]))
                              + len(e_wGG[0]) - np.trace(e_wGG[i]))
                else:
                    e2_GG = np.zeros((npw,npw), complex)
                    for ipw in range(npw):
                        e2_GG[ipw,:] = e_wGG[i][Gindex_G[ipw],:][Gindex_G]
                    local_E_q_we[i, iecut] = (np.log(np.linalg.det(e2_GG))
                              + len(e2_GG) - np.trace(e2_GG))

        df.wcomm.all_gather(local_E_q_we, E_q_we)

        if self.cuda:
            for iw in range(df.Nw_local):
                _gpaw.cuFree(self.cu.chi0_w[iw])

        del df, e_wGG

        E_q_e = np.zeros(self.necut)
        for iecut, ecut in enumerate(self.ecutlist_e):
            E_q_w = E_q_we[:,iecut].copy()
            if self.gauss_legendre is not None:
                E_q = np.sum(E_q_w * self.gauss_weights * self.transform) \
                      / (4*np.pi)
            else:   
                dws = self.w[1:] - self.w[:-1]
                E_q = np.dot((E_q_w[:-1] + E_q_w[1:])/2., dws) / (2.*np.pi)
    
                if self.extrapolate:
                    '''Fit tail to: Eq(w) = A**2/((w-B)**2 + C)**2'''
                    e1 = abs(E_q_w[-1])**0.5
                    e2 = abs(E_q_w[-2])**0.5
                    e3 = abs(E_q_w[-3])**0.5
                    w1 = self.w[-1]
                    w2 = self.w[-2]
                    w3 = self.w[-3]
                    B = (((e3*w3**2-e1*w1**2)/(e1-e3) -
                          (e2*w2**2-e1*w1**2)/(e1-e2))
                         / ((2*w3*e3-2*w1*e1)/(e1-e3) -
                            (2*w2*e2-2*w1*e1)/(e1-e2)))
                    C = ((w2-B)**2*e2 - (w1-B)**2*e1)/(e1-e2)
                    A = e1*((w1-B)**2+C)
                    if C > 0:
                        E_q -= A**2*(np.pi/(4*C**1.5)
                                     - (w1-B)/((w1-B)**2+C)/(2*C)
                                     - np.arctan((w1-B)/C**0.5)/(2*C**1.5)) \
                                     / (2*np.pi)
                    else:
                        E_q += A**2*((w1-B)/((w1-B)**2+C)/(2*C)
                                     +np.log((w1-B-abs(C)**0.5)/(w1-B+abs(C)**0.5))
                                     / (4*C*abs(C)**0.5)) / (2*np.pi)
    
            print >> self.txt, 'ecut %s : E_c(q) = %s eV' %(ecut, E_q.real)
            E_q_e[iecut] = E_q.real
        print >> self.txt

        if integrated:
            return E_q_e
        else:
            return E_q_we.real               

    def initialize_calculation(self, w, ecut, smooth_cut,
                               nbands, kcommsize, extrapolate,
                               gauss_legendre, frequency_cut, frequency_scale):
        if w is not None:
            assert (gauss_legendre is None and
                    frequency_cut is None and
                    frequency_scale is None)
        else:
            if gauss_legendre is None:
                gauss_legendre = 16
            self.gauss_points, self.gauss_weights = p_roots(gauss_legendre)
            if frequency_scale is None:
                frequency_scale = 2.0
            if frequency_cut is None:
                frequency_cut = 800.
            ys = 0.5 - 0.5 * self.gauss_points
            ys = ys[::-1]
            w = (-np.log(1-ys))**frequency_scale
            w *= frequency_cut/w[-1]
            alpha = (-np.log(1-ys[-1]))**frequency_scale/frequency_cut
            transform = (-np.log(1-ys))**(frequency_scale-1) \
                        / (1-ys)*frequency_scale/alpha
            self.transform = transform
            
        dummy = DF(calc=self.calc,
                   eta=0.0,
                   w=w * 1j,
                   q=[0.,0.,0.0001],
                   ecut=ecut,
                   optical_limit=True,
                   cell=self.cell,
                   hilbert_trans=False,
                   kcommsize=kcommsize)
        dummy.txt = devnull
        dummy.spin = 0
        dummy.initialize(simple_version=True)

        self.npw = dummy.npw
        self.ecut = ecut
        self.smooth_cut = smooth_cut
        self.w = w
        self.gauss_legendre = gauss_legendre
        self.frequency_cut = frequency_cut
        self.frequency_scale = frequency_scale
        self.extrapolate = extrapolate
        self.kcommsize = kcommsize
        self.nbands = nbands

        if self.cuda:
            self.cu = BaseCuda()
            print >> self.txt, 'Init Cuda! '
            print >> self.txt, '  Sync is  ', self.sync
            print >> self.txt, '  Nmultix :', self.nmultix
        else:
            self.cu = None

        print >> self.txt
        print >> self.txt, 'Planewave cutoff              : %s eV' % ecut
        if self.smooth_cut is not None:
            print >> self.txt, 'Smooth cutoff from            : %s x cutoff' \
                  % self.smooth_cut
        print >> self.txt, 'Number of Planewaves at Gamma : %s' % self.npw
        if self.nbands is None:
            print >> self.txt, 'Response function bands       :'\
                  + ' Equal to number of Planewaves'
        else:
            print >> self.txt, 'Response function bands       : %s' \
                  % self.nbands
        print >> self.txt, 'Frequencies'
        if self.gauss_legendre is not None:
            print >> self.txt, '    Gauss-Legendre integration '\
                  + 'with %s frequency points' % len(self.w)
            print >> self.txt, '    Frequency cutoff is '\
                  + '%s eV and scale (B) is %s' % (self.w[-1],
                                                  self.frequency_scale)
        else:
            print >> self.txt, '    %s specified frequency points' \
                  % len(self.w)
            print >> self.txt, '    Frequency cutoff is %s eV' \
                  % self.w[-1]
            if extrapolate:
                print >> self.txt, '    Squared Lorentzian extrapolation ' \
                      + 'to frequencies at infinity'
        print >> self.txt
        print >> self.txt, 'Parallelization scheme'
        print >> self.txt, '     Total CPUs        : %d' % dummy.comm.size
        if dummy.nkpt == 1:
            print >> self.txt, '     Band parsize      : %d' % dummy.kcomm.size
        else:
            print >> self.txt, '     Kpoint parsize    : %d' % dummy.kcomm.size
        print >> self.txt, '     Frequency parsize : %d' % dummy.wScomm.size
        print >> self.txt, 'Memory usage estimate'
        print >> self.txt, '     chi0_wGG(Q)       : %f M / cpu' \
              % (dummy.Nw_local * self.npw**2 * 16. / 1024**2)
        print >> self.txt
        del dummy

    def print_initialization(self):
        
        print >> self.txt, \
              '------------------------------------------------------'
        print >> self.txt, 'Non-self-consistent RPA correlation energy'
        print >> self.txt, \
              '------------------------------------------------------'
        print >> self.txt, 'Started at:  ', ctime()
        print >> self.txt
#        print >> self.txt, 'Atoms                          :   %s' \
#              % self.atoms.get_chemical_formula(mode="hill")
        print >> self.txt, 'Ground state XC functional     :   %s' \
              % self.calc.hamiltonian.xc.name
        print >> self.txt, 'Valence electrons              :   %s' \
              % self.setups.nvalence
        print >> self.txt, 'Number of Bands                :   %s' \
              % self.calc.wfs.bd.nbands
        print >> self.txt, 'Number of Converged Bands      :   %s' \
              % self.calc.input_parameters['convergence']['bands']
        print >> self.txt, 'Number of Spins                :   %s' \
              % self.nspins
        print >> self.txt, 'Number of k-points             :   %s' \
              % len(self.calc.wfs.bzk_kc)
        print >> self.txt, 'Number of q-points             :   %s' \
              % len(self.bz_q_points)
        print >> self.txt, 'Number of Irreducible k-points :   %s' \
              % len(self.calc.wfs.ibzk_kc)
        if self.qsym:
            print >> self.txt, 'Number of Irreducible q-points :   %s' \
                  % len(self.ibz_q_points)
        else:
            print >> self.txt, 'No reduction of q-points' 
        print >> self.txt
        for q, weight in zip(self.ibz_q_points, self.q_weights):
            print >> self.txt, 'q: [%1.4f %1.4f %1.4f] - weight: %1.3f' \
                  % (q[0],q[1],q[2], weight)
        print >> self.txt
        print >> self.txt, \
              '------------------------------------------------------'
        print >> self.txt, \
              '------------------------------------------------------'
        print >> self.txt

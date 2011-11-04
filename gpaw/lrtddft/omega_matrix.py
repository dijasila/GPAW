import sys
from math import sqrt
import numpy as np
import gpaw.mpi as mpi
MASTER = mpi.MASTER

from ase.parallel import paropen
from ase.units import Hartree

from gpaw import debug
import gpaw.mpi as mpi
from gpaw.poisson import PoissonSolver
from gpaw.output import initialize_text_stream
from gpaw.lrtddft.excitation import Excitation,ExcitationList
from gpaw.lrtddft.kssingle import KSSingles
from gpaw.transformers import Transformer
from gpaw.utilities import pack, pack2, packed_index
from gpaw.utilities.lapack import diagonalize
from gpaw.kohnsham_layouts import LrTDDFTLayouts
from gpaw.utilities.timing import Timer
from gpaw.xc import XC
from gpaw.parameters import InputParameters


import time

"""This module defines a Omega Matrix class."""

class OmegaMatrix:
    """
    Omega matrix in Casidas linear response formalism

    Parameters
      - calculator: the calculator object the ground state calculation
      - kss: the Kohn-Sham singles object
      - xc: the exchange correlation approx. to use
      - derivativeLevel: which level i of d^i Exc/dn^i to use
      - numscale: numeric epsilon for derivativeLevel=0,1
      - filehandle: the oject can be read from a filehandle
      - txt: output stream or file name
      - finegrid: level of fine grid to use. 0: nothing, 1 for poisson only,
        2 everything on the fine grid
      - comm: MPI communicator object for parallelization over eh-pairs.
    """
    def __init__(self,
                 calculator=None,
                 kss=None,
                 xc=None,
                 derivativeLevel=None,
                 numscale=0.001,
                 filehandle=None,
                 hdf5 = False,
                 txt=None,
                 finegrid=2,
                 eh_comm=None,
                 ):
        
        if not txt and calculator:
            txt = calculator.txt
        self.txt, firsttime = initialize_text_stream(txt, mpi.rank)

        if eh_comm == None:
            eh_comm = mpi.serial_comm

        self.eh_comm = eh_comm

        if filehandle is not None:
            self.kss = kss
            if hdf5:
                self.read_hdf5(fh=filehandle)
            else:
                self.read_gz(fh=filehandle)
            return None

        self.fullkss = kss
        self.finegrid = finegrid

        if calculator is None:
            return

        self.paw = calculator
        wfs = self.paw.wfs
        
        # handle different grid possibilities
        self.restrict = None
        self.poisson = PoissonSolver(nn=self.paw.hamiltonian.poisson.nn)
        if finegrid:
            self.poisson.set_grid_descriptor(self.paw.density.finegd)
            self.poisson.initialize()
            
            self.gd = self.paw.density.finegd
            if finegrid == 1:
                self.gd = wfs.gd
        else:
            self.poisson.set_grid_descriptor(wfs.gd)
            self.poisson.initialize()
            self.gd = wfs.gd
        self.restrict = Transformer(self.paw.density.finegd, wfs.gd,
                                    self.paw.input_parameters.stencils[0]
                                    ).apply

        if xc == 'RPA': 
            xc = None # enable RPA as keyword
        if xc is not None:
            self.xc = XC(xc)
            self.xc.initialize(self.paw.density, self.paw.hamiltonian,
                               wfs, self.paw.occupations)

            # check derivativeLevel
            if derivativeLevel is None:
                derivativeLevel= \
                    self.xc.get_functional().get_max_derivative_level()
            self.derivativeLevel=derivativeLevel
            # change the setup xc functional if needed
            # the ground state calculation may have used another xc
            if kss.npspins > kss.nvspins:
                spin_increased = True
            else:
                spin_increased = False
        else:
            self.xc = None

        self.numscale = numscale
    
        self.singletsinglet = False
        if kss.nvspins<2 and kss.npspins<2:
             # this will be a singlet to singlet calculation only
             self.singletsinglet=True

        nkss = len(kss)
        self.ij = range(self.eh_comm.rank, nkss, self.eh_comm.size)
        self.nij = len(self.ij)
        self.nkq = len(kss)
        self.Om = np.zeros((self.nij,self.nkq))
        # self.get_full()
        self.calculate()

    def calculate(self):
        """Calculate the full Omega matrix.
           With parallelization over eh-pairs, the full matrix is not
           collected."""

        self.paw.timer.start('Omega RPA')
        self.get_rpa()
        self.paw.timer.stop()

        if self.xc is not None:
            self.paw.timer.start('Omega XC')
            self.get_xc()
            self.paw.timer.stop()

    def get_full(self):
        """Collect the full Omega matrix within the eh-communicator."""

        self.paw.timer.start('Collect Omega')
        if self.eh_comm.size == 1:
            self.full = self.Om
        else:
            if self.eh_comm.rank == 0:
                self.full = np.zeros((self.nkq,self.nkq))
                self.full[0::self.eh_comm.size, :] = self.Om[:]
                for eh_rank in range(1, self.eh_comm.size):
                    ij = range(rank, self.nkq, self.eh_comm.size)
                    Om = np.zeros(len(nij), self.nkq)
                    self.eh_comm.receive(Om, rank, 222 + rank)
                    self.full[ij, :] = Om[:]
            else:
                self.eh_comm.ssend(self.Om, self.eh_comm.rank, 222 + self.eh_comm.rank)
                    
                
        self.paw.timer.stop('Collect Omega')

    def get_xc(self):
        """Add xc part of the coupling matrix"""

        # shorthands
        paw = self.paw
        wfs = paw.wfs
        gd = paw.density.finegd
        comm = gd.comm
        eh_comm = self.eh_comm
        
        fg = self.finegrid is 2
        kss = self.fullkss
        # nij = len(kss)

        Om_xc = self.Om
        # initialize densities
        # nt_sg is the smooth density on the fine grid with spin index

        if kss.nvspins==2:
            # spin polarised ground state calc.
            nt_sg = paw.density.nt_sg
        else:
            # spin unpolarised ground state calc.
            if kss.npspins==2:
                # construct spin polarised densities
                nt_sg = np.array([.5*paw.density.nt_sg[0],
                                  .5*paw.density.nt_sg[0]])
            else:
                nt_sg = paw.density.nt_sg
        # check if D_sp have been changed before
        D_asp = self.paw.density.D_asp
        for a, D_sp in D_asp.items():
            if len(D_sp) != kss.npspins:
                if len(D_sp) == 1:
                    D_asp[a] = np.array([0.5 * D_sp[0], 0.5 * D_sp[0]])
                else:
                    D_asp[a] = np.array([D_sp[0] + D_sp[1]])
                
        # restrict the density if needed
        if fg:
            nt_s = nt_sg
        else:
            nt_s = self.gd.zeros(nt_sg.shape[0])
            for s in range(nt_sg.shape[0]):
                self.restrict(nt_sg[s], nt_s[s])
            gd = paw.density.gd
                
        # initialize vxc or fxc

        if self.derivativeLevel==0:
            raise NotImplementedError
            if kss.npspins==2:
                v_g=nt_sg[0].copy()
            else:
                v_g=nt_sg.copy()
        elif self.derivativeLevel==1:
            pass
        elif self.derivativeLevel==2:
            raise NotImplementedError
            if kss.npspins==2:
                fxc=d2Excdnsdnt(nt_sg[0],nt_sg[1])
            else:
                fxc=d2Excdn2(nt_sg)
        else:
            raise ValueError('derivativeLevel can only be 0,1,2')

##        self.paw.my_nuclei = []

        ns=self.numscale
        xc=self.xc
        print >> self.txt, 'XC',self.nkq,'transitions'
        for my_ij, ij in enumerate(self.ij):
            print >> self.txt,'XC kss['+'%d'%ij+']' 

            timer = Timer()
            timer.start('init')
            timer2 = Timer()
                      
            if self.derivativeLevel == 1:
                # vxc is available
                # We use the numerical two point formula for calculating
                # the integral over fxc*n_ij. The results are
                # vvt_s        smooth integral
                # nucleus.I_sp atom based correction matrices (pack2)
                #              stored on each nucleus
                timer2.start('init v grids')
                vp_s=np.zeros(nt_s.shape,nt_s.dtype.char)
                vm_s=np.zeros(nt_s.shape,nt_s.dtype.char)
                if kss.npspins == 2: # spin polarised
                    nv_s = nt_s.copy()
                    nv_s[kss[ij].pspin] += ns * kss[ij].get(fg)
                    xc.calculate(gd, nv_s, vp_s)
                    nv_s = nt_s.copy()
                    nv_s[kss[ij].pspin] -= ns * kss[ij].get(fg)
                    xc.calculate(gd, nv_s, vm_s)
                else: # spin unpolarised
                    nv = nt_s + ns * kss[ij].get(fg)
                    xc.calculate(gd, nv, vp_s)
                    nv = nt_s - ns * kss[ij].get(fg)
                    xc.calculate(gd, nv, vm_s)
                vvt_s = (0.5 / ns) * (vp_s - vm_s)
                timer2.stop()

                # initialize the correction matrices
                timer2.start('init v corrections')
                I_asp = {}
                for a, P_ni in wfs.kpt_u[kss[ij].spin].P_ani.items():
                    # create the modified density matrix
                    Pi_i = P_ni[kss[ij].i]
                    Pj_i = P_ni[kss[ij].j]
                    P_ii = np.outer(Pi_i, Pj_i)
                    # we need the symmetric form, hence we can pack
                    P_p = pack(P_ii)
                    D_sp = self.paw.density.D_asp[a].copy()
                    D_sp[kss[ij].pspin] -= ns * P_p
                    setup = wfs.setups[a]
                    I_sp = np.zeros_like(D_sp)
                    self.xc.calculate_paw_correction(setup, D_sp, I_sp)
                    I_sp *= -1.0
                    D_sp = self.paw.density.D_asp[a].copy()
                    D_sp[kss[ij].pspin] += ns * P_p
                    self.xc.calculate_paw_correction(setup, D_sp, I_sp)
                    I_sp /= 2.0 * ns
                    I_asp[a] = I_sp
                timer2.stop()
                    
            timer.stop()
            t0 = timer.get_time('init')
            timer.start(ij)
            
            for kq in range(ij, self.nkq):
                weight = self.weight_Kijkq(ij, kq)
                
                if self.derivativeLevel == 0:
                    # only Exc is available
                    
                    if kss.npspins==2: # spin polarised
                        nv_g = nt_sg.copy()
                        nv_g[kss[ij].pspin] += kss[ij].get(fg)
                        nv_g[kss[kq].pspin] += kss[kq].get(fg)
                        Excpp = xc.get_energy_and_potential(
                            nv_g[0], v_g, nv_g[1], v_g)
                        nv_g = nt_sg.copy()
                        nv_g[kss[ij].pspin] += kss[ij].get(fg)
                        nv_g[kss[kq].pspin] -= kss[kq].get(fg)
                        Excpm = xc.get_energy_and_potential(\
                                            nv_g[0],v_g,nv_g[1],v_g)
                        nv_g = nt_sg.copy()
                        nv_g[kss[ij].pspin] -=\
                                        kss[ij].get(fg)
                        nv_g[kss[kq].pspin] +=\
                                        kss[kq].get(fg)
                        Excmp = xc.get_energy_and_potential(\
                                            nv_g[0],v_g,nv_g[1],v_g)
                        nv_g = nt_sg.copy()
                        nv_g[kss[ij].pspin] -= \
                                        kss[ij].get(fg)
                        nv_g[kss[kq].pspin] -=\
                                        kss[kq].get(fg)
                        Excpp = xc.get_energy_and_potential(\
                                            nv_g[0],v_g,nv_g[1],v_g)
                    else: # spin unpolarised
                        nv_g=nt_sg + ns*kss[ij].get(fg)\
                              + ns*kss[kq].get(fg)
                        Excpp = xc.get_energy_and_potential(nv_g,v_g)
                        nv_g=nt_sg + ns*kss[ij].get(fg)\
                              - ns*kss[kq].get(fg)
                        Excpm = xc.get_energy_and_potential(nv_g,v_g)
                        nv_g=nt_sg - ns*kss[ij].get(fg)\
                              + ns*kss[kq].get(fg)
                        Excmp = xc.get_energy_and_potential(nv_g,v_g)
                        nv_g=nt_sg - ns*kss[ij].get(fg)\
                              - ns*kss[kq].get(fg)
                        Excmm = xc.get_energy_and_potential(nv_g,v_g)

                    Om_xc[my_ij,kq] += weight *\
                                0.25*(Excpp-Excmp-Excpm+Excmm)/(ns*ns)
                              
                elif self.derivativeLevel == 1:
                    # vxc is available

                    timer2.start('integrate')
                    Om_xc[my_ij,kq] += weight*\
                                 self.gd.integrate(kss[kq].get(fg)*
                                                   vvt_s[kss[kq].pspin])
                    timer2.stop()

                    timer2.start('integrate corrections')
                    Exc = 0.
                    for a, P_ni in wfs.kpt_u[kss[kq].spin].P_ani.items():
                        # create the modified density matrix
                        Pk_i = P_ni[kss[kq].i]
                        Pq_i = P_ni[kss[kq].j]
                        P_ii = np.outer(Pk_i, Pq_i)
                        # we need the symmetric form, hence we can pack
                        # use pack as I_sp used pack2
                        P_p = pack(P_ii)
                        Exc += np.dot(I_asp[a][kss[kq].pspin], P_p)
                    Om_xc[my_ij, kq] += weight * self.gd.comm.sum(Exc)
                    timer2.stop()

                elif self.derivativeLevel == 2:
                    # fxc is available
                    if kss.npspins==2: # spin polarised
                        Om_xc[my_ij,kq] += weight *\
                            gd.integrate(kss[ij].get(fg)*
                                         kss[kq].get(fg)*
                                         fxc[kss[ij].pspin,kss[kq].pspin])
                    else: # spin unpolarised
                        Om_xc[my_ij,kq] += weight *\
                            gd.integrate(kss[ij].get(fg)*
                                         kss[kq].get(fg)*
                                         fxc)
                # XXX Is this needed?
                # if ij != kq:
                #    Om_xc[kq,ij] = Om_xc[ij,kq]
                
            timer.stop()
##            timer2.write()
            nij = self.nij
            if ij < (self.nkq-1):
                t = timer.get_time(ij) # time for nij-ij calculations
                t = .5*t*(nij-my_ij)  # estimated time for n*(n+1)/2, n=nij-(ij+1)
                print >> self.txt,'XC estimated time left',\
                      self.timestring(t0*(nij-my_ij-1)+t)


    def get_rpa(self):
        """calculate RPA part of the omega matrix"""

        # shorthands
        kss=self.fullkss
        finegrid=self.finegrid
        wfs = self.paw.wfs
        eh_comm = self.eh_comm
        
        # calculate omega matrix
        # nij = len(kss)
        print >> self.txt,'RPA',self.nkq,'transitions'
        
        Om = self.Om
        
        for my_ij, ij in enumerate(self.ij):
            print >> self.txt,'RPA kss['+'%d'%ij+']=', kss[ij]

            timer = Timer()
            timer.start('init')
            timer2 = Timer()
                      
            # smooth density including compensation charges
            timer2.start('with_compensation_charges 0')
            rhot_p = kss[ij].with_compensation_charges(
                finegrid is not 0)
            timer2.stop()
            
            # integrate with 1/|r_1-r_2|
            timer2.start('poisson')
            phit_p = np.zeros(rhot_p.shape, rhot_p.dtype.char)
            self.poisson.solve(phit_p, rhot_p, charge=None)
            timer2.stop()

            timer.stop()
            t0 = timer.get_time('init')
            timer.start(ij)

            if finegrid == 1:
                rhot = kss[ij].with_compensation_charges()
                phit = self.gd.zeros()
##                print "shapes 0=",phit.shape,rhot.shape
                self.restrict(phit_p,phit)
            else:
                phit = phit_p
                rhot = rhot_p

            for kq in range(ij, self.nkq):
                if kq != ij:
                    # smooth density including compensation charges
                    timer2.start('kq with_compensation_charges')
                    rhot = kss[kq].with_compensation_charges(
                        finegrid is 2)
                    timer2.stop()

                timer2.start('integrate')
                pre = 2.*sqrt(kss[ij].get_energy()*kss[kq].get_energy()*
                                  kss[ij].get_weight()*kss[kq].get_weight())
                Om[my_ij,kq]= pre * self.gd.integrate(rhot*phit)
##                print "int=",Om[ij,kq]
                timer2.stop()

                # Add atomic corrections
                timer2.start('integrate corrections 2')
                Ia = 0.
                for a, P_ni in wfs.kpt_u[kss[ij].spin].P_ani.items():
                    Pi_i = P_ni[kss[ij].i]
                    Pj_i = P_ni[kss[ij].j]
                    Dij_ii = np.outer(Pi_i, Pj_i)
                    Dij_p = pack(Dij_ii)
                    Pkq_ni = wfs.kpt_u[kss[kq].spin].P_ani[a]
                    Pk_i = Pkq_ni[kss[kq].i]
                    Pq_i = Pkq_ni[kss[kq].j]
                    Dkq_ii = np.outer(Pk_i, Pq_i)
                    Dkq_p = pack(Dkq_ii)
                    C_pp = wfs.setups[a].M_pp
                    #   ----
                    # 2 >      P   P  C    P  P
                    #   ----    ip  jr prst ks qt
                    #   prst
                    Ia += 2.0*np.dot(Dkq_p,np.dot(C_pp,Dij_p))
                timer2.stop()
                
                Om[my_ij,kq] += pre * self.gd.comm.sum(Ia)
                    
                if ij == kq:
                    Om[my_ij,kq] += kss[ij].get_energy()**2
                # else:
                #     Om[kq,ij]=Om[ij,kq]

            timer.stop()
##            timer2.write()
            nij = self.nij
            if ij < (self.nkq-1):
                t = timer.get_time(ij) # time for nij-ij calculations
                t = .5*t*(nij-my_ij)  # estimated time for n*(n+1)/2, n=nij-(ij+1)
                print >> self.txt,'RPA estimated time left',\
                      self.timestring(t0*(nij-my_ij-1)+t)

    def singlets_triplets(self):
        """Split yourself into singlet and triplet transitions"""

        if self.eh_comm is not None:
            raise NotImplementedError('Singlet triplet splitting with eh-parallelization')

        assert(self.fullkss.npspins == 2)
        assert(self.fullkss.nvspins == 1)
        
        # strip kss from down spins
        skss = KSSingles()
        tkss = KSSingles()
        map = []
        for ij, ks in enumerate(self.fullkss):
            if ks.pspin == ks.spin:
                skss.append((ks + ks) / sqrt(2))
                tkss.append((ks - ks) / sqrt(2))
                map.append(ij)
            
        nkss = len(skss)

        # define the singlet and the triplet omega-matrixes
        sOm = OmegaMatrix(kss=skss)
        sOm.full = np.empty((nkss, nkss))
        tOm = OmegaMatrix(kss=tkss)
        tOm.full = np.empty((nkss, nkss))
        for ij in range(nkss):
            for kl in range(nkss):
                sOm.full[ij, kl] = (self.full[map[ij], map[kl]] +
                                    self.full[map[ij], nkss + map[kl]])
                tOm.full[ij, kl] = (self.full[map[ij], map[kl]] -
                                    self.full[map[ij], nkss + map[kl]])
        return sOm, tOm

    def timestring(self, t):
        ti = int(t + 0.5)
        td = ti // 86400
        st = ''
        if td > 0:
            st += '%d' % td + 'd'
            ti -= td * 86400
        th = ti // 3600
        if th > 0:
            st += '%d' % th + 'h'
            ti -= th * 3600
        tm = ti // 60
        if tm > 0:
            st += '%d' % tm + 'm'
            ti -= tm * 60
        st += '%d' % ti + 's'
        return st

    def get_map(self, istart=None, jend=None, energy_range=None):
        """Return the reduction map for the given requirements"""

        self.istart = istart
        self.jend = jend
        if istart is None and jend is None and energy_range is None:
            return None, self.fullkss

        # reduce the matrix
        print >> self.txt,'# diagonalize: %d transitions original'\
                  % len(self.fullkss)

        if energy_range is None:
            if istart is None: istart = self.kss.istart
            if self.fullkss.istart > istart:
                raise RuntimeError('istart=%d has to be >= %d' %
                                   (istart, self.kss.istart))
            if jend is None: jend = self.kss.jend
            if self.fullkss.jend < jend:
                raise RuntimeError('jend=%d has to be <= %d' %
                                   (jend, self.kss.jend))

        else:
            try:
                emin, emax = energy_range
            except:
                emax = energy_range
                emin = 0.
            emin /= Hartree
            emax /= Hartree

        map= []
        kss = KSSingles()
        for ij, k in zip(range(len(self.fullkss)), self.fullkss):
            if energy_range is None:
                if k.i >= istart and k.j <= jend:
                    kss.append(k)
                    map.append(ij)
            else:
                if k.energy >= emin and k.energy < emax:
                    kss.append(k)
                    map.append(ij)
        kss.update()
        print >> self.txt, '# diagonalize: %d transitions now' % len(kss)
            
        return map, kss

    def diagonalize(self, istart=None, jend=None, energy_range=None):
        """Evaluate Eigenvectors and Eigenvalues:"""

        eh_comm = self.eh_comm
        if eh_comm.size > 1:
            if istart is not None or jend is not None \
               or energy_range is not None:
                raise NotImplementedError('Submatrix mapping not implemented for eh-parallelization')
            par = InputParameters()  # We need ScaLAPACK parameters
            sl_omega = par.parallel['sl_diagonalize']
            if sl_omega is None:
                sl_omega = par.parallel['sl_default']
            nkq = self.Om.shape[1]
            ksl = LrTDDFTLayouts(sl_omega, nkq, eh_comm)
            self.eigenvectors = self.Om.copy()  # Should duplicate be avoided
                                                # for memory?
            self.eigenvalues = np.zeros(nkq, dtype=float)
            ksl.diagonalize(self.eigenvectors, self.eigenvalues)
            # print "Shapes", self.Om.shape, self.eigenvectors.shape
            shape = self.eigenvectors.shape
            self.eigenvectors = self.eigenvectors.reshape((shape[1], shape[0]))
        else:
            map, kss = self.get_map(istart, jend, energy_range)
            nij = len(kss)
            if map is None:
                evec = self.Om.copy()
            else:
                evec = np.zeros((nij,nij))
                for ij in range(nij):
                    for kq in range(nij):
                        evec[ij,kq] = self.Om[map[ij],map[kq]]

            self.eigenvectors = evec        
            self.eigenvalues = np.zeros((len(kss)))
            self.kss = kss
            diagonalize(self.eigenvectors, self.eigenvalues)

    def Kss(self, kss=None):
        """Set and get own Kohn-Sham singles"""
        if kss is not None:
            self.fullkss = kss
        if(hasattr(self,'fullkss')):
            return self.fullkss
        else:
            return None
 
    def read_gz(self, filename=None, fh=None):
        """Read myself from a file"""
        if fh is None:
            f = open(filename, 'r')
        else:
            f = fh

        f.readline()
        nij = int(f.readline())
        full = np.zeros((nij,nij))
        for ij in range(nij):
            l = f.readline().split()
            for kq in range(ij,nij):
                full[ij,kq] = float(l[kq-ij])
                full[kq,ij] = full[ij,kq]
        self.full = full

        if fh is None:
            f.close()

    def write(self, filename=None, fh=None):
        """Write current state to a file."""

        self.get_full()
        if mpi.rank == mpi.MASTER:
            if fh is None:
                f = open(filename, 'w')
            else:
                f = fh

            f.write('# OmegaMatrix\n')
            nij = len(self.fullkss)
            f.write('%d\n' % nij)
            for ij in range(nij):
                for kq in range(ij,nij):
                    f.write(' %g' % self.full[ij,kq])
                f.write('\n')
            
            if fh is None:
                f.close()

    def read_hdf5(self, filename=None, fh=None):
        """Read myself from a file"""
        from gpaw.io.hdf5_highlevel import File, HyperslabSelection
        if fh is None:
            f = File(filename, 'r', mpi.world.get_c_object())
        else:
            f = fh

        dset = f['OmegaMatrix']
        full_shape = dset.shape
        nkq = full_shape[0]
        indices = (slice(self.eh_comm.rank, nkq, self.eh_comm.size),
                   slice(nkq))
        selection = HyperslabSelection(indices, dset.shape)
        mshape = selection.mshape
        self.Om = np.ndarray(mshape, dset.dtype)
        dset.read(self.Om, selection, collective=True)

        if fh is None:
            f.close()

    def write_hdf5(self, filename=None, fh=None):
        """Write current state to a file."""
        from gpaw.io.hdf5_highlevel import File, HyperslabSelection
        if fh is None:
            f = File(filename, 'w', mpi.world.get_c_object())
        else:
            f = fh

        dset = f.create_dataset('OmegaMatrix', (self.nkq, self.nkq), float)
        indices = (slice(self.eh_comm.rank, self.nkq, self.eh_comm.size),
                   slice(self.nkq))
        comm = self.paw.density.finegd.comm
        if comm.rank == 0:
            selection = HyperslabSelection(indices, dset.shape)
        else:
            selection = None
        dset.write(self.Om, selection, collective=True)
        
        if fh is None:
            f.close()

    def weight_Kijkq(self, ij, kq):
        """weight for the coupling matrix terms"""
        kss = self.fullkss
        return 2.*sqrt( kss[ij].get_energy() * kss[kq].get_energy() *
                        kss[ij].get_weight() * kss[kq].get_weight()   )

    def __str__(self):
        str='<OmegaMatrix> '
        if hasattr(self,'eigenvalues'):
            str += 'dimension '+ ('%d'%len(self.eigenvalues))
            str += "\neigenvalues: "
            for ev in self.eigenvalues:
                str += ' ' + ('%f'%(sqrt(ev) * Hartree))
        return str
            


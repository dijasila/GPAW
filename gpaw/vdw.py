import os
from math import pi, sqrt

import numpy as npy
from ase.units import Hartree

from gpaw.xc_functional import XCFunctional
#these are used for calculating the gradient
from gpaw.grid_descriptor import GridDescriptor
from gpaw.domain import Domain
from gpaw.xc_functional import XC3DGrid
from gpaw.transformers import Transformer
from gpaw.utilities import  check_unit_cell
import gpaw.mpi as mpi
import _gpaw


class VanDerWaals:
    def __init__(self, n_g, gd, xcname='revPBE', ncoarsen=0):
        """Van der Waals object.

        Bla-bla-bla, ref to paper, ...

        ... the density must be given with the shape
        spin,gridpoint_x,gridpoint_y,gridpoint_z, This class only
        works for non spin polarized calculations. In case of
        spinpolarized calcultions, one should ad the spin up and spin
        down densities and use that as input.

        Parameters
        ----------
        n_g : 3D ndarray of floats
            Electron density.
        gd : GridDescriptor
            Grid-descriptor object.
        xcname : string
            Name of the XC functional used.  Default value: revPBE.
        """
        
        self.gd = gd

        v_g = gd.empty()

        # GGA exchange and correlation:
        xc = XC3DGrid(xcname, gd)
        self.GGA_xc_energy = xc.get_energy_and_potential(n_g, v_g)
        self.a2_g = xc.a2_g

        # LDA correlation energy:
        c = XC3DGrid('None-C_PW', gd)
        self.LDA_c_energy = c.get_energy_and_potential(n_g, v_g)
        
        # GGA exchange energy:
        xname = {'revPBE': 'X_PBE_R-None','RPBE': 'X_RPBE-None'}[xcname]
        x = XC3DGrid(xname, gd)
        self.GGA_x_energy = x.get_energy_and_potential(n_g, v_g)

        ncut = 1.0e-7
        n_g.clip(ncut, npy.inf, out=n_g)

        kF_g = (3.0 * pi**2 * n_g)**(1.0 / 3.0)
        self.n_g = n_g
        self.q0_g = self.get_q0(kF_g)

        for n in range(ncoarsen):
            coarsegd = self.gd.coarsen()
            t = Transformer(self.gd, coarsegd)
            self.n_g, n_g = coarsegd.empty(), self.n_g
            self.q0_g, q0_g = coarsegd.empty(), self.q0_g
            t.apply(n_g, self.n_g)
            t.apply(q0_g, self.q0_g)
            self.gd = coarsegd
            
        self.phi_jk = self.get_phitab_from_1darrays()

    def get_prl_plot(self):
        ndelta, nD = self.phi_jk.shape
        nD8 = int(8.0 / self.deltaD)
        D = npy.linspace(0, nD8 * self.deltaD, nD8, endpoint=False)
        import pylab as p
        for delta, c in [(0, 'r'), (0.5, 'g'), (0.9, 'b')]:
            jdelta = int(delta / self.deltadelta + 0.5)
            p.plot(D, self.phimat[jdelta, :nD8], c + '-',
                   label=r'$\delta=%.1f$' % delta)
        p.plot(D, 0, 'k-')
        p.legend(loc='best')
        p.xlabel('D')
        p.ylabel(r'$4\pi D^2 \phi(\rm{Hartree})$')
        p.show()
        
    def get_energy(self, ncut=0.0005):
        #introduces periodic boundary conditions using
        #the minimum image convention

        gd = self.gd
        n_g = gd.collect(self.n_g)
        q0_g = gd.collect(self.q0_g)
        if mpi.rank != 0:
            n_g = gd.empty(global_array=True)
            q0_g = gd.empty(global_array=True)
        mpi.world.broadcast(n_g, 0)
        mpi.world.broadcast(q0_g, 0)

        n_c = self.n_g.shape
        R_gc = npy.empty(n_c + (3,))
        R_gc[..., 0] = (npy.arange(0, n_c[0]) * gd.h_c[0]).reshape((-1, 1, 1))
        R_gc[..., 1] = (npy.arange(0, n_c[1]) * gd.h_c[1]).reshape((-1, 1))
        R_gc[..., 2] = npy.arange(0, n_c[2]) * gd.h_c[2]

        mask_g = (self.n_g.ravel() > ncut)
        R_ic = R_gc.reshape((-1, 3)).compress(mask_g, axis=0)
        n_i = self.n_g.ravel().compress(mask_g)
        q0_i = self.q0_g.ravel().compress(mask_g)

        # Number of grid points:
        ni = len(n_i)

        # Number of pairs per processor:
        np = ni * (ni - 1) // 2 // mpi.size
        
        iA = 0
        for r in range(mpi.size):
            iB = iA + int(0.5 - iA + sqrt((iA - 0.5)**2 + 2 * np))
            if r == mpi.rank:
                break
            iA = iB

        assert iA <= iB
        
        if mpi.rank == mpi.size - 1:
            iB = ni
            
        E_cl = _gpaw.vdw(n_i, q0_i, R_ic, gd.domain.cell_c, gd.domain.pbc_c,
                         self.phi_jk, self.deltaD, self.deltadelta,
                         iA, iB)
        E_cl = mpi.world.sum(E_cl * gd.h_c.prod()**2)
        E_nl_c = (-self.GGA_xc_energy + E_cl + self.LDA_c_energy +
                  self.GGA_x_energy)
        return E_nl_c * Hartree
    

    def get_e_xc_LDA(self):
        e_xc_LDA=self.get_e_xc_LDA_c()+self.get_e_x_LDA()
        return e_xc_LDA
    
    def get_e_xc_LDA_c(self):
        #this is for one spin only
        #PW91 LDA correlation
        c=1.7099210
        n_up=self.n_g/2.0
        n_down=self.n_g/2.0
        #nt=abs(n_up+n_down)
        n=self.n_g
        #npy.choose(npy.less_equal(nt,0.00001),(nt,0.00001))
        r=(3./(4.*npy.pi*n))**(1./3.)
        zeta=(n_up-n_down)/n
        wz=((1.+zeta)**(4./3.)+(1.-zeta)**(4./3.)-2.)/(2.**(4./3.)-2.)
        res=self.e_PW92_LDA(r,0.031091,0.21370,7.5957,3.5876,1.6382,0.49294,1.)*(1.-wz*zeta**4.)
        res=res+self.e_PW92_LDA(r,0.015545,0.20548,14.1189,6.1977,3.3662,0.62517,1.)*wz*zeta**4.
        res=res-self.e_PW92_LDA(r,0.016887,0.11125,10.357,3.6231,0.88026,0.49671,1.)*wz*(1.-zeta**4.)/c
        return(res)
#function used by def eps_c_PW92_LDA (n_up,n_down):
    def e_PW92_LDA (self,r,t,u,v,w,x,y,p):
        return(-2.*t*(1.+u*r)*npy.log(1.+1./(2.*t*(v*npy.sqrt(r)+w*r+x*r**(3./2.)+y*r**(p+1.)))))
    
    def get_e_x_LDA(self):
        result = (-3./(4.*npy.pi)*(3.*npy.pi*npy.pi*self.n_g)**(1./3.))
        return result

    def get_q0(self, kF_g):
        #implementet as in PRL92(2004)246401-1
        e_xc_0 = self.get_e_xc_LDA()-self.get_e_x_LDA()*(-0.8491/9.0*self.a2_g/(2.0*kF_g*self.n_g)**2.0)
        q_0 = e_xc_0/self.get_e_x_LDA()*kF_g
        return q_0
    
    def get_phitab_from_1darrays(self, filename='phi_delta'):
        path = os.environ['VDW']
        #function that constucts phimat from files containing phi_delta(D)
        #The filename must be given as something+delta
        #
        file = open(path + '/grid.dat')
        phigrid = {}
        line = file.readline()
        while line:
            a = line.split()
            phigrid[a[0]] = float(a[1])
            line = file.readline()
        file.close()
        self.deltadelta = phigrid['deltadelta']
        self.deltaD=phigrid['deltaD']
        self.Dmax=phigrid['Dmax']
        self.deltamax=phigrid['deltamax']
        x = {}
        #filename='eta_2_phi_delta'
        faktor = 2.0*4.0*pi/pi**2.0
        for n in npy.arange(0.0,1.0,self.deltadelta):
            f = path+filename+str(n)+'.dat'
            data = self.read_array1d_from_txt_file(f)
            x[n] = npy.array(data[:])*faktor 
        #h=0.05 for D og delta 
        phimat = npy.zeros((len(x),len(x[0.0])))
        for n in range(0,phimat.shape[0]):
            for m in range(phimat.shape[1]):
                phimat[n,m] = x[n*0.05][m]
        return phimat

    def get_c6(self,n=1,ncut=0.0005):
        #Returns C6 in units of Hartree
        ncut=ncut
        h_c = self.h_c
        denstab=self.n_g
        nx, ny, nz = denstab[::n,::n,::n].shape
        print 'denstab.shape' ,denstab.shape
        N = nx * ny * nz
        print 'N' , N
        qtab_N = self.q0[::n,::n,::n].copy()
        print 'qtab_N.shape',qtab_N.shape
        qtab_N.shape = [N]

        denstab_N = denstab[::n,::n,::n].copy()
        denstab_N.shape = [N]
        print 'denstab_N.shape', denstab_N.shape
        qtab_N = npy.compress(npy.greater_equal(denstab_N,ncut),qtab_N)
        denstab_N = npy.compress(npy.greater_equal(denstab_N,ncut),denstab_N)
        #print denstab_N
        #print 'B:h_c[0]',h_c[0]
        #print denstab.shape
        print 'denstab_N.shape[0]', denstab_N.shape[0]
        C6 = 0.0
        C=(-12.*(4.*npy.pi/9.)**3)
        for m in range(denstab_N.shape[0]):
            #print C6
            C6 = C6+npy.sum(denstab_N[m]*denstab_N[:]*C/(qtab_N[m]**2*qtab_N[:]**2*(qtab_N[m]**2+qtab_N[:]**2)))
        #print 'C:h_c[0]',h_c[:], 'n=', n
        #print 'udenfor loop C6=',C6
        #print 'norm', n**6*h_c[0]**2.0*h_c[1]**2.0*h_c[2]**2.0
        C6 = -C6*n**6*h_c[0]**2.0*h_c[1]**2.0*h_c[2]**2.0
        #print denstab.shape
        Ry = 13.6058
        self.mik = qtab_N
        self.mikd = denstab_N
        return C6 ,'Ha*a0**6'

    def get_c6_coarse(self,n=1,ncut=0.0005):
        #Returns C6 in units of Hartree
        ncut=ncut
        h_c = self.h_c
        denstab =self.coarsen(self.n_g,n)
        print 'denstab.shape' ,denstab.shape
        nx, ny, nz = denstab.shape
        N = nx * ny * nz
        print 'N',N
        qtab_N = self.coarsen(self.q0.copy(),n)
        qtab_N.shape = [N]
        denstab_N = denstab.copy()
        denstab_N.shape = [N]
        print 'denstab_N.shape', denstab_N.shape
        qtab_N = npy.compress(npy.greater_equal(denstab_N,ncut),qtab_N)
        denstab_N = npy.compress(npy.greater_equal(denstab_N,ncut),denstab_N)
        C6 = 0.0
        C=(-12.*(4.*npy.pi/9.)**3)
        for m in range(denstab_N.shape[0]):
            C6 = C6+npy.sum(denstab_N[m]*denstab_N[:]*C/(qtab_N[m]**2*qtab_N[:]**2*(qtab_N[m]**2+qtab_N[:]**2)))
        C6 = -C6*n**6*h_c[0]**2.0*h_c[1]**2.0*h_c[2]**2.0
        Ry = 13.6058
        return C6 ,'Ha*a0**6'

    def read_array1d_from_txt_file(self,filename='Phi5_D0_10_1delta_0_09_01.dat'):
        file = open(filename)
        line = file.readline()
        filearray1D = []
        while line:
            filearray1D.append(float(line))
            line=file.readline()
        file.close()
        return filearray1D
        




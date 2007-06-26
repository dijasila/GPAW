from math import sqrt
import Numeric as num
import _gpaw
import gpaw.mpi as mpi
MASTER = mpi.MASTER

from ASE.Units import Convert
from gpaw import debug
from gpaw.poisson_solver import PoissonSolver
from gpaw.lrtddft.excitation import Excitation,ExcitationList
from gpaw.lrtddft.kssingle import KSSingles
from gpaw.lrtddft.omega_matrix import OmegaMatrix
from gpaw.utilities import packed_index
from gpaw.utilities.lapack import diagonalize
from gpaw.xc_functional import XC3DGrid, XCFunctional

"""This module defines a linear response TDDFT-class."""

class LrTDDFT(ExcitationList):
    """Linear Response TDDFT excitation class
    
    Input parameters:

    calculator:
      the calculator object after a ground state calculation
      
    nspins:
      number of spins considered in the calculation
      Note: Valid only for unpolarised ground state calculation

    eps:
      Minimal occupation difference for a transition (default 0.001)

    istart:
      First occupied state to consider
    jend:
      Last unoccupied state to consider
      
    xc:
      Exchange-Correlation approximation in the Kernel
    derivativeLevel:
      0: use Exc, 1: use vxc, 2: use fxc  if available

    filename:
      read from a file
    """
    def __init__(self,
                 calculator=None,
                 nspins=None,
                 eps=0.001,
                 istart=0,
                 jend=None,
                 xc=None,
                 derivativeLevel=None,
                 numscale=0.001,
                 filename=None):

        if filename is None:

            ExcitationList.__init__(self,calculator)

            self.calculator=None
            self.nspins=None
            self.eps=None
            self.istart=None
            self.jend=None
            self.xc=None
            self.derivativeLevel=None
            self.numscale=numscale
            self.update(calculator,nspins,eps,istart,jend,
                        xc,derivativeLevel,numscale)

        else:
            self.read(filename)

    def analyse(self,what=None,min=.1):
        if what is None:
            what = range(len(self))
        elif type(what)==type(1):
            what=[what]
            
        for i in what:
            print self[i].analyse(min=min)
            
    def update(self,
               calculator=None,
               nspins=None,
               eps=0.001,
               istart=0,
               jend=None,
               xc=None,
               derivativeLevel=None,
               numscale=0.001):

        changed=False
        if self.calculator!=calculator or \
           self.nspins != nspins or \
           self.eps != eps or \
           self.istart != istart or \
           self.jend != jend :
            changed=True

        if not changed: return

        self.calculator = calculator
        self.out = calculator.out
        self.nspins = nspins
        self.eps = eps
        self.istart = istart
        self.jend = jend
        self.xc = xc
        self.derivativeLevel=derivativeLevel
        self.numscale=numscale
        self.kss = KSSingles(calculator=calculator,
                             nspins=nspins,
                             eps=eps,
                             istart=istart,
                             jend=jend)
        self.Om = OmegaMatrix(self.calculator,self.kss,
                              self.xc,self.derivativeLevel,self.numscale)
##        self.diagonalize()

    def diagonalize(self, istart=None, jend=None):
        self.istart = istart
        self.jend = jend
        self.Om.diagonalize(istart,jend)
        
        # remove old stuff
        while len(self): self.pop()

        for j in range(len(self.Om.kss)):
            self.append(LrTDDFTExcitation(self.Om,j))

    def get_Om(self):
        return self.Om

    def Read(self, filename=None, fh=None):
        return self.read(filename,fh)
        
    def read(self, filename=None, fh=None):
        """Read myself from a file"""
        if mpi.rank == mpi.MASTER:
            self.Ha = Convert(1., 'Hartree', 'eV')
            
            if fh is None:
                if filename.endswith('.gz'):
                    import gzip
                    f = gzip.open(filename)
                else:
                    f = open(filename, 'r')
            else:
                f = fh

            f.readline()
            self.xc = f.readline().replace('\n','')
            self.eps = float(f.readline())
            self.kss = KSSingles(filehandle=f)
            self.Om = OmegaMatrix(kss=self.kss,filehandle=f)
            self.Om.Kss(self.kss)

            # check if already diagonalized
            p = f.tell()
            s = f.readline()
            if s != '# Eigenvalues\n':
                # go back to previous position
                f.seek(p)
            else:
                # load the eigenvalues
                n = int(f.readline().split()[0])
                for i in range(n):
                    l = f.readline().split()
                    E = float(l[0])
                    me = [float(l[1]),float(l[2]),float(l[3])]
                    print E,me
                    self.append(LrTDDFTExcitation(e=E,m=me))
                    
                # load the eigenvectors
                pass

            if fh is None:
                f.close()

    def SPA(self):
        """Return the excitation list according to the
        single pole approximation. See e.g.:
        Grabo et al, Theochem 501 (2000) 353-367
        """
        spa = self.kss
        for i in range(len(spa)):
            E = sqrt(self.Om.full[i][i])
            print "<SPA> E was",spa[i].GetEnergy()*27.211," and is ",E*27.211
            spa[i].SetEnergy(sqrt(self.Om.full[i][i]))
        return spa

    def __str__(self):
        string = ExcitationList.__str__(self)
        string += '# derived from:\n'
        string += self.kss.__str__()
        return string

    def Write(self, filename=None, fh=None):
        return self.write(filename,fh)
    
    def write(self, filename=None, fh=None):
        """Write current state to a file.

        'filename' is the filename. If the filename ends in .gz,
        the file is automatically saved in compressed gzip format.

        'fh' is a filehandle. This can be used to write into already
        opened files. 
        """
        if mpi.rank == mpi.MASTER:
            if fh is None:
                if filename.endswith('.gz'):
                    import gzip
                    f = gzip.open(filename,'wb')
                else:
                    f = open(filename, 'w')
            else:
                f = fh

            f.write('# LrTDDFT\n')
            xc = self.xc
            if xc is None: xc = 'RPA'
            f.write(xc+'\n')
            f.write('%g' % self.eps + '\n')
            self.kss.write(fh=f)
            self.Om.write(fh=f)

            if len(self):
                f.write('# Eigenvalues\n')
                istart = self.istart
                if istart is None: istart = self.kss.istart
                jend = self.jend
                if jend is None: jend = self.kss.jend
                f.write('%d %d %d'%(len(self),istart,jend)+'\n')
                for ex in self:
                    print 'ex=',ex
                    f.write(ex.outstring())
                f.write('# Eigenvectors\n')
                for ex in self:
                    for w in ex.f:
                        f.write('%g '%w)
                    f.write('\n')

            if fh is None:
                f.close()

def d2Excdnsdnt(dup,ddn):
    """Second derivative of Exc polarised"""
    res=[[0, 0], [0, 0]]
    for ispin in range(2):
        for jspin in range(2):
            res[ispin][jspin]=num.zeros(dup.shape,num.Float)
            _gpaw.d2Excdnsdnt(dup, ddn, ispin, jspin, res[ispin][jspin])
    return res

def d2Excdn2(den):
    """Second derivative of Exc unpolarised"""
    res=num.zeros(den.shape,num.Float)
    _gpaw.d2Excdn2(den, res)
    return res

class LrTDDFTExcitation(Excitation):
    def __init__(self,Om=None,i=None,
                 e=None,m=None):
        # define from the diagonalized Omega matrix
        if Om is not None:
            if i is None:
                raise RuntimeError
        
            self.energy=sqrt(Om.eigenvalues[i])
            self.f = Om.eigenvectors[i]
            self.kss = Om.kss
            self.me = 0.
            for f,k in zip(self.f,self.kss):
                self.me += f * k.me

            return

        # define from energy and matrix element
        if e is not None:
            if m is None:
                raise RuntimeError
            self.energy = e
            self.me = m
            return

        raise RuntimeError

    def outstring(self):
        str = '%g ' % self.energy
        str += '  '
        for m in self.me:
            str += ' %g' % m
        str += '\n'
        return str
        
    def __str__(self):
        m2 = num.sum(self.me*self.me)
        m = sqrt(m2)
        if m>0: me = self.me/m
        else:   me = self.me
        str = "<LrTDDFTExcitation> om=%g[eV] |me|=%g (%.2f,%.2f,%.2f)" % \
              (self.energy*27.211,m,me[0],me[1],me[2])
        return str

    def analyse(self,min=.1):
        """Return an analysis string of the excitation"""
        s='E=%.3f'%(self.energy*27.211)+' eV, f=%.3g'\
           %(self.GetOscillatorStrength()[0])+'\n'

        def sqr(x): return x*x
        spin = ['u','d'] 
        min2 = sqr(min)
        rest = num.sum(self.f**2)
        for f,k in zip(self.f,self.kss):
            f2 = sqr(f)
            if f2>min2:
                s += '  %d->%d ' % (k.i,k.j) + spin[k.pspin] + ' ' 
                s += '%.3g \n'%f2
                rest -= f2
        s+='  rest=%.3g'%rest
        return s
        





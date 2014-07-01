"""Kohn-Sham single particle excitations realated objects.

"""
import sys
from math import pi, sqrt

import numpy as np
from ase.units import Bohr, Hartree, alpha
from ase.parallel import world, parprint

import gpaw.mpi as mpi
from gpaw.utilities import packed_index
from gpaw.lrtddft.excitation import Excitation, ExcitationList
from gpaw.pair_density import PairDensity
from gpaw.fd_operators import Gradient
from gpaw.utilities.tools import coordinates


class KSSingles(ExcitationList):
    """Kohn-Sham single particle excitations

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
    energy_range:
      The energy range [emin, emax] or emax for KS transitions to use as basis
    """

    def __init__(self,
                 calculator=None,
                 nspins=None,
                 eps=0.001,
                 istart=0,
                 jend=None,
                 energy_range=None,
                 filehandle=None,
                 txt=None):

        self.eps = None

        if filehandle is not None:
            self.read(fh=filehandle)
            return None

        ExcitationList.__init__(self, calculator, txt=txt)
        
        if calculator is None:
            return  # leave the list empty

        # deny hybrids as their empty states are wrong
        gsxc = calculator.hamiltonian.xc
        hybrid = hasattr(gsxc, 'hybrid') and gsxc.hybrid > 0.0
#        assert(not hybrid)

        # parallelization over spin or k-points is not yet supported
        if calculator.wfs.gd.comm.size != world.size:
            raise RuntimeError(
                """Parallelization over spin or k-points is not yet supported.
Use parallel={'domain': world.size} in the calculator.""")

        # XXX is this still needed ?
        error = calculator.wfs.eigensolver.error
        criterion = (calculator.input_parameters['convergence']['eigenstates']
                     * calculator.wfs.nvalence)
        if error > criterion:
            raise RuntimeError('The wfs error is larger than ' +
                               'the convergence criterion (' +
                               str(error) + ' > ' + str(criterion) + ')')

        self.select(nspins, eps, istart, jend, energy_range)

        trkm = self.get_trk()
        self.txt.write('KSS TRK sum %g (%g,%g,%g)\n' %
                       (np.sum(trkm) / 3., trkm[0], trkm[1], trkm[2]))
        pol = self.get_polarizabilities(lmax=3)
        self.txt.write(
            'KSS polarisabilities(l=0-3) %g, %g, %g, %g\n' %
            tuple(pol.tolist()))

    def select(self, nspins=None, eps=0.001,
               istart=0, jend=None, energy_range=None):
        """Select KSSingles according to the given criterium."""

        paw = self.calculator
        wfs = paw.wfs
        self.dtype = wfs.dtype
        self.kpt_u = wfs.kpt_u

        if self.kpt_u[0].psit_nG is None:
            raise RuntimeError('No wave functions in calculator!')

        # criteria
        emin = -sys.float_info.max
        emax = sys.float_info.max
        if energy_range is not None:
            try:
                emin, emax = energy_range
                emin /= Hartree
                emax /= Hartree
            except:
                emax = energy_range / Hartree
        self.istart = istart
        if jend is None:
            self.jend = sys.maxint
        else:
            self.jend = jend
        self.eps = eps

        # here, we need to take care of the spins also for
        # closed shell systems (Sz=0)
        # vspin is the virtual spin of the wave functions,
        #       i.e. the spin used in the ground state calculation
        # pspin is the physical spin of the wave functions
        #       i.e. the spin of the excited states
        self.nvspins = wfs.nspins
        self.npspins = wfs.nspins
        fijscale = 1
        ispins = [0]
        if self.nvspins < 2:
            if nspins > self.nvspins:
                self.npspins = nspins
                fijscale = 0.5
                ispins = [0, 1]

        # now select
        for ispin in ispins:
            for kpt in self.kpt_u:
                pspin = max(kpt.s, ispin)
                f_n = kpt.f_n
                eps_n = kpt.eps_n
                for i, fi in enumerate(f_n):
                    for j in range(i + 1, len(f_n)):
                        fij = fi - f_n[j]
                        epsij = eps_n[j] - eps_n[i]
                        if (fij > eps and
                            epsij >= emin and epsij < emax and
                            i >= self.istart and j <= self.jend):
                            # this is an accepted transition
                            ks = KSSingle(i, j, pspin, kpt, paw,
                                              fijscale=fijscale)
                            self.append(ks)
##                            parprint(('\r' + str(len(self))), end='')
##                            sys.stdout.flush()

    def read(self, filename=None, fh=None):
        """Read myself from a file"""
        if fh is None:
            if filename.endswith('.gz'):
                import gzip
                f = gzip.open(filename)
            else:
                f = open(filename, 'r')
        else:
            f = fh

        try:
            assert(f.readline().strip() == '# KSSingles')
        except:
            raise RuntimeError(f.name + ' is not a ' + 
                               self.__class__.__name__ + ' data file')
        words = f.readline().split()
        if len(words) == 1:
            # old output style for real wave functions (finite systems)
            self.dtype = float
        else:
            self.dtype = complex
            self.eps = float(f.readline())
        n = int(words[0])
        self.npspins = 1
        for i in range(n):
            kss = KSSingle(string=f.readline(), dtype=self.dtype)
            self.append(kss)
            self.npspins = max(self.npspins, kss.pspin + 1)
        self.update()

        if fh is None:
            f.close()

    def update(self):
        istart = self[0].i
        jend = 0
        npspins = 1
        nvspins = 1
        for kss in self:
            istart = min(kss.i, istart)
            jend = max(kss.j, jend)
            if kss.pspin == 1:
                npspins = 2
            if kss.spin == 1:
                nvspins = 2
        self.istart = istart
        self.jend = jend
        self.npspins = npspins
        self.nvspins = nvspins

        if hasattr(self, 'energies'):
            del(self.energies)

    def set_arrays(self):
        if hasattr(self, 'energies'):
            return
        energies = []
        fij = []
        me = []
        mur = []
        muv = []
        magn = []
        for k in self:
            energies.append(k.energy)
            fij.append(k.fij)
            me.append(k.me)
            mur.append(k.mur)
            if k.muv is not None:
                muv.append(k.muv)
            if k.magn is not None:
                magn.append(k.magn)
        self.energies = np.array(energies)
        self.fij = np.array(fij)
        self.me = np.array(me)
        self.mur = np.array(mur)
        if len(muv):
            self.muv = np.array(muv)
        else:
            self.muv = None
        if len(magn):
            self.magn = np.array(magn)
        else:
            self.magn = None

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
                    f = gzip.open(filename, 'wb')
                else:
                    f = open(filename, 'w')
            else:
                f = fh

            f.write('# KSSingles\n')
            if self.dtype == float:
                f.write('{0:d}\n'.format(len(self)))
            else:
                f.write('{0:d} complex\n'.format(len(self)))
                f.write('{0}\n'.format(self.eps))
            for kss in self:
                f.write(kss.outstring())
            
            if fh is None:
                f.close()

 
class KSSingle(Excitation, PairDensity):
    """Single Kohn-Sham transition containing all it's indicees

    ::

      pspin=physical spin
      spin=virtual  spin, i.e. spin in the ground state calc.
      kpt=the Kpoint object
      fijscale=weight for the occupation difference::
      me  = sqrt(fij*epsij) * <i|r|j>
      mur = - <i|r|a>
      muv = - <i|nabla|a>/omega_ia with omega_ia>0
      magn = <i|[r x nabla]|a> / (2 m_e c)
    """
    def __init__(self, iidx=None, jidx=None, pspin=None, kpt=None,
                 paw=None, string=None, fijscale=1, dtype=float):
        
        if string is not None:
            self.fromstring(string, dtype)
            return None

        # normal entry
        
        PairDensity.__init__(self, paw)
        wfs = paw.wfs
        PairDensity.initialize(self, kpt, iidx, jidx)

        self.pspin = pspin
        
        f = kpt.f_n
        self.fij = (f[iidx] - f[jidx]) * fijscale
        e = kpt.eps_n
        self.energy = e[jidx] - e[iidx]

        # calculate matrix elements -----------

        gd = wfs.gd
        self.gd = gd

        # length form ..........................

        # course grid contribution
        # <i|r|j> is the negative of the dipole moment (because of negative
        # e- charge)
        me = - gd.calculate_dipole_moment(self.get())

        # augmentation contributions
        ma = np.zeros(me.shape, dtype=me.dtype)
        pos_av = paw.atoms.get_positions() / Bohr
        for a, P_ni in kpt.P_ani.items():
            Ra = pos_av[a]
            Pi_i = P_ni[self.i].conj()
            Pj_i = P_ni[self.j]
            Delta_pL = wfs.setups[a].Delta_pL
            ni = len(Pi_i)
            ma0 = 0
            ma1 = np.zeros(me.shape, dtype=me.dtype)
            for i in range(ni):
                for j in range(ni):
                    pij = Pi_i[i] * Pj_i[j]
                    ij = packed_index(i, j, ni)
                    # L=0 term
                    ma0 += Delta_pL[ij, 0] * pij
                    # L=1 terms
                    if wfs.setups[a].lmax >= 1:
                        # see spherical_harmonics.py for
                        # L=1:y L=2:z; L=3:x
                        ma1 += np.array([Delta_pL[ij, 3], Delta_pL[ij, 1],
                                         Delta_pL[ij, 2]]) * pij
            ma += sqrt(4 * pi / 3) * ma1 + Ra * sqrt(4 * pi) * ma0
        gd.comm.sum(ma)

        self.me = sqrt(self.energy * self.fij) * (me + ma)

        self.mur = - (me + ma)

        # velocity form .............................

        me = np.zeros(self.mur.shape, dtype=self.mur.dtype)

        # get derivatives
        dtype = self.wfj.dtype
        dwfj_cg = gd.empty((3), dtype=dtype)
        if not hasattr(gd, 'ddr'):
            gd.ddr = [Gradient(gd, c, dtype=dtype).apply for c in range(3)]
        for c in range(3):
            gd.ddr[c](self.wfj, dwfj_cg[c], kpt.phase_cd)
            me[c] = gd.integrate(self.wfi.conj() * dwfj_cg[c])

        # augmentation contributions
        ma = np.zeros(me.shape, dtype=me.dtype)
        for a, P_ni in kpt.P_ani.items():
            Pi_i = P_ni[self.i].conj()
            Pj_i = P_ni[self.j]
            nabla_iiv = paw.wfs.setups[a].nabla_iiv
            for c in range(3):
                for i1, Pi in enumerate(Pi_i):
                    for i2, Pj in enumerate(Pj_i):
                        ma[c] += Pi * Pj * nabla_iiv[i1, i2, c]
        gd.comm.sum(ma)
        
        self.muv = - (me + ma) / self.energy

        # magnetic transition dipole ................

        magn = np.zeros(me.shape, dtype=me.dtype)
        r_cg, r2_g = coordinates(gd)

        wfi_g = self.wfi.conj()
        for ci in range(3):
            cj = (ci + 1) % 3
            ck = (ci + 2) % 3
            magn[ci] = gd.integrate(wfi_g * r_cg[cj] * dwfj_cg[ck] -
                                    wfi_g * r_cg[ck] * dwfj_cg[cj]  )
        # augmentation contributions
        ma = np.zeros(magn.shape, dtype=magn.dtype)
        for a, P_ni in kpt.P_ani.items():
            Pi_i = P_ni[self.i].conj()
            Pj_i = P_ni[self.j]
            rnabla_iiv = paw.wfs.setups[a].rnabla_iiv
            for c in range(3):
                for i1, Pi in enumerate(Pi_i):
                    for i2, Pj in enumerate(Pj_i):
                        ma[c] += Pi * Pj * rnabla_iiv[i1, i2, c]
        gd.comm.sum(ma)
        
        self.magn = -alpha / 2. * (magn + ma)

    def __add__(self, other):
        """Add two KSSingles"""
        result = self.copy()
        result.me = self.me + other.me
        result.mur = self.mur + other.mur
        result.muv = self.muv + other.muv
        return result

    def __sub__(self, other):
        """Subtract two KSSingles"""
        result = self.copy()
        result.me = self.me - other.me
        result.mur = self.mur - other.mur
        result.muv = self.muv - other.muv
        return result

    def __mul__(self, x):
        """Multiply a KSSingle with a number"""
        if type(x) == type(0.) or type(x) == type(0):
            result = self.copy()
            result.me = self.me * x
            result.mur = self.mur * x
            result.muv = self.muv * x
            return result
        else:
            return RuntimeError('not a number')
        
    def __div__(self, x):
        return self.__mul__(1. / x)

    def copy(self):
        return KSSingle(string=self.outstring())

    def fromstring(self, string, dtype=float):
        l = string.split()
        self.i = int(l.pop(0))
        self.j = int(l.pop(0))
        self.pspin = int(l.pop(0))
        self.spin = int(l.pop(0))
        if dtype == float:
            self.k = 0
            self.weight = 1
        else:
            self.k = int(l.pop(0))
            self.weight = float(l.pop(0))
        self.energy = float(l.pop(0))
        self.fij = float(l.pop(0))
        if len(l) == 3: # old writing style
            self.me = np.array([float(l.pop(0)) for i in range(3)])
            self.mur = - self.me / sqrt(self.energy * self.fij)
            self.muv = None
            self.magn = None
        else:
            self.mur = np.array([dtype(l.pop(0)) for i in range(3)])
            self.me = - self.mur * sqrt(self.energy * self.fij)
            self.muv = np.array([dtype(l.pop(0)) for i in range(3)])
            if len(l): 
                self.magn = np.array([dtype(l.pop(0)) for i in range(3)])
            else:
                self.magn = None
        return None

    def outstring(self):
        if self.mur.dtype == float:
            string = '{:d} {:d}  {:d} {:d}  {:g} {:g}'.format(
                self.i, self.j, self.pspin, self.spin, self.energy, self.fij)
        else:
            string = '{:d} {:d}  {:d} {:d} {:d} {:g}  {:g} {:g}'.format(
                self.i, self.j, self.pspin, self.spin, self.k, self.weight,
                self.energy, self.fij)
        string += '  '

        def format_me(me):
            string = ''
            if me.dtype == float:
                for m in me:
                    string += ' {0:.5e}'.format(m)
            else:
                for m in me:
                    string += ' {0.real:.5e}{0.imag:+.5e}j'.format(m)
            return string
                
        string += '  ' + format_me(self.mur)
        string += '  ' + format_me(self.muv)
        string += '  ' + format_me(self.magn)
        string += '\n'

        return string
        
    def __str__(self):
        string = "# <KSSingle> %d->%d %d(%d) eji=%g[eV]" % \
            (self.i, self.j, self.pspin, self.spin,
             self.energy * Hartree)
        if self.me.dtype == float:
            string += ' (%g,%g,%g)' % (self.me[0], self.me[1], self.me[2])
        else:
            string += ' kpt={0:d} w={1:g}'.format(self.k, self.weight)
            string += ' ('
            for c, m in enumerate(self.me):
                string += '{0.real:.5e}{0.imag:+.5e}j'.format(m)
                if c < 2:
                    string +=','
            string +=')'
        return string
    
    #####################
    ## User interface: ##
    #####################

    def get_weight(self):
        return self.fij


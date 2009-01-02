from math import pi

import numpy as npy
from numpy.fft import fftn

from ase.units import Hartree
from pair_density import PairDensity2 as PairDensity
from gpaw.poisson import PoissonSolver
from gpaw.utilities import pack, unpack
from gpaw.utilities.tools import pick, construct_reciprocal, dagger, fill
from gpaw.utilities.gauss import Gaussian
from gpaw.utilities.blas import r2k
from gpaw.mpi import rank, MASTER

def get_vxc(paw, spin=0, U=None):
    """Calculate matrix elements of the xc-potential."""

    if U is not None: # Rotate xc matrix
        return npy.dot(dagger(U), npy.dot(get_vxc(paw, spin), U))
    
    psit_nG = paw.kpt_u[spin].psit_nG[:]
    nt_g = paw.density.nt_sg[spin]
    vxct_g = paw.finegd.zeros()
    paw.hamiltonian.xc.get_energy_and_potential(nt_g, vxct_g)
    vxct_G = paw.gd.empty()
    paw.hamiltonian.restrict(vxct_g, vxct_G)
    Vxc_nn = npy.zeros((paw.nbands, paw.nbands))

    # Fill in upper triangle
    r2k(0.5 * paw.gd.dv, psit_nG, vxct_G * psit_nG, 0.0, Vxc_nn)
    paw.gd.comm.sum(Vxc_nn)

    # Fill in lower triangle
    for n in range(paw.nbands - 1):
        Vxc_nn[n:, n] = Vxc_nn[n, n:]

    # Add atomic PAW corrections
    for nucleus in paw.my_nuclei:
        D_sp = nucleus.D_sp[:]
        H_sp = npy.zeros_like(D_sp)
        nucleus.setup.xc_correction.calculate_energy_and_derivatives(
            D_sp, H_sp)
        H_ii = unpack(H_sp[spin])
        P_ni = nucleus.P_uni[spin]
        Vxc_nn += npy.dot(P_ni, npy.dot(H_ii, P_ni.T))
    return Vxc_nn * Hartree


class Coulomb:
    """Class used to evaluate two index coulomb integrals"""
    def __init__(self, gd, poisson=None):
        """Class should be initialized with a grid_descriptor 'gd' from
           the gpaw module.
        """        
        self.gd = gd
        self.poisson = poisson

    def load(self, method):
        """Make sure all necessary attributes have been initialized"""
        
        # ensure that requested method is valid
        assert method in ('real', 'recip_gauss', 'recip_ewald'),\
            str(method) + ' is an invalid method name,\n' +\
            'use either real, recip_gauss, or recip_ewald'

        if method.startswith('recip'):
            if self.gd.comm.size > 1:
                raise RuntimeError('Cannot do parallel FFT, ' +\
                                   'use method=\'real\'')
            if not hasattr(self, 'k2'):
                self.k2, self.N3 = construct_reciprocal(self.gd)
                
            if method.endswith('ewald') and not hasattr(self, 'ewald'):
                # cutoff radius
                rc = 0.5 * npy.average(self.gd.domain.cell_c)
                # ewald potential: 1 - cos(k rc)
                self.ewald = (npy.ones(self.gd.n_c) - 
                              npy.cos(npy.sqrt(self.k2) * rc))
                # lim k -> 0 ewald / k2 
                self.ewald[0, 0, 0] = 0.5 * rc**2

            if method.endswith('gauss') and not hasattr(self, 'ng'):
                gauss = Gaussian(self.gd)
                self.ng = gauss.get_gauss(0) / npy.sqrt(4 * pi)
                self.vg = gauss.get_gauss_pot(0) / npy.sqrt(4 * pi)
        
        else: # method == 'real'
            if not hasattr(self, 'solve'):
                if self.poisson is not None:
                    self.solve = self.poisson.solve
                else:
                    solver = PoissonSolver(nn=2)
                    solver.initialize(self.gd, load_gauss=True)
                    self.solve = solver.solve


    def coulomb(self, n1, n2=None, Z1=None, Z2=None, method='recip_gauss'):
        """Evaluates the coulomb integral of n1 and n2

        The coulomb integral is defined by::

                                      *
                      /    /      n1(r)  n2(r')
          (n1 | n2) = | dr | dr'  -------------,
                      /    /         |r - r'|
                      
        where n1 and n2 could be complex.

        real:
           Evaluate directly in real space using gaussians to neutralize
           density n2, such that the potential can be generated by standard
           procedures
              
        recip_ewald:
           Evaluate by Fourier transform.
           Divergence at division by k^2 is avoided by utilizing the Ewald /
           Tuckermann trick, which formaly requires the densities to be
           localized within half of the unit cell.

        recip_gauss:
           Evaluate by Fourier transform.
           Divergence at division by k^2 is avoided by removing total charge
           of n1 and n2 with gaussian density ng::
           
                                                   *          *    *
            (n1|n2) = (n1 - Z1 ng|n2 - Z2 ng) + (Z2 n1 + Z1 n2 - Z1 Z2 ng | ng)

           The evaluation of the integral (n1 - Z1 ng|n2 - Z2 ng) is done in
           k-space using FFT techniques.
        """
        self.load(method)
        # determine integrand using specified method
        if method == 'real':
            I = self.gd.zeros()
            if n2 == None: n2 = n1; Z2 = Z1
            self.solve(I, n2, charge=Z2, eps=1e-12, zero_initial_phi=True)
            I *= n1.conj()
        elif method == 'recip_ewald':
            n1k = fftn(n1)
            if n2 == None: n2k = n1k
            else: n2k = fftn(n2)
            I = n1k.conj() * n2k * self.ewald * 4 * pi / (self.k2 * self.N3)
        elif method == 'recip_gauss':
            # Determine total charges
            if Z1 == None: Z1 = self.gd.integrate(n1)
            if Z2 == None and n2 != None: Z2 = self.gd.integrate(n2)

            # Determine the integrand of the neutral system
            # (n1 - Z1 ng)* int dr'  (n2 - Z2 ng) / |r - r'|
            nk1 = fftn(n1 - Z1 * self.ng)
            if n2 == None:
                I = abs(nk1)**2 * 4 * pi / (self.k2 * self.N3)
            else:
                nk2 = fftn(n2 - Z2 * self.ng)
                I = nk1.conj() * nk2 * 4 * pi / (self.k2 * self.N3)

            # add the corrections to the integrand due to neutralization
            if n2 == None:
                I += (2 * npy.real(npy.conj(Z1) * n1) -
                      abs(Z1)**2 * self.ng) * self.vg
            else:
                I += (npy.conj(Z1) * n2 + Z2 * n1.conj() -
                      npy.conj(Z1) * Z2 * self.ng) * self.vg
        else:
             raise RuntimeError, 'Method %s unknown' % method
         
        if n1.dtype.char == float and (n2 == None or
                                       n2.dtype.char == float):
            return npy.real(self.gd.integrate(I))
        else:
            return self.gd.integrate(I)


## # coulomb integral types:
## 'ijij', # Direct
## 'ijji', # Exchange
## 'iijj', # iijj
## 'iiij', # Semiexchange
## 'ikjk', # Extra
class Coulomb4:
    """Determine four-index Coulomb integrals"""
    def __init__(self, paw, spin=0):
        paw.set_positions()
        paw.initialize_wave_functions()
        
        self.kpt = paw.kpt_u[spin]
        self.pd = PairDensity(paw, finegrid=True)
        self.nt12_G = paw.gd.empty()
        self.nt34_G = paw.gd.empty()
        self.rhot12_g = paw.finegd.empty()
        self.rhot34_g = paw.finegd.empty()
        self.potential_g = paw.finegd.empty()
        self.psum = paw.gd.comm.sum
        self.my_nuclei = paw.my_nuclei
        self.integrate = paw.finegd.integrate
        self.poisson_solve = paw.hamiltonian.poisson.solve
        self.u = spin
        
    def get_integral(self, n1, n2, n3, n4, order=0):
        """Get four-index coulomb integral.

        Indices can be vectors or scalars.
        If order == 0, return the Coulomb integrals::
        
          C4(ijkl) = \iint drdr' / |r-r'| i(r) j*(r) k*(r') l(r')

        else, return::

          V_{ijkl} = \iint drdr' / |r-r'| i*(r) j*(r') k(r) l(r')

        Notice that V_ijkl = C4(kijl)
        """
        if order != 0:
            n1, n2, n3, n4 = n3, n1, n2, n4
        
        self.pd.initialize(self.kpt, n1, n2)
        self.pd.get_coarse(self.nt12_G)
        self.pd.add_compensation_charges(self.nt12_G, self.rhot12_g)
        
        self.pd.initialize(self.kpt, n3, n4)
        self.pd.get_coarse(self.nt34_G)
        self.pd.add_compensation_charges(self.nt34_G, self.rhot34_g)

        self.poisson_solve(self.potential_g, self.rhot34_g, charge=None,
                           eps=1e-12, zero_initial_phi=True)
        I = self.integrate(self.rhot12_g * self.potential_g)

        # Add atomic corrections
        Ia = 0.0
        for nucleus in self.my_nuclei:
            #   ----
            # 2 >     P   P  C    P  P
            #   ----   1i  2j ijkl 3k 4l
            #   ijkl 
            P_ni = nucleus.P_uni[self.u]
            D12_p = pack(npy.outer(pick(P_ni, n1), pick(P_ni, n2)), 1e3)
            D34_p = pack(npy.outer(pick(P_ni, n3), pick(P_ni, n4)), 1e3)
            Ia += 2 * npy.dot(D12_p, npy.dot(nucleus.setup.M_pp, D34_p))
        I += self.psum(Ia)

        return I


def symmetry(i, j, k, l):
    """Uniqify index order.

    Permute indices into unique ordering, conserving the symmetry of
    the Coulomb kernel."""
    ijkl = npy.array((i, j, k, l), int)
    a = npy.argmin(ijkl)
    conj = False
    if a == 1:
        npy.take(ijkl, (1, 0, 3, 2), out=ijkl)
    elif a == 2:
        conj = True
        npy.take(ijkl, (2, 3, 0, 1), out=ijkl)
    elif a == 3:
        conj = True
        npy.take(ijkl, (3, 2, 1, 0), out=ijkl)

    if ijkl[0] == ijkl[1] and ijkl[3] < ijkl[2]:
        npy.take(ijkl, (0, 1, 3, 2), out=ijkl)
    elif ijkl[2] == ijkl[3] and ijkl[2] < ijkl[1]:
        conj = not conj
        npy.take(ijkl, (2, 3, 0, 1), out=ijkl)
    
    return tuple(ijkl), conj


def reduce_pairs(pairs):
    p = 0
    while p < len(pairs):
        i, j, k, l = pairs[p]
        ijkl, conj = symmetry(i, j, k, l)
        if ijkl == (i, j, k, l):
            p += 1
        else:
            pairs.pop(p)


def coulomb_dict(paw, U_nj, pairs, spin=0, done={}):
    coulomb = Coulomb4(paw, spin)
    for ijkl in pairs:
        ni, nj, nk, nl = U_nj[:, ijkl].T
        done[ijkl] = coulomb.get_integral(nk, ni, nj, nl) * Hartree
    return done


def unfold(N, done, dtype=float):
    V = npy.empty([N, N, N, N], dtype)
    for i in xrange(N):
        for j in xrange(N):
            for k in xrange(N):
                for l in xrange(N):
                    ijkl, conj = symmetry(i, j, k, l)
                    if conj:
                        V[i, j, k, l] = npy.conj(done.get(ijkl, 0))
                    else:
                        V[i, j, k, l] = done.get(ijkl, 0)
    return V


from gpaw.utilities.tools import symmetrize
from gpaw.utilities import packed_index, unpack2
from gpaw.utilities.blas import r2k
class HF:
    def __init__(self, paw):
        paw.set_positions()
        
        self.nspins       = paw.nspins
        self.nbands       = paw.nbands
        self.my_nuclei    = paw.my_nuclei
        self.restrict     = paw.hamiltonian.restrict
        self.pair_density = PairDensity(paw, finegrid=True)
        self.dv           = paw.gd.dv
        self.dtype        = paw.dtype 

        # Allocate space for matrices
        self.nt_G   = paw.gd.empty()
        self.rhot_g = paw.finegd.empty()
        self.vt_G   = paw.gd.empty()
        self.vt_g   = paw.finegd.empty()
        self.poisson_solve = paw.hamiltonian.poisson.solve

    def apply(self, kpt):
        H_nn = npy.zeros((self.nbands, self.nbands), self.dtype)
        self.soft_pseudo(kpt, H_nn, H_nn)
        self.atomic_val_val(kpt, H_nn)
        self.atomic_val_core(kpt, H_nn)
        return H_nn * Hartree

    def soft_pseudo(self, kpt, H_nn, h_nn):
        pd = self.pair_density
        deg = 2 / self.nspins
        fmin = 1e-9
        Htpsit_nG = npy.zeros(kpt.psit_nG.shape, self.dtype)

        for n1 in range(self.nbands):
            psit1_G = kpt.psit_nG[n1]
            f1 = kpt.f_n[n1] / deg
            for n2 in range(n1, self.nbands):
                psit2_G = kpt.psit_nG[n2]
                f2 = kpt.f_n[n2] / deg
                if f1 < fmin and f2 < fmin:
                    continue
                
                dc = 1 + (n1 != n2)
                pd.initialize(kpt, n1, n2)
                pd.get_coarse(self.nt_G)
                pd.add_compensation_charges(self.nt_G, self.rhot_g)
                self.poisson_solve(self.vt_g, -self.rhot_g,
                                   charge=-float(n1 == n2), eps=1e-12,
                                   zero_initial_phi=True)
                self.restrict(self.vt_g, self.vt_G)
                Htpsit_nG[n1] += f2 * self.vt_G * psit2_G
                if n1 != n2:
                    Htpsit_nG[n2] += f1 * self.vt_G * psit1_G

                for nucleus in self.my_nuclei:
                    P_ni = nucleus.P_uni[kpt.u]
                    v_L = npy.zeros((nucleus.setup.lmax + 1)**2)
                    nucleus.ghat_L.integrate(self.vt_g, v_L)
                    v_ii = unpack(npy.dot(nucleus.setup.Delta_pL, v_L))
                    h_nn[:, n1] += f2 * npy.dot(P_ni, npy.dot(v_ii, P_ni[n2]))
                    if n1 != n2:
                        h_nn[:,n2] += f1 * npy.dot(P_ni,npy.dot(v_ii,P_ni[n1]))
                    
        symmetrize(h_nn) # Grrrr why!!! XXX

        # Fill in lower triangle
        r2k(0.5 * self.dv, kpt.psit_nG[:], Htpsit_nG, 1.0, H_nn)

        # Fill in upper triangle
        fill(H_nn, 'upper')

    def atomic_val_val(self, kpt, H_nn):
        deg = 2 / self.nspins
        for nucleus in self.my_nuclei:
            P_ni = nucleus.P_uni[kpt.u]
            D_p  = nucleus.D_sp[kpt.s]
            D_ii = unpack2(D_p)
            H_p  = npy.zeros_like(D_p)
            ni = len(D_ii)
            for i1 in range(ni):
                for i2 in range(ni):
                    A = 0.0
                    for i3 in range(ni):
                        p13 = packed_index(i1, i3, ni)
                        for i4 in range(ni):
                            p24 = packed_index(i2, i4, ni)
                            A += nucleus.setup.M_pp[p13, p24] * D_ii[i3, i4]
                    p12 = packed_index(i1, i2, ni)
                    H_p[p12] -= 2 / deg * A / ((i1 != i2) + 1)
            H_nn += npy.dot(P_ni, npy.inner(unpack(H_p), P_ni.conj()))

    def atomic_val_core(self, kpt, H_nn):
        for nucleus in self.my_nuclei:
            P_ni = nucleus.P_uni[kpt.u]
            dH_ii = unpack(-nucleus.setup.X_p)
            H_nn += npy.dot(P_ni, npy.inner(dH_ii, P_ni.conj()))

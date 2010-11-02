# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""This module provides all the classes and functions associated with the
evaluation of exact exchange.

The eXact-eXchange energy functional is::

                                         *  _       _     * _        _
           __                /        psi  (r) psi (r) psi (r') psi (r')
       -1 \                  |  _  _      n       m       m        n
 E   = --  ) delta     f  f  | dr dr' ---------------------------------  (5.1)
  xx    2 /__     s s   n  m |                    _   _
           nm      n m       /                   |r - r'|
         
The action of the non-local exchange potential on an orbital is::

               /                         __
 ^             | _    _  _        _     \      _       _
 V   psi (r) = |dr' V(r, r') psi (r') =  ) V  (r) psi (r)               (5.3)
  xx    n      |                n       /__ nm       m
               /                         m

where::

                          _     * _
              __     psi (r) psi (r')
   _  _      \          m       m
 V(r, r') = - )  f   ----------------                                   (5.4a)
             /__  m       _   _
              m          |r - r'|
              
and::

                        * _       _
                /    psi (r) psi (r')
     _          | _     m       n
 V  (r) = -  f  |dr' ----------------                                   (5.4b)
  nm          m |         _   _
                /        |r - r'|

Equation numbers as in
'Exact Exchange in Density Functional Calculations'
Masters Thesis by Carsten Rostgaard, CAMP 2006

In PAW, equation (5.3) above transforms to::

   ^     ~      --               /
   v  | psi > = >  f  delta      |
    xx     n    --  m      s ,s  |
                m           n  m \

                   ~        ~       -- /
                   v  (r) |psi  > + >  |
                    nm        m     -- |
                                     a \

                            --  ~a    --      a  /    ~a    ~       a
                            >  |p  >  >  Delta   | dr g (r) v  (r) P
                            --   i    --      L  /     L     nm     mi
                            i i   1    L                              2
                             1 2

                            --     a       a           a
                       +    >     D     * C         * D
                            --     i i     i i i i     i i
                          i i i i   1 2     1 3 2 4     3 4
                           1 2 3 4

                            --      a      a
                       +    >      X    * D
                            --      i i    i i
                            i i      1 2    1 2
                             1 2

                             core-core  \ \
                       +    E           | |
                             xx         / /
"""

import numpy as np

from ase.units import Hartree
from gpaw.atom.configurations import core_states
from gpaw.gaunt import make_gaunt
from gpaw.utilities import hartree, packed_index, unpack, unpack2, pack, pack2
from gpaw.utilities.blas import gemm, r2k
from gpaw.pair_density import PairDensity2 as PairDensity
from gpaw.poisson import PoissonSolver, PoissonFFTSolver
from gpaw.utilities.tools import apply_subspace_mask, symmetrize

usefft = False
verbose = False

## for debug
## from gpaw.mpi.mpiprint import mpiprint

def dummy_interpolate(nt_G, nt_g):
    nt_g[:] = nt_G


class EXX:
    """EXact eXchange.

    Class offering methods for selfconsistent evaluation of the
    exchange energy."""
    
    def __init__(self, density, hamiltonian, wfs, atoms,
                 energy_only=False, use_finegrid=True):
        # Initialize class-attributes
        self.nspins        = wfs.nspins
        self.nbands        = wfs.nbands
        self.interpolate   = density.interpolate
        self.restrict      = hamiltonian.restrict
        self.integrate     = wfs.gd.integrate
        self.fineintegrate = density.finegd.integrate
        ksum, dsum = wfs.kpt_comm.sum, wfs.gd.comm.sum
        self.psum          = lambda x: ksum(dsum(x)) # Sum over all processors
        self.energy_only   = energy_only
        self.use_finegrid  = use_finegrid
        self.pair_density  = PairDensity(density, atoms, use_finegrid)
        self.poisson_solve = hamiltonian.poisson.solve
        self.density = density
        self.Exx = 0.0
        self.Ekin = 0.0
        self.qs0 = None

        # Set correct Poisson solver
        if usefft:
            if use_finegrid:
                solver = PoissonFFTSolver()
                solver.set_grid_descriptor(density.finegd)
                solver.initialize()
                self.poisson_solve = solver.solve
            else:
                solver = PoissonFFTSolver()
                solver.set_grid_descriptor(wfs.gd)
                solver.initialize()
                self.poisson_solve = solver.solve
        elif not use_finegrid:
            solver = PoissonSolver(nn=hamiltonian.poisson.nn)
            solver.set_grid_descriptor(wfs.gd)
            solver.initialize()
            self.poisson_solve = solver.solve
            
        # Allocate space for matrices
        self.nt_G = wfs.gd.empty()# Pseudo density on coarse grid
        self.rhot_g = density.finegd.empty()# Comp. pseudo density on fine grid
        self.vt_G = wfs.gd.empty()# Pot. of comp. pseudo density on coarse grid
        self.vt_g =density.finegd.empty()# Pot. of comp. ps. dens. on fine grid

        # Overwrites in case of coarse grid Poisson solver
        if not use_finegrid:
            self.fineintegrate = wfs.gd.integrate
            self.interpolate = dummy_interpolate
            self.rhot_g = wfs.gd.empty()
            self.vt_g = self.vt_G
        
        # For rotating the residuals we need the diagonal Fock potentials
        if not energy_only:
            for kpt in wfs.kpt_u:
                kpt.vt_nG = wfs.gd.zeros(wfs.nbands)

    def grr(self, wfs, kpt, Htpsit_nG, hamiltonian):
        nbands = wfs.nbands
        domain_comm = self.density.gd.comm
        H_nn = np.zeros((nbands, nbands))
        wfs.kin.apply(kpt.psit_nG, Htpsit_nG, kpt.phase_cd)
        hamiltonian.apply_local_potential(kpt.psit_nG, Htpsit_nG, kpt.s)
        self.apply(kpt, Htpsit_nG, H_nn, hamiltonian.dH_asp,
                   hamiltonian.xc.xcfunc.hybrid)
        r2k(0.5 * wfs.gd.dv, kpt.psit_nG, Htpsit_nG, 1.0, H_nn)
        for a, P_ni in kpt.P_ani.items():
            dH_p = unpack(hamiltonian.dH_asp[a][kpt.s])
            gemm(1.0, P_ni, np.dot(P_ni, dH_p), 1.0, H_nn, 'c')
        domain_comm.sum(H_nn, 0)
        if kpt.f_n is not None:
            apply_subspace_mask(H_nn, kpt.f_n)
        return H_nn

    def apply(self, kpt, Htpsit_nG, H_nn, dH_asp, hybrid):
        """Apply exact exchange operator."""

        # Initialize method-attributes
        psit_nG = kpt.psit_nG   # Wave functions
        Exx = Ekin = 0.0        # Energy of eXact eXchange and kinetic energy
        deg = 2 // self.nspins   # Spin degeneracy
        f_n = kpt.f_n           # Occupation number
        s   = kpt.s             # Global spin index
        pd  = self.pair_density # Class for handling pair densities
        fmin= 1e-9              # Occupations less than this counts as empty

        if f_n is None:
            return

        P_ani = kpt.P_ani
        setups = self.density.setups
        domain_comm = self.density.gd.comm

        if not self.energy_only:
            kpt.vxx_ani = vxx_ani = {}
            kpt.vxx_anii = vxx_anii = {}
            for a, P_ni in P_ani.items():
                nbands, ni = P_ni.shape
                vxx_ani[a] = np.zeros((nbands, ni))
                vxx_anii[a] = np.zeros((nbands, ni, ni))

        # Determine pseudo-exchange
        for n1 in range(self.nbands):
            psit1_G = psit_nG[n1]
            f1 = f_n[n1] * hybrid / deg
            for n2 in range(n1, self.nbands):
                psit2_G = psit_nG[n2]
                f2 = f_n[n2] * hybrid / deg
                if f1 < fmin and f2 < fmin:
                    continue # Don't do anything if both occupations are small
                
                dc = (1 + (n1 != n2)) * deg / hybrid  # double count factor
                pd.initialize(kpt, n1, n2)

                # Determine current exchange density
                pd.get_coarse(self.nt_G)
                pd.add_compensation_charges(self.nt_G, self.rhot_g)

                # Re-use stored potential if possible
                zero_phi = True
                if 0: #n1 == n2 and hasattr(self, 'vt_unG'):
                    zero_phi = False
                    if self.use_finegrid:
                        self.interpolate(self.vt_unG[u, n2] / f2, self.vt_g)
                    else:
                        np.divide(self.vt_unG[u, n2], f2, self.vt_g)
                    
                # Determine exchange potential:
                iter = self.poisson_solve(self.vt_g, -self.rhot_g,
                                          charge=-float(n1 == n2), eps=1e-12,
                                          zero_initial_phi=zero_phi)
                if verbose:
                    print 'EXX: n1=%s, n2=%s, iterations=%s' % (n1, n2, iter)

                if self.use_finegrid:
                    self.restrict(self.vt_g, self.vt_G)
                else:
                    assert self.vt_G is self.vt_g

                # Integrate the potential on fine and coarse grids
                int_fine = self.fineintegrate(self.vt_g * self.rhot_g)
                int_coarse = self.integrate(self.vt_G * self.nt_G)
                if domain_comm.rank == 0: # Only add to energy on master CPU
                    Exx += 0.5 * f1 * f2 * dc * int_fine
                    Ekin -= f1 * f2 * dc * int_coarse
                if not self.energy_only:
                    Htpsit_nG[n1] += f2 * self.vt_G * psit2_G
                    if n1 != n2:
                        Htpsit_nG[n2] += f1 * self.vt_G * psit1_G
                    else:
                        kpt.vt_nG[n2] = f2 * self.vt_G

                    # Update the vxx_uni and vxx_unii vectors of the nuclei,
                    # used to determine the atomic hamiltonian, and the 
                    # residuals
                    if self.use_finegrid:
                        ghat = self.density.ghat
                    else:
                        ghat = self.density.Ghat
                    v_aL = ghat.dict()
                    ghat.integrate(self.vt_g, v_aL)
                    for a, v_L in v_aL.items():
                        v_ii = unpack(np.dot(setups[a].Delta_pL, v_L))
                        v_ni = vxx_ani[a]
                        v_nii = vxx_anii[a]
                        P_ni = P_ani[a]
                        v_ni[n1] += f2 * np.dot(v_ii, P_ni[n2])
                        if n1 != n2:
                            v_ni[n2] += f1 * np.dot(v_ii, P_ni[n1])
                        else:
                            # XXX Check this:
                            v_nii[n1] = f2 * v_ii

        # Apply the atomic corrections to the energy and the Hamiltonian matrix
        for a, P_ni in P_ani.items():
            setup = setups[a]

            # Add non-trivial corrections the Hamiltonian matrix
            if not self.energy_only:
                h_nn = symmetrize(np.inner(P_ni, vxx_ani[a]))
                H_nn += h_nn
                Ekin -= np.dot(f_n, np.diagonal(h_nn))
                dH_p = dH_asp[a][s]

            # Get atomic density and Hamiltonian matrices
            D_p  = self.density.D_asp[a][s]
            D_ii = unpack2(D_p)
            ni = len(D_ii)
            
            # Add atomic corrections to the valence-valence exchange energy
            # --
            # >  D   C     D
            # --  ii  iiii  ii
            for i1 in range(ni):
                for i2 in range(ni):
                    A = 0.0
                    for i3 in range(ni):
                        p13 = packed_index(i1, i3, ni)
                        for i4 in range(ni):
                            p24 = packed_index(i2, i4, ni)
                            A += setup.M_pp[p13, p24] * D_ii[i3, i4]
                    if not self.energy_only:
                        p12 = packed_index(i1, i2, ni)
                        dH_p[p12] -= 2 * hybrid / deg * A / ((i1 != i2) + 1)
                        Ekin += 2 * hybrid / deg * D_ii[i1, i2] * A
                    Exx -= hybrid / deg * D_ii[i1, i2] * A
            
            # Add valence-core exchange energy
            # --
            # >  X   D
            # --  ii  ii
            Exx -= hybrid * np.dot(D_p, setup.X_p)
            if not self.energy_only:
                dH_p -= hybrid * setup.X_p
                Ekin += hybrid * np.dot(D_p, setup.X_p)

            # Add core-core exchange energy
            if s == 0:
                Exx += hybrid * setup.ExxC

        # Update the class attributes
        if self.qs0 is None:
            self.qs0 = (kpt.q, kpt.s)
        if (kpt.q, kpt.s) == self.qs0:
            self.Exx = 0.0
            self.Ekin = 0.0
        self.Exx += self.psum(Exx)
        self.Ekin += self.psum(Ekin)

    def force_kpoint(self, kpt, hybrid):
        """Force due to exact exchange operator"""
        raise NotImplementedError

        deg = 2 / self.nspins
        u = kpt.u
        F_ac = np.zeros((self.Na, 3))
        fmin = 1.e-10
        pd = self.pair_density

        for n1 in range(self.nbands):
            psit1_G = kpt.psit_nG[n1]
            f1 = kpt.f_n[n1] * hybrid / deg
            for n2 in range(n1, self.nbands):
                psit2_G = kpt.psit_nG[n2]
                f2 = kpt.f_n[n2] * hybrid / deg
                if f1 < fmin and f2 < fmin:
                    continue

                # Re-determine all of the exhange potentials
                dc = (1 + (n1 != n2)) * deg / hybrid
                pd.initialize(kpt, n1, n2)
                pd.get_coarse(self.nt_G)
                pd.add_compensation_charges(self.nt_G, self.rhot_g)
                self.poisson_solve(self.vt_g, -self.rhot_g,
                                   charge=-float(n1 == n2), eps=1e-12,
                                   zero_initial_phi=True)
                if self.use_finegrid:
                    self.restrict(self.vt_g, self.vt_G)
                else:
                    assert self.vt_G is self.vt_g

                # Determine force contribution from exchange potential
                for nucleus in self.ghat_nuclei:
                    if self.use_finegrid:
                        ghat_L = nucleus.ghat_L
                    else:
                        ghat_L = nucleus.Ghat_L
                    
                    if nucleus.in_this_domain:
                        lmax = nucleus.setup.lmax
                        F_Lc = np.zeros(((lmax + 1)**2, 3))
                        ghat_L.derivative(self.vt_g, F_Lc)
                        D_ii = np.outer(nucleus.P_uni[u, n1],
                                         nucleus.P_uni[u, n2])
                        D_p = pack(D_ii, tolerance=1e30)
                        Q_L = np.dot(D_p, nucleus.setup.Delta_pL)
                        F_ac[nucleus.a] -= (f1 * f2 * dc * np.dot(Q_L, F_Lc))
                    else:
                        ghat_L.derivative(self.vt_g, None)

                # Add force contribution from the change in projectors
                for nucleus in self.ghat_nuclei:
                    v_L = np.zeros((nucleus.setup.lmax + 1)**2)
                    if self.use_finegrid:
                        nucleus.ghat_L.integrate(self.vt_g, v_L)
                    else:
                        nucleus.Ghat_L.integrate(self.vt_G, v_L)
                    
                    if nucleus.in_this_domain:
                        v_ii = unpack(np.dot(nucleus.setup.Delta_pL, v_L))

                        ni = nucleus.setup.ni
                        F_ic = np.zeros((ni, 3))
                        nucleus.pt_i.derivative(psit1_G, F_ic)
                        F_ic.shape = (ni * 3,)
                        F_iic = np.dot(v_ii, np.outer(
                            nucleus.P_uni[u, n2], F_ic))

                        F_ic[:] = 0.0
                        F_ic.shape =(ni, 3)
                        nucleus.pt_i.derivative(psit2_G, F_ic)
                        F_ic.shape = (ni * 3,)
                        F_iic += np.dot(v_ii, np.outer(
                            nucleus.P_uni[u, n1], F_ic))

                        # F_iic *= 2.0
                        F_iic.shape = (ni, ni, 3)
                        for i in range(ni):
                            F_ac[nucleus.a] -= f1 * f2 * dc * F_iic[i, i].real

                    else:
                        nucleus.pt_i.derivative(psit1_G, None)
                        nucleus.pt_i.derivative(psit2_G, None)
        return F_ac

    def adjust_residual(self, pR_G, dR_G, kpt, n):
        dR_G += kpt.vt_nG[n] * pR_G

    def rotate(self, kpt, U_nn):
        if not hasattr(kpt, 'vxx_anii'):
            return
        # Rotate EXX related stuff
        vt_nG = kpt.vt_nG
        gemm(1.0, vt_nG.copy(), U_nn, 0.0, vt_nG)
        for v_ni in kpt.vxx_ani.values():
            gemm(1.0, v_ni.copy(), U_nn, 0.0, v_ni)
        for v_nii in kpt.vxx_anii.values():
            gemm(1.0, v_nii.copy(), U_nn, 0.0, v_nii)


def atomic_exact_exchange(atom, type = 'all'):
    """Returns the exact exchange energy of the atom defined by the
       instantiated AllElectron object 'atom'
    """
    gaunt = make_gaunt(lmax=max(atom.l_j)) # Make gaunt coeff. list
    Nj = len(atom.n_j)                     # The total number of orbitals

    # determine relevant states for chosen type of exchange contribution
    if type == 'all':
        nstates = mstates = range(Nj)
    else:
        Njcore = core_states(atom.symbol) # The number of core orbitals
        if type == 'val-val':
            nstates = mstates = range(Njcore, Nj)
        elif type == 'core-core':
            nstates = mstates = range(Njcore)
        elif type == 'val-core':
            nstates = range(Njcore,Nj)
            mstates = range(Njcore)
        else:
            raise RuntimeError('Unknown type of exchange: ', type)

    # Arrays for storing the potential (times radius)
    vr = np.zeros(atom.N)
    vrl = np.zeros(atom.N)
    
    # do actual calculation of exchange contribution
    Exx = 0.0
    for j1 in nstates:
        # angular momentum of first state
        l1 = atom.l_j[j1]

        for j2 in mstates:
            # angular momentum of second state
            l2 = atom.l_j[j2]

            # joint occupation number
            f12 = .5 * atom.f_j[j1] / (2. * l1 + 1) * \
                       atom.f_j[j2] / (2. * l2 + 1)

            # electron density times radius times length element
            nrdr = atom.u_j[j1] * atom.u_j[j2] * atom.dr
            nrdr[1:] /= atom.r[1:]

            # potential times radius
            vr[:] = 0.0

            # L summation
            for l in range(l1 + l2 + 1):
                # get potential for current l-value
                hartree(l, nrdr, atom.beta, atom.N, vrl)

                # take all m1 m2 and m values of Gaunt matrix of the form
                # G(L1,L2,L) where L = {l,m}
                G2 = gaunt[l1**2:(l1+1)**2, l2**2:(l2+1)**2, l**2:(l+1)**2]**2

                # add to total potential
                vr += vrl * np.sum(G2)

            # add to total exchange the contribution from current two states
            Exx += -.5 * f12 * np.dot(vr, nrdr)

    # double energy if mixed contribution
    if type == 'val-core': Exx *= 2.

    # return exchange energy
    return Exx


def constructX(gen):
    """Construct the X_p^a matrix for the given atom.

    The X_p^a matrix describes the valence-core interactions of the
    partial waves.
    """
    # initialize attributes
    uv_j = gen.vu_j    # soft valence states * r:
    lv_j = gen.vl_j    # their repective l quantum numbers
    Nvi  = 0 
    for l in lv_j:
        Nvi += 2 * l + 1   # total number of valence states (including m)

    # number of core and valence orbitals (j only, i.e. not m-number)
    Njcore = gen.njcore
    Njval  = len(lv_j)

    # core states * r:
    uc_j = gen.u_j[:Njcore]
    r, dr, N, beta = gen.r, gen.dr, gen.N, gen.beta

    # potential times radius
    vr = np.zeros(N)
        
    # initialize X_ii matrix
    X_ii = np.zeros((Nvi, Nvi))

    # make gaunt coeff. list
    lmax = max(gen.l_j[:Njcore] + gen.vl_j)
    gaunt = make_gaunt(lmax=lmax)

    # sum over core states
    for jc in range(Njcore):
        lc = gen.l_j[jc]

        # sum over first valence state index
        i1 = 0
        for jv1 in range(Njval):
            lv1 = lv_j[jv1] 

            # electron density 1 times radius times length element
            n1c = uv_j[jv1] * uc_j[jc] * dr
            n1c[1:] /= r[1:]

            # sum over second valence state index
            i2 = 0
            for jv2 in range(Njval):
                lv2 = lv_j[jv2]
                
                # electron density 2
                n2c = uv_j[jv2] * uc_j[jc] * dr
                n2c[1:] /= r[1:]
            
                # sum expansion in angular momenta
                for l in range(min(lv1, lv2) + lc + 1):
                    # Int density * potential * r^2 * dr:
                    hartree(l, n2c, beta, N, vr)
                    nv = np.dot(n1c, vr)
                    
                    # expansion coefficients
                    A_mm = X_ii[i1:i1 + 2 * lv1 + 1, i2:i2 + 2 * lv2 + 1]
                    for mc in range(2 * lc + 1):
                        for m in range(2 * l + 1):
                            G1c = gaunt[lv1**2:(lv1 + 1)**2,
                                        lc**2 + mc, l**2 + m]
                            G2c = gaunt[lv2**2:(lv2 + 1)**2,
                                        lc**2 + mc, l**2 + m]
                            A_mm += nv * np.outer(G1c, G2c)
                i2 += 2 * lv2 + 1
            i1 += 2 * lv1 + 1

    # pack X_ii matrix
    X_p = pack2(X_ii, tolerance=1e-8)
    return X_p


def H_coulomb_val_core(paw, u=0):
    """Short description here.

    ::

                     core   *    *
             //       --   i(r) k(r') k(r) j (r')
       H   = || drdr' >   ----------------------
        ij   //       --          |r - r'|
                      k
    """
    H_nn = np.zeros((paw.wfs.nbands, paw.wfs.nbands), dtype=paw.wfs.dtype)
    for a, P_ni in paw.wfs.kpt_u[u].P_ani.items():
        X_ii = unpack(paw.wfs.setups[a].X_p)
        H_nn += np.dot(P_ni.conj(), np.dot(X_ii, P_ni.T))
    paw.wfs.gd.comm.sum(H_nn)
    return H_nn * Hartree

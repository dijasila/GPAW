# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""This module defines a ``KPoint`` class."""

from math import pi, sqrt
from cmath import exp

import Numeric as num
import LinearAlgebra as linalg
from RandomArray import random, seed
from multiarray import innerproduct as inner # avoid the dotblas version!

from gpaw import mpi
from gpaw.operators import Gradient
from gpaw.transformers import Transformer
from gpaw.utilities import unpack
from gpaw.utilities.blas import axpy, rk, gemm
from gpaw.utilities.complex import cc, real
from gpaw.utilities.lapack import diagonalize

class KPoint:
    """Class for a singel **k**-point.

    The ``KPoint`` class takes care of all wave functions for a
    certain **k**-point and a certain spin."""
    
    def __init__(self, gd, weight, s, k, u, k_c, typecode):
        """Construct **k**-point object.

        Parameters:
         ============ =======================================================
         ``gd``       Descriptor for wave-function grid.
         ``weight``   Weight of this **k**-point.
         ``s``        Spin index: up or down (0 or 1).
         ``k``        **k**-point index.
         ``u``        Combined spin and **k**-point index.
         ``k_c``      scaled **k**-point vector (coordinates scaled to
                      [-0.5:0.5] interval).
         ``typecode`` Data type of wave functions (``Float`` or ``Complex``).
         ============ =======================================================

        Note that ``s`` and ``k`` are global spin/**k**-point indices,
        whereas ``u`` is a local spin/**k**-point pair index for this
        processor.  So if we have `S` spins and `K` **k**-points, and
        the spins/**k**-points are parallelized over `P` processors
        (``kpt_comm``), then we have this equation relating ``s``,
        ``k`` and ``u``::

           rSK
           --- + u = sK + k,
            P

        where `r` is the processor rank within ``kpt_comm``.  The
        total number of spin/**k**-point pairs, `SK`, is always a
        multiple of the number of processors, `P`.

        Attributes:
         ============= =======================================================
         ``phase_cd``  Bloch phase-factors for translations - axis ``c=0,1,2``
                       and direction ``d=0,1``.
         ``eps_n``     Eigenvalues.
         ``f_n``       Occupation numbers.

         ``psit_nG``   Wave functions.

         ``nbands``    Number of bands.
         ============= =======================================================

        Parallel stuff:
         ======== =======================================================
         ``comm`` MPI-communicator for domain.
         ``root`` Rank of the CPU that does the matrix diagonalization of
                  ``H_nn`` and the Cholesky decomposition of ``S_nn``.
         ======== =======================================================
        """

        self.gd = gd
        self.weight = weight
        self.typecode = typecode
        
        self.phase_cd = num.ones((3, 2), num.Complex)
        if typecode == num.Float:
            # Gamma-point calculation:
            self.k_c = None
        else:
            sdisp_cd = self.gd.domain.sdisp_cd
            for c in range(3):
                for d in range(2):
                    self.phase_cd[c, d] = exp(2j * pi *
                                              sdisp_cd[c, d] * k_c[c])
            self.k_c = k_c

        self.s = s  # spin index
        self.k = k  # k-point index
        self.u = u  # combined spin and k-point index

        # Which CPU does overlap-matrix Cholesky-decomposition and
        # Hamiltonian-matrix diagonalization?
        self.comm = self.gd.comm
        self.root = u % self.comm.size
        
        self.psit_nG = None

    def allocate(self, nbands):
        """Allocate arrays."""
        self.nbands = nbands
        self.eps_n = num.empty(nbands, num.Float)
        self.f_n = num.empty(nbands, num.Float)
        
    def adjust_number_of_bands(self, nbands, pt_nuclei, my_nuclei):
        """Adjust the number of states.

        If we are starting from atomic orbitals, then the desired
        number of bands (``nbands``) will most likely differ from the
        number of current atomic orbitals (``self.nbands``).  If this
        is the case, then new arrays are allocated:

        * Too many bands: The bands with the lowest eigenvalues are
          used.
        * Too few bands: Extra random wave functions are added.
        """
        
        if nbands == self.nbands:
            return

        nao = self.nbands  # number of atomic orbitals
        nmin = min(nao, nbands)

        tmp_nG = self.psit_nG
        self.psit_nG = self.gd.empty(nbands, self.typecode)
        self.psit_nG[:nmin] = tmp_nG[:nmin]

        tmp_n = self.eps_n
        self.allocate(nbands)
        self.eps_n[:nmin] = tmp_n[:nmin]

        extra = nbands - nao
        if extra > 0:
            # Generate random wave functions:
            self.eps_n[nao:] = self.eps_n[nao - 1] + 0.5
            self.random_wave_functions(self.psit_nG[nao:])

            #Calculate projections
            for nucleus in pt_nuclei:
                if nucleus.in_this_domain:
                    P_ni = nucleus.P_uni[self.u,nao:]
                    P_ni[:] = 0.0
                    nucleus.pt_i.integrate(self.psit_nG[nao:], P_ni, self.k)
                else:
                    nucleus.pt_i.integrate(self.psit_nG[nao:], None, self.k)
                    
            #Orthonormalize
            self.orthonormalize(my_nuclei)

    def random_wave_functions(self, psit_nG):
        """Generate random wave functions"""

        gd1 = self.gd.coarsen()
        gd2 = gd1.coarsen()

        psit_G1 = gd1.empty(typecode=self.typecode)
        psit_G2 = gd2.empty(typecode=self.typecode)

        interpolate2 = Transformer(gd2, gd1, 1, self.typecode).apply
        interpolate1 = Transformer(gd1, self.gd, 1, self.typecode).apply

        shape = tuple(gd2.n_c)

        scale = sqrt(12 / num.product(gd2.domain.cell_c))

        seed(1, 2 + mpi.rank)

        for psit_G in psit_nG:
            if self.typecode == num.Float:
                psit_G2[:] = (random(shape) - 0.5) * scale
            else:
                psit_G2.real = (random(shape) - 0.5) * scale
                psit_G2.imag = (random(shape) - 0.5) * scale

            interpolate2(psit_G2, psit_G1, self.phase_cd)
            interpolate1(psit_G1, psit_G, self.phase_cd)

    
    def orthonormalize(self, my_nuclei):
        """Orthonormalize wave functions."""
        S_nn = num.zeros((self.nbands, self.nbands), self.typecode)

        # Fill in the lower triangle:
        rk(self.gd.dv, self.psit_nG, 0.0, S_nn)
        
        for nucleus in my_nuclei:
            P_ni = nucleus.P_uni[self.u]

            S_nn += num.dot(P_ni, cc(inner(nucleus.setup.O_ii, P_ni)))
        
        self.comm.sum(S_nn, self.root)

        if self.comm.rank == self.root:
            # inverse returns a non-contigous matrix - grrrr!  That is
            # why there is a copy.  Should be optimized with a
            # different lapack call to invert a triangular matrix XXXXX
            S_nn[:] = linalg.inverse(
                linalg.cholesky_decomposition(S_nn)).copy()

        self.comm.broadcast(S_nn, self.root)
        
        gemm(1.0, self.psit_nG.copy(), S_nn, 0.0, self.psit_nG)

        for nucleus in my_nuclei:
            P_ni = nucleus.P_uni[self.u]
            gemm(1.0, P_ni.copy(), S_nn, 0.0, P_ni)


    def add_to_density(self, nt_G):
        """Add contribution to pseudo electron-density."""
        if self.typecode is num.Float:
            for psit_G, f in zip(self.psit_nG, self.f_n):
                axpy(f, psit_G**2, nt_G)  # nt_G += f * psit_G**2
        else:
            for psit_G, f in zip(self.psit_nG, self.f_n):
                nt_G += f * (psit_G * num.conjugate(psit_G)).real
                
    def add_to_kinetic_energy_density(self, taut_G, ddr):
        """Add contribution to pseudo kinetic energy density.
        The derivative object ddr must be given.
        """
        for psit_G, f in zip(self.psit_nG, self.f_n):
            d_G = self.gd.empty()
            for c in range(3):
                ddr[c](psit_G,d_G)
                if self.typecode is num.Float:
                    taut_G += f * d_G[c]**2
                else:
                    taut_G += f * (d_G * num.conjugate(d_G)).real
                
    def create_atomic_orbitals(self, nao, nuclei):
        """Initialize the wave functions from atomic orbitals.

        Create ``nao`` atomic orbitals."""

        # Allocate space for wave functions, occupation numbers,
        # eigenvalues and projections:
        self.allocate(nao)
        self.psit_nG = self.gd.zeros(nao, self.typecode)
        
        # fill in the atomic orbitals:
        nao0 = 0
        for nucleus in nuclei:
            nao1 = nao0 + nucleus.get_number_of_atomic_orbitals()
            nucleus.create_atomic_orbitals(self.psit_nG[nao0:nao1], self.k)
            nao0 = nao1
        assert nao0 == nao

    def create_random_orbitals(self, nbands):
        """Initialize all the wave functions from random numbers"""

        self.allocate(nbands)
        self.psit_nG = self.gd.zeros(nbands, self.typecode)
        self.random_wave_functions(self.psit_nG)        

    def apply_hamiltonian(self, hamiltonian, a_nG, b_nG):
        """Apply Hamiltonian to wave functions."""

        b_nG[:] = 0.0
        hamiltonian.kin.apply(a_nG, b_nG, self.phase_cd)
        b_nG += a_nG * hamiltonian.vt_sG[self.s]
        
        for nucleus in hamiltonian.pt_nuclei:
            # Apply the non-local part:
            nucleus.apply_hamiltonian(a_nG, b_nG, self.s, self.k)

    def apply_overlap(self, pt_nuclei, a_nG, b_nG):
        """Apply overlap operator to wave functions."""

        b_nG[:] = a_nG
        
        for nucleus in pt_nuclei:
            # Apply the non-local part:
            nucleus.apply_overlap(a_nG, b_nG, self.k)

            
    def apply_inverse_overlap(self, pt_nuclei, a_nG, b_nG):
        """Apply overlap operator to wave functions."""

        b_nG[:] = a_nG
        
        for nucleus in pt_nuclei:
            # Apply the non-local part:
            nucleus.apply_inverse_overlap(a_nG, b_nG, self.k)

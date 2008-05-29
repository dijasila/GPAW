# -*- coding: utf-8 -*-
# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""This module will go away some dat in the future.  The few methods
in the PAWExtra class should be moved to the PAW class or other
modules."""

import sys

import numpy as npy

import gpaw.io
import gpaw.mpi as mpi
from gpaw.xc_functional import XCFunctional
from gpaw.density import Density
from gpaw.utilities import pack

MASTER = 0


class PAWExtra:
    def get_fermi_level(self):
        """Return the Fermi-level."""
        e = self.occupation.get_fermi_level()
        if e is None:
            # Zero temperature calculation - return vacuum level:
            e = 0.0
        return e * self.Ha

    def write(self, filename, mode=''):
        """use mode='all' to write the wave functions"""
        self.timer.start('IO')
        gpaw.io.write(self, filename, mode)
        self.timer.stop('IO')
        
    def get_reference_energy(self):
        return self.Eref * self.Ha
    
    def get_ibz_kpoints(self):
        """Return array of k-points in the irreducible part of the BZ."""
        return self.ibzk_kc

    def get_wave_function_array(self, n, k, s):
        """Return pseudo-wave-function array.
        
        For the parallel case find the rank in kpt_comm that contains
        the (k,s) pair, for this rank, collect on the corresponding
        domain a full array on the domain master and send this to the
        global master."""

        kpt_rank, u = divmod(k + self.nkpts * s, self.nmyu)

        psit_nG = self.kpt_u[u].psit_nG
        if psit_nG is None:
            raise RuntimeError('This calculator has no wave functions!')

        if self.world.size == 1:
            return psit_nG[n][:]

        if self.kpt_comm.rank == kpt_rank:
            psit_G = self.gd.collect(psit_nG[n][:])

            if kpt_rank == MASTER:
                if self.master:
                    return psit_G

            # Domain master send this to the global master
            if self.domain.comm.rank == MASTER:
                self.kpt_comm.send(psit_G, MASTER, 1398)

        if self.master:
            # allocate full wavefunction and receive
            psit_G = self.gd.empty(dtype=self.dtype, global_array=True)
            self.kpt_comm.receive(psit_G, kpt_rank, 1398)
            return psit_G

    def collect_eigenvalues(self, k=0, s=0):
        """Return eigenvalue array.

        For the parallel case find the rank in kpt_comm that contains
        the (k,s) pair, for this rank, collect on the corresponding
        domain a full array on the domain master and send this to the
        global master."""

        kpt_rank, u = divmod(k + self.nkpts * s, self.nmyu)

        if kpt_rank == MASTER:
            return self.kpt_u[u].eps_n

        if self.kpt_comm.rank == kpt_rank:
            # Domain master send this to the global master
            if self.domain.comm.rank == MASTER:
                self.kpt_comm.send(self.kpt_u[u].eps_n, MASTER, 1301)
        elif self.master:
            eps_n = npy.zeros(self.nbands)
            self.kpt_comm.receive(eps_n, kpt_rank, 1301)
            return eps_n

    def collect_occupations(self, k=0, s=0):
        """Return occupation array.

        For the parallel case find the rank in kpt_comm that contains
        the (k,s) pair, for this rank, collect on the corresponding
        domain a full array on the domain master and send this to the
        global master."""

        kpt_rank, u = divmod(k + self.nkpts * s, self.nmyu)

        if kpt_rank == MASTER:
            return self.kpt_u[u].f_n

        if self.kpt_comm.rank == kpt_rank:
            # Domain master send this to the global master
            if self.domain.comm.rank == MASTER:
                self.kpt_comm.send(self.kpt_u[u].f_n, MASTER, 1313)
        elif self.master:
            f_n = npy.zeros(self.nbands)
            self.kpt_comm.receive(f_n, kpt_rank, 1313)
            return f_n

    def get_wannier_integrals(self, c, s, k, k1, G):
        """Calculate integrals for maximally localized Wannier functions."""

        assert s <= self.nspins

        kpt_rank, u = divmod(k + self.nkpts * s, self.nmyu)
        kpt_rank1, u1 = divmod(k1 + self.nkpts * s, self.nmyu)

        # XXX not for the kpoint/spin parallel case
        assert self.kpt_comm.size == 1
        
        # Get pseudo part
        Z_nn = self.gd.wannier_matrix(self.kpt_u[u].psit_nG,
                                      self.kpt_u[u1].psit_nG, c, G)

        # Add corrections
        for nucleus in self.my_nuclei:
            Z_nn += nucleus.wannier_correction(G, c, u, u1)

        self.gd.comm.sum(Z_nn, MASTER)
            
        return Z_nn

    def get_xc_difference(self, xcname):
        """Calculate non-selfconsistent XC-energy difference."""
        xc = self.hamiltonian.xc
        oldxcfunc = xc.xcfunc

        if isinstance(xcname, str):
            newxcfunc = XCFunctional(xcname, self.nspins)
        else:
            newxcfunc = xcname

        newxcfunc.set_non_local_things(self, energy_only=True)

        xc.set_functional(newxcfunc)
        for setup in self.setups:
            setup.xc_correction.xc.set_functional(newxcfunc)

        if newxcfunc.hybrid > 0.0 and not self.nuclei[0].ready:
            self.set_positions(npy.array([n.spos_c * self.domain.cell_c
                                          for n in self.nuclei]))

        vt_g = self.finegd.empty()  # not used for anything!
        nt_sg = self.density.nt_sg
        if self.nspins == 2:
            Exc = xc.get_energy_and_potential(nt_sg[0], vt_g, nt_sg[1], vt_g)
        else:
            Exc = xc.get_energy_and_potential(nt_sg[0], vt_g)

        for nucleus in self.my_nuclei:
            D_sp = nucleus.D_sp
            H_sp = npy.zeros(D_sp.shape) # not used for anything!
            xc_correction = nucleus.setup.xc_correction
            Exc += xc_correction.calculate_energy_and_derivatives(D_sp, H_sp)

        Exc = self.domain.comm.sum(Exc)

        for kpt in self.kpt_u:
            newxcfunc.apply_non_local(kpt)
        Exc += newxcfunc.get_non_local_energy()

        xc.set_functional(oldxcfunc)
        for setup in self.setups:
            setup.xc_correction.xc.set_functional(oldxcfunc)

        return self.Ha * (Exc - self.Exc)

    def get_grid_spacings(self):
        return self.a0 * self.gd.h_c

    def get_exact_exchange(self):
        dExc = self.get_xc_difference('EXX') / self.Ha
        Exx = self.Exc + dExc
        for nucleus in self.nuclei:
            Exx += nucleus.setup.xc_correction.Exc0
        return Exx

    def get_weights(self):
        return self.weight_k #???

    def initialize_from_wave_functions(self):
        """Initialize density and Hamiltonian from wave functions"""
        
        self.set_positions()
        self.density.move()
        self.density.update(self.kpt_u, self.symmetry)
##         if self.wave_functions_initialized:
##             self.density.move()
##             self.density.update(self.kpt_u, self.symmetry)
##         else:
##             # no wave-functions: restart from LCAO
##             self.initialize_wave_functions()

    def totype(self, dtype):
        """Converts all the dtype dependent quantities of Paw
        (Laplacian, wavefunctions etc.) to dtype"""

        from gpaw.operators import Laplace

        if dtype not in [float, complex]:
            raise RuntimeError('PAW can be converted only to Float or Complex')

        self.dtype = dtype

        # Hamiltonian
        nn = self.stencils[0]
        self.hamiltonian.kin = Laplace(self.gd, -0.5, nn, dtype)

        # Nuclei
        for nucleus in self.nuclei:
            nucleus.dtype = dtype
            nucleus.ready = False

        # reallocate only my_nuclei (as the others are not allocated at all)
        for nucleus in self.my_nuclei:
            nucleus.reallocate(self.nmybands)

        self.set_positions()

        # Wave functions
        for kpt in self.kpt_u:
            kpt.dtype = dtype
            kpt.psit_nG = npy.array(kpt.psit_nG[:], dtype)

        # Eigensolver
        # !!! FIX ME !!!
        # not implemented yet...

    def read_wave_functions(self, mode='gpw'):
        """Read wave functions one by one from seperate files"""

        for u in range(self.nmyu):
            kpt = self.kpt_u[u]
            kpt.psit_nG = self.gd.empty(self.nbands, self.dtype)
            # Read band by band to save memory
            s = kpt.s
            k = kpt.k
            for n, psit_G in enumerate(kpt.psit_nG):
                psit_G[:] = gpaw.io.read_wave_function(self.gd, s, k, n, mode)
                
    def warn(self, string=None):
        if not string:
            string = "somethings wrong"
        print >> self.txt, "WARNING >>"
        print >> self.txt, string
        print >> self.txt, "WARNING <<"
                
    def wave_function_volumes(self):
        """Return the volume needed by the wave functions"""
        nu = self.nkpts * self.nspins
        volumes = npy.empty((nu,self.nbands))

        for k in range(nu):
            for n, psit_G in enumerate(self.kpt_u[k].psit_nG):
                volumes[k, n] = self.gd.integrate(psit_G**4)

                # atomic corrections
                for nucleus in self.my_nuclei:
                    # make sure the integrals are there
                    nucleus.setup.four_phi_integrals()
                    P_i = nucleus.P_uni[k, n]
                    ni = len(P_i)
                    P_ii = npy.outer(P_i, P_i)
                    P_p = pack(P_ii)
                    I = 0
                    for i1 in range(ni):
                        for i2 in range(ni):
                            I += P_ii[i1, i2] * npy.dot(P_p,
                                             nucleus.setup.I4_iip[i1, i2])
                volumes[k, n] += I
                
        return 1. / volumes

import numpy as np
from numpy.linalg import inv, solve

from gpaw.io import Reader

from gpaw.utilities.scalapack import (pblas_simple_hemm, pblas_simple_gemm,
                                      scalapack_inverse, scalapack_solve,
                                      scalapack_zero, pblas_tran,
                                      scalapack_set)


def create_propagator(name, **kwargs):
    if isinstance(name, Propagator):
        return name
    elif isinstance(name, dict):
        kwargs.update(name)
        return create_propagator(**kwargs)
    elif name == 'sicn':
        return SICNPropagator(**kwargs)
    elif name == 'ecn':
        return ECNPropagator(**kwargs)
    else:
        raise RuntimeError('Unknown propagator: %s' % name)


def equal(a, b, eps=1e-16):
    return abs(a - b) < eps


class Propagator(object):

    def __init__(self):
        return

    def initialize(self, paw):
        self.timer = paw.timer
        self.fxc = paw.fxc
        self.log = paw.log

    def propagate(self, time, time_step):
        raise RuntimeError('Virtual member function called')


class LCAOPropagator(Propagator):

    def __init__(self):
        return

    def initialize(self, paw):
        Propagator.initialize(self, paw)
        self.wfs = paw.wfs
        self.density = paw.density
        self.hamiltonian = paw.hamiltonian

    def update_projectors(self):
        self.timer.start('LCAO update projectors')
        # Loop over all k-points
        for k, kpt in enumerate(self.wfs.kpt_u):
            self.wfs.atomic_correction.calculate_projections(self.wfs, kpt)
        self.timer.stop('LCAO update projectors')

    def get_hamiltonian(self, kpt):
        eig = self.wfs.eigensolver
        H_MM = eig.calculate_hamiltonian_matrix(self.hamiltonian, self.wfs,
                                                kpt, root=-1)
        return H_MM

    def update_hamiltonian(self):
        self.update_projectors()
        self.density.update(self.wfs)
        self.hamiltonian.update(self.density)


class ReplayPropagator(LCAOPropagator):

    def __init__(self, filename):
        self.reader = Reader(filename)
        version = self.reader.version
        if version != 1:
            raise RuntimeError('Unknown version %s' % version)
        self.readi = 1
        self.readN = len(self.reader)

    def _align_read(self, time):
        while self.readi < self.readN:
            r = self.reader[self.readi]
            if equal(r.time, time):
                break
            self.readi += 1
        if self.readi == self.readN:
            raise RuntimeError('Time not found: %f' % time)

    def propagate(self, time, time_step):
        next_time = time + time_step
        self._align_read(next_time)
        r = self.reader[self.readi].wave_functions
        self.wfs.read_wave_functions(r)
        self.wfs.read_occupations(r)
        self.readi += 1
        self.update_hamiltonian()
        return next_time

    def __del__(self):
        self.reader.close()


class ECNPropagator(LCAOPropagator):

    def __init__(self):
        return

    def initialize(self, paw, hamiltonian=None):
        LCAOPropagator.initialize(self, paw)
        if hamiltonian is not None:
            self.hamiltonian = hamiltonian

        self.blacs = self.wfs.ksl.using_blacs
        if self.blacs:
            self.ksl = ksl = self.wfs.ksl
            nao = ksl.nao
            nbands = ksl.bd.nbands
            mynbands = ksl.bd.mynbands
            blocksize = ksl.blocksize

            from gpaw.blacs import Redistributor
            self.log('BLACS Parallelization')

            # Parallel grid descriptors
            grid = ksl.blockgrid
            assert grid.nprow * grid.npcol == self.wfs.ksl.block_comm.size
            # FOR DEBUG
            self.MM_descriptor = grid.new_descriptor(nao, nao, nao, nao)
            self.mm_block_descriptor = grid.new_descriptor(nao, nao, blocksize,
                                                           blocksize)
            self.Cnm_block_descriptor = grid.new_descriptor(nbands, nao,
                                                            blocksize,
                                                            blocksize)
            # self.CnM_descriptor = ksl.blockgrid.new_descriptor(nbands,
            #     nao, mynbands, nao)
            self.mM_column_descriptor = ksl.single_column_grid.new_descriptor(
                nao, nao, ksl.naoblocksize, nao)
            self.CnM_unique_descriptor = ksl.single_column_grid.new_descriptor(
                nbands, nao, mynbands, nao)

            # Redistributors
            self.mm2MM = Redistributor(ksl.block_comm,
                                       self.mm_block_descriptor,
                                       self.MM_descriptor)  # XXX FOR DEBUG
            self.MM2mm = Redistributor(ksl.block_comm,
                                       self.MM_descriptor,
                                       self.mm_block_descriptor)  # FOR DEBUG
            self.Cnm2nM = Redistributor(ksl.block_comm,
                                        self.Cnm_block_descriptor,
                                        self.CnM_unique_descriptor)
            self.CnM2nm = Redistributor(ksl.block_comm,
                                        self.CnM_unique_descriptor,
                                        self.Cnm_block_descriptor)
            self.mM2mm = Redistributor(ksl.block_comm,
                                       self.mM_column_descriptor,
                                       self.mm_block_descriptor)

            for kpt in self.wfs.kpt_u:
                scalapack_zero(self.mm_block_descriptor, kpt.S_MM, 'U')
                scalapack_zero(self.mm_block_descriptor, kpt.T_MM, 'U')

    def propagate(self, time, time_step):
        for k, kpt in enumerate(self.wfs.kpt_u):
            H_MM = self.wfs.eigensolver.calculate_hamiltonian_matrix(
                self.hamiltonian, self.wfs, kpt, add_kinetic=False, root=-1)
            self.propagate_wfs(kpt.C_nM, kpt.C_nM, kpt.S_MM, H_MM,
                               time_step)
        return time

    def propagate_wfs(self, sourceC_nM, targetC_nM, S_MM, H_MM, dt):
        self.timer.start('Linear solve')

        if self.blacs:
            # XXX, Preallocate
            target_blockC_nm = self.Cnm_block_descriptor.empty(dtype=complex)
            temp_blockC_nm = self.Cnm_block_descriptor.empty(dtype=complex)
            temp_block_mm = self.mm_block_descriptor.empty(dtype=complex)
            if self.density.gd.comm.rank != 0:
                # XXX Fake blacks nbands, nao, nbands, nao grid because some
                # weird asserts
                # (these are 0,x or x,0 arrays)
                sourceC_nM = self.CnM_unique_descriptor.zeros(dtype=complex)

            # 1. target = (S+0.5j*H*dt) * source
            # Wave functions to target
            self.CnM2nm.redistribute(sourceC_nM, temp_blockC_nm)

            # XXX It can't be this f'n hard to symmetrize a matrix (tri2full)
            # Remove upper diagonal
            scalapack_zero(self.mm_block_descriptor, H_MM, 'U')
            # Lower diagonal matrix:
            temp_block_mm[:] = S_MM - (0.5j * dt) * H_MM
            scalapack_set(self.mm_block_descriptor, temp_block_mm, 0, 0, 'U')
            # Note it's strictly lower diagonal matrix
            # Add transpose of H
            pblas_tran(-0.5j * dt, H_MM, 1.0, temp_block_mm,
                       self.mm_block_descriptor, self.mm_block_descriptor)
            # Add transpose of S
            pblas_tran(1.0, S_MM, 1.0, temp_block_mm,
                       self.mm_block_descriptor, self.mm_block_descriptor)

            pblas_simple_gemm(self.Cnm_block_descriptor,
                              self.mm_block_descriptor,
                              self.Cnm_block_descriptor,
                              temp_blockC_nm,
                              temp_block_mm,
                              target_blockC_nm)
            # 2. target = (S-0.5j*H*dt)^-1 * target
            # temp_block_mm[:] = S_MM + (0.5j*dt) * H_MM
            # XXX It can't be this f'n hard to symmetrize a matrix (tri2full)
            # Lower diagonal matrix:
            temp_block_mm[:] = S_MM + (0.5j * dt) * H_MM
            # Not it's stricly lower diagonal matrix:
            scalapack_set(self.mm_block_descriptor, temp_block_mm, 0, 0, 'U')
            # Add transpose of H:
            pblas_tran(+0.5j * dt, H_MM, 1.0, temp_block_mm,
                       self.mm_block_descriptor, self.mm_block_descriptor)
            # Add transpose of S
            pblas_tran(1.0, S_MM, 1.0, temp_block_mm,
                       self.mm_block_descriptor, self.mm_block_descriptor)

            scalapack_solve(self.mm_block_descriptor,
                            self.Cnm_block_descriptor,
                            temp_block_mm,
                            target_blockC_nm)

            if self.density.gd.comm.rank != 0:  # XXX is this correct?
                # XXX Fake blacks nbands, nao, nbands, nao grid because some
                # weird asserts
                # (these are 0,x or x,0 arrays)
                target = self.CnM_unique_descriptor.zeros(dtype=complex)
            else:
                target = targetC_nM
            self.Cnm2nM.redistribute(target_blockC_nm, target)
            self.density.gd.comm.broadcast(targetC_nM, 0)  # Is this required?
        else:
            # Note: The full equation is conjugated (therefore -+, not +-)
            targetC_nM[:] = \
                solve(S_MM - 0.5j * H_MM * dt,
                      np.dot(S_MM + 0.5j * H_MM * dt,
                             sourceC_nM.T.conjugate())).T.conjugate()

        self.timer.stop('Linear solve')

    def blacs_mm_to_global(self, H_mm):
        # Someone could verify that this works and remove the error.
        raise NotImplementedError('Method untested and thus unreliable')
        target = self.MM_descriptor.empty(dtype=complex)
        self.mm2MM.redistribute(H_mm, target)
        self.wfs.world.barrier()
        return target

    def blacs_nm_to_global(self, C_nm):
        # Someone could verify that this works and remove the error.
        raise NotImplementedError('Method untested and thus unreliable')
        target = self.CnM_unique_descriptor.empty(dtype=complex)
        self.Cnm2nM.redistribute(C_nm, target)
        self.wfs.world.barrier()
        return target


class SICNPropagator(ECNPropagator):

    def __init__(self):
        return

    def initialize(self, paw):
        ECNPropagator.initialize(self, paw)
        # Allocate kpt.C2_nM arrays
        for k, kpt in enumerate(self.wfs.kpt_u):
            kpt.C2_nM = np.empty_like(kpt.C_nM)

    def propagate(self, time, dt):
        # --------------
        # Predictor step
        # --------------
        # 1. Store current C_nM
        self.save_wfs()  # kpt.C2_nM = kpt.C_nM
        for k, kpt in enumerate(self.wfs.kpt_u):
            # H_MM(t) = <M|H(t)|M>
            kpt.H0_MM = self.get_hamiltonian(kpt)
            if self.fxc is not None:
                kpt.H0_MM += kpt.deltaXC_H_MM
            # 2. Solve Psi(t+dt) from (S_MM - 0.5j*H_MM(t)*dt) Psi(t+dt) =
            #                         (S_MM + 0.5j*H_MM(t)*dt) Psi(t)
            self.propagate_wfs(kpt.C_nM, kpt.C_nM, kpt.S_MM, kpt.H0_MM, dt)
        # ---------------
        # Propagator step
        # ---------------
        # 1. Calculate H(t+dt)
        self.update_hamiltonian()
        # 2. Estimate H(t+0.5*dt) ~ H(t) + H(t+dT)
        for k, kpt in enumerate(self.wfs.kpt_u):
            kpt.H0_MM *= 0.5
            #  Store this to H0_MM and maybe save one extra H_MM of
            # memory?
            kpt.H0_MM += 0.5 * self.get_hamiltonian(kpt)

            if self.fxc is not None:
                kpt.H0_MM += 0.5 * kpt.deltaXC_H_MM

            # 3. Solve Psi(t+dt) from
            # (S_MM - 0.5j*H_MM(t+0.5*dt)*dt) Psi(t+dt)
            #    = (S_MM + 0.5j*H_MM(t+0.5*dt)*dt) Psi(t)
            self.propagate_wfs(kpt.C2_nM, kpt.C_nM, kpt.S_MM, kpt.H0_MM, dt)

        # 4. Calculate new density and Hamiltonian
        self.update_hamiltonian()
        # TODO: this Hamiltonian does not contain fxc
        return time + dt

    def save_wfs(self):
        for k, kpt in enumerate(self.wfs.kpt_u):
            kpt.C2_nM[:] = kpt.C_nM


class TaylorPropagator(Propagator):

    def __init__(self):
        raise NotImplementedError('TaylorPropagator not implemented')

    def initialize(self, paw):
        if 1:
            # XXX to propagator class
            if self.propagator == 'taylor' and self.blacs:
                # cholS_mm = self.mm_block_descriptor.empty(dtype=complex)
                for kpt in self.wfs.kpt_u:
                    kpt.invS_MM = kpt.S_MM.copy()
                    scalapack_inverse(self.mm_block_descriptor,
                                      kpt.invS_MM, 'L')
            if self.propagator == 'taylor' and not self.blacs:
                tmp = inv(self.wfs.kpt_u[0].S_MM)
                self.wfs.kpt_u[0].invS = tmp

    def taylor_propagator(self, sourceC_nM, targetC_nM, S_MM, H_MM, dt):
        self.timer.start('Taylor propagator')

        if self.blacs:
            # XXX, Preallocate
            target_blockC_nm = self.Cnm_block_descriptor.empty(dtype=complex)
            if self.density.gd.comm.rank != 0:
                # XXX Fake blacks nbands, nao, nbands, nao grid because some
                # weird asserts
                # (these are 0,x or x,0 arrays)
                sourceC_nM = self.CnM_unique_descriptor.zeros(dtype=complex)

            # Zeroth order taylor to target
            self.CnM2nm.redistribute(sourceC_nM, target_blockC_nm)

            # XXX, preallocate, optimize use of temporal arrays
            temp_blockC_nm = target_blockC_nm.copy()
            temp2_blockC_nm = target_blockC_nm.copy()

            order = 4
            assert self.wfs.kd.comm.size == 1
            for n in range(order):
                # Multiply with hamiltonian
                pblas_simple_hemm(self.mm_block_descriptor,
                                  self.Cnm_block_descriptor,
                                  self.Cnm_block_descriptor,
                                  H_MM,
                                  temp_blockC_nm,
                                  temp2_blockC_nm, side='R')
                # XXX: replace with not simple gemm
                temp2_blockC_nm *= -1j * dt / (n + 1)
                # Multiply with inverse overlap
                pblas_simple_hemm(self.mm_block_descriptor,
                                  self.Cnm_block_descriptor,
                                  self.Cnm_block_descriptor,
                                  self.wfs.kpt_u[0].invS_MM,  # XXX
                                  temp2_blockC_nm,
                                  temp_blockC_nm, side='R')
                target_blockC_nm += temp_blockC_nm
            if self.density.gd.comm.rank != 0:  # Todo: Change to gd.rank
                # XXX Fake blacks nbands, nao, nbands, nao grid because
                # some weird asserts
                # (these are 0,x or x,0 arrays)
                target = self.CnM_unique_descriptor.zeros(dtype=complex)
            else:
                target = targetC_nM
            self.Cnm2nM.redistribute(target_blockC_nm, target)

            self.density.gd.comm.broadcast(targetC_nM, 0)
        else:
            assert self.wfs.kd.comm.size == 1
            if self.density.gd.comm.rank == 0:
                targetC_nM[:] = sourceC_nM[:]
                tempC_nM = sourceC_nM.copy()
                order = 4
                for n in range(order):
                    tempC_nM[:] = \
                        np.dot(self.wfs.kpt_u[0].invS,
                               np.dot(H_MM, 1j * dt / (n + 1) *
                                      tempC_nM.T.conjugate())).T.conjugate()
                    targetC_nM += tempC_nM
            self.density.gd.comm.broadcast(targetC_nM, 0)

        self.timer.stop('Taylor propagator')

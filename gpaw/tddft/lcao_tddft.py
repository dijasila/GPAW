from gpaw import GPAW
from gpaw.external_potential import ConstantElectricField
import numpy as np
from math import sqrt
from gpaw.utilities.blas import gemm
from gpaw.utilities import unpack
from numpy.linalg import inv, eig
from numpy import dot, eye, array, asarray, zeros
from gpaw.mixer import DummyMixer
from math import pi
from ase.parallel import paropen
from gpaw.mpi import world
from gpaw.tddft.units import attosec_to_autime
from numpy.linalg import solve
from gpaw.utilities.scalapack import pblas_simple_hemm, pblas_simple_gemm, \
                                     scalapack_inverse, scalapack_solve, \
                                     scalapack_zero, pblas_tran, scalapack_set
                                     
from gpaw.utilities.tools import tri2full 

import sys

def print_matrix(M, file=None, rank=0):
    # XXX Debugging stuff. Remove.
    if world.rank == 0:
        if file is not None:
            f = open(file,'w')
        else:
            f = sys.stdout
        a, b = M.shape
        for i in range(a):
            for j in range(b):
                print >>f, "%.7f" % M[i][j].real, "%.7f" % M[i][j].imag,
            print >>f
        if file is not None:
            f.close()


def verify(data1, data2, id, uplo='B'):
    # Debugging stuff. Remove
    if uplo=='B':
        err = sum(abs(data1-data2).ravel()**2)
    else:
        err = 0
        N,M = data1.shape
        for i in range(N):
            for j in range(i,M):
                if uplo == 'L':
                    if i >= j:
                        err += abs(data1[i][j]-data2[i][j])**2
                if uplo == 'U':
                    if i <= j:
                        err += abs(data1[i][j]-data2[i][j])**2
    if err > 1e-7:
        print "verify err", err
    if err > 1e-5:
       print "Parallel assert failed: ", id, " norm: ", err
       print "Data from proc ", world.rank
       print "First", data1
       print "Second", data2
       print "Diff", data1-data2
       assert False

def mpiverify(data, id):
    # Do some debugging when running on two procs XXX REMOVE
    if world.size == 2:
        if world.rank == 0:
            temp = -data.copy()
        else:
            temp = data.copy()
        world.sum(temp)
        err = sum(abs(temp).ravel()**2)
        if err > 1e-10:
            if world.rank == 0:
                print "Parallel assert failed: ", id, " norm: ", sum(temp.ravel()**2)
            print "Data from proc ", world.rank
            print data
            assert False
            
class KickHamiltonian:
    def __init__(self, calc, ext):
        self.ext = ext
        self.vt_sG = [ ext.get_potential(gd=calc.density.gd) ]
        self.dH_asp = {}

        # This code is copy-paste from hamiltonian.update
        for a, D_sp in calc.density.D_asp.items():
            setup = calc.hamiltonian.setups[a]
            vext = ext.get_taylor(spos_c=calc.hamiltonian.spos_ac[a, :])
            # Taylor expansion to the zeroth order
            self.dH_asp[a] = [ vext[0][0] * sqrt(4 * pi) * setup.Delta_pL[:, 0] ]
            if len(vext) > 1:
                # Taylor expansion to the first order
                Delta_p1 = np.array([setup.Delta_pL[:, 1],
                                     setup.Delta_pL[:, 2],
                                     setup.Delta_pL[:, 3]])
                self.dH_asp[a] += sqrt(4 * pi / 3) * np.dot(vext[1], Delta_p1)


class LCAOTDDFT(GPAW):
    def __init__(self, filename=None, propagator_debug=False, propagator='cn', **kwargs):
        GPAW.__init__(self, filename, **kwargs)
        self.propagator_debug = propagator_debug
        self.kick_strength = [0.0, 0.0, 0.0]
        self.tddft_initialized = False

        # XXX Make propagator class
        plist = {'cn': self.linear_propagator, # Doesn't work with blacs yet
                 'taylor': self.taylor_propagator} # Not very good, but works with blacs.
        self.propagator_text = propagator
        self.propagator = plist[self.propagator_text]

        # Restarting from a file
        if filename is not None:
            self.initialize()
            self.set_positions()

    def linear_propagator(self, sourceC_nM, targetC_nM, S_MM, H_MM, dt):
        self.timer.start('Linear solve')
        # XXX Debugging stuff. Remove
        if self.propagator_debug:
            if self.blacs:
                globalH_MM = self.blacs_mm_to_global(H_MM)
                globalS_MM = self.blacs_mm_to_global(S_MM) 
                if world.rank == 0:
                    tri2full(globalS_MM, 'L')
                    tri2full(globalH_MM, 'L')
                    U_MM = dot(inv(globalS_MM-0.5j*globalH_MM*dt), globalS_MM+0.5j*globalH_MM*dt)
                    debugC_nM = dot(sourceC_nM, U_MM.T.conjugate())
                    #print "PASS PROPAGATOR"
                    #debugC_nM = sourceC_nM.copy()
            else:
                if world.rank == 0:
                    U_MM = dot(inv(S_MM-0.5j*H_MM*dt), S_MM+0.5j*H_MM*dt)
                    debugC_nM = dot(sourceC_nM, U_MM.T.conjugate())
                #print "PASS PROPAGATOR"
                #debugC_nM = sourceC_nM.copy()

        if self.blacs:
            target_blockC_nm = self.Cnm_block_descriptor.empty(dtype=complex) # XXX, Preallocate
            temp_blockC_nm = self.Cnm_block_descriptor.empty(dtype=complex) # XXX, Preallocate
            temp_block_mm = self.mm_block_descriptor.empty(dtype=complex)
            if self.density.gd.comm.rank != 0: 
                # XXX Fake blacks nbands, nao, nbands, nao grid because some weird asserts
                # (these are 0,x or x,0 arrays)
                sourceC_nM = self.CnM_unique_descriptor.zeros(dtype=complex)

            # 1. target = (S+0.5j*H*dt) * source
            # Wave functions to target
            self.CnM2nm.redistribute(sourceC_nM, temp_blockC_nm) 

            # XXX It can't be this f'n hard to symmetrize a matrix (tri2full)
            scalapack_zero(self.mm_block_descriptor, H_MM, 'U') # Remove upper diagonal
            temp_block_mm[:] = S_MM - (0.5j*dt) * H_MM  # Lower diagonal matrix
            scalapack_set(self.mm_block_descriptor, temp_block_mm, 0, 0, 'U') # Not it's stricly lower diagonal matrix           
            pblas_tran(-0.5j*dt, H_MM, 1.0, temp_block_mm, self.mm_block_descriptor, self.mm_block_descriptor) # Add transpose of H
            pblas_tran(1.0, S_MM, 1.0, temp_block_mm, self.mm_block_descriptor, self.mm_block_descriptor) # Add transpose of S

            pblas_simple_gemm(self.Cnm_block_descriptor, 
                              self.mm_block_descriptor, 
                              self.Cnm_block_descriptor, 
                              temp_blockC_nm, 
                              temp_block_mm, 
                              target_blockC_nm)
            # 2. target = (S-0.5j*H*dt)^-1 * target
            #temp_block_mm[:] = S_MM + (0.5j*dt) * H_MM
            # XXX It can't be this f'n hard to symmetrize a matrix (tri2full)
            temp_block_mm[:] = S_MM + (0.5j*dt) * H_MM  # Lower diagonal matrix
            scalapack_set(self.mm_block_descriptor, temp_block_mm, 0, 0, 'U') # Not it's stricly lower diagonal matrix           
            pblas_tran(+0.5j*dt, H_MM, 1.0, temp_block_mm, self.mm_block_descriptor, self.mm_block_descriptor) # Add transpose of H
            pblas_tran(1.0, S_MM, 1.0, temp_block_mm, self.mm_block_descriptor, self.mm_block_descriptor) # Add transpose of S

            scalapack_solve(self.mm_block_descriptor, 
                            self.Cnm_block_descriptor, 
                            temp_block_mm,
                            target_blockC_nm)

            if self.density.gd.comm.rank != 0: # XXX is this correct?
                # XXX Fake blacks nbands, nao, nbands, nao grid because some weird asserts
                # (these are 0,x or x,0 arrays)
                target = self.CnM_unique_descriptor.zeros(dtype=complex)
            else:
                target = targetC_nM
            self.Cnm2nM.redistribute(target_blockC_nm, target)
            self.density.gd.comm.broadcast(targetC_nM, 0)
        else:
            # Note: The full equation is conjugated (therefore -+, not +-)
            targetC_nM[:] = solve(S_MM-0.5j*H_MM*dt, np.dot(S_MM+0.5j*H_MM*dt, sourceC_nM.T.conjugate())).T.conjugate()
        
        # XXX Debugging stuff. Remove
        if self.propagator_debug:
             if world.rank == 0:
                 verify(targetC_nM, debugC_nM, "Linear solver propagator vs. reference")

        self.timer.stop('Linear solve')

    def taylor_propagator(self, sourceC_nM, targetC_nM, S_MM, H_MM, dt):
        self.timer.start('Taylor propagator')
        # XXX Debugging stuff. Remove
        if self.propagator_debug:
            if self.blacs:
                globalH_MM = self.blacs_mm_to_global(H_MM)
                globalS_MM = self.blacs_mm_to_global(S_MM) 
                if world.rank == 0:
                    tri2full(globalS_MM, 'L')
                    tri2full(globalH_MM, 'L')
                    U_MM = dot(inv(globalS_MM-0.5j*globalH_MM*dt), globalS_MM+0.5j*globalH_MM*dt)
                    debugC_nM = dot(sourceC_nM, U_MM.T.conjugate())
                    #print "PASS PROPAGATOR"
                    #debugC_nM = sourceC_nM.copy()
            else:
                if world.rank == 0:
                    U_MM = dot(inv(S_MM-0.5j*H_MM*dt), S_MM+0.5j*H_MM*dt)
                    debugC_nM = dot(sourceC_nM, U_MM.T.conjugate())
                #print "PASS PROPAGATOR"
                #debugC_nM = sourceC_nM.copy()

        if self.blacs:
            target_blockC_nm = self.Cnm_block_descriptor.empty(dtype=complex) # XXX, Preallocate
            if self.density.gd.comm.rank != 0: 
                # XXX Fake blacks nbands, nao, nbands, nao grid because some weird asserts
                # (these are 0,x or x,0 arrays)
                sourceC_nM = self.CnM_unique_descriptor.zeros(dtype=complex)

            # Zeroth order taylor to target
            self.CnM2nm.redistribute(sourceC_nM, target_blockC_nm) 

            # XXX, preallocate, optimize use of temporal arrays
            temp_blockC_nm = target_blockC_nm.copy()
            temp2_blockC_nm = target_blockC_nm.copy()

            order = 4
            assert self.wfs.kpt_comm.size == 1
            for n in range(order):
                # Multiply with hamiltonian
                pblas_simple_hemm(self.mm_block_descriptor, 
                                  self.Cnm_block_descriptor, 
                                  self.Cnm_block_descriptor, 
                                  H_MM, 
                                  temp_blockC_nm, 
                                  temp2_blockC_nm, side='R') 
                # XXX: replace with not simple gemm
                temp2_blockC_nm *= -1j*dt/(n+1) 
                # Multiply with inverse overlap
                pblas_simple_hemm(self.mm_block_descriptor, 
                                  self.Cnm_block_descriptor,
                                  self.Cnm_block_descriptor, 
                                  self.wfs.kpt_u[0].invS_MM, # XXX
                                  temp2_blockC_nm, 
                                  temp_blockC_nm, side='R')
                target_blockC_nm += temp_blockC_nm
            if self.density.gd.comm.rank != 0: # Todo: Change to gd.rank
                # XXX Fake blacks nbands, nao, nbands, nao grid because some weird asserts
                # (these are 0,x or x,0 arrays)
                target = self.CnM_unique_descriptor.zeros(dtype=complex)
            else:
                target = targetC_nM
            self.Cnm2nM.redistribute(target_blockC_nm, target)

            self.density.gd.comm.broadcast(targetC_nM, 0)
        else:
            assert self.wfs.kpt_comm.size == 1
            if self.density.gd.comm.rank == 0:
                targetC_nM[:] = sourceC_nM[:]
                tempC_nM = sourceC_nM.copy()
                order = 4
                for n in range(order):
                    tempC_nM[:] = np.dot(self.wfs.kpt_u[0].invS, np.dot(H_MM, 1j*dt/(n+1)*tempC_nM.T.conjugate())).T.conjugate()
                    targetC_nM += tempC_nM
            self.density.gd.comm.broadcast(targetC_nM, 0)
                
        if self.propagator_debug:
             if world.rank == 0:
                 verify(targetC_nM, debugC_nM, "Linear solver propagator vs. reference")

        self.timer.stop('Taylor propagator')

    def absorption_kick(self, strength):
        self.tddft_init()
        self.kick_strength = strength

        # magnitude
        magnitude = np.sqrt(strength[0]*strength[0] 
                             + strength[1]*strength[1] 
                             + strength[2]*strength[2])

        # normalize
        direction = strength / magnitude

        if world.rank == 0:
            print "Applying absorbtion kick"
            print "Magnitude: ", magnitude
            print "Direction: ", direction

        # Create hamiltonian object for absorbtion kick
        kick_hamiltonian = KickHamiltonian(self, ConstantElectricField(magnitude, direction=direction))
        for k, kpt in enumerate(self.wfs.kpt_u):
            Vkick_MM = self.wfs.eigensolver.calculate_hamiltonian_matrix(kick_hamiltonian, self.wfs, kpt, add_kinetic=False, root=-1)
            for i in range(10):
                self.propagator(kpt.C_nM, kpt.C_nM, kpt.S_MM, Vkick_MM, 0.1)

    def blacs_mm_to_global(self, H_mm):
        target = self.MM_descriptor.empty(dtype=complex)
        self.mm2MM.redistribute(H_mm, target)
        world.barrier()
        return target

    def blacs_nm_to_global(self, C_nm):
        target = self.CnM_unique_descriptor.empty(dtype=complex)
        self.Cnm2nM.redistribute(C_nm, target)
        world.barrier()
        return target

    def tddft_init(self):
        if not self.tddft_initialized:
            if world.rank == 0:
                print "Initializing real time LCAO TD-DFT calculation."
                print "XXX Warning: Array use not optimal for memory."
                print "XXX Taylor propagator probably doesn't work"
                print "XXX ...and no arrays are listed in memory estimate yet."
            self.blacs = self.wfs.ksl.using_blacs
            if self.blacs:
                self.ksl = ksl = self.wfs.ksl    
                nao = ksl.nao
                nbands = ksl.bd.nbands
                mynbands = ksl.bd.mynbands
                blocksize = ksl.blocksize

                from gpaw.blacs import Redistributor
                if world.rank == 0:
                    print "BLACS Parallelization"

                # Parallel grid descriptors
                self.MM_descriptor = ksl.blockgrid.new_descriptor(nao, nao, nao, nao) # FOR DEBUG
                self.mm_block_descriptor = ksl.blockgrid.new_descriptor(nao, nao, blocksize, blocksize)
                self.Cnm_block_descriptor = ksl.blockgrid.new_descriptor(nbands, nao, blocksize, blocksize)
                #self.CnM_descriptor = ksl.blockgrid.new_descriptor(nbands, nao, mynbands, nao)
                self.mM_column_descriptor = ksl.single_column_grid.new_descriptor(nao, nao, ksl.naoblocksize, nao)
                self.CnM_unique_descriptor = ksl.single_column_grid.new_descriptor(nbands, nao, mynbands, nao)

                # Redistributors
                self.mm2MM =  Redistributor(ksl.block_comm, self.mm_block_descriptor, self.MM_descriptor) # XXX FOR DEBUG
                self.MM2mm =  Redistributor(ksl.block_comm, self.MM_descriptor, self.mm_block_descriptor) # XXX FOR DEBUG
                self.Cnm2nM = Redistributor(ksl.block_comm, self.Cnm_block_descriptor, self.CnM_unique_descriptor) 
                self.CnM2nm = Redistributor(ksl.block_comm, self.CnM_unique_descriptor, self.Cnm_block_descriptor) 
                self.mM2mm =  Redistributor(ksl.block_comm, self.mM_column_descriptor, self.mm_block_descriptor)

                for kpt in self.wfs.kpt_u:
                    scalapack_zero(self.mm_block_descriptor, kpt.S_MM,'U')
                    scalapack_zero(self.mm_block_descriptor, kpt.T_MM,'U')

                if self.propagator_text == 'taylor' and self.blacs: # XXX to propagator class
                    cholS_mm = self.mm_block_descriptor.empty(dtype=complex)
                    for kpt in self.wfs.kpt_u:
                        kpt.invS_MM = kpt.S_MM.copy()
                        scalapack_inverse(self.mm_block_descriptor, kpt.invS_MM, 'L')
                    if self.propagator_debug:
                        if world.rank == 0:
                            print "XXX Doing serial inversion of overlap matrix."
                        self.timer.start('Invert overlap (serial)')
                        invS2_MM = self.MM_descriptor.empty(dtype=complex)
                        for kpt in self.wfs.kpt_u:
                            #kpt.S_MM[:] = 128.0*(2**world.rank)
                            self.mm2MM.redistribute(self.wfs.S_qMM[kpt.q], invS2_MM)
                            world.barrier()
                            if world.rank == 0:
                                tri2full(invS2_MM,'L')
                                invS2_MM[:] = inv(invS2_MM.copy())
                                self.invS2_MM = invS2_MM
                            kpt.invS2_MM = ksl.mmdescriptor.empty(dtype=complex)
                            self.MM2mm.redistribute(invS2_MM, kpt.invS2_MM)
                            verify(kpt.invS_MM, kpt.invS2_MM, "overlap par. vs. serial", 'L')
                        self.timer.stop('Invert overlap (serial)')
                        if world.rank == 0:
                            print "XXX Overlap inverted."
                if self.propagator_text == 'taylor' and not self.blacs:
                    tmp = inv(self.wfs.kpt_u[0].S_MM)
                    self.wfs.kpt_u[0].invS = tmp

            # Reset the density mixer
            self.density.mixer = DummyMixer()    
            self.tddft_initialized = True
            for k, kpt in enumerate(self.wfs.kpt_u):
                kpt.C2_nM = kpt.C_nM.copy()
                #kpt.firstC_nM = kpt.C_nM.copy()

    def update_projectors(self):
        self.timer.start('LCAO update projectors') 
        # Loop over all k-points
        for k, kpt in enumerate(self.wfs.kpt_u):
            for a, P_ni in kpt.P_ani.items():
                P_ni.fill(117)
                gemm(1.0, kpt.P_aMi[a], kpt.C_nM, 0.0, P_ni, 'n')
        self.timer.stop('LCAO update projectors') 

    def save_wfs(self):
        for k, kpt in enumerate(self.wfs.kpt_u):
            kpt.C2_nM[:] = kpt.C_nM

    def update_hamiltonian(self):
        self.update_projectors()
        self.density.update(self.wfs)
        self.hamiltonian.update(self.density)

    def propagate(self, dt, max_steps, out='lcao.dm'):
        dt *= attosec_to_autime
        self.dt = dt
        
        self.dm_file = paropen(out,'w')
        self.tddft_init()

        header = '# Kick = [%22.12le, %22.12le, %22.12le]\n' \
               % (self.kick_strength[0], self.kick_strength[1], \
                  self.kick_strength[2])
        header += '# %15s %15s %22s %22s %22s\n' \
               % ('time', 'norm', 'dmx', 'dmy', 'dmz')
        self.dm_file.write(header)
        self.dm_file.flush()

        dm0 = None # Initial dipole moment
        time = 0
        steps = 0
        self.timer.start('Propagate')
        while 1:
            dm = self.density.finegd.calculate_dipole_moment(self.density.rhot_g)
            if dm0 is None:
                dm0 = dm
            norm = self.density.finegd.integrate(self.density.rhot_g)
            line = '%20.8lf %20.8le %22.12le %22.12le %22.12le\n' % (time, norm, dm[0], dm[1], dm[2])
            if world.rank == 0:            
                print >>self.dm_file, line,
                print line,
                self.dm_file.flush()

            # ---------------------------------------------------------------------------- 
            # Predictor step
            # ----------------------------------------------------------------------------
            # 1. Calculate H(t)
            self.save_wfs() # kpt.C2_nM = kpt.C_nM
            self.update_hamiltonian()
            # 2. H_MM(t) = <M|H(t)|H>
            #    Solve Psi(t+dt) from (S_MM - 0.5j*H_MM(t)*dt) Psi(t+dt) = (S_MM + 0.5j*H_MM(t)*dt) Psi(t)
            for k, kpt in enumerate(self.wfs.kpt_u):
                kpt.H0_MM = self.wfs.eigensolver.calculate_hamiltonian_matrix(self.hamiltonian, self.wfs, kpt, root=-1)
                self.propagator(kpt.C_nM, kpt.C_nM, kpt.S_MM, kpt.H0_MM, dt)
            # ----------------------------------------------------------------------------
            # Propagator step
            # ----------------------------------------------------------------------------
            # 1. Calculate H(t+dt)
            self.update_hamiltonian()
            # 2. Estimate H(t+0.5*dt) ~ H(t) + H(t+dT)
            for k, kpt in enumerate(self.wfs.kpt_u):
                H_MM = 0.5 * kpt.H0_MM + \
                       0.5 * self.wfs.eigensolver.calculate_hamiltonian_matrix( \
                                 self.hamiltonian, self.wfs, kpt, root=-1)
                # 3. Solve Psi(t+dt) from (S_MM - 0.5j*H_MM(t+0.5*dt)*dt) Psi(t+dt) = (S_MM + 0.5j*H_MM(t+0.5*dt)*dt) Psi(t)
                self.propagator(kpt.C2_nM, kpt.C_nM, kpt.S_MM, H_MM, dt)

            steps += 1
            time += dt
            if steps > max_steps:
                self.timer.stop('Propagate')
                break

            # Call registered callback functions
            self.call_observers(steps)

        self.call_observers(steps, final=True)
        self.dm_file.close()



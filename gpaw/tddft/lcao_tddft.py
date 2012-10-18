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

DEBUG_FULL_INV = False

def verify(data1, data2, id):
    err = sum(abs(data1-data2).ravel()**2)
    if err > 1e-10:
       print "Parallel assert failed: ", id, " norm: ", err
       print "Data from proc ", world.rank
       print "First", data1
       print "Second", data2
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
    def __init__(self, **kwargs):
        GPAW.__init__(self, **kwargs)
        self.kick_strength = [0.0, 0.0, 0.0]
        self.tddft_initialized = False

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

        print "gamma", self.wfs.basis_functions.gamma

        # Create hamiltonian object for absorbtion kick
        kick_hamiltonian = KickHamiltonian(self, ConstantElectricField(magnitude, direction=direction))
        
        Vkick_MM = self.wfs.eigensolver.calculate_hamiltonian_matrix(kick_hamiltonian, self.wfs, self.wfs.kpt_u[0], add_kinetic=False, root=-1)
        mpiverify(Vkick_MM,"Vkick_MM")
  
        # Apply kick
        if DEBUG_FULL_INV:
            Ukick_MM = dot(inv(self.S_MM-0.5j*0.1*Vkick_MM), self.S_MM+0.5j*0.1*Vkick_MM)
            temp_C_nM = self.C_nM.copy()
            for i in range(0,10):
                temp_C_nM = dot(temp_C_nM, Ukick_MM.T)

        for i in range(0,10):
            self.C_nM = solve(self.S_MM-0.5j*0.1*Vkick_MM, np.dot(self.S_MM+0.5j*0.1*Vkick_MM, self.C_nM.T)).T     

        if DEBUG_FULL_INV:
            verify(temp_C_nM, self.C_nM, "Absorbtion propagator")

        mpiverify(self.C_nM, "C_nm after kick")

    def tddft_init(self):
        if not self.tddft_initialized:
             self.density.mixer = DummyMixer()    # Reset the density mixer
             self.S_MM = self.wfs.S_qMM[0]        # Get the overlap matrix
             self.C_nM = self.wfs.kpt_u[0].C_nM   # Get the LCAO matrix
             self.tddft_initialized = True

    def propagate(self, dt, max_steps, out='lcao.dm'):
        dt *= attosec_to_autime
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
            mpiverify(dm, "dipole moment")
            if dm0 is None:
                dm0 = dm
            norm = self.density.finegd.integrate(self.density.rhot_g)
            time += dt
            line = '%20.8lf %20.8le %22.12le %22.12le %22.12le\n' % (time, norm, dm[0], dm[1], dm[2])
            if world.rank == 0:            
                print >>self.dm_file, line,
                print line,
                self.dm_file.flush()

            self.timer.start('LCAO update projectors') 
            # Update projectors, (is this needed?)
            for a, P_ni in self.wfs.kpt_u[0].P_ani.items():
                P_ni.fill(117)
                gemm(1.0, self.wfs.kpt_u[0].P_aMi[a], self.C_nM, 0.0, P_ni, 'n')

            self.timer.stop('LCAO update projectors') 
            self.wfs.kpt_u[0].C_nM[:] = self.C_nM
            mpiverify(self.C_nM, "C_nM first")
            self.density.update(self.wfs)
            self.hamiltonian.update(self.density)


            tempC_nM = self.C_nM.copy()
            H_MM = self.wfs.eigensolver.calculate_hamiltonian_matrix(self.hamiltonian, self.wfs, self.wfs.kpt_u[0], root=-1)

            if DEBUG_FULL_INV:
                U_MM = dot(inv(self.S_MM-0.5j*H_MM*dt), self.S_MM+0.5j*H_MM*dt)
                mpiverify(U_MM, "U_MM first")
                debugC_nM = dot(self.C_nM, U_MM.T)

            self.timer.start('Linear solve')
            self.wfs.kpt_u[0].C_nM = solve(self.S_MM-0.5j*H_MM*dt, np.dot(self.S_MM+0.5j*H_MM*dt, \
                                                                          self.wfs.kpt_u[0].C_nM.T)).T
            self.timer.stop('Linear solve')
            mpiverify(H_MM, "H_MM first")
            if DEBUG_FULL_INV:          
                verify(self.wfs.kpt_u[0].C_nM, debugC_nM, "Predictor propagator")

            for a, P_ni in self.wfs.kpt_u[0].P_ani.items():
                P_ni.fill(117)
                gemm(1.0, self.wfs.kpt_u[0].P_aMi[a], self.wfs.kpt_u[0].C_nM, 0.0, P_ni, 'n')

            self.density.update(self.wfs)
            self.hamiltonian.update(self.density)
            H_MM = 0.5 * H_MM + 0.5 * self.wfs.eigensolver.calculate_hamiltonian_matrix(self.hamiltonian, self.wfs, self.wfs.kpt_u[0], root=-1)
            mpiverify(H_MM, "H_MM first")

            if DEBUG_FULL_INV:
                U_MM = dot(inv(self.S_MM-0.5j*H_MM*dt), self.S_MM+0.5j*H_MM*dt)
                mpiverify(U_MM, "U_MM first")
                debugC_nM = dot(self.C_nM, U_MM.T)

            self.timer.start('Linear solve')
            self.C_nM = solve(self.S_MM-0.5j*H_MM*dt, np.dot(self.S_MM+0.5j*H_MM*dt,
                                                             self.C_nM.T)).T
            self.timer.stop('Linear solve')


            if DEBUG_FULL_INV:          
                verify(debugC_nM, self.C_nM, "Propagator")

            mpiverify(self.C_nM, "C_nM second")
            steps += 1
            if steps > max_steps:
                self.self.timer.stop('Propagate')
                break
        self.dm_file.close()



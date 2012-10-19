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

DEBUG_FULL_INV = True

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
    def __init__(self, filename=None, propagator_debug=True, propagator='numpysolve_CN', **kwargs):
        GPAW.__init__(self, filename, **kwargs)
        self.kick_strength = [0.0, 0.0, 0.0]
        self.tddft_initialized = False
        plist = {'numpysolve_CN': self.linear_propagator}
        self.propagator = plist[propagator]

        # Restarting from a file
        if filename is not None:
            self.initialize()
            self.set_positions()

    def linear_propagator(self, sourceC_nM, targetC_nM, S_MM, H_MM, dt):
        self.timer.start('Linear solve')

        if DEBUG_FULL_INV:
            mpiverify(H_MM, "H_MM first")
            U_MM = dot(inv(S_MM-0.5j*H_MM*dt), S_MM+0.5j*H_MM*dt)
            mpiverify(U_MM, "U_MM first")
            debugC_nM = dot(sourceC_nM, U_MM.T.conjugate())

        targetC_nM[:] = solve(S_MM-0.5j*H_MM*dt, np.dot(S_MM+0.5j*H_MM*dt, sourceC_nM.T.conjugate())).T.conjugate()

        if DEBUG_FULL_INV:
             verify(targetC_nM, debugC_nM, "Linear solver propagator vs. reference")

        self.timer.stop('Linear solve')

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
        for k, kpt in enumerate(self.wfs.kpt_u):
            Vkick_MM = self.wfs.eigensolver.calculate_hamiltonian_matrix(kick_hamiltonian, self.wfs, kpt, add_kinetic=False, root=-1)
            for i in range(10):
                self.propagator(kpt.C_nM, kpt.C_nM, kpt.S_MM, Vkick_MM, 0.1)
            mpiverify(Vkick_MM,"Vkick_MM")
  
    def tddft_init(self):
        if not self.tddft_initialized:
             self.density.mixer = DummyMixer()    # Reset the density mixer
             self.tddft_initialized = True
             for k, kpt in enumerate(self.wfs.kpt_u):
                 kpt.C2_nM = kpt.C_nM.copy()
                 kpt.firstC_nM = kpt.C_nM.copy()


    def update_projectors(self):
        self.timer.start('LCAO update projectors') 
        # Loop over all k-points
        for k, kpt in enumerate(self.wfs.kpt_u):
            for a, P_ni in kpt.P_ani.items():
                P_ni.fill(117)
                gemm(1.0, kpt.P_aMi[a], kpt.C_nM, 0.0, P_ni, 'n')
            mpiverify(kpt.C_nM, "C_nM first")
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
            if steps > max_steps:
                self.timer.stop('Propagate')
                break
        self.dm_file.close()



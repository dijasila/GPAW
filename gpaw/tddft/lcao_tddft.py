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

#de = open('debug.new','w')

class KickHamiltonian:
    def __init__(self, calc, ext):
        self.ext = ext
        self.vt_sG = [ ext.get_potential(gd=calc.density.gd) ]
        #print >> de, "vt_sG", self.vt_sG[0]
        self.dH_asp = {}

        # This code is copy-paste from hamiltonian.update
        for a, D_sp in calc.density.D_asp.items():
            print "Kick proc ", world.rank, " atom ", a
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
        #print >>de, "dH_asp", self.dH_asp


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
        
        Vkick_MM = self.wfs.eigensolver.calculate_hamiltonian_matrix(kick_hamiltonian, self.wfs, self.wfs.kpt_u[0], add_kinetic=False)
        print "V_KICK_MM", Vkick_MM
        #print >>de, "Vkick_MM", Vkick_MM
   
        # Apply kick
        Ukick_MM = dot(inv(self.S_MM-0.5j*0.1*Vkick_MM), self.S_MM+0.5j*0.1*Vkick_MM)
        for i in range(0,10):
            self.C_nM = dot(self.C_nM, Ukick_MM.T)

        
    def tddft_init(self):
        if not self.tddft_initialized:
             self.density.mixer = DummyMixer()    # Reset the density mixer
             self.S_MM = self.wfs.S_qMM[0]        # Get the overlap matrix
             #print >> de, "initS", self.S_MM
             self.C_nM = self.wfs.kpt_u[0].C_nM   # Get the LCAO matrix
             #print >> de, "initC", self.C_nM

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
        while 1:
            dm = self.density.finegd.calculate_dipole_moment(self.density.rhot_g)
            if dm0 is None:
                dm0 = dm
            norm = self.density.finegd.integrate(self.density.rhot_g)
            time += dt
            line = '%20.8lf %20.8le %22.12le %22.12le %22.12le\n' % (time, norm, dm[0], dm[1], dm[2])
            print >>self.dm_file, line,
            print line,
            self.dm_file.flush()


            self.timer.start('LCAO update projectors') 
            # Update projectors, (is this needed?)
            for a, P_ni in self.wfs.kpt_u[0].P_ani.items():
                P_ni.fill(117)
                gemm(1.0, self.wfs.kpt_u[0].P_aMi[a], self.C_nM, 0.0, P_ni, 'n')
                #print >>de, "P_ni", P_ni
            self.timer.stop('LCAO update projectors') 
            self.wfs.kpt_u[0].C_nM[:] = self.C_nM
            self.density.update(self.wfs)
            self.hamiltonian.update(self.density)


            tempC_nM = self.C_nM.copy()
            self.timer.start('LCAO Predictor step') 
            H_MM = self.wfs.eigensolver.calculate_hamiltonian_matrix(self.hamiltonian, self.wfs, self.wfs.kpt_u[0])
            U_MM = dot(inv(self.S_MM-0.5j*H_MM*dt), self.S_MM+0.5j*H_MM*dt)
            self.wfs.kpt_u[0].C_nM = dot(self.wfs.kpt_u[0].C_nM, U_MM.T)
            for a, P_ni in self.wfs.kpt_u[0].P_ani.items():
                P_ni.fill(117)
                gemm(1.0, self.wfs.kpt_u[0].P_aMi[a], self.wfs.kpt_u[0].C_nM, 0.0, P_ni, 'n')
                #print >>de, "P_ni", P_ni
            self.timer.stop('LCAO Predictor step') 
            self.timer.start('Propagate')
            self.density.update(self.wfs)
            self.hamiltonian.update(self.density)
            H_MM = 0.5 * H_MM + 0.5 * self.wfs.eigensolver.calculate_hamiltonian_matrix(self.hamiltonian, self.wfs, self.wfs.kpt_u[0])
            U_MM = dot(inv(self.S_MM-0.5j*H_MM*dt), self.S_MM+0.5j*H_MM*dt)
            self.C_nM = dot(self.C_nM, U_MM.T)
            self.timer.stop('Propagate')
            steps += 1
            if steps > max_steps:
                break
        self.dm_file.close()



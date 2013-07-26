from ase.parallel import parprint
from ase.units import Hartree, Bohr, _eps0, _c, _aut
from gpaw import PoissonConvergenceError
from gpaw.fd_operators import Gradient
#from gpaw.poisson import PoissonSolver
from poisson_corr import PoissonSolver
from gpaw.grid_descriptor import GridDescriptor
from gpaw.transformers import Transformer
from gpaw.io import open as gpaw_io_open
from gpaw.tddft.units import attosec_to_autime, autime_to_attosec
from gpaw.utilities.blas import axpy
from gpaw.utilities.ewald import madelung
from gpaw.utilities.gauss import Gaussian
from gpaw.utilities.tools import construct_reciprocal
from gpaw.utilities import mlsqr
from math import pi
from numpy.fft import fftn, ifftn, fft2, ifft2
from string import split
import _gpaw
import gpaw.mpi as mpi
import numpy as np
import sys
from gpaw.tddft.units import autime_to_attosec
from polarizable_material import *


# in atomic units, 1/(4*pi*e_0) = 1
_eps0_au = 1.0 / (4.0 * np.pi)
_maxL    = 4 # 1 for monopole, 4 for dipole, 9 for quadrupole

# This helps in telling the classical quantitites from the quantum ones
class PoissonOrganizer():
    def __init__(self, PoissonSolver=None):
        self.poissonSolver = PoissonSolver
        self.gd            = None
        self.density       = None

# Inherits the properties normal PoissonSolver, and adds a contribution
# from classical polarization charge density 
class FDTDPoissonSolver(PoissonSolver):
    def __init__(self, nn=3,
                        relax='J',
                        eps=2e-10,
                        classicalMaterial=None,
                        tag='fdtd.poisson',
                        remove_moment=_maxL,
                        debugPlots=False,
                        coupling='both'):
        PoissonSolver.__init__(self)
        if classicalMaterial==None:
            self.classMat = PolarizableMaterial()
        else:
            self.classMat = classicalMaterial
        self.setCalculationMode('solve')
        self.remove_moment = remove_moment
        self.tag      = tag
        self.time     = 0.0
        self.timestep = 0.0
        self.rank     = None
        self.rank     = mpi.rank
        self.dm_file  = None
        self.kick     = None
        self.debugPlots = debugPlots
        assert(coupling in ['none', 'both', 'Cl2Qm', 'Qm2Cl'])
        self.coupling = coupling
        self.maxiter  = 2000
        # From now on, only handle the quantities via self.qm or self.cl 
        self.qm       = PoissonOrganizer(PoissonSolver) # Default solver
        self.cl       = PoissonOrganizer()
    
    def set_grid_descriptor(self, qmgd=None,
                                     clgd=None):
        if qmgd != None:
            self.qm.poissonSolver.set_grid_descriptor(self, qmgd)
        if clgd != None:
            self.qm.poissonSolver.set_grid_descriptor(self, clgd)

    def initializePropagation(self, timestep,
                                       kick,
                                       time = 0.0,
                                       dmClfname='dmCl.dat'):
        self.time      = time
        self.timestep  = timestep * attosec_to_autime
        self.kick      = kick
        self.dmClfname = dmClfname
        
        # dipole moment file
        if self.rank == 0:
            if self.dm_file is not None and not self.dm_file.closed:
                raise RuntimeError('Dipole moment file is already open')
            if self.time == 0.0:
                mode = 'w'
            else:
                mode = 'a'
            self.dm_file = file(self.dmClfname, mode)
            if self.dm_file.tell() == 0:
                header = '# Kick = [%22.12le, %22.12le, %22.12le]\n' % \
                            (self.kick[0], self.kick[1], self.kick[2])
                header += '# %15s %15s %22s %22s %22s\n'             % \
                            ('time', 'norm', 'dmx', 'dmy', 'dmz')
                self.dm_file.write(header)
                self.dm_file.flush()

    def finalizePropagation(self):
        if self.rank == 0:
            self.dm_file.close()
            self.dm_file = None
    
    def setCalculationMode(self, calcMode):
        # Three calculation modes are available:
        #  1) solve:     just solve the Poisson equation with
        #                given quantum+classical rho
        #  2) iterate:   iterate classical density so that the Poisson
        #                equation for quantum+classical rho is satisfied
        #  3) propagate: propagate classical density in time, and solve
        #                the new Poisson equation
        assert(calcMode=='solve' or
               calcMode=='iterate' or
               calcMode=='propagate')
        self.calcMode = calcMode

    # The density object must be attached, so that the electric field
    # from all-electron density can be calculated    
    def setDensity(self, density):
        self.density = density
    
    # Depending on the coupling scheme, the quantum and classical potentials
    # are added in different ways
    def sumPotentials(self, phi):
        if self.coupling in ['both', 'Cl2Qm']:
            phi[:] = np.add(self.qm.phi, self.cl.phi) 
        else:
            phi[:] = self.qm.phi[:]
    
    # Depending on the coupling scheme, the quantum and classical potentials
    # contribute to the electic field in different ways
    def solveCoupledElectricField(self):
        if self.coupling in ['none', 'Cl2Qm']: # field from classical charge
            self.classMat.solveElectricField(self.cl.phi)
        else:                          # field from classical+quantum charge
            self.classMat.solveElectricField(np.add(self.qm.phi, self.cl.phi))
    
    # Just solve it 
    def solve_solve(self, phi,
                            rho,
                            charge=None,
                            eps=None,
                            maxcharge=1e-6,
                            zero_initial_phi=False,
                            calcMode=None):
        self.qm.phi = self.qm.gd.empty()
        self.cl.phi = self.cl.gd.empty()
        self.qm.poissonSolver.solve(self, self.qm.phi,
                                          rho,
                                          charge,
                                          eps,
                                          maxcharge,
                                          zero_initial_phi,
                                          self.remove_moment)
        self.cl.poissonSolver.solve(self.cl.phi,
                                    self.classMat.sign * self.classMat.rhoCl,
                                    charge,
                                    eps,
                                    maxcharge,
                                    zero_initial_phi,
                                    self.remove_moment)
        self.sumPotentials(phi[:])
    
    # Iterate classical and quantum potentials until convergence
    def solve_iterate(self, phi,
                              rho,
                              charge=None,
                              eps=None,
                              maxcharge=1e-6,
                              zero_initial_phi=False,
                              calcMode=None):
        self.qm.phi = self.gd.zeros()
        self.cl.phi = self.cl.gd.zeros()
        self.qm.poissonSolver.solve(self, self.qm.phi,
                                          rho,
                                          charge,
                                          eps,
                                          maxcharge,
                                          zero_initial_phi,
                                          self.remove_moment)
        self.cl.poissonSolver.solve(self.cl.phi,
                                    self.classMat.sign * self.classMat.rhoCl,
                                    charge,
                                    eps,
                                    maxcharge,
                                    zero_initial_phi,
                                    self.remove_moment)
        self.sumPotentials(phi[:])

        oldRho = rho
        poissonClCnt = 0
            
        while True:
            ## field from the potential: use the classical potential if no
            #       coupling from quantum to classical charge is requested 
            self.classMat.solveElectricField(phi)     # E = -Div[Vh]

            ## Polarizations P0_j and Ptot
            self.classMat.solvePolarizations()        # P = (eps - eps0)E

            ## Classical charge density
            self.classMat.solveRho()                  # n = -Grad[P]
                
            ## Update electrostatic potential         # nabla^2 Vh = -4*pi*n
            self.qm.poissonSolver.solve(self, self.qm.phi,
                                              rho,
                                              charge,
                                              eps,
                                              maxcharge,
                                              zero_initial_phi,
                                              self.remove_moment)
            self.cl.poissonSolver.solve(self.cl.phi,
                                      self.classMat.sign*self.classMat.rhoCl,
                                      charge,
                                      eps,
                                      maxcharge,
                                      zero_initial_phi,
                                      self.remove_moment)
            self.sumPotentials(phi[:])
            
            # Mix potential
            try:
                self.mixPhi
            except:
                self.mixPhi = simpleMixer(0.10, phi)

            phi = self.mixPhi.mix(phi)
                
            ## Check convergence
            poissonClCnt = poissonClCnt + 1
            dRho = self.gd.integrate(
                abs(rho - oldRho + self.classMat.sign * self.classMat.rhoCl))
                
            if(abs(dRho)<1e-3):
                break
            oldRho = rho + self.classMat.sign * self.classMat.rhoCl

        if True:
            parprint("Poisson equation solved in %i iterations" \
                     " for the classical part" % (poissonClCnt))
            
    def solve_propagate(self, phi,
                                rho,
                                charge=None,
                                eps=None,
                                maxcharge=1e-6,
                                zero_initial_phi=False,
                                calcMode=None):
        if self.debugPlots and np.floor(self.time/self.timestep) % 50 == 0:
                self.plotThings(phi, rho)

        # 1) P(t) from P(t-dt) and J(t-dt/2)
        self.classMat.propagatePolarizations(self.timestep)
                
        # 2) n(t) from P(t)
        self.classMat.solveRho()
                
        # 3a) V(t) from n(t)
        self.qm.phi = self.qm.gd.zeros()
        self.cl.phi = self.cl.gd.zeros()
        self.qm.poissonSolver.solve(self, self.qm.phi,
                                          rho,
                                          charge,
                                          eps,
                                          maxcharge,
                                          zero_initial_phi,
                                          self.remove_moment)
        self.cl.poissonSolver.solve(self.cl.phi,
                                    self.classMat.sign * self.classMat.rhoCl,
                                    charge,
                                    eps,
                                    maxcharge,
                                    zero_initial_phi,
                                    self.remove_moment)
        self.sumPotentials(phi[:])
                
        # 4a) E(r) from V(t):      E = -Div[Vh]
        self.solveCoupledElectricField()
                
        # 4b) Apply the kick by changing the electric field
        if self.time==0:
            self.classMat.kickElectricField(self.timestep, self.kick)
                    
        # 5) J(t+dt/2) from J(t-dt/2) and P(t)
        self.classMat.propagateCurrents(self.timestep)

        # Write updated dipole moment into file
        self.updateDipoleMomentFile(rho)
                
        # Update timer
        self.time = self.time + self.timestep
                
        # Do not propagate before the next time step
        self.setCalculationMode('solve')             
                                

    def solve(self, phi,
                     rho,
                     charge=None,
                     eps=None,
                     maxcharge=1e-6,
                     zero_initial_phi=False,
                     calcMode=None):
        
        self.qm.phi = None
        self.cl.phi = None
        
        # if requested, switch to new calculation mode
        if calcMode != None:
            self.dm_file = file('dmCl.%s.dat' % (self.tag), mode)
            if self.dm_file.tell() == 0:
                header = '# Kick = [%22.12le, %22.12le, %22.12le]\n' % \
                                   (self.kick[0], self.kick[1], self.kick[2])
                header += '# %15s %15s %22s %22s %22s\n'             % \
                                   ('time', 'norm', 'dmx', 'dmy', 'dmz')
                self.dm_file.write(header)
                self.dm_file.flush()

        if self.density == None:
            print 'FDTDPoissonSolver requires a density object.' \
                  ' Use setDensity routine to initialize it.'
            raise

        # assign grid descriptor
        if not self.classMat.initialized:
            self.qm.gd = self.gd
            self.cl.gd = GridDescriptor(self.qm.gd.N_c,
                                        self.qm.gd.cell_cv,
                                        self.qm.gd.pbc_c,
                                        self.qm.gd.comm,
                                        self.qm.gd.parsize_c)
            self.cl.poissonSolver = PoissonSolver()
            self.cl.poissonSolver.set_grid_descriptor(self.cl.gd)
            self.cl.poissonSolver.initialize()
            self.classMat.initialize(self.cl.gd)
            self.mixPhi = simpleMixer(0.10, phi)
            
        if(self.calcMode=='solve'): # do not modify the polarizable material
            self.solve_solve(phi,
                             rho,
                             charge=None,
                             eps=None,
                             maxcharge=1e-6,
                             zero_initial_phi=False,
                             calcMode=None)

        elif(self.calcMode=='iterate'): # find self-consistent density
            self.solve_iterate(phi,
                               rho,
                               charge=None,
                               eps=None,
                               maxcharge=1e-6,
                               zero_initial_phi=False,
                               calcMode=None)
        
        elif(self.calcMode=='propagate'): # propagate one time step
            self.solve_propagate(phi,
                                 rho,
                                 charge=None,
                                 eps=None,
                                 maxcharge=1e-6,
                                 zero_initial_phi=False,
                                 calcMode=None)
                
    def updateDipoleMomentFile(self, rho):
        
        norm = self.gd.integrate( rho + 
                                  self.classMat.sign * self.classMat.rhoCl )
        # The sign (-1) comes from the sign of electron charge
        dm=self.density.finegd.calculate_dipole_moment(self.density.rhot_g) \
        +[-1.0*self.classMat.sign*self.gd.integrate(self.classMat.pTotal[0]),
          -1.0*self.classMat.sign*self.gd.integrate(self.classMat.pTotal[1]),
          -1.0*self.classMat.sign*self.gd.integrate(self.classMat.pTotal[2])]
        if self.rank == 0:
            line = '%20.8lf %20.8le %22.12le %22.12le %22.12le\n' \
                 % (self.time, norm, dm[0], dm[1], dm[2])
            self.dm_file.write(line)
            self.dm_file.flush()

    def read(self, paw,
                    filename = 'poisson'):
        
        world     = paw.wfs.world
        master    = (world.rank == 0)
        parallel  = (world.size > 1)
        self.rank = paw.wfs.world.rank

        r = gpaw_io_open(filename, 'r', world)

        version = r['version']
        parprint('reading poisson gpw-file... version: %f', version)
        print self.qm.poissonSolver is None
        self.classMat.initialize(self.qm.gd)
        
        # Read self.classMat.rhoCl
        if self.gd.comm.rank == 0:
                big_rhoCl = np.array(r.get('ClassicalMaterialRho'),
                                     dtype=float)
        else:
                big_rhoCl = None
        self.gd.distribute(big_rhoCl,
                           self.classMat.rhoCl)
                
        # Read self.classMat.pTotal
        if self.gd.comm.rank == 0:
                big_pTotal = np.array(r.get('pTotal'),
                                      dtype=float)
        else:
                big_pTotal = None
        self.gd.distribute(big_pTotal,
                           self.classMat.pTotal)
        
        # Read self.classMat.polarizations
        if self.gd.comm.rank == 0:
                big_polarizations = np.array(r.get('polarizations'),
                                             dtype=float)
        else:
                big_polarizations = None
        self.gd.distribute(big_polarizations,
                           self.classMat.polarizations)
        
        r.close
         
    def write(self, paw,
                     filename = 'poisson'):
        #parprint('Writing FDTDPoissonSolver data to %s' % (filename))
        rho         = self.classMat.rhoCl
        world       = paw.wfs.world
        domain_comm = paw.wfs.gd.comm
        kpt_comm    = paw.wfs.kpt_comm
        band_comm   = paw.wfs.band_comm
    
        master = (world.rank == 0)
        parallel = (world.size > 1)
        
        w = gpaw_io_open(filename, 'w', world)
        w['history'] = 'FDTDPoissonSolver restart file'
        w['version'] = 1
        w['lengthunit'] = 'Bohr'
        w['energyunit'] = 'Hartree'
        w['DataType'] = 'Float'
        
        # Create dimensions for various netCDF variables:
        ng = self.gd.get_size_of_global_array()
        
        # Write the classical charge density
        w.dimension('ngptsx', ng[0])
        w.dimension('ngptsy', ng[1])
        w.dimension('ngptsz', ng[2])
        w.add('ClassicalMaterialRho',
              ('ngptsx', 'ngptsy', 'ngptsz'),
              dtype=float,
              write=master)
        if kpt_comm.rank == 0:
            rhoCl = self.gd.collect(self.classMat.rhoCl)
            if master:
                w.fill(rhoCl)

        # Write the total polarization
        w.dimension('3', 3)
        w.dimension('ngptsx', ng[0])
        w.dimension('ngptsy', ng[1])
        w.dimension('ngptsz', ng[2])
        w.add('pTotal',
              ('3', 'ngptsx', 'ngptsy', 'ngptsz'),
              dtype=float,
              write=master)
        if kpt_comm.rank == 0:
            pTotal = self.gd.collect(self.classMat.pTotal)
            if master:
                w.fill(pTotal)

        # Write the partial polarizations
        w.dimension('3', 3)
        w.dimension('Nj', self.classMat.Nj)
        w.dimension('ngptsx', ng[0])
        w.dimension('ngptsy', ng[1])
        w.dimension('ngptsz', ng[2])
        w.add('polarizations',
              ('3', 'Nj', 'ngptsx', 'ngptsy', 'ngptsz'),
              dtype=float,
              write=master)
        if kpt_comm.rank == 0:
            polarizations = self.gd.collect(self.classMat.polarizations)
            if master:
                w.fill(polarizations)

        w.close()
        world.barrier()
        


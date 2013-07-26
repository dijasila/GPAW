from ase.parallel import parprint
from ase.units import Hartree, Bohr, _eps0, _c, _aut
from gpaw.utilities.gauss import Gaussian

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
#from gpaw.utilities.gauss import Gaussian
from gpaw.utilities.gpts import get_number_of_grid_points
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
from gpaw.mpi import world
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
        self.cell          = None
        self.spacing_def   = None
        self.spacing       = None

# Inherits the properties normal PoissonSolver, and adds a contribution
# from classical polarization charge density 
class FDTDPoissonSolver(PoissonSolver):
    def __init__(self, nn=3,
                        relax='J',
                        eps=2e-10,
                        classicalMaterial=None,
                        cell=None,
                        cl_spacing=1.20,
                        qm_spacing=0.30,
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
        self.cl             = PoissonOrganizer()        
        self.cl.spacing_def = cl_spacing*np.ones(3)/Bohr
        self.qm             = PoissonOrganizer(PoissonSolver) # Default solver
        self.qm.spacing_def = qm_spacing*np.ones(3)/Bohr
        self.qm.cell        = np.array(cell)/Bohr
        
        # Create grid descriptor for the classical part
        _cell = np.array(cell) / Bohr
        self.cl.spacing = self.cl.spacing_def
        if np.size(_cell)==3:
            self.cl.cell = np.diag(_cell)
        else:
            self.cl.cell = _cell

        N_c = get_number_of_grid_points(self.cl.cell, self.cl.spacing)
        self.cl.spacing = np.diag(self.cl.cell)/N_c
        self.cl.gd = GridDescriptor(N_c, self.cl.cell, False, world, None)

    def set_grid_descriptor(self, qmgd=None,
                                     clgd=None):
        if qmgd != None:
            self.qm.poissonSolver.set_grid_descriptor(self, qmgd)

        if clgd != None:
            self.cl.poissonSolver.set_grid_descriptor(self, clgd)

    # the quantum simulation cell is determined by the two corners v1 and v2
    def cut_cell(self, atoms_in, _v1, _v2):
        qmh = self.qm.spacing_def
        v1  = np.array(_v1)/Bohr
        v2  = np.array(_v2)/Bohr
        
        # Sanity check: quantum box must be inside the classical one
        assert(all([v1[w] <= v2[w] and
                    v1[w] >= 0 and
                    v2[w] <= np.diag(self.cl.cell)[w] for w in range(3)]))
        
        # Create new Atoms object
        atoms_out = atoms_in.copy()
        
        # Quantum grid is probably not yet created
        if not self.qm.gd:
            self.qm.cell = np.zeros((3,3))
            for w in range(3):
                self.qm.cell[w,w] = v2[w]-v1[w]
        
            N_c = get_number_of_grid_points(self.qm.cell, qmh)
            self.qm.spacing = np.diag(self.qm.cell)/N_c
        else:
            self.qm.cell = self.qm.gd.cell_cv
            N_c = self.qm.gd.N_c
            self.qm.spacing = self.qm.gd.get_grid_spacings()
        
        # Ratios of the user-given spacings
        hratios = self.cl.spacing_def/qmh
        self.num_refinements = 1+int(round(np.log(hratios[0])/np.log(2.0)))
        assert([int(round(np.log(hratios[w])/np.log(2.0)))==self.num_refinements for w in range(3)])

        # Classical corner indices must be divisable with numb
        if any(self.cl.spacing/self.qm.spacing >= 3):
            numb=1
        elif any(self.cl.spacing/self.qm.spacing >= 2):
            numb = 2
        else:
            numb = 4
        
        # the index mismatch of the two simulation cells
        self.num_indices = numb * np.ceil((np.array(v2)-np.array(v1))/self.cl.spacing/numb)
        
        self.num_indices_1 = numb * np.floor(np.array(v1)/self.cl.spacing/numb)
        self.num_indices_2 = numb * np.ceil(np.array(v2)/self.cl.spacing/numb)
        self.num_indices = self.num_indices_2-self.num_indices_1
        
        # center, left, and right points of the suggested quantum grid
        cp = 0.5*(np.array(v1)+np.array(v2))
        lp = cp - 0.5*self.num_indices*self.cl.spacing 
        rp = cp + 0.5*self.num_indices*self.cl.spacing
                
        # indices in the classical grid restricting the quantum grid
        self.shift_indices_1 = np.floor(lp/self.cl.spacing)
        self.shift_indices_2 = self.shift_indices_1 + self.num_indices

        # sanity checks
        assert(all([self.shift_indices_1[w] >= 0 and
                    self.shift_indices_2[w] <= self.cl.gd.N_c[w] for w in range(3)])), \
                    "Could not find appropriate quantum grid. Maybe you can move it further away from the boundary."
        
        # corner coordinates
        self.qm.corner1 = self.shift_indices_1 * self.cl.spacing
        self.qm.corner2 = self.shift_indices_2 * self.cl.spacing
               
        # new quantum grid
        for w in range(3):
            self.qm.cell[w,w] = (self.shift_indices_2[w]-self.shift_indices_1[w])*self.cl.spacing[w]
        self.qm.spacing = self.cl.spacing/hratios
        N_c = get_number_of_grid_points(self.qm.cell, self.qm.spacing)
 
        atoms_out.set_cell(np.diag(self.qm.cell)*Bohr)
        atoms_out.positions = atoms_in.get_positions() - self.qm.corner1*Bohr
        
        parprint("Quantum box readjustment:")
        parprint("  Given cell/atomic coordinates:");
        parprint("             [%10.5f %10.5f %10.5f]" % tuple(np.diag(atoms_in.get_cell())))
        for s,c in zip(atoms_in.get_chemical_symbols(), atoms_in.get_positions()):
            parprint("           %s %10.5f %10.5f %10.5f" % (s, c[0], c[1], c[2]))
        parprint("  Readjusted cell/atomic coordinates:");
        parprint("             [%10.5f %10.5f %10.5f]" % tuple(np.diag(atoms_out.get_cell())))
        for s,c in zip(atoms_out.get_chemical_symbols(), atoms_out.get_positions()):
            parprint("           %s %10.5f %10.5f %10.5f" % (s, c[0], c[1], c[2]))
        
        parprint("  Given corner points:       (%10.5f %10.5f %10.5f) - (%10.5f %10.5f %10.5f)" % (tuple(np.concatenate((v1, v2))*Bohr)))
        parprint("  Readjusted corner points:  (%10.5f %10.5f %10.5f) - (%10.5f %10.5f %10.5f)" % (tuple(np.concatenate((self.qm.corner1, self.qm.corner2))*Bohr)))
        parprint("  Indices in classical grid: (%10i %10i %10i) - (%10i %10i %10i)" % (tuple(np.concatenate((self.shift_indices_1, self.shift_indices_2)))))
        parprint("  Grid points in classical grid: (%10i %10i %10i)" % (tuple(self.cl.gd.N_c)))
        parprint("  Grid points in quantum grid:   (%10i %10i %10i)" % (tuple(N_c)))
        
        parprint("  Spacings in quantum grid:    (%10.5f %10.5f %10.5f)" % (tuple(np.diag(self.qm.cell)*Bohr/N_c)))
        parprint("  Spacings in classical grid:  (%10.5f %10.5f %10.5f)" % (tuple(np.diag(self.cl.cell)*Bohr/get_number_of_grid_points(self.cl.cell, self.cl.spacing))))
        parprint("  Ratios of cl/qm spacings:    (%10i %10i %10i)" % (tuple(hratios)))
        parprint("                             = (%10.2f %10.2f %10.2f)" % (tuple((np.diag(self.cl.cell)*Bohr/get_number_of_grid_points(self.cl.cell, self.cl.spacing))/(np.diag(self.qm.cell)*Bohr/N_c))))
        parprint("  Needed number of refinements: %10i" % self.num_refinements)

        #   First, create the quantum grid equivalent griddescriptor object self.cl.subgd.
        #   Then coarsen it until its h_cv equals that of self.cl.gd.
        #   Finally, map the points between clgd and coarsened subgrid.
        
        subcell_cv = np.diag(self.qm.corner2-self.qm.corner1)
        N_c = get_number_of_grid_points(subcell_cv, self.cl.spacing)
        N_c = self.shift_indices_2 - self.shift_indices_1
        self.cl.subgds = []
        self.cl.subgds.append(GridDescriptor(N_c, subcell_cv, False, world, None))

        parprint("  N_c/spacing of the subgrid:           %3i %3i %3i / %.4f %.4f %.4f" %
                  (self.cl.subgds[0].N_c[0], self.cl.subgds[0].N_c[1], self.cl.subgds[0].N_c[2],
                   self.cl.subgds[0].h_cv[0][0]*Bohr, self.cl.subgds[0].h_cv[1][1]*Bohr, self.cl.subgds[0].h_cv[2][2]*Bohr))
        parprint("  shape from the subgrid:           %3i %3i %3i" % (tuple(self.cl.subgds[0].empty().shape)))

        self.cl.coarseners = []
        self.cl.refiners = []
        for n in range(self.num_refinements):
            self.cl.subgds.append(self.cl.subgds[n].refine())
            self.cl.refiners.append(Transformer(self.cl.subgds[n], self.cl.subgds[n+1]))
            
            parprint("  refiners[%i] can perform the transformation (%3i %3i %3i) -> (%3i %3i %3i)" % (\
                     n,
                     self.cl.subgds[n].empty().shape[0], self.cl.subgds[n].empty().shape[1], self.cl.subgds[n].empty().shape[2],
                     self.cl.subgds[n+1].empty().shape[0], self.cl.subgds[n+1].empty().shape[1], self.cl.subgds[n+1].empty().shape[2]))
            self.cl.coarseners.append(Transformer(self.cl.subgds[n+1], self.cl.subgds[n]))
        self.cl.coarseners[:] = self.cl.coarseners[::-1]
        
        # Now extend the grid in order to handle the zero boundary conditions that the refiner assumes
        # The default interpolation order
        self.extend_nn = Transformer(GridDescriptor([8, 8, 8], [1, 1, 1], False, world, None),
                                     GridDescriptor([8, 8, 8], [1, 1, 1], False, world, None).coarsen()).nn
        
        #if self.extend_nn<2:
        #    self.extended_num_indices = self.num_indices+4*self.extend_nn
        #elif self.extend_nn<4:
        #    self.extended_num_indices = self.num_indices+2*self.extend_nn
        #else:
        #    self.extended_num_indices = self.num_indices+self.extend_nn
        self.extended_num_indices = self.num_indices+[2, 2, 2]
        
        # center, left, and right points of the suggested quantum grid
        extended_cp = 0.5*(np.array(v1)+np.array(v2))
        extended_lp = extended_cp - 0.5*(self.extended_num_indices) * self.cl.spacing 
        extended_rp = extended_cp + 0.5*(self.extended_num_indices) * self.cl.spacing
        
        # indices in the classical grid restricting the quantum grid
        self.extended_shift_indices_1 = np.floor(extended_lp/self.cl.spacing)
        self.extended_shift_indices_2 = self.extended_shift_indices_1 + self.extended_num_indices

        # sanity checks
        assert(all([self.extended_shift_indices_1[w] >= 0 and
                    self.extended_shift_indices_2[w] <= self.cl.gd.N_c[w] for w in range(3)])), \
                    "Could not find appropriate quantum grid. Maybe you can move it further away from the boundary."
        
        # corner coordinates
        self.qm.extended_corner1 = self.extended_shift_indices_1 * self.cl.spacing
        self.qm.extended_corner2 = self.extended_shift_indices_2 * self.cl.spacing
        N_c = self.extended_shift_indices_2 - self.extended_shift_indices_1
               
        self.cl.extended_subgds = []
        self.cl.extended_refiners = []
        extended_subcell_cv = np.diag(self.qm.extended_corner2 - self.qm.extended_corner1)

        self.cl.extended_subgds.append(GridDescriptor(N_c, extended_subcell_cv, False, world, None))

        for n in range(self.num_refinements):
            self.cl.extended_subgds.append(self.cl.extended_subgds[n].refine())
            self.cl.extended_refiners.append(Transformer(self.cl.extended_subgds[n], self.cl.extended_subgds[n+1]))
            parprint("  extended_refiners[%i] can perform the transformation (%3i %3i %3i) -> (%3i %3i %3i)" % (\
                     n,
                     self.cl.extended_subgds[n].empty().shape[0], self.cl.extended_subgds[n].empty().shape[1], self.cl.extended_subgds[n].empty().shape[2],
                     self.cl.extended_subgds[n+1].empty().shape[0], self.cl.extended_subgds[n+1].empty().shape[1], self.cl.extended_subgds[n+1].empty().shape[2]))
        
        parprint("  N_c/spacing of the refined subgrid:   %3i %3i %3i / %.4f %.4f %.4f" %
                  (self.cl.subgds[-1].N_c[0], self.cl.subgds[-1].N_c[1], self.cl.subgds[-1].N_c[2],
                  self.cl.subgds[-1].h_cv[0][0]*Bohr, self.cl.subgds[-1].h_cv[1][1]*Bohr, self.cl.subgds[-1].h_cv[2][2]*Bohr))
        parprint("  shape from the refined subgrid:       %3i %3i %3i" % (tuple(self.cl.subgds[-1].empty().shape)))
        
        self.extended_deltaIndex = 2**(self.num_refinements)*self.extend_nn
        parprint("self.extended_deltaIndex = %i" % self.extended_deltaIndex)
        
        qgpts = self.cl.subgds[-2].N_c
    
        # Assure that one returns to the original shape
        dmygd = self.cl.subgds[-1].coarsen()
        for n in range(self.num_refinements-1):
            dmygd = dmygd.coarsen()
        
        parprint("  N_c/spacing of the coarsened subgrid: %3i %3i %3i / %.4f %.4f %.4f" %
                  (dmygd.N_c[0], dmygd.N_c[1], dmygd.N_c[2],
                   dmygd.h_cv[0][0]*Bohr, dmygd.h_cv[1][1]*Bohr, dmygd.h_cv[2][2]*Bohr))

        return atoms_out, self.qm.spacing[0]*Bohr, qgpts
                
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
    def sumPotentials(self, phi, moments, doplots=False):

        if self.coupling in ['both', 'Cl2Qm']:

            # 1) Store the quantum grid for later use
            #print 'Phase 1: Store the quantum grid for later use'
            qm_phi_copy = self.qm.phi.copy()

            # 2) From classical to quantum grid:
            #    Create suitable subgrid of the self.cl.gd, whose refined version
            #    yields the quantum grid. During the refinement, interpolate the
            #    values, and finally copy the interpolated values to the quantum grid.
            #print 'Phase 2: From classical to quantum grid'
            #cl_phi_copy = self.cl.phi[self.shift_indices_1[0]:self.shift_indices_2[0]-1,
            #                          self.shift_indices_1[1]:self.shift_indices_2[1]-1,
            #                          self.shift_indices_1[2]:self.shift_indices_2[2]-1].copy()
            cl_phi_copy2 = self.cl.phi[self.extended_shift_indices_1[0]:self.extended_shift_indices_2[0]-1,
                                       self.extended_shift_indices_1[1]:self.extended_shift_indices_2[1]-1,
                                       self.extended_shift_indices_1[2]:self.extended_shift_indices_2[2]-1].copy()
            
            if doplots:
                import matplotlib.pyplot as plt         
                
                plt.clf()
                p = cl_phi_copy
                n = p.shape
                #r = self.qm.gd.get_grid_point_coordinates().transpose((1, 2, 3, 0))+self.qm.corner1
                plt.plot(np.linspace(0, 1.0, n[0]), p[:, n[1]/2, n[2]/2], 'o-', label = 'cl.phi.copy (0) / %i %i %i' % ( \
                                    cl_phi_copy.shape[0], cl_phi_copy.shape[1], cl_phi_copy.shape[2]))
                for nn in range(self.num_refinements):
                    cl_phi_copy = self.cl.refiners[nn].apply(cl_phi_copy)
                    p = cl_phi_copy
                    n = p.shape
                    plt.plot(np.linspace(0, 1.0, n[0]), p[:, n[1]/2, n[2]/2], 'o-', label = 'cl.phi.copy (%i) / %i %i %i' % (nn+1,
                                    cl_phi_copy.shape[0], cl_phi_copy.shape[1], cl_phi_copy.shape[2]))
            
                p = cl_phi_copy2[self.extended_deltaIndex:-self.extended_deltaIndex]
                n = p.shape
                #r = self.qm.gd.get_grid_point_coordinates().transpose((1, 2, 3, 0))+self.qm.corner1
                plt.plot(np.linspace(0, 1.0, n[0]), p[:, n[1]/2, n[2]/2], 'o-', label = 'cl.phi.copy (0) / %i %i %i' % ( \
                                    cl_phi_copy2.shape[0],
                                    cl_phi_copy2.shape[1],
                                    cl_phi_copy2.shape[2]))
                for nn in range(self.num_refinements):
                    cl_phi_copy2 = self.cl.extended_refiners[nn].apply(cl_phi_copy2)
                    p = cl_phi_copy2
                    n = p.shape
                    plt.plot(np.linspace(0, 1.0, n[0]), p[:, n[1]/2, n[2]/2], 'o-', label = 'cl.phi.copy (%i) / %i %i %i' % (nn+1,
                                    cl_phi_copy2.shape[0],
                                    cl_phi_copy2.shape[1],
                                    cl_phi_copy2.shape[2]))
                plt.legend()
                plt.show()
            else:
                #for n in range(self.num_refinements):
                #    cl_phi_copy = self.cl.refiners[n].apply(cl_phi_copy)
                    
                for n in range(self.num_refinements):
                    cl_phi_copy2 = self.cl.extended_refiners[n].apply(cl_phi_copy2)
            
            #self.qm.phi[:] += cl_phi_copy[:]
            self.qm.phi[:] += cl_phi_copy2[self.extended_deltaIndex:-self.extended_deltaIndex,
                                           self.extended_deltaIndex:-self.extended_deltaIndex,
                                           self.extended_deltaIndex:-self.extended_deltaIndex
                                           ]
            
            
            # 3a) From quantum to classical grid outside the overlapping region:
            #     Calculate the multipole moments of the quantum density, then
            #     determine the potential that they generate at the outside points.
            #print 'Phase 3a: From quantum to classical grid outside the overlapping region'
            if moments==None:
                # calculate the multipole moments from self.rho?
                pass
            else:
                center = self.qm.corner1 + 0.5*self.qm.gd.cell_cv.sum(0)
                p = np.sum(np.array([m*Gaussian(self.cl.gd, center=center).get_gauss_pot(l) for m,l in zip(moments, range(self.remove_moment))]), axis=0)
                p[self.shift_indices_1[0]:self.shift_indices_2[0]-1,
                  self.shift_indices_1[1]:self.shift_indices_2[1]-1,
                  self.shift_indices_1[2]:self.shift_indices_2[2]-1] = 0.0
                self.cl.phi[:] += p[:]
	                
            # 3b) From quantum to classical grid inside the overlapping region:
            #     The quantum values in qm_phi_copy (i.e. the values before the
            #     classical values were added) are now added to phi_cl at those
            #     points that are common to both grids.
            #print 'Phase 3b: From quantum to classical grid inside the overlapping region'
            for n in range(self.num_refinements):
                qm_phi_copy = self.cl.coarseners[n].apply(qm_phi_copy)
                
            self.cl.phi[self.shift_indices_1[0]:self.shift_indices_2[0]-1,
                        self.shift_indices_1[1]:self.shift_indices_2[1]-1,
                        self.shift_indices_1[2]:self.shift_indices_2[2]-1] += qm_phi_copy[:]
                                        
            phi[:] = self.qm.phi[:]
        else:
            phi[:] = self.qm.phi[:]
        
        return qm_phi_copy, cl_phi_copy2[self.extended_deltaIndex:-self.extended_deltaIndex,
                                         self.extended_deltaIndex:-self.extended_deltaIndex,
                                         self.extended_deltaIndex:-self.extended_deltaIndex]
    
    # Depending on the coupling scheme, the quantum and classical potentials
    # contribute to the electic field in different ways
    def solveCoupledElectricField(self):
        if self.coupling in ['none', 'Cl2Qm']: # field from classical charge
            self.classMat.solveElectricField(self.cl.phi)
        else:                          # field from classical+quantum charge
            #new_phi = self.qm.gd.zeros()
            #self.sumPotentials(new_phi) #np.add(self.qm.phi, self.cl.phi)
            #self.classMat.solveElectricField(new_phi)
            self.classMat.solveElectricField(self.cl.phi)
    
    # Just solve it 
    def solve_solve(self, phi,
                            rho,
                            charge=None,
                            eps=None,
                            maxcharge=1e-6,
                            zero_initial_phi=False,
                            calcMode=None):
        self.qm.phi = phi.copy() #self.qm.gd.empty()
        self.cl.phi = self.cl.gd.empty()
        niter, moments = self.qm.poissonSolver.solve(self, self.qm.phi,
                                                     rho,
                                                     charge,
                                                     eps,
                                                     maxcharge,
                                                     zero_initial_phi,
                                                     self.remove_moment)
        self.cl.poissonSolver.solve(self.cl.phi,
                                    self.classMat.sign * self.classMat.rhoCl,
                                    0.0, #charge,
                                    eps,
                                    0.0, #maxcharge,
                                    zero_initial_phi,
                                    self.remove_moment)
        self.sumPotentials(phi[:], moments)
    
    # Iterate classical and quantum potentials until convergence
    def solve_iterate(self, phi,
                              rho,
                              charge=None,
                              eps=None,
                              maxcharge=1e-6,
                              zero_initial_phi=False,
                              calcMode=None):
        self.qm.phi = phi #self.gd.zeros()
        try:
            self.cl.phi
        except:
            print 'not initialized'
            self.cl.phi = self.cl.gd.zeros()
        
        if False: # artificial rhos for debugging how they add up 
            import matplotlib.pyplot as plt
            r = self.qm.gd.get_grid_point_coordinates().transpose((1, 2, 3, 0)) + self.qm.corner1
            n = rho.shape
            cc = r[n[0]/2, n[1]/2, n[2]/2] #+ self.qm.corner1
            #self.qm.phi = (r[:,:,:,0]-cc[0]) * np.exp(-0.10*np.sum((r[:,:,:,:]-cc) * (r[:,:,:,:]-cc), axis=3))  # ~ x*exp(-r^2)
            
            if True:
                #rho = np.sum(np.array([0.1*float(l)*Gaussian(self.qm.gd).get_gauss(l) for l in np.arange(0, 10, 1)]), axis=0)
                #rho = -0.25*(r[:,:,:,0]-cc[0]) * np.cos(5.0/(0.50+np.sum((r[:,:,:,:]-cc) * (r[:,:,:,:]-cc), axis=3))) #* \
                #    #np.exp(-0.05*np.sum((r[:,:,:,:]-cc) * (r[:,:,:,:]-cc), axis=3)) # ~ x*sin(-r^2) * exp(-r^2)
                rho = np.exp(-0.05*np.sum((r[:,:,:,:]-cc) * (r[:,:,:,:]-cc), axis=3)) # ~ x*sin(-r^2) * exp(-r^2)
                p = rho
                n = p.shape
                r = self.qm.gd.get_grid_point_coordinates().transpose((1, 2, 3, 0))  + self.qm.corner1

                plt.plot(r[:, 0,0,0], p[:, n[1]/2, n[2]/2], 'bo-', label = 'rho*10')
                
                niter, moments = self.qm.poissonSolver.solve(self,
                                                             self.qm.phi,
                                                             rho,
                                                             charge,
                                                             eps,
                                                             maxcharge,
                                                             zero_initial_phi,
                                                             remove_moment=self.remove_moment)
                print moments
                p = self.qm.phi
                n = p.shape
                r = self.qm.gd.get_grid_point_coordinates().transpose((1, 2, 3, 0)) + self.qm.corner1
                plt.plot(r[:, 0,0,0], p[:, n[1]/2, n[2]/2], 'ro-', label = 'phi')
                center = self.qm.corner1 + 0.5*self.qm.gd.cell_cv.sum(0)

                p = np.sum(np.array([m*Gaussian(self.cl.gd, center=center).get_gauss_pot(l) for m,l in zip(moments, range(self.remove_moment))]), axis=0)
                p[self.shift_indices_1[0]:self.shift_indices_2[0]-1,
                  self.shift_indices_1[1]:self.shift_indices_2[1]-1,
                  self.shift_indices_1[2]:self.shift_indices_2[2]-1] = 0.0
                r = self.cl.gd.get_grid_point_coordinates().transpose((1, 2, 3, 0))
                n = p.shape
                plt.plot(r[:, 0,0,0], p[:, n[1]/2, n[2]/2], 'ko-', label = 'cl.phi')
                plt.legend()
                plt.show()
                sys.exit()
                
        if False:
            import matplotlib.pyplot as plt
            p = self.qm.phi
            n = p.shape
            r = self.qm.gd.get_grid_point_coordinates().transpose((1, 2, 3, 0))+self.qm.corner1
            plt.plot(r[:, 0,0,0], p[:, n[1]/2, n[2]/2], 'b', label = 'qm.phi')
            p = self.cl.phi
            n = p.shape
            r = self.cl.gd.get_grid_point_coordinates().transpose((1, 2, 3, 0))
            plt.plot(r[:, 0,0,0], p[:, n[1]/2, n[2]/2], 'ro', label = 'cl.phi')
            plt.legend()
            plt.show()
        
        niter, moments = self.qm.poissonSolver.solve(self, self.qm.phi,
                                                     rho,
                                                     charge,
                                                     eps,
                                                     maxcharge,
                                                     zero_initial_phi,
                                                     self.remove_moment)
        self.cl.poissonSolver.solve(self.cl.phi,
                                    self.classMat.sign * self.classMat.rhoCl,
                                    0.0, #charge,
                                    eps,
                                    0.0, #maxcharge,
                                    zero_initial_phi,
                                    self.remove_moment)
        self.sumPotentials(phi[:], moments)
        #phi[:] = self.qm.phi[:]

        oldRho_qm = rho.copy()
        oldRho_cl = self.classMat.rhoCl.copy()
        
        poissonClCnt = 0

            
        while True:
            ## field from the potential: use the classical potential if no
            #       coupling from quantum to classical charge is requested 
            self.classMat.solveElectricField(self.cl.phi)     # E = -Div[Vh]

            ## Polarizations P0_j and Ptot
            self.classMat.solvePolarizations()        # P = (eps - eps0)E

            ## Classical charge density
            self.classMat.solveRho()                  # n = -Grad[P]
                
            ## Update electrostatic potential         # nabla^2 Vh = -4*pi*n
            niter, moments = self.qm.poissonSolver.solve(self, self.qm.phi,
                                                         rho,
                                                         charge,
                                                         eps,
                                                         maxcharge,
                                                         zero_initial_phi,
                                                         self.remove_moment)
            
            self.cl.poissonSolver.solve(self.cl.phi,
                                        self.classMat.sign*self.classMat.rhoCl,
                                        0.0, #charge,
                                        eps,
                                        0.0, #maxcharge,
                                        zero_initial_phi,
                                        self.remove_moment)            

            self.sumPotentials(phi[:], moments)
            
            # Mix potential
            try:
                self.mixPhi
            except:
                self.mixPhi = simpleMixer(0.10, phi)

            phi = self.mixPhi.mix(phi)
                
            ## Check convergence
            poissonClCnt = poissonClCnt + 1
            
            dRho = self.qm.gd.integrate(abs(rho - oldRho_qm)) + \
                    self.cl.gd.integrate(abs(self.classMat.rhoCl - oldRho_cl))
            
            if(abs(dRho)<1e-3):
                break
            oldRho_qm = rho.copy()
            oldRho_cl = (self.classMat.sign * self.classMat.rhoCl).copy()


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
        
        if False: # artificial rhos for debugging how they add up 
            r = self.qm.gd.get_grid_point_coordinates().transpose((1, 2, 3, 0))
            n = rho.shape
            cc = r[n[0]/2, n[1]/2, n[2]/2]
            #self.qm.phi = (r[:,:,:,0]-cc[0]) * np.exp(-0.10*np.sum((r[:,:,:,:]-cc) * (r[:,:,:,:]-cc), axis=3))  # ~ x*exp(-r^2)
            rho = -0.05*(0.1*r[:,:,:,0]-cc[0]) * np.sin(0.10*np.sum((r[:,:,:,:]-cc) * (r[:,:,:,:]-cc), axis=3)) * \
                                                 np.exp(-0.05*np.sum((r[:,:,:,:]-cc) * (r[:,:,:,:]-cc), axis=3)) # ~ x*sin(-r^2) * exp(-r^2)
        
            r = self.cl.gd.get_grid_point_coordinates().transpose((1, 2, 3, 0))
            n = r.shape
            cc = r[n[0]/2, n[1]/2, n[2]/2]
            #self.qm.phi = (r[:,:,:,0]-cc[0]) * np.exp(-0.10*np.sum((r[:,:,:,:]-cc) * (r[:,:,:,:]-cc), axis=3))  # ~ x*exp(-r^2)
            self.classMat.rhoCl = -0.05*(0.1*r[:,:,:,0]-cc[0]) * np.sin(0.03*np.sum((r[:,:,:,:]-cc) * (r[:,:,:,:]-cc), axis=3)) * \
                                            np.exp(-0.12*np.sum((r[:,:,:,:]-cc) * (r[:,:,:,:]-cc), axis=3)) # ~ x*sin(-r^2) * exp(-r^2)
        
        # 3a) V(t) from n(t)       
        niter, moments = self.qm.poissonSolver.solve(self, self.qm.phi,
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
        
        doPlots2 = False and np.round(self.time/self.timestep)%20==0

        if False: # artificial phis for debugging how they add up 
            r = self.cl.gd.get_grid_point_coordinates().transpose((1, 2, 3, 0))
            n = self.cl.phi.shape
            cc = r[n[0]/2, n[1]/2, n[2]/2]
            self.cl.phi = 0.75*np.exp(-0.01*np.sum((r[:,:,:,:]-cc) * (r[:,:,:,:]-cc), axis=3)) # ~ exp(-r^2)
            
            r = self.qm.gd.get_grid_point_coordinates().transpose((1, 2, 3, 0))
            n = self.qm.phi.shape
            cc = r[n[0]/2, n[1]/2, n[2]/2]
            #self.qm.phi = (r[:,:,:,0]-cc[0]) * np.exp(-0.10*np.sum((r[:,:,:,:]-cc) * (r[:,:,:,:]-cc), axis=3))  # ~ x*exp(-r^2)
            self.qm.phi = -0.05*(0.1*r[:,:,:,0]-cc[0]) * np.sin(0.10*np.sum((r[:,:,:,:]-cc) * (r[:,:,:,:]-cc), axis=3)) * \
                                                         np.exp(-0.05*np.sum((r[:,:,:,:]-cc) * (r[:,:,:,:]-cc), axis=3)) # ~ x*sin(-r^2) * exp(-r^2)
        
        aa = 2
        bb = 3
        
        if doPlots2:
            import matplotlib.pyplot as plt
            plt.ylim(-3.00, 0.10)
            p = self.qm.phi
            n = p.shape
            ny = aa*n[1]/bb
            nz = aa*n[2]/bb
            r = self.qm.gd.get_grid_point_coordinates().transpose((1, 2, 3, 0))+self.qm.corner1
            plt.plot(r[:, 0,0,0], p[:, ny, nz], 'b.-', label = 'qm.phi.0')
            p = self.cl.phi
            n = p.shape
            ny = aa*n[1]/bb
            nz = aa*n[2]/bb
            r = self.cl.gd.get_grid_point_coordinates().transpose((1, 2, 3, 0))
            plt.plot(r[:, 0,0,0], p[:, ny, nz], 'r.-', label = 'cl.phi.0')
        
        qm_phi_coarsed, cl_phi_refined = self.sumPotentials(phi[:], moments, doplots=False)
        
        if False and doPlots2:
            import matplotlib.pyplot as plt
            p0 = qm_phi_coarsed
            p  = self.cl.gd.zeros()
            n = p.shape
            ny = aa*n[1]/bb
            nz = aa*n[2]/bb
            r = self.cl.gd.get_grid_point_coordinates().transpose((1, 2, 3, 0))
            
            p[self.shift_indices_1[0]:self.shift_indices_2[0]-1,
              self.shift_indices_1[1]:self.shift_indices_2[1]-1,
              self.shift_indices_1[2]:self.shift_indices_2[2]-1] = p0
            
            plt.plot(r[:, 0,0,0], -20*p[:, ny, nz], 'b:', label = '-20*qm_phi_coarsed')
            
            center = self.qm.corner1 + 0.5*self.qm.gd.cell_cv.sum(0)
            p = np.sum(np.array([m*Gaussian(self.cl.gd, center=center).get_gauss_pot(l) for m,l in zip(moments, range(self.remove_moment))]), axis=0)
            p[self.shift_indices_1[0]:self.shift_indices_2[0]-1,
              self.shift_indices_1[1]:self.shift_indices_2[1]-1,
              self.shift_indices_1[2]:self.shift_indices_2[2]-1] = 0.0
            n = p.shape
            ny = aa*n[1]/bb
            nz = aa*n[2]/bb
            r = self.cl.gd.get_grid_point_coordinates().transpose((1, 2, 3, 0))
            plt.plot(r[:, 0,0,0], p[:, ny, nz], 'm:', label = 'qm_phi_extended')
            
            p = cl_phi_refined
            n = p.shape
            ny = aa*n[1]/bb
            nz = aa*n[2]/bb
            r = self.qm.gd.get_grid_point_coordinates().transpose((1, 2, 3, 0))+self.qm.corner1
            plt.plot(r[:, 0,0,0], -p[:, ny, nz], 'r:', label = '-cl_phi_refined')
        
        if doPlots2:
            import matplotlib.pyplot as plt
            p = self.qm.phi
            n = p.shape
            ny = aa*n[1]/bb
            nz = aa*n[2]/bb
            r = self.qm.gd.get_grid_point_coordinates().transpose((1, 2, 3, 0))+self.qm.corner1
            plt.plot(r[:, 0,0,0], p[:, ny, nz]-0.0, 'g.-', label = 'qm.phi.1')
            
            p = self.cl.phi
            n = p.shape
            ny = aa*n[1]/bb
            nz = aa*n[2]/bb
            r = self.cl.gd.get_grid_point_coordinates().transpose((1, 2, 3, 0))
            plt.plot(r[:, 0,0,0], p[:, ny, nz]-0.0, 'k.-', label = 'cl.phi.1')
            plt.title('Potentials at propagation, time t=%.1f as' % (self.time * autime_to_attosec))
            if doPlots2:
                plt.legend(loc=4)
                plt.show()
                plt.clf()
                #sys.exit()
                        
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
        
        #self.qm.phi = None
        #self.cl.phi = None

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

            # Setup the Poisson solver, and attach the grid descriptor 
            self.cl.poissonSolver = PoissonSolver()
            self.cl.poissonSolver.set_grid_descriptor(self.cl.gd)
            self.cl.poissonSolver.initialize()
            self.classMat.initialize(self.cl.gd)
            self.mixPhi = simpleMixer(0.10, phi)

            if False:
                qr = self.qm.gd.get_grid_point_coordinates().transpose((1, 2, 3, 0))
                cr = self.cl.gd.get_grid_point_coordinates().transpose((1, 2, 3, 0))
                print qr[:,0,0,0]
                print cr[:,0,0,0]
                sys.exit()
        
            
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
        
        # classical contribution. Note the different origin.
        r_gv = self.cl.gd.get_grid_point_coordinates().transpose((1, 2, 3, 0))-self.qm.corner1
        dmcl = -1.0*self.classMat.sign*np.array([self.cl.gd.integrate(np.multiply(r_gv[:, :, :, w]+self.qm.corner1[w], self.classMat.rhoCl)) for w in range(3)])
        
        # add quantum contribution
        dm = self.density.finegd.calculate_dipole_moment(self.density.rhot_g) + dmcl
        norm = self.qm.gd.integrate( rho ) + self.classMat.sign * self.cl.gd.integrate(self.classMat.rhoCl )
        
        # The sign (-1) comes from the sign of electron charge
        #dm=self.density.finegd.calculate_dipole_moment(self.density.rhot_g) \
        #+[-1.0*self.classMat.sign*self.cl.gd.integrate(self.classMat.pTotal[0]),
        #  -1.0*self.classMat.sign*self.cl.gd.integrate(self.classMat.pTotal[1]),
        #  -1.0*self.classMat.sign*self.cl.gd.integrate(self.classMat.pTotal[2])]
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
        self.classMat.initialize(self.cl.gd)
        
        # Read self.classMat.rhoCl
        if self.cl.gd.comm.rank == 0:
                big_rhoCl = np.array(r.get('ClassicalMaterialRho'),
                                     dtype=float)
        else:
                big_rhoCl = None
        self.cl.gd.distribute(big_rhoCl,
                              self.classMat.rhoCl)
                
        # Read self.classMat.pTotal
        if self.cl.gd.comm.rank == 0:
                    big_pTotal = np.array(r.get('pTotal'),
                                          dtype=float)
        else:
                big_pTotal = None
        self.cl.gd.distribute(big_pTotal,
                           self.classMat.pTotal)
        
        # Read self.classMat.polarizations
        if self.cl.gd.comm.rank == 0:
                big_polarizations = np.array(r.get('polarizations'),
                                             dtype=float)
        else:
                big_polarizations = None
        self.cl.gd.distribute(big_polarizations,
                           self.classMat.polarizations)
        
        r.close
         
    def write(self, paw,
                     filename = 'poisson'):
        #parprint('Writing FDTDPoissonSolver data to %s' % (filename))
        rho         = self.classMat.rhoCl
        world       = paw.wfs.world
        domain_comm = self.cl.gd.comm
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
        ng = self.cl.gd.get_size_of_global_array()
        
        # Write the classical charge density
        w.dimension('ngptsx', ng[0])
        w.dimension('ngptsy', ng[1])
        w.dimension('ngptsz', ng[2])
        w.add('ClassicalMaterialRho',
              ('ngptsx', 'ngptsy', 'ngptsz'),
              dtype=float,
              write=master)
        if kpt_comm.rank == 0:
            rhoCl = self.cl.gd.collect(self.classMat.rhoCl)
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
            pTotal = self.cl.gd.collect(self.classMat.pTotal)
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
            polarizations = self.cl.gd.collect(self.classMat.polarizations)
            if master:
                w.fill(polarizations)

        w.close()
        world.barrier()
        


from gpaw.utilities.gauss import Gaussian
import numpy as np

class PotentialCoupler():
    def __init__(self,
                  qm, # PoissonOrganizer
                  cl, # PoissonOrganizer
                  index_offset_1,
                  index_offset_2,
                  extended_index_offset_1,
                  extended_index_offset_2,
                  extended_delta_index,
                  num_refinements,
                  remove_moment_qm,
                  remove_moment_cl,
                  coupling_level,
                  rank):
        self.qm = qm
        self.cl = cl
        self.index_offset_1 = index_offset_1
        self.index_offset_2 = index_offset_2
        self.extended_index_offset_1 = extended_index_offset_1
        self.extended_index_offset_2 = extended_index_offset_2
        self.extended_delta_index = extended_delta_index
        self.num_refinements = num_refinements
        self.remove_moment_qm = remove_moment_qm
        self.remove_moment_cl = remove_moment_cl
        self.coupling_level = coupling_level
        self.rank = rank
        assert(self.coupling_level in ['both', 'qm2cl', 'cl2qm', 'none'])

        # These are used to remember the previous solutions
        self.old_local_phi_qm_qmgd = self.qm.gd.zeros()
        self.old_local_phi_cl_clgd = self.cl.gd.zeros()
        self.old_local_phi_qm_clgd = self.cl.gd.zeros()
    
    # Add quantum and classical parts (usually potentials)
    def couple(self, qm_qmgd,
                      qm_clgd,
                      cl_qmgd,
                      cl_clgd):
        if self.coupling_level=='both':
            tot_qmgd = cl_qmgd + qm_qmgd
            tot_clgd = cl_clgd + qm_clgd
        elif self.coupling_level=='qm2cl':
            tot_qmgd = qm_qmgd
            tot_clgd = cl_clgd + qm_clgd
        elif self.coupling_level=='cl2qm':
            tot_qmgd = cl_qmgd + qm_qmgd
            tot_clgd = cl_clgd
        else: # self.coupling_level=='none'
            tot_qmgd = qm_qmgd
            tot_clgd = cl_clgd
         
        return tot_qmgd, tot_clgd
    
# Coarsen quantum potential to suit the classical grid,
# and refine classical potential to suit the quantum grid 
class NewPotentialCoupler(PotentialCoupler):

    def __init__(self, qm,
                        cl,
                        index_offset_1,
                        index_offset_2,
                        extended_index_offset_1,
                        extended_index_offset_2,
                        extended_delta_index,
                        num_refinements,
                        remove_moment_qm,
                        remove_moment_cl,
                        coupling_level,
                        rank):
        PotentialCoupler.__init__(self,
                                  qm,
                                  cl,
                                  index_offset_1,
                                  index_offset_2,
                                  extended_index_offset_1,
                                  extended_index_offset_2,
                                  extended_delta_index,
                                  num_refinements,
                                  remove_moment_qm,
                                  remove_moment_cl,
                                  coupling_level,
                                  rank)

    def getPotential(self, local_rho_qm_qmgd, local_rho_cl_clgd, **kwargs):
        # Quantum potential
        local_phi_qm_qmgd = self.old_local_phi_qm_qmgd
        niter_qm, moments_qm = self.qm.poisson_solver.solve(phi=local_phi_qm_qmgd,
                                                            rho=local_rho_qm_qmgd,
                                                            remove_moment=self.remove_moment_qm,
                                                            **kwargs)
        self.old_local_phi_qm_qmgd = local_phi_qm_qmgd.copy()

        # Classical potential
        local_phi_cl_clgd = self.old_local_phi_cl_clgd
        niter_cl, moments_cl = self.cl.poisson_solver.solve(phi=local_phi_cl_clgd,
                                                            rho=local_rho_cl_clgd,
                                                            remove_moment=self.remove_moment_cl,
                                                            **kwargs)
        self.old_local_phi_cl_clgd = local_phi_cl_clgd.copy()

        # Transfer classical potential into quantum subsystem
        global_phi_cl_clgd = self.cl.gd.collect(local_phi_cl_clgd)
        local_phi_cl_qmgd = self.qm.gd.zeros()
        global_phi_cl_qmgd = self.qm.gd.zeros(global_array = True)

        if self.rank == 0:
            # Extract the overlapping region from the classical potential
            global_phi_cl_clgd_refined = global_phi_cl_clgd[self.extended_index_offset_1[0]:self.extended_index_offset_2[0] - 1,
                                                            self.extended_index_offset_1[1]:self.extended_index_offset_2[1] - 1,
                                                            self.extended_index_offset_1[2]:self.extended_index_offset_2[2] - 1].copy()
            
            for n in range(self.num_refinements):
                global_phi_cl_clgd_refined = self.cl.extended_refiners[n].apply(global_phi_cl_clgd_refined)
            
            global_phi_cl_qmgd = global_phi_cl_clgd_refined[self.extended_delta_index:-self.extended_delta_index,
                                                            self.extended_delta_index:-self.extended_delta_index,
                                                            self.extended_delta_index:-self.extended_delta_index]

        self.qm.gd.distribute(global_phi_cl_qmgd, local_phi_cl_qmgd)

        # Transfer quantum potential into classical subsystem
        global_rho_qm_qmgd = self.qm.gd.collect(local_rho_qm_qmgd)
        global_rho_qm_clgd = self.cl.gd.zeros(global_array = True)
        local_rho_qm_clgd = self.cl.gd.zeros()
        
        if self.rank == 0:
            # Coarsen the quantum density
            global_rho_qm_qmgd_coarsened = global_rho_qm_qmgd.copy()
            for n in range(self.num_refinements):
                global_rho_qm_qmgd_coarsened = self.cl.coarseners[n].apply(global_rho_qm_qmgd_coarsened)
            
            # Add the coarsened quantum density
            global_rho_qm_clgd[self.index_offset_1[0]:self.index_offset_2[0] - 1,
                               self.index_offset_1[1]:self.index_offset_2[1] - 1,
                               self.index_offset_1[2]:self.index_offset_2[2] - 1] = global_rho_qm_qmgd_coarsened[:]
            
        # Distribute the combined density to all processes
        self.cl.gd.distribute(global_rho_qm_clgd, local_rho_qm_clgd)
        
        # Solve potential
        local_phi_qm_clgd = self.old_local_phi_qm_clgd
        niter_qm_clgd = self.cl.poisson_solver.solve(phi=local_phi_qm_clgd,
                                                     rho=local_rho_qm_clgd,
                                                     remove_moment=None,
                                                     **kwargs)
        self.old_local_phi_qm_clgd = local_phi_qm_clgd.copy()

        # Sum quantum and classical potentials 
        local_phi_tot_qmgd, local_phi_tot_clgd = self.couple(qm_qmgd = local_phi_qm_qmgd,
                                                             qm_clgd = local_phi_qm_clgd,
                                                             cl_qmgd = local_phi_cl_qmgd,
                                                             cl_clgd = local_phi_cl_clgd)
        
        return local_phi_tot_qmgd, local_phi_tot_clgd, (niter_qm, niter_cl, niter_qm_clgd)


# Refine classical potential to suit the quantum grid, and use the multipole 
# expansion to extrapolate the quantum potential into classical grid
class MultipolesPotentialCoupler(PotentialCoupler):

    def __init__(self, qm,
                        cl,
                        index_offset_1,
                        index_offset_2,
                        extended_index_offset_1,
                        extended_index_offset_2,
                        extended_delta_index,
                        num_refinements,
                        remove_moment_qm,
                        remove_moment_cl,
                        coupling_level,
                        rank):
        PotentialCoupler.__init__(self,
                                  qm,
                                  cl,
                                  index_offset_1,
                                  index_offset_2,
                                  extended_index_offset_1,
                                  extended_index_offset_2,
                                  extended_delta_index,
                                  num_refinements,
                                  remove_moment_qm,
                                  remove_moment_cl,
                                  coupling_level,
                                  rank)


    def getPotential(self, local_rho_qm_qmgd, local_rho_cl_clgd, **kwargs):        
        # Quantum potential
        local_phi_qm_qmgd = self.old_local_phi_qm_qmgd
        niter_qm, moments_qm = self.qm.poisson_solver.solve(phi=local_phi_qm_qmgd,
                                                            rho=local_rho_qm_qmgd,
                                                            remove_moment=self.remove_moment_qm,
                                                            **kwargs)
        self.old_local_phi_qm_qmgd = local_phi_qm_qmgd.copy()

        # Classical potential
        local_phi_cl_clgd = self.old_local_phi_cl_clgd
        niter_cl, moments_cl = self.cl.poisson_solver.solve(phi=local_phi_cl_clgd,
                                                            rho=local_rho_cl_clgd,
                                                            remove_moment=self.remove_moment_cl,
                                                            **kwargs)
        self.old_local_phi_cl_clgd = local_phi_cl_clgd.copy()

        # Transfer classical potential into quantum subsystem
        global_phi_cl_clgd = self.cl.gd.collect(local_phi_cl_clgd)
        local_phi_cl_qmgd = self.qm.gd.zeros()
        global_phi_cl_qmgd = self.qm.gd.zeros(global_array = True)

        if self.rank == 0:
            # Extract the overlapping region from the classical potential
            global_phi_cl_clgd_refined = global_phi_cl_clgd[self.extended_index_offset_1[0]:self.extended_index_offset_2[0] - 1,
                                                            self.extended_index_offset_1[1]:self.extended_index_offset_2[1] - 1,
                                                            self.extended_index_offset_1[2]:self.extended_index_offset_2[2] - 1].copy()
            
            for n in range(self.num_refinements):
                global_phi_cl_clgd_refined = self.cl.extended_refiners[n].apply(global_phi_cl_clgd_refined)
            
            global_phi_cl_qmgd = global_phi_cl_clgd_refined[self.extended_delta_index:-self.extended_delta_index,
                                                            self.extended_delta_index:-self.extended_delta_index,
                                                            self.extended_delta_index:-self.extended_delta_index]


        self.qm.gd.distribute(global_phi_cl_qmgd, local_phi_cl_qmgd)

        # Transfer quantum potential into classical subsystem            

        # Outside overlapping region: multipole expansion
        center = self.qm.corner1 + 0.5*self.qm.gd.cell_cv.sum(0)
        global_phi_qm_clgd = np.sum(np.array([m * Gaussian(self.cl.gd_global,
                                                          center=self.qm.corner1 + 0.5*self.qm.gd.cell_cv.sum(0)).get_gauss_pot(l) \
                                             for m, l in zip(moments_qm, range(self.remove_moment_qm))]), axis=0)      
                
        # Inside overlapping region: coarsen
        global_phi_qm_qmgd = self.qm.gd.collect(local_phi_qm_qmgd)
        local_phi_qm_clgd = self.cl.gd.empty()
        
        if self.rank == 0:
            # Coarsen the quantum density
            global_phi_qm_qmgd_coarsened = global_phi_qm_qmgd.copy()
            for n in range(self.num_refinements):
                global_phi_qm_qmgd_coarsened = self.cl.coarseners[n].apply(global_phi_qm_qmgd_coarsened)
            
            # Add the coarsened quantum density
            global_phi_qm_clgd[self.index_offset_1[0]:self.index_offset_2[0] - 1,
                               self.index_offset_1[1]:self.index_offset_2[1] - 1,
                               self.index_offset_1[2]:self.index_offset_2[2] - 1] = global_phi_qm_qmgd_coarsened[:]
        
        # Distribute the combined potential to all processes
        self.cl.gd.distribute(global_phi_qm_clgd, local_phi_qm_clgd)
        
        # Sum quantum and classical potentials
        local_phi_tot_qmgd, local_phi_tot_clgd = self.couple(qm_qmgd = local_phi_qm_qmgd,
                                                             qm_clgd = local_phi_qm_clgd,
                                                             cl_qmgd = local_phi_cl_qmgd,
                                                             cl_clgd = local_phi_cl_clgd)

        return local_phi_tot_qmgd, local_phi_tot_clgd, (niter_qm, niter_cl)




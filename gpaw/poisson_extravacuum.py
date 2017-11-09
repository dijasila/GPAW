import numpy as np

from gpaw.transformers import Transformer
from gpaw.fd_operators import Laplace

from gpaw.utilities.extend_grid import extended_grid_descriptor, \
    extend_array, deextend_array


class ExtraVacuumPoissonSolver:
    """Wrapper around PoissonSolver extending the vacuum size.
    
       """

    def __init__(self, gpts, poissonsolver_large, poissonsolver_small=None, coarses=0,
                 nn_coarse=3, nn_refine=3, nn_laplace=3,
                 use_aux_grid=True):
        # TODO: Alternative options: vacuum size and h
        # Multiply gpts by 2 to get gpts on fine grid
        self.N_large_fine_c = np.array(gpts, dtype=int) * 2
        self.Ncoar = coarses  # coar == coarse
        if self.Ncoar > 0:
            self.use_coarse = True
        else:
            self.use_coarse = False
        self.ps_small_fine = poissonsolver_small
        self.ps_large_coar = poissonsolver_large  # coar == coarse
        self.nn_coarse = nn_coarse
        self.nn_refine = nn_refine
        self.nn_laplace = nn_laplace
        self.use_aux_grid = use_aux_grid

    def set_grid_descriptor(self, gd):
        # If non-periodic boundary conditions is used,
        # there is problems with auxiliary grid.
        # Maybe with use_aux_grid=False it would work?
        if gd.pbc_c.any():
            raise NotImplementedError('Only non-periodic boundary '
                                      'conditions are tested')

        self.gd_small_fine = gd
        assert np.all(self.gd_small_fine.N_c <= self.N_large_fine_c), \
            'extended grid has to be larger than the original one'

        if self.use_coarse:
            # 1.1. Construct coarse chain on the small grid
            self.coarser_i = []
            gd = self.gd_small_fine
            N_c = self.N_large_fine_c.copy()
            for i in range(self.Ncoar):
                gd2 = gd.coarsen()
                self.coarser_i.append(Transformer(gd, gd2, self.nn_coarse))
                N_c /= 2
                gd = gd2
            self.gd_small_coar = gd
        else:
            self.gd_small_coar = self.gd_small_fine
            N_c = self.N_large_fine_c

        # 1.2. Construct coarse extended grid
        self.gd_large_coar, _, _ = extended_grid_descriptor(self.gd_small_coar, N_c=N_c)

        # Initialize poissonsolvers
        self.ps_large_coar.set_grid_descriptor(self.gd_large_coar)
        if not self.use_coarse:
            return
        self.ps_small_fine.set_grid_descriptor(self.gd_small_fine)

        if self.use_aux_grid:
            # 2.1. Construct an auxiliary grid that is the small grid plus
            # a buffer region allowing Laplace and refining with the used stencils
            buf = self.nn_refine
            for i in range(self.Ncoar):
                buf = 2 * buf + self.nn_refine
            buf += self.nn_laplace
            div = 2**self.Ncoar
            if buf % div != 0:
                buf += div - buf % div
            N_c = self.gd_small_fine.N_c + 2 * buf
            if np.any(N_c > self.N_large_fine_c):
                self.use_aux_grid = False
                N_c = self.N_large_fine_c
            self.gd_aux_fine, _, _ = extended_grid_descriptor(self.gd_small_fine, N_c=N_c)
        else:
            self.gd_aux_fine, _, _ = extended_grid_descriptor(self.gd_small_fine, N_c=self.N_large_fine_c)

        # 2.2 Construct Laplace on the aux grid
        self.laplace_aux_fine = Laplace(self.gd_aux_fine, - 0.25 / np.pi, self.nn_laplace)

        # 2.3 Construct refine chain
        self.refiner_i = []
        gd = self.gd_aux_fine
        N_c = gd.N_c.copy()
        for i in range(self.Ncoar):
            gd2 = gd.coarsen()
            self.refiner_i.append(Transformer(gd2, gd, self.nn_refine))
            N_c /= 2
            gd = gd2
        self.refiner_i = self.refiner_i[::-1]
        self.gd_aux_coar = gd

        if self.use_aux_grid:
            # 2.4 Construct large coarse grid from aux grid
            self.gd_large_coar_from_aux, _, _ = extended_grid_descriptor(self.gd_aux_coar, N_c=self.gd_large_coar.N_c)
            assert np.all(self.gd_large_coar_from_aux.N_c == self.gd_large_coar.N_c) and np.all(self.gd_large_coar_from_aux.h_cv == self.gd_large_coar.h_cv)

    def initialize(self):
        # Allocate arrays
        self.phi_large_coar_g = self.gd_large_coar.zeros()

        # Initialize poissonsolvers
        self.ps_large_coar.initialize(self.gd_large_coar)
        if not self.use_coarse:
            return
        self.ps_small_fine.initialize(self.gd_small_fine)

    def solve(self, phi, rho, **kwargs):
        phi_small_fine_g = phi
        rho_small_fine_g = rho

        if self.use_coarse:
            # 1.1. Coarse rho on the small grid
            tmp_g = rho_small_fine_g
            for coarser in self.coarser_i:
                tmp_g = coarser.apply(tmp_g)
            rho_small_coar_g = tmp_g
        else:
            rho_small_coar_g = rho_small_fine_g

        # 1.2. Extend rho to the large grid
        rho_large_coar_g = self.gd_large_coar.zeros()
        extend_array(self.gd_small_coar, self.gd_large_coar, rho_small_coar_g, rho_large_coar_g)

        # 1.3 Solve potential on the large coarse grid
        niter_large = self.ps_large_coar.solve(self.phi_large_coar_g, rho_large_coar_g, **kwargs)

        if not self.use_coarse:
            deextend_array(self.gd_small_fine, self.gd_large_coar, phi_small_fine_g, self.phi_large_coar_g)
            return niter_large

        if self.use_aux_grid:
            # 2.1 De-extend the potential to the auxiliary grid
            phi_aux_coar_g = self.gd_aux_coar.empty()
            deextend_array(self.gd_aux_coar, self.gd_large_coar_from_aux, phi_aux_coar_g, self.phi_large_coar_g)
        else:
            phi_aux_coar_g = self.phi_large_coar_g

        # 3.1 Refine the potential
        tmp_g = phi_aux_coar_g
        for refiner in self.refiner_i:
            tmp_g = refiner.apply(tmp_g)
        phi_aux_fine_g = tmp_g

        # 3.2 Calculate the corresponding density with Laplace
        # (the refined coarse density would not accurately match with the potential)
        rho_aux_fine_g = self.gd_aux_fine.empty()
        self.laplace_aux_fine.apply(phi_aux_fine_g, rho_aux_fine_g)

        # 3.3 De-extend the potential and density to the small grid
        cor_phi_small_fine_g = self.gd_small_fine.empty()
        deextend_array(self.gd_small_fine, self.gd_aux_fine, cor_phi_small_fine_g, phi_aux_fine_g)
        cor_rho_small_fine_g = self.gd_small_fine.empty()
        deextend_array(self.gd_small_fine, self.gd_aux_fine, cor_rho_small_fine_g, rho_aux_fine_g)

        # 3.4 Remove the correcting density and potential
        rho_small_fine_g -= cor_rho_small_fine_g
        phi_small_fine_g -= cor_phi_small_fine_g

        # 3.5 Solve potential on the small grid
        niter_small = self.ps_small_fine.solve(phi_small_fine_g, rho_small_fine_g, **kwargs)

        # 3.6 Correct potential and density
        phi_small_fine_g += cor_phi_small_fine_g
        #rho_small_fine_g += cor_rho_small_fine_g

        return (niter_large, niter_small)

    def estimate_memory(self, mem):
        # TODO
        return

    def get_description(self):
        lines = ['%s' % (self.__class__.__name__)]
        return '\n'.join(lines)

    def todict(self):
        d = {'name': self.__class__.__name__}
        d['gpts'] = self.N_large_fine_c / 2
        d['coarses'] = self.Ncoar
        d['nn_coarse'] = self.nn_coarse
        d['nn_refine'] = self.nn_refine
        d['nn_laplace'] = self.nn_laplace
        d['use_aux_grid'] = self.use_aux_grid
        d['poissonsolver_large'] = self.ps_large_coar.todict()
        d['poissonsolver_small'] = self.ps_small_fine.todict()
        return d

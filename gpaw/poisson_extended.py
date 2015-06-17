import numpy as np
from ase.units import Bohr
from gpaw.poisson import PoissonSolver
from gpaw.utilities.gauss import Gaussian
from gpaw.utilities.timing import nulltimer

from ase.utils.timing import timer

from gpaw.utilities.extend_grid import extended_grid_descriptor, \
    extend_array, deextend_array



class ExtendedPoissonSolver(PoissonSolver):
    def __init__(self, nn=3, relax='J', eps=2e-10, maxiter=1000,
                 moment_corrections=None,
                 extendedgpts=None,
                 extendedhistory=False,
                 timer=nulltimer):

        PoissonSolver.__init__(self, nn=nn, relax=relax,
                               eps=eps, maxiter=maxiter,
                               remove_moment=None)

        self.timer = timer

        extendedcomm = None

        if moment_corrections is None:
            self.moment_corrections = None
        elif isinstance(moment_corrections, int):
            self.moment_corrections = [{'moms': range(moment_corrections),
                                        'center': None}]
        else:
            self.moment_corrections = moment_corrections

        self.is_extended = False
        self.requires_broadcast = False
        if extendedgpts is not None:
            self.is_extended = True
            self.extendedgpts = extendedgpts
            self.extendedcomm = extendedcomm
            self.extendedhistory = extendedhistory
            if self.extendedcomm is not None:
                self.requires_broadcast = True

    def set_grid_descriptor(self, gd):
        if self.is_extended:
            self.gd_original = gd
            gd, _, _ = extended_grid_descriptor(gd,
                                                N_c=self.extendedgpts,
                                                extcomm=self.extendedcomm)
        PoissonSolver.set_grid_descriptor(self, gd)

    def get_description(self):
        description = PoissonSolver.get_description(self)

        lines = [description]

        if self.is_extended:
            lines.extend(['    Extended %d*%d*%d grid' %
                          tuple(self.gd.N_c)])
            lines.extend(['    Remember history is %s' %
                          self.extendedhistory])

        if self.moment_corrections:
            lines.extend(['    %d moment corrections:' %
                          len(self.moment_corrections)])
            lines.extend(['      %s' %
                          ('[%s] %s' %
                           ('center' if mom['center'] is None
                            else (', '.join(['%.2f' % (x * Bohr)
                                             for x in mom['center']])),
                            mom['moms']))
                          for mom in self.moment_corrections])

        return '\n'.join(lines)

    @timer('Poisson initialize')
    def initialize(self, load_gauss=False):
        PoissonSolver.initialize(self, load_gauss=load_gauss)

        if self.is_extended:
            if not self.gd.orthogonal or self.gd.pbc_c.any():
                raise NotImplementedError('Only orthogonal unit cells' +
                                          'and non-periodic boundary' +
                                          'conditions are tested')
            self.rho_g = self.gd.zeros()
            self.phi_g = self.gd.zeros()

        if self.moment_corrections is not None:
            if not self.gd.orthogonal or self.gd.pbc_c.any():
                raise NotImplementedError('Only orthogonal unit cells' +
                                          'and non-periodic boundary' +
                                          'conditions are tested')
            self.load_moment_corrections_gauss()

    @timer('Load moment corrections')
    def load_moment_corrections_gauss(self):
        if not hasattr(self, 'gauss_i'):
            self.gauss_i = []
            mask_ir = []
            r_ir = []
            self.mom_ij = []

            for rmom in self.moment_corrections:
                if rmom['center'] is None:
                    center = None
                else:
                    center = np.array(rmom['center'])
                mom_j = rmom['moms']
                gauss = Gaussian(self.gd, center=center)
                self.gauss_i.append(gauss)
                r_ir.append(gauss.r.ravel())
                mask_ir.append(self.gd.zeros(dtype=int).ravel())
                self.mom_ij.append(mom_j)

            r_ir = np.array(r_ir)
            mask_ir = np.array(mask_ir)

            Ni = r_ir.shape[0]
            Nr = r_ir.shape[1]

            for r in range(Nr):
                i = np.argmin(r_ir[:, r])
                mask_ir[i, r] = 1

            self.mask_ig = []
            for i in range(Ni):
                mask_r = mask_ir[i]
                mask_g = mask_r.reshape(self.gd.n_c)
                self.mask_ig.append(mask_g)

                # big_g = self.gd.collect(mask_g)
                # if self.gd.comm.rank == 0:
                #     big_g.dump('mask_%dg' % (i))

    def solve(self, phi, rho, charge=None, eps=None, maxcharge=1e-6,
              zero_initial_phi=False):
        if self.is_extended:
            self.rho_g[:] = 0
            if not self.extendedhistory:
                self.phi_g[:] = 0

            self.timer.start('Extend array')
            extend_array(rho, self.gd_original, self.rho_g, self.gd)
            self.timer.stop('Extend array')

            retval = self._solve(self.phi_g, self.rho_g, charge,
                                 eps, maxcharge, zero_initial_phi)

            self.timer.start('Deextend array')
            deextend_array(phi, self.gd_original, self.phi_g, self.gd)
            self.timer.stop('Deextend array')

            return retval
        else:
            return self._solve(phi, rho, charge, eps, maxcharge,
                               zero_initial_phi)

    @timer('Solve')
    def _solve(self, phi, rho, charge=None, eps=None, maxcharge=1e-6,
               zero_initial_phi=False):
        if eps is None:
            eps = self.eps

        if self.moment_corrections:
            assert not self.gd.pbc_c.any()

            self.timer.start('Multipole moment corrections')

            rho_neutral = rho * 0.0
            phi_cor_k = []
            for gauss, mask_g, mom_j in zip(self.gauss_i, self.mask_ig,
                                            self.mom_ij):
                rho_masked = rho * mask_g
                for mom in mom_j:
                    phi_cor_k.append(gauss.remove_moment(rho_masked, mom))
                rho_neutral += rho_masked

            # Remove multipoles for better initial guess
            for phi_cor in phi_cor_k:
                phi -= phi_cor

            self.timer.stop('Multipole moment corrections')

            self.timer.start('Solve neutral')
            niter = self.solve_neutral(phi, rho_neutral, eps=eps)
            self.timer.stop('Solve neutral')

            self.timer.start('Multipole moment corrections')
            # correct error introduced by removing multipoles
            for phi_cor in phi_cor_k:
                phi += phi_cor
            self.timer.stop('Multipole moment corrections')

            return niter
        else:
            return PoissonSolver.solve(self, phi, rho, charge,
                                       eps, maxcharge,
                                       zero_initial_phi)

    def estimate_memory(self, mem):
        PoissonSolver.estimate_memory(self, mem)
        gdbytes = self.gd.bytecount()
        if self.is_extended:
            mem.subnode('extended arrays',
                        2*gdbytes)
        if self.moment_corrections is not None:
            mem.subnode('moment_corrections masks',
                        len(self.moment_corrections)*gdbytes)

    def __repr__(self):
        template = 'ExtendedPoissonSolver(relax=\'%s\', nn=%s, eps=%e)'
        representation = template % (self.relax, repr(self.nn), self.eps)
        return representation

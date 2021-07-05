from typing import Any, Dict, Optional, List, Union

import numpy as np
from ase.units import Bohr
from ase.utils.timing import Timer
from gpaw.poisson import _PoissonSolver
from gpaw.utilities.gauss import Gaussian
from gpaw.utilities.timing import nulltimer, NullTimer

from ase.utils.timing import timer


class MomentCorrectionPoissonSolver(_PoissonSolver):
    """MomentCorrectionPoissonSolver

    Parameter syntax:

    moment_corrections = [{'moms': moms_list1, 'center': center1},
                          {'moms': moms_list2, 'center': center2},
                          ...]
    Here moms_listX is list of integers of multipole moments to be corrected
    at centerX.

    Important: provide timer for PoissonSolver to analyze the cost of
    the multipole moment corrections and grid extension to your system!

    """

    def __init__(self,
                 poissonsolver: _PoissonSolver,
                 moment_corrections: Optional[Union[int, List[Dict[str, Any]]]] = None,
                 timer: Union[NullTimer, Timer] = nulltimer):

        self._initialized = False
        self.poissonsolver = poissonsolver
        self.gd = self.poissonsolver.gd
        self.timer = timer

        if moment_corrections is None:
            self.moment_corrections = None
        elif isinstance(moment_corrections, int):
            self.moment_corrections = [{'moms': range(moment_corrections),
                                        'center': None}]
        else:
            self.moment_corrections = moment_corrections

    def set_grid_descriptor(self, gd):
        self.poissonsolver.set_grid_descriptor(gd)

    def get_description(self) -> str:
        description = self.poissonsolver.get_description()

        lines = [description]

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
    def _init(self):
        if self._initialized:
            return
        self.poissonsolver._init()

        if self.moment_corrections is not None:
            if not self.gd.orthogonal or self.gd.pbc_c.any():
                raise NotImplementedError('Only orthogonal unit cells '
                                          'and non-periodic boundary '
                                          'conditions are tested')
            self.load_moment_corrections_gauss()

        self._initialized = True

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

                # Uncomment this to see masks on grid
                # big_g = self.gd.collect(mask_g)
                # if self.gd.comm.rank == 0:
                #     big_g.dump('mask_%dg' % (i))

    def solve(self, phi, rho, charge=None, maxcharge=1e-6,
              zero_initial_phi=False):
        self._init()
        return self._solve(phi, rho, charge, maxcharge,
                           zero_initial_phi)

    @timer('Solve')
    def _solve(self, phi, rho, charge=None, maxcharge=1e-6,
               zero_initial_phi=False):
        self._init()

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
            niter = self.poissonsolver.solve(phi, rho_neutral)
            self.timer.stop('Solve neutral')

            self.timer.start('Multipole moment corrections')
            # correct error introduced by removing multipoles
            for phi_cor in phi_cor_k:
                phi += phi_cor
            self.timer.stop('Multipole moment corrections')

            return niter
        else:
            return self.poissonsolver.solve(phi, rho, charge,
                                            maxcharge,
                                            zero_initial_phi)

    def estimate_memory(self, mem):
        self.poissonsolver.estimate_memory(mem)
        gdbytes = self.gd.bytecount()
        if self.moment_corrections is not None:
            mem.subnode('moment_corrections masks',
                        len(self.moment_corrections) * gdbytes)

    def __repr__(self):
        if self.moment_corrections is None or len(self.moment_corrections) == 0:
            corrections_str = 'no corrections'
        else:
            if len(self.moment_corrections) == 1:
                m = self.moment_corrections[0]
                corrections_str = f'{repr(m["moms"]) @ {repr(m["center"])}}'
            else:
                corrections_str = f'{len(self.moment_corrections)} corrections'

        representation = f'MomentCorrectionPoissonSolver ({corrections_str})'
        return representation

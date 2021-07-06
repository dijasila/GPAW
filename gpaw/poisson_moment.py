from typing import Any, Dict, Optional, List, Union

import numpy as np
from ase.units import Bohr
from ase.utils.timing import Timer
from gpaw.poisson import _PoissonSolver, create_poisson_solver
from gpaw.utilities.gauss import Gaussian
from gpaw.utilities.timing import nulltimer, NullTimer

from ase.utils.timing import timer


def dict_sanitize(dict_in: Dict[str, Any]) -> Dict[str, Any]:
    """ sanitize the moment correction dictionary """

    center = dict_in['center']
    if center is not None:
        center = np.asarray(center)

    dict_out = dict(moms=np.asarray(dict_in['moms']), center=center)

    return dict_out


def dict_A_to_au(dict_in: Dict[str, Any]) -> Dict[str, Any]:
    """ convert a moment correction dictionary from Ångström to units of Bohr """

    center = dict_in['center']
    if center is not None:
        center = np.asarray(center) / Bohr

    dict_out = dict(moms=dict_in['moms'], center=center)

    return dict_out


def dict_au_to_A(dict_in: Dict[str, Any]) -> Dict[str, Any]:
    """ convert a moment correction dictionary from units of Bohr to Ångström """

    center = dict_in['center']
    if center is not None:
        center = center * Bohr

    dict_out = dict(moms=dict_in['moms'], center=center)

    return dict_out


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
        self.poissonsolver = create_poisson_solver(poissonsolver)
        self.timer = timer

        gd = getattr(self.poissonsolver, 'gd', None)
        if gd is not None:
            self.gd = gd

        if moment_corrections is None:
            self.moment_corrections = None
        elif isinstance(moment_corrections, int):
            _moment_corrections = {'moms': range(moment_corrections), 'center': None}
            self.moment_corrections = [dict_sanitize(_moment_corrections)]
        elif isinstance(moment_corrections, list):
            assert all(['moms' in mom and 'center' in mom for mom in moment_corrections]), \
                   f'{self.__class__.__name__}: each element in moment_correction must be a dictionary' \
                   'with the keys "moms" and "center"'

            # Convert to Bohr units
            self.moment_corrections = [dict_A_to_au(dict_sanitize(mom)) for mom in moment_corrections]
        else:
            raise ValueError(f'{self.__class__.__name__}: moment_correction must be a list of dictionaries')

    def todict(self):
        d = {'name': 'MomentCorrectionPoissonSolver',
             'poissonsolver': self.poissonsolver.todict()}
        if self.moment_corrections:
            mom_corr = [dict_au_to_A(mom) for mom in self.moment_corrections]
            d['moment_corrections'] = mom_corr

        return d

    def set_grid_descriptor(self, gd):
        self.poissonsolver.set_grid_descriptor(gd)
        self.gd = gd

    def get_description(self) -> str:
        description = self.poissonsolver.get_description()

        lines = [description]

        if self.moment_corrections:
            lines.extend([f'    {len(self.moment_corrections)} moment corrections:'])
            lines.extend([f'      {self.describe_dict(mom)}' for mom in self.moment_corrections])

        return '\n'.join(lines)

    def describe_dict(self,
                      mom: Dict[str, Any]) -> str:
        if mom['center'] is None:
            center = 'center'
        else:
            center = ', '.join([f'{x:.2f}' for x in mom['center'] * Bohr])

        moms = mom['moms']
        if np.allclose(np.diff(moms), 1):
            # Increasing sequence
            moms = f'range({moms[0]}, {moms[-1]+1})'
        else:
            # List of integers
            _moms = ', '.join([f'{m:.0f}' for m in moms])
            moms = f'({_moms})'
        return f'[{center}] {moms}'

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

    def solve(self, phi, rho, **kwargs):
        self._init()
        return self._solve(phi, rho, **kwargs)

    @timer('Solve')
    def _solve(self, phi, rho, **kwargs):
        self._init()
        timer = kwargs.get('timer', None)
        if timer is None:
            timer = self.timer

        if self.moment_corrections:
            assert not self.gd.pbc_c.any()

            timer.start('Multipole moment corrections')

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

            timer.stop('Multipole moment corrections')

            timer.start('Solve neutral')
            niter = self.poissonsolver.solve(phi, rho_neutral)
            timer.stop('Solve neutral')

            timer.start('Multipole moment corrections')
            # correct error introduced by removing multipoles
            for phi_cor in phi_cor_k:
                phi += phi_cor
            timer.stop('Multipole moment corrections')

            return niter
        else:
            return self.poissonsolver.solve(phi, rho, **kwargs)

    def estimate_memory(self, mem):
        self.poissonsolver.estimate_memory(mem)
        gdbytes = self.gd.bytecount()
        if self.moment_corrections is not None:
            mem.subnode('moment_corrections masks',
                        len(self.moment_corrections) * gdbytes)

    def __repr__(self):
        if self.moment_corrections is None:
            corrections_str = 'no corrections'
        else:
            if len(self.moment_corrections) < 2:
                m = self.moment_corrections[0]
                corrections_str = f'{repr(m["moms"]) @ {repr(m["center"])}}'
            else:
                corrections_str = f'{len(self.moment_corrections)} corrections'

        representation = f'MomentCorrectionPoissonSolver ({corrections_str})'
        return representation

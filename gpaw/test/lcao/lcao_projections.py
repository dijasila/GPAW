from __future__ import print_function
import numpy as np
from ase.build import molecule

from gpaw import GPAW
from gpaw.poisson import FDPoissonSolver
from gpaw.lcao.projected_wannier import get_lcao_projections_HSP

atoms = molecule('C2H2')
atoms.center(vacuum=3.0)
calc = GPAW(gpts=(32, 32, 48),
            experimental={'niter_fixdensity': 2},
            poissonsolver=FDPoissonSolver(),
            eigensolver='rmm-diis')
atoms.set_calculator(calc)
atoms.get_potential_energy()

V_qnM, H_qMM, S_qMM, P_aqMi = get_lcao_projections_HSP(
    calc, bfs=None, spin=0, projectionsonly=False)


# Test H and S
eig = sorted(np.linalg.eigvals(np.linalg.solve(S_qMM[0], H_qMM[0])).real)
eig_ref = np.array([-17.879421874585983, -13.248808896902093,
                    -11.431277688074198, -7.125788588305689,
                    -7.125788588305654, 0.5927294174024342,
                    0.5927294174024489, 3.925079093706861,
                    7.450988940069014, 26.73430106000541])
print(eig)
assert np.allclose(eig, eig_ref)

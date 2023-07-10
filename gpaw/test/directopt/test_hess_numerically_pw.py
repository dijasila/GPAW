import pytest

from ase import Atoms
from gpaw import GPAW, PW
from gpaw.directmin.derivatives import Derivatives
from gpaw.directmin.fdpw.directmin import DirectMin
from gpaw.mom import prepare_mom_calculation
import numpy as np


@pytest.mark.do
def test_hess_numerically_lcao(in_tmp_dir):
    """
    Test complex numerical Hessian
    w.r.t rotation parameters in LCAO

    :param in_tmp_dir:
    :return:
    """

    atoms = Atoms('H', positions=[[0, 0, 0]])
    atoms.center(vacuum=5.0)
    atoms.set_pbc(False)

    calc = GPAW(xc='PBE',
                mode=PW(300, force_complex_dtype=False),
                h=0.25,
                convergence={'energy': np.inf,
                             'eigenstates': np.inf,
                             'density': np.inf,
                             'minimum iterations': 1},
                spinpol=False,
                eigensolver=DirectMin(convergelumo=True),
                occupations={'name': 'fixed-uniform'},
                mixer={'backend': 'no-mixing'},
                nbands=2,
                symmetry='off',
                )
    atoms.calc = calc
    atoms.get_potential_energy()

    calc.set(eigensolver=DirectMin(exstopt=True))
    f_sn = [calc.get_occupation_numbers(spin=s).copy() / 2
            for s in range(calc.wfs.nspins)]
    prepare_mom_calculation(calc, atoms, f_sn)
    atoms.get_potential_energy()

    numder = Derivatives(calc.wfs.eigensolver.iloop_outer, calc.wfs)

    hess_n = numder.get_numerical_derivatives(
        calc.wfs.eigensolver.iloop_outer,
        calc.hamiltonian,
        calc.wfs,
        calc.density,
        what2calc='hessian'
    )
    hess_a = numder.get_analytical_derivatives(
        calc.wfs.eigensolver.iloop_outer,
        calc.hamiltonian,
        calc.wfs,
        calc.density,
        what2calc='hessian'
    )
    # hess_nt = np.asarray([[1.32720630e+00, -1.93947467e-11],
    #                      [3.95786680e-09, 1.14599176e+00]])
    assert hess_a == pytest.approx(hess_n[0], abs=0.2)

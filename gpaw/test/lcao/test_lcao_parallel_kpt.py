import pytest
from ase.build import molecule

from gpaw import GPAW, LCAO, FermiDirac, KohnShamConvergenceError
from gpaw.utilities import compiled_with_sl
from gpaw.forces import calculate_forces
from gpaw.mpi import world

# Calculates energy and forces for various parallelizations

pytestmark = pytest.mark.skipif(world.size < 4,
                                reason='world.size < 4')


def test_lcao_lcao_parallel_kpt():
    tolerance = 4e-5

    parallel = dict()

    basekwargs = dict(mode=LCAO(atomic_correction='dense'),
                      maxiter=3,
                      nbands=6,
                      h=0.3,
                      kpts=(4, 4, 4),  # 8 kpts in the IBZ
                      parallel=parallel)

    Eref = None
    Fref_av = None

    def run(formula='H2O', vacuum=1.0, cell=None, pbc=1, **morekwargs):
        print(formula, parallel)
        system = molecule(formula)
        kwargs = dict(basekwargs)
        kwargs.update(morekwargs)
        calc = GPAW(**kwargs)
        system.calc = calc
        system.center(vacuum)
        if cell is None:
            system.center(vacuum)
        else:
            system.set_cell(cell)
        system.set_pbc(pbc)

        try:
            system.get_potential_energy()
        except KohnShamConvergenceError:
            pass

        E = calc.hamiltonian.e_total_free
        F_av = calculate_forces(calc.wfs, calc.density,
                                calc.hamiltonian)

        nonlocal Eref, Fref_av
        if Eref is None:
            Eref = E
            Fref_av = F_av

        eerr = abs(E - Eref)
        ferr = abs(F_av - Fref_av).max()

        if calc.wfs.world.rank == 0:
            print('Energy', E)
            print()
            print('Forces')
            print(F_av)
            print()
            print('Errs', eerr, ferr)

        if eerr > tolerance or ferr > tolerance:
            if eerr > tolerance:
                if calc.wfs.world.rank == 0:
                    print('Failed!')
                    print('E = %f, Eref = %f' % (E, Eref))
                msg = 'Energy err larger than tolerance: %f' % eerr
            if ferr > tolerance:
                if calc.wfs.world.rank == 0:
                    print('Failed!')
                    print('Forces:')
                    print(F_av)
                    print()
                    print('Ref forces:')
                    print(Fref_av)
                    print()
                msg = 'Force err larger than tolerance: %f' % ferr
            if calc.wfs.world.rank == 0:
                print()
                print('Args:')
                print(formula, vacuum, cell, pbc, morekwargs)
                print(parallel)
            raise AssertionError(msg)

    # reference:
    # kpt-parallelization = 8,
    # state-parallelization = 1,
    # domain-decomposition = (1,1,1)
    run()

    # kpt-parallelization = 2,
    # state-parallelization = 2,
    # domain-decomposition = (1,2,1)
    parallel['band'] = 2 if world.size >= 2 else 1
    parallel['domain'] = (1, 2 if world.size >= 4 else 1, 1)
    run()

    if compiled_with_sl():
        # kpt-parallelization = 2,
        # state-parallelization = 2,
        # domain-decomposition = (1,2,1)
        # with blacs
        parallel['sl_default'] = (2 if world.size >= 2 else 1,
                                  2 if world.size >= 4 else 1, 2)
        run()

    # perform spin polarization test
    parallel = dict()

    basekwargs = dict(mode=LCAO(atomic_correction='sparse'),
                      maxiter=3,
                      nbands=6,
                      kpts=(4, 1, 1),
                      parallel=parallel)

    Eref = None
    Fref_av = None

    OH_kwargs = dict(formula='NH2', vacuum=1.5, pbc=1, spinpol=1,
                     occupations=FermiDirac(width=0.1))

    # reference:
    # kpt-parallelization = 4,
    # spin-polarization = 2,
    # state-parallelization = 1,
    # domain-decomposition = (1, 1, 1)
    run(**OH_kwargs)

    # kpt-parallelization = 2,
    # spin-polarization = 2,
    # state-parallelization = 1,
    # domain-decomposition = (1, 2, 1)
    parallel['domain'] = (1, 2, 1)
    run(**OH_kwargs)

    # kpt-parallelization = 2,
    # spin-polarization = 2,
    # state-parallelization = 2,
    # domain-decomposition = (1, 1, 1)
    del parallel['domain']
    parallel['band'] = 2
    run(**OH_kwargs)  # test for forces is failing in this case!

    if compiled_with_sl():
        # kpt-parallelization = 2,
        # spin-polarization = 2,
        # state-parallelization = 2,
        # domain-decomposition = (1, 2, 1)
        # with blacs
        parallel['domain'] = (1, 2, 1)
        parallel['sl_default'] = (2, 1, 2)
        run(**OH_kwargs)

        # kpt-parallelization = 2,
        # spin-polarization = 2,
        # state-parallelization = 2,
        # domain-decomposition = (1, 2, 1)
        # with blacs
        parallel['sl_default'] = (2, 2, 2)
        run(**OH_kwargs)

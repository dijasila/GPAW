import warnings

from ase.units import Ha
from gpaw.directmin.tools import sort_orbitals_according_to_occ_kpt


def do_if_converged(eigensolver_name, wfs, ham, dens, log):
    if eigensolver_name == 'etdm':
        if hasattr(wfs.eigensolver, 'e_sic'):
            e_sic = wfs.eigensolver.e_sic
        else:
            e_sic = 0.0
        energy = ham.get_energy(
            0.0, wfs, kin_en_using_band=False, e_sic=e_sic)
        wfs.calculate_occupation_numbers(dens.fixed)
        wfs.eigensolver.get_canonical_representation(
            ham, wfs, dens, sort_eigenvalues=True)
        wfs.eigensolver.update_ks_energy(ham, wfs, dens)
        energy_converged = ham.get_energy(
            0.0, wfs, kin_en_using_band=False, e_sic=e_sic)
        energy_diff_after_scf = abs(energy - energy_converged) * Ha
        if energy_diff_after_scf > 1.0e-6:
            warnings.warn('Jump in energy of %f eV detected at the end of '
                          'SCF after getting canonical orbitals, SCF '
                          'might have converged to the wrong solution '
                          'or achieved energy convergence to the correct '
                          'solution above 1.0e-6 eV'
                          % (energy_diff_after_scf))

        log('\nOccupied states converged after'
            ' {:d} e/g evaluations'.format(wfs.eigensolver.eg_count))
    elif eigensolver_name == 'directmin':
        solver = wfs.eigensolver
        occ_name = getattr(wfs.occupations, 'name', None)
        if wfs.mode == 'fd' or wfs.mode == 'pw':
            solver.choose_optimal_orbitals(wfs)
            niter1 = solver.eg_count
            niter2 = 0
            niter3 = 0

            iloop1 = solver.iloop is not None
            iloop2 = solver.iloop_outer is not None
            if iloop1:
                niter2 = solver.total_eg_count_iloop
            if iloop2:
                niter3 = solver.total_eg_count_iloop_outer

            if iloop1 and iloop2:
                log(
                    '\nOccupied states converged after'
                    ' {:d} KS and {:d} SIC e/g '
                    'evaluations'.format(niter3,
                                         niter2 + niter3))
            elif not iloop1 and iloop2:
                log(
                    '\nOccupied states converged after'
                    ' {:d} e/g evaluations'.format(niter3))
            elif iloop1 and not iloop2:
                log(
                    '\nOccupied states converged after'
                    ' {:d} KS and {:d} SIC e/g '
                    'evaluations'.format(niter1, niter2))
            else:
                log(
                    '\nOccupied states converged after'
                    ' {:d} e/g evaluations'.format(niter1))
            if solver.converge_unocc:
                log('Converge unoccupied states:')
                max_er = wfs.eigensolver.error
                max_er *= Ha ** 2 / wfs.nvalence
                solver.run_unocc(ham, wfs, dens, max_er, log)
            else:
                solver.initialized = False
                log('Unoccupied states are not converged.')
            rewrite_psi = True
            sic_calc = 'SIC' in solver.func_settings['name']
            if sic_calc:
                rewrite_psi = False
            solver.get_canonical_representation(
                ham, wfs, dens, rewrite_psi)
            solver._e_entropy = \
                wfs.calculate_occupation_numbers(dens.fixed)
            if not sic_calc and occ_name:
                for kpt in wfs.kpt_u:
                    sort_orbitals_according_to_occ_kpt(wfs, kpt)
                solver._e_entropy = \
                    wfs.calculate_occupation_numbers(dens.fixed)
            solver.get_energy_and_tangent_gradients(
                ham, wfs, dens)


def check_eigensolver_state(eigensolver_name, wfs, ham, dens, log):

    solver = wfs.eigensolver
    name = eigensolver_name
    if name == 'etdm' or name == 'directmin':
        solver.eg_count = 0
        solver.globaliters = 0

        if hasattr(solver, 'iloop'):
            if solver.iloop is not None:
                solver.iloop.total_eg_count = 0
        if hasattr(solver, 'iloop_outer'):
            if solver.iloop_outer is not None:
                solver.iloop_outer.total_eg_count = 0

        wfs.eigensolver.check_assertions(wfs, dens)
        if name == 'etdm':
            if wfs.eigensolver.dm_helper is None:
                wfs.eigensolver.initialize_dm_helper(wfs, ham, dens, log)
        else:
            if not solver.initialized:
                solver.init_me(wfs, ham, dens, log)

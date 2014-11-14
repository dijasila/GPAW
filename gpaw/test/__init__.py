import os
import gc
import sys
import time
import signal
import traceback

import numpy as np

from gpaw.atom.generator import Generator
from gpaw.atom.configurations import parameters, tf_parameters
from gpaw.utilities import devnull, compiled_with_sl
from gpaw import setup_paths
from gpaw import mpi
import gpaw


def equal(x, y, tolerance=0, fail=True, msg=''):
    """Compare x and y."""

    if not np.isfinite(x - y).any() or (np.abs(x - y) > tolerance).any():
        msg = (msg + '%s != %s (error: |%s| > %.9g)' %
               (x, y, x - y, tolerance))
        if fail:
            raise AssertionError(msg)
        else:
            sys.stderr.write('WARNING: %s\n' % msg)


def findpeak(x, y):
    dx = x[1] - x[0]
    i = y.argmax()
    a, b, c = np.polyfit([-1, 0, 1], y[i - 1:i + 2], 2)
    assert a < 0
    x = -0.5 * b / a
    return dx * (i + x), a * x**2 + b * x + c

    
def gen(symbol, exx=False, name=None, **kwargs):
    if mpi.rank == 0:
        if 'scalarrel' not in kwargs:
            kwargs['scalarrel'] = True
        g = Generator(symbol, **kwargs)
        if 'orbital_free' in kwargs:
            g.run(exx=exx, name=name, use_restart_file=False,
                  **tf_parameters.get(symbol, {'rcut': 0.9}))
        else:
            g.run(exx=exx, name=name, use_restart_file=False,
                  **parameters[symbol])
    mpi.world.barrier()
    if setup_paths[0] != '.':
        setup_paths.insert(0, '.')


def wrap_pylab(names=[]):
    """Use Agg backend and prevent windows from popping up."""
    import matplotlib
    matplotlib.use('Agg')
    import pylab

    def show(names=names):
        if names:
            name = names.pop(0)
        else:
            name = 'fig.png'
        pylab.savefig(name)

    pylab.show = show


tests = [
    'gemm_complex.py',
    'ase3k_version.py',
    'kpt.py',
    'mpicomm.py',
    'numpy_core_multiarray_dot.py',
    'fileio/hdf5_noncontiguous.py',
    'cg2.py',
    'laplace.py',
    'lapack.py',
    'eigh.py',
    'parallel/submatrix_redist.py',
    'second_derivative.py',
    'parallel/parallel_eigh.py',
    'gp2.py',
    'blas.py',
    'Gauss.py',
    'nabla.py',
    'dot.py',
    'mmm.py',
    'lxc_fxc.py',
    'pbe_pw91.py',
    'gradient.py',
    'erf.py',
    'lf.py',
    'fsbt.py',
    'parallel/compare.py',
    'integral4.py',
    'zher.py',
    'gd.py',
    'pw/interpol.py',
    'screened_poisson.py',
    'xc.py',
    'XC2.py',
    'yukawa_radial.py',
    'dump_chi0.py',
    'vdw/potential.py',
    'lebedev.py',
    'fileio/hdf5_simple.py',
    'occupations.py',
    'derivatives.py',
    'parallel/realspace_blacs.py',
    'pw/reallfc.py',
    'parallel/pblas.py',
    'non_periodic.py',
    'spectrum.py',
    'pw/lfc.py',
    'gauss_func.py',
    'multipoletest.py',
    'noncollinear/xcgrid3d.py',
    'cluster.py',
    'poisson.py',
    'parallel/overlap.py',
    'parallel/scalapack.py',
    'gauss_wave.py',
    'transformations.py',
    'parallel/blacsdist.py',
    'ut_rsh.py',
    'pbc.py',
    'noncollinear/xccorr.py',
    'atoms_too_close.py',
    'harmonic.py',
    'proton.py',
    'timing.py',
    'parallel/ut_parallel.py',
    'ut_csh.py',
    'lcao_density.py',
    'parallel/hamiltonian.py',
    'pw/stresstest.py',
    'pw/fftmixer.py',
    'usesymm.py',
    'coulomb.py',
    'xcatom.py',
    'force_as_stop.py',
    'vdwradii.py',
    'ase3k.py',
    'numpy_zdotc_graphite.py',
    'eed.py',
    'gemv.py',
    'fileio/idiotproof_setup.py',
    'ylexpand.py',
    'keep_htpsit.py',
    'gga_atom.py',
    'hydrogen.py',
    'restart2.py',
    'aeatom.py',
    'plt.py',
    'ds_beta.py',
    'multipoleH2O.py',
    'noncollinear/h.py',
    'stdout.py',
    'lcao_largecellforce.py',
    'parallel/scalapack_diag_simple.py',
    'fixdensity.py',
    'pseudopotential/ah.py',
    'lcao_restart.py',
    'wfs_io.py',
    'lrtddft2.py',
    'fileio/file_reference.py',
    'cmrtest/cmr_test2.py',
    'restart.py',
    'broydenmixer.py',
    'pw/fulldiagk.py',
    'external_potential.py',
    'mixer.py',
    'parallel/lcao_projections.py',
    'lcao_h2o.py',
    'h2o_xas.py',
    'wfs_auto.py',
    'pw/fulldiag.py',
    'symmetry_ft.py',
    'aluminum_EELS_RPA.py',
    'ewald.py',
    'symmetry.py',
    'revPBE.py',
    'tf_mode_pbc.py',
    'tf_mode.py',
    'bee1.py',
    'nonselfconsistentLDA.py',
    'aluminum_EELS_ALDA.py',
    'spin_contamination.py',
    'inducedfield_lrtddft.py',
    'H_force.py',
    'usesymm2.py',
    'mgga_restart.py',
    'fixocc.py',
    'spinFe3plus.py',
    'fermisplit.py',
    'Cl_minus.py',
    'h2o_xas_recursion.py',
    'nonselfconsistent.py',
    'spinpol.py',
    'exx_acdf.py',
    'cg.py',
    'kptpar.py',
    'elf.py',
    'blocked_rmm_diis.py',
    'pw/slab.py',
    'si.py',
    'lcao_bsse.py',
    'parallel/lcao_hamiltonian.py',
    'degeneracy.py',
    'refine.py',
    'gemm.py',
    'al_chain.py',
    'fileio/parallel.py',
    'fixmom.py',
    'exx_unocc.py',
    'davidson.py',
    'aedensity.py',
    'pw/h.py',
    'apmb.py',
    'pseudopotential/hgh_h2o.py',
    'ed_wrapper.py',
    'pw/bulk.py',
    'ne_gllb.py',
    'ed.py',
    'lcao_force.py',
    'fileio/restart_density.py',
    'rpa_energy_Ni.py',
    'be_nltd_ip.py',
    'test_ibzqpt.py',
    'si_primitive.py',
    'inducedfield_td.py',
    'ehrenfest_nacl.py',
    'fd2lcao_restart.py',
    'gw_method.py',
    'constant_electric_field.py',
    'complex.py',
    'vdw/quick.py',
    'bse_aluminum.py',
    'Al2_lrtddft.py',
    'ralda_energy_N2.py',
    'gw_ppa.py',
    'parallel/lcao_complicated.py',
    'bulk.py',
    'scfsic_h2.py',
    'lcao_bulk.py',
    '2Al.py',
    'kssingles_Be.py',
    'relax.py',
    'pw/mgo_hybrids.py',
    'dscf_lcao.py',
    '8Si.py',
    'partitioning.py',
    'lxc_xcatom.py',
    'gllbatomic.py',
    'guc_force.py',
    'ralda_energy_Ni.py',
    'simple_stm.py',
    'ed_shapes.py',
    'restart_band_structure.py',
    'exx.py',
    'Hubbard_U.py',
    'rpa_energy_Si.py',
    'dipole.py',
    'IP_oxygen.py',
    'rpa_energy_Na.py',
    'parallel/fd_parallel.py',
    'parallel/lcao_parallel.py',
    'atomize.py',
    'excited_state.py',
    'ne_disc.py',
    'tpss.py',
    'td_na2.py',
    'exx_coarse.py',
    'pplda.py',
    'si_xas.py',
    'mgga_sc.py',
    'Hubbard_U_Zn.py',
    'lrtddft.py',
    'parallel/fd_parallel_kpt.py',
    'pw/hyb.py',
    'Cu.py',
    'response_na_plasmon.py',
    'bse_diamond.py',
    'fermilevel.py',
    'parallel/ut_hsblacs.py',
    'ralda_energy_H2.py',
    'diamond_absorption.py',
    'ralda_energy_Si.py',
    'ldos.py',
    'revPBE_Li.py',
    'parallel/lcao_parallel_kpt.py',
    'h2o_dks.py',
    'nsc_MGGA.py',
    'diamond_gllb.py',
    'MgO_exx_fd_vs_pw.py',
    'vdw/quick_spin.py',
    'bse_sym.py',
    'parallel/ut_hsops.py',
    'LDA_unstable.py',
    'au02_absorption.py',
    'wannierk.py',
    'bse_vs_lrtddft.py',
    'aluminum_testcell.py',
    'pygga.py',
    'ut_tddft.py',
    'rpa_energy_N2.py',
    'vdw/ar2.py',
    'parallel/diamond_gllb.py',
    'beefvdw.py',
    'pw/si_stress.py',
    'chi0.py',
    'scfsic_n2.py',
    'transport.py',
    'lrtddft3.py',
    'nonlocalset.py',
    'lb94.py',
    'AA_exx_enthalpy.py',
    'lcao_tdgllbsc.py',
    'bse_silicon.py',
    'gwsi.py',
    'maxrss.py',
    'pw/moleculecg.py',
    'potential.py',
    'pes.py',
    'lcao_pair_and_coulomb.py',
    'asewannier.py',
    'exx_q.py',
    'pw/davidson_pw.py',
    'neb.py',
    'diamond_eps.py',
    'wannier_ethylene.py',
    'muffintinpot.py',
    'nscfsic.py',
    'coreeig.py',
    'bse_MoS2_cut.py',
    'parallel/scalapack_mpirecv_crash.py',
    'cmrtest/cmr_test.py',
    'cmrtest/cmr_test3.py',
    'cmrtest/cmr_test4.py',
    'cmrtest/cmr_append.py',
    'cmrtest/Li2_atomize.py',
    ]

#'fractional_translations.py',
#'graphene_EELS.py', disabled while work is in progress on response code
#'mbeef.py',

#'fractional_translations_med.py',
#'fractional_translations_big.py',

#'eigh_perf.py', # Requires LAPACK 3.2.1 or later
# XXX https://trac.fysik.dtu.dk/projects/gpaw/ticket/230
#'parallel/scalapack_pdlasrt_hang.py',
#'dscf_forces.py',
#'stark_shift.py',


exclude = []

# not available on Windows
if os.name in ['ce', 'nt']:
    exclude += ['maxrss.py']

if mpi.size > 1:
    exclude += ['maxrss.py',
                'pes.py',
                'diamond_eps.py',
                'nscfsic.py',
                'coreeig.py',
                'asewannier.py',
                'wannier_ethylene.py',
                'muffintinpot.py',
                'stark_shift.py',
                'exx_q.py',
                'potential.py',
                #'cmrtest/cmr_test3.py',
                #'cmrtest/cmr_append.py',
                #'cmrtest/Li2_atomize.py',  # started to hang May 2014
                'lcao_pair_and_coulomb.py',
                'bse_MoS2_cut.py',
                'pw/moleculecg.py',
                'pw/davidson_pw.py',
                # scipy.weave fails often in parallel due to
                # ~/.python*_compiled
                # https://github.com/scipy/scipy/issues/1895
                'scipy_test.py']

if mpi.size > 2:
    exclude += ['neb.py']

if mpi.size < 4:
    exclude += ['parallel/fd_parallel.py',
                'parallel/lcao_parallel.py',
                'parallel/pblas.py',
                'parallel/scalapack.py',
                'parallel/scalapack_diag_simple.py',
                'parallel/realspace_blacs.py',
                'AA_exx_enthalpy.py',
                'bse_aluminum.py',
                'bse_diamond.py',
                'bse_silicon.py',
                'bse_vs_lrtddft.py',
                'fileio/parallel.py',
                'parallel/diamond_gllb.py',
                'parallel/lcao_parallel_kpt.py',
                'parallel/fd_parallel_kpt.py']


if mpi.size != 4:
    exclude += ['parallel/scalapack_mpirecv_crash.py']
    exclude += ['parallel/scalapack_pdlasrt_hang.py']

if mpi.size == 1 or not compiled_with_sl():
    exclude += ['parallel/submatrix_redist.py']

if mpi.size != 1 and not compiled_with_sl():
    exclude += ['ralda_energy_H2.py',
                'ralda_energy_N2.py',
                'ralda_energy_Ni.py',
                'ralda_energy_Si.py',
                'bse_sym.py',
                'bse_silicon.py',
                'gwsi.py',
                'rpa_energy_N2.py',
                'pw/fulldiag.py',
                'pw/fulldiagk.py',
                'au02_absorption.py']

if sys.version_info < (2, 6):
    exclude.append('transport.py')
    
if np.__version__ < '1.6.0':
    exclude.append('chi0.py')

exclude = set(exclude)
    
#for test in exclude:
#    if test in tests:
#        tests.remove(test)


class TestRunner:
    def __init__(self, tests, stream=sys.__stdout__, jobs=1,
                 show_output=False):
        if mpi.size > 1:
            assert jobs == 1
        self.jobs = jobs
        self.show_output = show_output
        self.tests = tests
        self.failed = []
        self.skipped = []
        self.garbage = []
        if mpi.rank == 0:
            self.log = stream
        else:
            self.log = devnull
        self.n = max([len(test) for test in tests])

    def run(self):
        self.log.write('=' * 77 + '\n')
        if not self.show_output:
            sys.stdout = devnull
        ntests = len(self.tests)
        t0 = time.time()
        if self.jobs == 1:
            self.run_single()
        else:
            # Run several processes using fork:
            self.run_forked()

        sys.stdout = sys.__stdout__
        self.log.write('=' * 77 + '\n')
        self.log.write('Ran %d tests out of %d in %.1f seconds\n' %
                       (ntests - len(self.tests) - len(self.skipped),
                        ntests, time.time() - t0))
        self.log.write('Tests skipped: %d\n' % len(self.skipped))
        if self.failed:
            self.log.write('Tests failed: %d\n' % len(self.failed))
        else:
            self.log.write('All tests passed!\n')
        self.log.write('=' * 77 + '\n')
        return self.failed

    def run_single(self):
        while self.tests:
            test = self.tests.pop(0)
            try:
                self.run_one(test)
            except KeyboardInterrupt:
                self.tests.append(test)
                break

    def run_forked(self):
        j = 0
        pids = {}
        while self.tests or j > 0:
            if self.tests and j < self.jobs:
                test = self.tests.pop(0)
                pid = os.fork()
                if pid == 0:
                    exitcode = self.run_one(test)
                    os._exit(exitcode)
                else:
                    j += 1
                    pids[pid] = test
            else:
                try:
                    while True:
                        pid, exitcode = os.wait()
                        if pid in pids:
                            break
                except KeyboardInterrupt:
                    for pid, test in pids.items():
                        os.kill(pid, signal.SIGHUP)
                        self.write_result(test, 'STOPPED', time.time())
                        self.tests.append(test)
                    break
                if exitcode == 512:
                    self.failed.append(pids[pid])
                elif exitcode == 256:
                    self.skipped.append(pids[pid])
                del pids[pid]
                j -= 1

    def run_one(self, test):
        exitcode_ok = 0
        exitcode_skip = 1
        exitcode_fail = 2

        if self.jobs == 1:
            self.log.write('%*s' % (-self.n, test))
            self.log.flush()

        t0 = time.time()
        filename = gpaw.__path__[0] + '/test/' + test

        failed = False
        skip = False

        if test in exclude:
            self.register_skipped(test, t0)
            return exitcode_skip

        try:
            loc = {}
            execfile(filename, loc)
            loc.clear()
            del loc
            self.check_garbage()
        except KeyboardInterrupt:
            self.write_result(test, 'STOPPED', t0)
            raise
        except ImportError, ex:
            module = ex.args[0].split()[-1].split('.')[0]
            if module in ['scipy', 'cmr', '_gpaw_hdf5']:
                skip = True
            else:
                failed = True
        except Exception:
            failed = True

        mpi.ibarrier(timeout=60.0)  # guard against parallel hangs

        me = np.array(failed)
        everybody = np.empty(mpi.size, bool)
        mpi.world.all_gather(me, everybody)
        failed = everybody.any()
        skip = mpi.world.sum(int(skip))

        if failed:
            self.fail(test, np.argwhere(everybody).ravel(), t0)
            exitcode = exitcode_fail
        elif skip:
            self.register_skipped(test, t0)
            exitcode = exitcode_skip
        else:
            self.write_result(test, 'OK', t0)
            exitcode = exitcode_ok

        return exitcode

    def register_skipped(self, test, t0):
        self.write_result(test, 'SKIPPED', t0)
        self.skipped.append(test)
    
    def check_garbage(self):
        gc.collect()
        n = len(gc.garbage)
        self.garbage += gc.garbage
        del gc.garbage[:]
        assert n == 0, ('Leak: Uncollectable garbage (%d object%s) %s' %
                        (n, 's'[:n > 1], self.garbage))

    def fail(self, test, ranks, t0):
        if mpi.rank in ranks:
            if sys.version_info >= (2, 4, 0, 'final', 0):
                tb = traceback.format_exc()
            else:  # Python 2.3! XXX
                tb = ''
                traceback.print_exc()
        else:
            tb = ''
        if mpi.size == 1:
            text = 'FAILED!\n%s\n%s%s' % ('#' * 77, tb, '#' * 77)
            self.write_result(test, text, t0)
        else:
            tbs = {tb: [0]}
            for r in range(1, mpi.size):
                if mpi.rank == r:
                    mpi.send_string(tb, 0)
                elif mpi.rank == 0:
                    tb = mpi.receive_string(r)
                    if tb in tbs:
                        tbs[tb].append(r)
                    else:
                        tbs[tb] = [r]
            if mpi.rank == 0:
                text = ('FAILED! (rank %s)\n%s' %
                        (','.join([str(r) for r in ranks]), '#' * 77))
                for tb, ranks in tbs.items():
                    if tb:
                        text += ('\nRANK %s:\n' %
                                 ','.join([str(r) for r in ranks]))
                        text += '%s%s' % (tb, '#' * 77)
                self.write_result(test, text, t0)

        self.failed.append(test)

    def write_result(self, test, text, t0):
        t = time.time() - t0
        if self.jobs > 1:
            self.log.write('%*s' % (-self.n, test))
        self.log.write('%10.3f  %s\n' % (t, text))


if __name__ == '__main__':
    TestRunner(tests).run()

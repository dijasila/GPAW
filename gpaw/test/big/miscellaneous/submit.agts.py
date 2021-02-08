"""Submit tests from the test suite that were removed because they were
too long."""

from myqueue.workflow import run


def workflow():
    return [
        task('H2Al110.py'),
        task('dscf_CO.py'),
        task('revtpss_tpss_scf.py'),
        task('ltt.py'),
        task('scalapack.py', cores=16),
        task('pblacs_oblong.py@64:5m')]

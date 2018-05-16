"""Submit tests from the test suite that were removed because they were
too long."""

from myqueue.job import Job


def workflow():
    return [
        task('H2Al110.py'),
        task('dscf_CO.py'),
        task('revtpss_tpss_scf.py'),
        task('ltt.py'),
        task('pblacs_oblong.py@64:5m')]

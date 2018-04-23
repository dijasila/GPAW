"""Submit tests from the test suite that were removed because they were
too long."""

from myqueue.job import Job


def workflow():
    return [
        Job('H2Al110.py'),
        Job('dscf_CO.py'),
        Job('revtpss_tpss_scf.py'),
        Job('ltt.py'),
        Job('pblacs_oblong.py@64x5s')]

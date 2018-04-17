from q2.job import Job


def workflow():
    return [
        Job('H2Al110.py'),
        Job('dscf_CO.py'),
        Job('revtpss_tpss_scf.py'),
        Job('ltt.py'),
        Job('pblacs_oblong.py@64x5s')]

# Submit tests from the test suite that were remove becase they were
# too long.
def agts(queue):
    queue.add('H2Al110.py')
    queue.add('dscf_CO.py')
    queue.add('revtpss_tpss_scf.py')
    queue.add('ltt.py')
    queue.add('pblacs_oblong.py', walltime=5, ncpus=64)

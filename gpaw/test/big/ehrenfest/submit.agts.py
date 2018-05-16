from myqueue.job import Job


def workflow():
    return [
        task('h2_osc.py@8:2h'),
        task('n2_osc.py@40:15h'),
        task('na2_md.py@8:2h'),
        task('na2_osc.py@8:40h')]

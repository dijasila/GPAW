from myqueue.job import Job


def workflow():
    return [
        Job('h2_osc.py@8x2h'),
        Job('n2_osc.py@40x15h'),
        Job('na2_md.py@8x2h'),
        Job('na2_osc.py@8x40h')]

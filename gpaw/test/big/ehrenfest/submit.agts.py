from myqueue.job import Job


def workflow():
    return [
        Job('h2_osc.py@8:2h'),
        Job('n2_osc.py@40:15h'),
        Job('na2_md.py@8:2h'),
        Job('na2_osc.py@8:40h')]

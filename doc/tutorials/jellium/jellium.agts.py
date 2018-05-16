from myqueue.job import Job


def workflow():
    return [
        Job('bulk.py@4:6m'),
        Job('surface.py@4:6m'),
        Job('sigma.py', deps=['bulk.py', 'surface.py']),
        Job('fig2.py', deps=['sigma.py'])]

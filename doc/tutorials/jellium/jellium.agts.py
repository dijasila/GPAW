from myqueue.job import Job


def workflow():
    return [
        task('bulk.py@4:6m'),
        task('surface.py@4:6m'),
        task('sigma.py', deps=['bulk.py', 'surface.py']),
        task('fig2.py', deps=['sigma.py'])]

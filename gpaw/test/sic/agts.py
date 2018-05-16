def workflow():
    from myqueue.task import task
    return [task('scfcis_n2.py@8:10m')]

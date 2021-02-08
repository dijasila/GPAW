def workflow():
    from myqueue.workflow import run
    return [task('scfsic_n2.py@8:10m')]

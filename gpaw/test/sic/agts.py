def workflow():
    from myqueue.job import Job
    return [Job('scfcis_n2.py@8:10m')]

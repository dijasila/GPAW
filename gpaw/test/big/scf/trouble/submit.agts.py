from myqueue.workflow import run


def workflow():
    runs = [run(script='run.py', args=[cores], cores=cores, tmax=tmax)
            for cores, tmax in [(1, '1h'),
                                (4, '5h'),
                                (8, '12h'),
                                (16, '10h')]]
    run(script='analyse.py', deps=runs)

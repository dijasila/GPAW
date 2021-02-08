from myqueue.workflow import run


def workflow():
    runs = [run(script='run.py+16', cores=16, tmax='10h'),
            run(script='run.py+8', cores=8, tmax='12h'),
            run(script='run.py+4', cores=4, tmax='5h'),
            run(script='run.py+1', tmax='1h')]
    run(script='analyse.py', deps=runs)

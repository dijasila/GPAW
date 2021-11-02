from myqueue.workflow import run


def workflow():
    with run(script='nio_pbe.py'):
        run(script='nio_gw.py', tmax='1d', cores=16)

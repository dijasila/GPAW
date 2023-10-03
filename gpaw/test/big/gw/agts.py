from myqueue.workflow import run


def workflow():
    with run(script='nio_pbe.py', cores=16):
        run(script='nio_gw.py', tmax='1h', cores=24)

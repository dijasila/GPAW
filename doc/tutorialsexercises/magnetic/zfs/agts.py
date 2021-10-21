from myqueue.workflow import run


def workflow():
    run(script='diamond_nv_minus.py', cores=16, tmax='4h')
    with run(script='biradical.py', cores=16, tmax='4h'):
        run(script='plot.py')

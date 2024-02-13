from myqueue.workflow import run


def workflow():
    with run(script='run-vary-potential.py', cores=72, tmax='4h'):
        run(script='plot-overview.py')
        run(script='plot-traces.py')
    with run(script='run-vary-charge.py', cores=72, tmax='4h'):
        run(script='plot-delta-ne-phi.py')
        run(script='plot-charge-potential.py')

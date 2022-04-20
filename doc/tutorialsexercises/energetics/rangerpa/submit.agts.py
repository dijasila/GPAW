# web-page: Ec_rpa.png
from myqueue.workflow import run


def workflow():
    with run(script='si.groundstate.py'):
        with run(script='si.range_rpa.py', cores=8, tmax='30m'):
            run(script='si.compare.py')
            run(script='plot_ec.py')

# Creates: Ec_rpa.png
from myqueue.workflow import run


def workflow():
    return [task('si.groundstate.py'),
            task('si.range_rpa.py@8:30m', deps='si.groundstate.py'),
            task('si.compare.py', deps='si.range_rpa.py'),
            task('plot_ec.py', deps='si.range_rpa.py')]

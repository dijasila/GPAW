from myqueue.workflow import run


def workflow():
    with run(script='shg_MoS2.py', cores=8):
        run(script='shg_plot.py')

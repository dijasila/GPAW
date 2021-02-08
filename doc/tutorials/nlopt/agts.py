from myqueue.workflow import run


def workflow():
    t1 = task('shg_MoS2.py', cores=8)
    t2 = task('shg_plot.py', deps=[t1])
    return [t1, t2]

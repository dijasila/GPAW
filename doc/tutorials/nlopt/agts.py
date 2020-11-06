from myqueue.task import task


def create_tasks():
    t1 = task('shg_MoS2.py', cores=8)
    t2 = task('shg_plot.py', deps=[t1])
    return [t1, t2]

from myqueue.task import task


def create_tasks():
    t1 = task('diamond_nv_minus.py', cores=16, tmax='4h')
    t2 = task('biradical.py', cores=8, tmax='4h')
    t3 = task('plot.py', deps=[t2])
    return [t1, t2, t3]

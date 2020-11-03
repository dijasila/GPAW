from myqueue.task import task


def create_tasks():
    t1 = task('shg_MoS2.py', cores=8)
    return [t1]

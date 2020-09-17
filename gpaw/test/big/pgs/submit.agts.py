from myqueue.task import task


def create_tasks():
    t1 = task('systems.py')
    t2 = task('analyse.py', tmax='1h', deps=t1)
    return [t1, t2]

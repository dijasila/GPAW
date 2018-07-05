from myqueue.task import task
def create_tasks():
    return [task('run.py', tmax='5h')]

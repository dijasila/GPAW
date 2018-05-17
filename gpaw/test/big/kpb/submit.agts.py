from myqueue.task import task


def create_tasks():
    return [
        task('molecules.py+0@1:3h'),
        task('molecules.py+1@1:3h'),
        task('check.py', deps='molecules.py+0,molecules.py+1')]

from myqueue.task import task


def create_tasks():
    return [
        task('basis.py@1:10m'),
        task('ag55.py@48:2h', deps='basis.py'),
        task('fig1.py', deps='ag55.py'),

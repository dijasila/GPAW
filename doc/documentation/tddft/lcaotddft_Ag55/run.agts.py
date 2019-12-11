from myqueue.task import task


def create_tasks():
    return [
        task('basis.py@1:10m'),
        task('gs.py@48:30m'),
        task('td.py@48:4h', deps='gs.py'),
        task('spec.py@1:1m', deps='td.py'),
        task('plot_spec.py@1:1m', deps='spec.py'),
        ]

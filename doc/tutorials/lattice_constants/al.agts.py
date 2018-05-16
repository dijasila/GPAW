from myqueue.task import task


def workflow():
    return [task('al.py@8:12h'),
            task('al_analysis.py', deps='al.py')]

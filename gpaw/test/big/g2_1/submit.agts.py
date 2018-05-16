from myqueue.job import Job


def workflow():
    jobs = [task('g21gpaw.py+{}@1:40h'.format(i)) for i in range(4)]
    return jobs + [task('analyse.py', deps=jobs)]

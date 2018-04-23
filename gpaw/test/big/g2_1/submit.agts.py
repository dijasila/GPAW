from myqueue.job import Job


def workflow():
    jobs = [Job('g21gpaw.py+{}@1x40m'.format(i)) for i in range(4)]
    return jobs + [Job('analyse.py', deps=jobs)]

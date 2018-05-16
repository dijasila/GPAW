from myqueue.job import Job


def workflow():
    return [Job('al.py@8:12h'),
            Job('al_analysis.py', deps=['al.py'])]

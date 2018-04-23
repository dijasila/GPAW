# Creates: CO_Au111.csv


def workflow():
    from myqueue.job import Job
    return [
        Job('CO_Au111.agts.py')]


if __name__ == '__main__':
    from ase.optimize.test.CO_Au111 import *

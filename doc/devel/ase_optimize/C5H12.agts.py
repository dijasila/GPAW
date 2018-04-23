# Creates: C5H12-gpaw.csv


def workflow():
    from myqueue.job import Job
    return [
        Job('C5H12.agts.py@8x25s')]


if __name__ == '__main__':
    from ase.optimize.test.C5H12 import *

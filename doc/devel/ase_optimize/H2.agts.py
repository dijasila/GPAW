# Creates: H2-emt.csv, H2-gpaw.csv


def workflow():
    from myqueue.job import Job
    return [
        Job('H2.agts.py@8x25s')]


if __name__ == '__main__':
    from ase.optimize.test.H2 import *

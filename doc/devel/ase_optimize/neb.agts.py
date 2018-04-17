# Creates: neb-emt.csv, neb-gpaw.csv


def workflow():
    from q2.job import Job
    return [
        Job('neb.agts.py@12x15m')]


if __name__ == '__main__':
    from ase.optimize.test.neb import *

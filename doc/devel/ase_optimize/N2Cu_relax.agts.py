# Creates: N2Cu-N2.csv, N2Cu-surf.csv


def workflow():
    from myqueue.job import Job
    return [
        Job('N2Cu_relax.agts.py')]


if __name__ == '__main__':
    from ase.optimize.test.N2Cu_relax import *

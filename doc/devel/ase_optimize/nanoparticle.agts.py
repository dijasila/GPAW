# Creates: nanoparticle.csv


def workflow():
    from q2.job import Job
    return [
        Job('nanoparticle.agts.py@8x2m')]


if __name__ == '__main__':
    from ase.optimize.test.nanoparticle import *

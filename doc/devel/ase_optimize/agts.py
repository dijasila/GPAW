# Creates: systems.db


def workflow():
    from myqueue.job import Job
    return [Job('agts.py'),
            Job('run_tests_emt.py', deps='agts.py'),
            Job('run_tests.py+1@8x1d', deps='agts.py'),
            Job('run_tests.py+2@8x1d', deps='agts.py'),
            Job('analyze.py',
                deps=['run_tests_emt.py', 'run_tests.py+1', 'run_tests.py+2'])]


if __name__ == '__main__':
    from ase.optimize.test.systems import create_database
    create_database()  # creates systems.db

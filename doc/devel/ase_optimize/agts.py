# Creates: systems.db


def create_tasks():
    from myqueue.task import task
    return [task('agts.py'),
            task('run_tests_emt.py', deps='agts.py'),
            task('run_tests.py+1@8:1d', deps='agts.py'),
            task('run_tests.py+2@8:1d', deps='agts.py'),
            task('analyze.py',
                 deps=['run_tests_emt.py',
                       'run_tests.py+1', 'run_tests.py+2'])]


if __name__ == '__main__':
    from ase.optimize.test.systems import create_database
    create_database()  # creates systems.db

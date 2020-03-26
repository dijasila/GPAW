# Creates: systems.db
from ase.optimize.test.test import all_optimizers


def create_tasks():
    from myqueue.task import task
    tasks = [task('agts.py'),
             task('run_tests_emt.py', deps='agts.py')]
    deps = ['run_tests_emt.py']
    for name in all_optimizers:
        if name == 'Berny':
            continue
        tasks.append(task(f'run_tests.py+{name}@8:1d',
                          deps='agts.py'))
        deps.append(f'run_tests.py+{name}')
    tasks.append(task('analyze.py', deps=deps))
    return tasks


if __name__ == '__main__':
    from ase.optimize.test.systems import create_database
    create_database()  # creates systems.db

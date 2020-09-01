from myqueue.task import task


def create_tasks():
    t1 = task('cu_calc.py', cores=4)
    t2 = task('cu_plot.py', deps=t1)
    t3 = task('cu_agts.py', deps=t1)
    return [t1, t2, t3]


def check():
    from ase.io import read
    energies = []
    k = 20
    for name in ['ITM', 'TM', 'FD-0.05', 'MV-0.2']:
        e = read(f'Cu-{name}-{k}.txt').get_potential_energy()
        energies.append(e)
    assert max(energies) - min(energies) < 0.001


if __name__ == '__main__':
    check()

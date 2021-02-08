from myqueue.workflow import run


def workflow():
    t1 = task('cu_calc.py', cores=4, tmax='1h')
    t2 = task('cu_plot.py', deps=t1, creates='cu.png')
    t3 = task('cu_agts.py', deps=t1)
    return [t1, t2, t3]


def check():
    import numpy as np
    from ase.io import read
    energies = []
    k = 20
    for name in ['ITM', 'FD-0.05', 'MV-0.2']:
        e = read(f'Cu-{name}-{k}.txt').get_potential_energy()
        energies.append(e)
    # Extrapolate TM:
    e19 = read('Cu-TM-19.txt').get_potential_energy()
    e20 = read('Cu-TM-20.txt').get_potential_energy()
    e = np.polyval(np.polyfit([20**-2, 19**-2], [e20, e19], 1), 0)
    energies.append(e)
    assert max(energies) - min(energies) < 0.001


if __name__ == '__main__':
    check()

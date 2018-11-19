# Creates: planaraverages.png, energies.png
from myqueue.task import task


def create_tasks():
    tasks = [task('gaas.py+1@8:1h'),
             task('gaas.py+2@8:1h'),
             task('gaas.py+31@24:2h'),
             task('gaas.py+4@24:4h'),
             task('electrostatics.py@1:15m', deps=['gaas.py+1',
                                                   'gaas.py+2',
                                                   'gaas.py+3',
                                                   'gaas.py+4']),
             task('plot_potentials.py', deps='electrostatics'),
             task('plot_energies.py', deps='electrostatics')]
    return tasks

def agts(queue):
    queue.add('gaas.py 1', ncpus=8, walltime=1)
    queue.add('gaas.py 2', ncpus=16, walltime=2)
    queue.add('gaas.py 3', ncpus=24, walltime=3)
    gaas = queue.add('gaas.py 4', ncpus=24, walltime=6)
    elc = queue.add('electrostatics.py', deps=gaas)
    queue.add('plot_potentials.py', deps=elc, creates='planaraverages.png')
    queue.add('plot_energies.py', deps=elc, creates='energies.png')

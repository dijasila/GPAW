def agts(queue):
    queue.add('gaas.py 1', ncpus=8, walltime=1)
    queue.add('gaas.py 2', ncpus=16, walltime=12)
    queue.add('gaas.py 3', ncpus=24, walltime=24)
    gaas444 = queue.add('gaas.py 4', ncpus=32, walltime=60)
    fnv = queue.add('fnv.py', deps=gaas444)
    queue.add('compare.py', deps=fnv)
    queue.add('plot_potentials.py', deps=gaas, creates='planaraverages.png')
    queue.add('plot_energies.py', deps=gaas, creates='energies.png')

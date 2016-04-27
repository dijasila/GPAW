def agts(queue):
    gaas = queue.add('gaas222.py', ncpus=8, walltime=60)
    queue.add('compare.py', deps=gaas)
    queue.add('plot_potentials.py', deps=gaas, creates='planaraverages.png')

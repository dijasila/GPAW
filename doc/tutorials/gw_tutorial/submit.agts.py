def agts(queue):
    conv = queue.add('convergence.py', ncpus=4, walltime=10*60)
    queue.add('plot_convergence.py', deps=conv, creates=['Si_EXX.png', 'Si_GW.png'])
    freq = queue.add('frequency.py', ncpus=4, walltime=10*60)
    queue.add('plot_frequency.py', deps=freq, creates='Si_w.png')

def agts(queue):
    gs_N2 = queue.add('gs_N2.py', ncpus=1, walltime=200)
    w = queue.add('frequency.py', deps=gs_N2, walltime=200)
    f = queue.add('con_freq.py', deps=gs_N2, walltime=200)
    rpa_N2 = queue.add('rpa_N2.py', deps=gs_N2, ncpus=16, walltime=10 * 60)
    queue.add('plot_w.py', deps=[w, f], creates='integration.png')
    queue.add('extrapolate.py', deps=rpa_N2, creates='extrapolate.png')

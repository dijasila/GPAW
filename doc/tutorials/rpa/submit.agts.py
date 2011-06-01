def agts(queue):
    gs_N2 = queue.add('gs_N2.py', ncpus=8, walltime=47 * 60)
    w = queue.add('frequency.py', deps=gs_N2, walltime=25)
    queue.add('N8_B2.0.py', deps=gs_N2, walltime=10)
    queue.add('N16_B1.2.py', deps=gs_N2, walltime=10)
    queue.add('N16_B2.0.py', deps=gs_N2, walltime=10)
    rpa = queue.add('rpa_N2.py', deps=gs_N2, ncpus=16, walltime=47 * 60)
    queue.add('plot_w.py', deps=w, creates='integration.png')
    queue.add('extrapolate.py', deps=rpa, creates='extrapolate.png')

    gs_Kr = queue.add('gs_Kr.py', ncpus=8, walltime=120)
    rpa = queue.add('rpa_Kr.py', deps=gs_Kr, ncpus=8, walltime=47 * 60)
    queue.add('hf_Kr.py', deps=gs_Kr, ncpus=8, walltime=300)
    queue.add('extrapolate_Kr.py', deps=rpa, creates='extrapolate_Kr.png')

def agts(queue):
    gs = queue.add('gs.py', ncpus=8, walltime=10)
    td0 = queue.add('td.py', deps=[gs], ncpus=8, walltime=30)
    td = queue.add('tdc.py', deps=[td0], ncpus=8, walltime=30)
    queue.add('td_replay.py', deps=[td], ncpus=8, walltime=30)
    spec = queue.add('spectrum.py', deps=[td], ncpus=1, walltime=2)
    fdm = queue.add('td_fdm_replay.py', deps=[td], ncpus=1, walltime=5)
    ksd = queue.add('ksd_init.py', deps=[gs], ncpus=1, walltime=5)
    queue.add('tcm_plot.py', deps=[ksd, fdm, spec], ncpus=1, walltime=2,
              creates=['tcm_1.12.png', 'tcm_2.48.png'])
    ind = queue.add('ksd_ind.py', deps=[ksd, fdm], ncpus=1, walltime=2)
    queue.add('ind_plot.py', deps=[ind], ncpus=1, walltime=2,
              creates=['ind_1.12.png', 'ind_2.48.png'])

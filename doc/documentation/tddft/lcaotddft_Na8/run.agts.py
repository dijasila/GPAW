from q2.job import Job


def workflow():
    return [
        Job('gs.py@8x10s'),
        Job('td.py@8x30s', deps=['gs.py']),
        Job('tdc.py@8x30s', deps=['td.py']),
        Job('td_replay.py@8x30s', deps=['tdc.py']),
        Job('spectrum.py@1x2s', deps=['tdc.py']),
        Job('td_fdm_replay.py@1x5s', deps=['tdc.py']),
        Job('ksd_init.py@1x5s', deps=['gs.py']),
        Job('fdm_ind.py@1x2s', deps=['td_fdm_replay.py']),
        Job('spec_plot.py@1x2s', deps=['spectrum.py']),
        Job('tcm_plot.py@1x2s', deps=['ksd_init.py', 'td_fdm_replay.py', 'spectrum.py']),
        Job('ind_plot.py@1x2s', deps=['fdm_ind.py'])]

def agts(queue):
    gs = queue.add('gs.py', ncpus=8, walltime=10)
    td0 = queue.add('td.py', deps=[gs], ncpus=8, walltime=30)
    td = queue.add('tdc.py', deps=[td0], ncpus=8, walltime=30)
    queue.add('td_replay.py', deps=[td], ncpus=8, walltime=30)
    spec = queue.add('spectrum.py', deps=[td], ncpus=1, walltime=2)
    fdm = queue.add('td_fdm_replay.py', deps=[td], ncpus=1, walltime=5)
    ksd = queue.add('ksd_init.py', deps=[gs], ncpus=1, walltime=5)
    ind = queue.add('fdm_ind.py', deps=[fdm], ncpus=1, walltime=2)
    queue.add('spec_plot.py', deps=[spec], ncpus=1, walltime=2,
              creates=['spec.png'])
    queue.add('tcm_plot.py', deps=[ksd, fdm, spec], ncpus=1, walltime=2,
              creates=['tcm_1.12.png', 'tcm_2.48.png',
                       'table_1.12.txt', 'table_2.48.txt'])
    queue.add('ind_plot.py', deps=[ind], ncpus=1, walltime=2,
              creates=['ind_1.12.png', 'ind_2.48.png'])

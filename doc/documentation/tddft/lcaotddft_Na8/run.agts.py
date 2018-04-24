from myqueue.job import Job


def workflow():
    return [
        Job('gs.py@8x10m'),
        Job('td.py@8x30m', deps=['gs.py']),
        Job('tdc.py@8x30m', deps=['td.py']),
        Job('td_replay.py@8x30m', deps=['tdc.py']),
        Job('spectrum.py@1x2m', deps=['tdc.py']),
        Job('td_fdm_replay.py@1x5m', deps=['tdc.py']),
        Job('ksd_init.py@1x5m', deps=['gs.py']),
        Job('fdm_ind.py@1x2m', deps=['td_fdm_replay.py']),
        Job('spec_plot.py@1x2m', deps=['spectrum.py']),
        Job('tcm_plot.py@1x2m',
            deps=['ksd_init.py', 'td_fdm_replay.py', 'spectrum.py']),
        Job('ind_plot.py@1x2m', deps=['fdm_ind.py'])]

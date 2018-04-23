from myqueue.job import Job


def workflow():
    return [
        Job('gs_N2.py@8x30s'),
        Job('frequency.py@1x3m', deps=['gs_N2.py']),
        Job('con_freq.py@2x16m', deps=['gs_N2.py']),
        Job('rpa_N2.py@32x20m', deps=['gs_N2.py']),
        Job('plot_w.py', deps=['frequency.py', 'con_freq.py']),
        Job('plot_con_freq.py', deps=['con_freq.py']),
        Job('extrapolate.py', deps=['rpa_N2.py'])]

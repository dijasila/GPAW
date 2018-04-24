from myqueue.job import Job


def workflow():
    return [
        Job('gs_N2.py@8x30m'),
        Job('frequency.py@1x3h', deps=['gs_N2.py']),
        Job('con_freq.py@2x16h', deps=['gs_N2.py']),
        Job('rpa_N2.py@32x20h', deps=['gs_N2.py']),
        Job('plot_w.py', deps=['frequency.py', 'con_freq.py']),
        Job('plot_con_freq.py', deps=['con_freq.py']),
        Job('extrapolate.py', deps=['rpa_N2.py'])]

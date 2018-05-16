from myqueue.job import Job


def workflow():
    return [
        task('gs_N2.py@8:30m'),
        task('frequency.py@1:3h', deps=['gs_N2.py']),
        task('con_freq.py@2:16h', deps=['gs_N2.py']),
        task('rpa_N2.py@32:20h', deps=['gs_N2.py']),
        task('plot_w.py', deps=['frequency.py', 'con_freq.py']),
        task('plot_con_freq.py', deps=['con_freq.py']),
        task('extrapolate.py', deps=['rpa_N2.py'])]

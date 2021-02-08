from myqueue.workflow import run


def workflow():
    return [task('gllbsc_si_simple.py@1:5m'),
            task('gllbsc_si_band_edges.py@1:5m')]

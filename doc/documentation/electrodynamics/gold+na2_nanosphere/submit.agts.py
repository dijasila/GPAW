from myqueue.job import Job


def workflow():
    return [
        Job('calculate.py@8:1h'),
        Job('plot_geom.py', deps=['calculate.py']),
        Job('plot.py', deps=['calculate.py'])]

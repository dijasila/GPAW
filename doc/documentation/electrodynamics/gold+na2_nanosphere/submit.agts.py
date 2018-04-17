from q2.job import Job


def workflow():
    return [
        Job('calculate.py@8x1m'),
        Job('plot_geom.py', deps=['calculate.py']),
        Job('plot.py', deps=['calculate.py'])]

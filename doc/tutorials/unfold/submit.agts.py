from myqueue.job import Job


def workflow():
    return [
        Job('gs_3x3_defect.py@16x5m'),
        Job('unfold_3x3_defect.py@16x10m', deps=['gs_3x3_defect.py']),
        Job('plot_sf.py', deps=['unfold_3x3_defect.py'])]

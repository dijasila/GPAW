from myqueue.job import Job


def workflow():
    return [
        Job('gs_3x3_defect.py@16x5s'),
        Job('unfold_3x3_defect.py@16x10s', deps=['gs_3x3_defect.py']),
        Job('plot_sf.py', deps=['unfold_3x3_defect.py'])]

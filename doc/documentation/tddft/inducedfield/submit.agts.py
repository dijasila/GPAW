from myqueue.job import Job


def workflow():
    return [
        Job('timepropagation_calculate.py@8x1h'),
        Job('timepropagation_continue.py@8x1h',
            deps=['timepropagation_calculate.py']),
        Job('timepropagation_postprocess.py@8x5m',
            deps=['timepropagation_continue.py']),
        Job('timepropagation_plot.py@1x5m',
            deps=['timepropagation_postprocess.py']),
        Job('casida_calculate.py@8x1h'),
        Job('casida_postprocess.py@8x5m', deps=['casida_calculate.py']),
        Job('casida_plot.py@1x5m', deps=['casida_postprocess.py'])]

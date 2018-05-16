from myqueue.job import Job


def workflow():
    return [
        Job('timepropagation_calculate.py@8:1h'),
        Job('timepropagation_continue.py@8:1h',
            deps=['timepropagation_calculate.py']),
        Job('timepropagation_postprocess.py@8:5m',
            deps=['timepropagation_continue.py']),
        Job('timepropagation_plot.py@1:5m',
            deps=['timepropagation_postprocess.py']),
        Job('casida_calculate.py@8:1h'),
        Job('casida_postprocess.py@8:5m', deps=['casida_calculate.py']),
        Job('casida_plot.py@1:5m', deps=['casida_postprocess.py'])]

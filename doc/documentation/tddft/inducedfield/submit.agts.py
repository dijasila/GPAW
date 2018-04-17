from q2.job import Job


def workflow():
    return [
        Job('timepropagation_calculate.py@8x1m'),
        Job('timepropagation_continue.py@8x1m',
            deps=['timepropagation_calculate.py']),
        Job('timepropagation_postprocess.py@8x5s',
            deps=['timepropagation_continue.py']),
        Job('timepropagation_plot.py@1x5s',
            deps=['timepropagation_postprocess.py']),
        Job('casida_calculate.py@8x1m'),
        Job('casida_postprocess.py@8x5s', deps=['casida_calculate.py']),
        Job('casida_plot.py@1x5s', deps=['casida_postprocess.py'])]

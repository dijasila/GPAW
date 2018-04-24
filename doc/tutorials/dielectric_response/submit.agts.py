from myqueue.job import Job


def workflow():
    return [
        Job('plot_freq.py'),
        Job('silicon_ABS_simpleversion.py'),
        Job('plot_silicon_ABS_simple.py',
            deps=['silicon_ABS_simpleversion.py']),
        Job('silicon_ABS.py@16x1h'),
        Job('plot_ABS.py', deps=['silicon_ABS.py']),
        Job('aluminum_EELS.py@8x1h'),
        Job('plot_aluminum_EELS_simple.py', deps=['aluminum_EELS.py']),
        Job('graphite_EELS.py@8x1h'),
        Job('plot_EELS.py', deps=['graphite_EELS.py']),
        Job('tas2_dielectric_function.py@8x15m'),
        Job('graphene_dielectric_function.py@8x15m')]

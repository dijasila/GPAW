from myqueue.job import Job


def workflow():
    return [
        Job('gs_Si.py@4x20m'),
        Job('eps_Si.py@4x6h', deps=['gs_Si.py']),
        Job('plot_Si.py@1x10m', deps=['eps_Si.py']),
        Job('gs_MoS2.py@4x1h'),
        Job('pol_MoS2.py@64x33h', deps=['gs_MoS2.py']),
        Job('plot_MoS2.py@1x10m', deps=['pol_MoS2.py']),
        Job('get_2d_eps.py@1x8h', deps=['gs_MoS2.py']),
        Job('plot_2d_eps.py@1x10m', deps=['get_2d_eps.py']),
        Job('alpha_MoS2.py@1x10m', deps=['gs_MoS2.py'])]

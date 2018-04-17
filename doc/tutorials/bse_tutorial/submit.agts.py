from q2.job import Job


def workflow():
    return [
        Job('gs_Si.py@4x20s'),
        Job('eps_Si.py@4x4m', deps=['gs_Si.py']),
        Job('plot_Si.py@1x10s', deps=['eps_Si.py']),
        Job('gs_MoS2.py@4x1m'),
        Job('pol_MoS2.py@64x33m', deps=['gs_MoS2.py']),
        Job('plot_MoS2.py@1x10s', deps=['pol_MoS2.py']),
        Job('get_2d_eps.py@1x8m', deps=['gs_MoS2.py']),
        Job('plot_2d_eps.py@1x10s', deps=['get_2d_eps.py']),
        Job('alpha_MoS2.py@1x10s', deps=['gs_MoS2.py'])]

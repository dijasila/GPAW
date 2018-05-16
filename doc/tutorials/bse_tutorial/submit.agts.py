from myqueue.job import Job


def workflow():
    return [
        Job('gs_Si.py@4:20m'),
        Job('eps_Si.py@4:6h', deps=['gs_Si.py']),
        Job('plot_Si.py@1:10m', deps=['eps_Si.py']),
        Job('gs_MoS2.py@4:1h'),
        Job('pol_MoS2.py@64:33h', deps=['gs_MoS2.py']),
        Job('plot_MoS2.py@1:10m', deps=['pol_MoS2.py']),
        Job('get_2d_eps.py@1:8h', deps=['gs_MoS2.py']),
        Job('plot_2d_eps.py@1:10m', deps=['get_2d_eps.py']),
        Job('alpha_MoS2.py@1:10m', deps=['gs_MoS2.py'])]
